from pathlib import Path
import sys

import numpy as np
import pandas as pd

from estimation2d import compute_2d_acf_ccf_diagnostics
from simulation2d import simulate_subgaussian_sas_noise_2d


def parse_ims_timestamp(path):
    """
    Parses IMS file names such as ``2003.10.22.12.06.24`` as timestamps.
    """
    return pd.to_datetime(Path(path).name, format="%Y.%m.%d.%H.%M.%S", errors="coerce")


def _read_ims_segment(path):
    segment = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return segment.apply(pd.to_numeric, errors="coerce")


def load_ims_bearing1_rms(
    ims_folder=Path("ims/1st_test/1st_test"),
    cache_path=Path("ims_set1_bearing1_rms_hi.csv"),
    force_recompute=False,
):
    """
    Loads or computes the RMS health index for Bearing 1 from IMS Set No. 1.

    Bearing 1 is represented by channels 0 and 1. Each raw file corresponds
    to one one-second vibration segment, and one RMS value per channel becomes
    one health-index observation.
    """
    ims_folder = Path(ims_folder)
    cache_path = Path(cache_path)

    metadata = {
        "ims_folder": str(ims_folder),
        "cache_path": str(cache_path),
        "loaded_from": None,
        "raw_folder_exists": ims_folder.exists(),
        "raw_file_count": 0,
        "first_file_channels": None,
        "first_file_rows": None,
        "message": "",
    }

    raw_files = []
    if ims_folder.exists():
        raw_files = sorted(
            [path for path in ims_folder.iterdir() if path.is_file()],
            key=lambda path: (parse_ims_timestamp(path), path.name),
        )
        metadata["raw_file_count"] = len(raw_files)

        if raw_files:
            first_segment = _read_ims_segment(raw_files[0])
            metadata["first_file_channels"] = int(first_segment.shape[1])
            metadata["first_file_rows"] = int(first_segment.shape[0])

    if cache_path.exists() and not force_recompute:
        df = pd.read_csv(cache_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        metadata["loaded_from"] = "cache"
        metadata["message"] = "Loaded RMS health index from cache CSV."
        return df, metadata

    if not raw_files:
        metadata["loaded_from"] = "missing"
        metadata["message"] = (
            "IMS raw files were not found. Set IMS_FOLDER to the correct "
            "IMS Set No. 1 directory or provide the cache CSV."
        )
        return pd.DataFrame(columns=["t", "filename", "timestamp", "S1", "S2"]), metadata

    rows = []

    for t, path in enumerate(raw_files):
        segment = _read_ims_segment(path)

        if segment.shape[1] < 2:
            raise ValueError(f"IMS segment {path} has fewer than two channels.")

        selected = segment.iloc[: min(20480, len(segment)), :2].to_numpy(dtype=float)
        rms = np.sqrt(np.nanmean(selected**2, axis=0))

        rows.append({
            "t": t,
            "filename": path.name,
            "timestamp": parse_ims_timestamp(path),
            "S1": rms[0],
            "S2": rms[1],
        })

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)

    metadata["loaded_from"] = "raw_files"
    metadata["message"] = "Computed RMS health index from raw IMS files and saved cache CSV."

    return df, metadata


def prepare_trimmed_hi(hi_df, start_index=245):
    """
    Returns a two-dimensional HI array after chronological trimming.
    """
    if hi_df.empty:
        return np.empty((0, 2), dtype=float), pd.DataFrame()

    trimmed = hi_df.iloc[int(start_index):].copy().reset_index(drop=True)
    trimmed["t_trimmed"] = np.arange(1, len(trimmed) + 1)
    hi = trimmed[["S1", "S2"]].to_numpy(dtype=float)

    return hi, trimmed


def try_midast_change_detection(
    hi,
    search_roots=(Path("."), Path("references")),
    test_name="KSTest",
    window_size=100,
    shift=2,
    n_bkps=2,
    alpha=0.1,
    shift_group=5,
):
    """
    Tries to run the MIDAST ChangeDetector. If the package is unavailable,
    returns a structured diagnostic instead of failing the notebook.
    """
    searched_paths = []

    candidate_roots = [Path(root).resolve() for root in search_roots]
    for root in list(candidate_roots):
        if root.exists():
            for path in root.rglob("multivariate_statistical_test_method.py"):
                candidate_roots.append(path.parents[2])

    ChangeDetector = None
    import_error = None

    for root in [None] + candidate_roots:
        if root is not None:
            root_str = str(root)
            searched_paths.append(root_str)
            if root_str not in sys.path:
                sys.path.append(root_str)

        try:
            from MIDAST.src.multivariate_statistical_test_method import ChangeDetector as _ChangeDetector
            ChangeDetector = _ChangeDetector
            break
        except Exception as exc:
            import_error = exc

    if ChangeDetector is None:
        return {
            "available": False,
            "change_points_estimated": None,
            "results_df": None,
            "message": (
                "Full MIDAST implementation is not available in the current workspace. "
                "Therefore, reference regime-change points are used."
            ),
            "import_error": repr(import_error),
            "searched_paths": searched_paths,
        }

    detection_df = pd.DataFrame(
        np.asarray(hi, dtype=float),
        columns=["Zmienna_1_diff", "Zmienna_2_diff"],
    ).reset_index()

    detector = ChangeDetector(test_name=test_name)
    results_df = detector.fit(
        df=detection_df,
        window_size=window_size,
        shift=shift,
    )
    change_points = detector.analyze_results(
        results_df=results_df,
        alpha=alpha,
        max_no_changes=n_bkps,
        shift_group=shift_group,
    )

    return {
        "available": True,
        "change_points_estimated": list(change_points),
        "results_df": results_df,
        "message": "MIDAST ChangeDetector was available and executed.",
        "import_error": None,
        "searched_paths": searched_paths,
    }


def simulate_varp_paths_from_estimate(est, n_simulations=100, random_state=12345):
    """
    Simulates trajectories from an estimated VAR(p) model and transforms them
    back to the observed HI scale using estimated trend and scale functions.
    """
    rng = np.random.default_rng(random_state)

    Theta_hats = np.asarray(est["Theta_hats"], dtype=float)
    T_hat = np.asarray(est["T_hat"], dtype=float)
    SC_hat = np.asarray(est["SC_hat"], dtype=float)
    R3_hat = np.asarray(est["R3_hat"], dtype=float)

    p = Theta_hats.shape[0]
    n_obs = T_hat.shape[0]
    alpha_hat = float(est["alpha_hat"])

    Sigma_hat = est.get("Sigma_hat")
    if Sigma_hat is None:
        Sigma_hat = np.cov(R3_hat.T)

    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)

    min_eig = np.min(np.linalg.eigvalsh(Sigma_hat))
    if min_eig <= 1e-8:
        Sigma_hat = Sigma_hat + np.eye(2) * (1e-8 - min_eig)

    simulations = np.zeros((n_simulations, n_obs, 2), dtype=float)

    for sim_idx in range(n_simulations):
        noise = simulate_subgaussian_sas_noise_2d(
            N=n_obs,
            alpha=alpha_hat,
            Sigma=Sigma_hat,
            random_state=rng,
        )

        R2_sim = np.zeros((n_obs, 2), dtype=float)

        for t in range(n_obs):
            fitted = np.zeros(2, dtype=float)

            for lag in range(1, min(p, t) + 1):
                fitted += Theta_hats[lag - 1] @ R2_sim[t - lag]

            R2_sim[t] = fitted + noise[t]

        simulations[sim_idx] = T_hat + SC_hat * R2_sim

    return simulations


def quantile_band_coverage(S, q05, q95):
    """
    Computes componentwise coverage of observations inside 5%-95% bands.
    """
    S = np.asarray(S, dtype=float)
    q05 = np.asarray(q05, dtype=float)
    q95 = np.asarray(q95, dtype=float)

    rows = []
    for idx, component in enumerate(["S1", "S2"]):
        mask = np.isfinite(S[:, idx]) & np.isfinite(q05[:, idx]) & np.isfinite(q95[:, idx])
        inside = (S[mask, idx] >= q05[mask, idx]) & (S[mask, idx] <= q95[mask, idx])
        coverage = 100.0 * np.mean(inside) if len(inside) else np.nan

        rows.append({
            "component": component,
            "coverage_percent": coverage,
            "outside_percent": 100.0 - coverage if np.isfinite(coverage) else np.nan,
            "n_points": int(np.sum(mask)),
        })

    return pd.DataFrame(rows)


def moving_block_bootstrap_diagnostics_ci(
    X,
    max_lags=20,
    n_reps=50,
    alpha=0.05,
    block_length=None,
    random_state=12345,
):
    """
    Moving-block bootstrap confidence intervals for robust ACF/CCF diagnostics.

    This helper can be computationally expensive because each bootstrap sample
    recomputes robust ACF/CCF values. The notebook keeps it optional.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (N, 2).")

    rng = np.random.default_rng(random_state)
    n_obs = X.shape[0]

    if block_length is None:
        block_length = int(np.ceil(n_obs ** (1.0 / 3.0)))

    n_blocks = int(np.ceil(n_obs / block_length))
    keys = ["acf_1", "ccf_12", "ccf_21", "acf_2"]
    values = {key: [] for key in keys}

    for _ in range(int(n_reps)):
        starts = rng.integers(0, n_obs - block_length + 1, size=n_blocks)
        bootstrap_sample = np.concatenate([
            X[start:start + block_length]
            for start in starts
        ], axis=0)[:n_obs]

        diagnostics = compute_2d_acf_ccf_diagnostics(
            bootstrap_sample,
            max_lags=max_lags,
        )

        for key in keys:
            values[key].append(diagnostics[key])

    ci = {}
    for key in keys:
        arr = np.asarray(values[key])
        ci[key] = {
            "lower": np.quantile(arr, alpha / 2.0, axis=0),
            "upper": np.quantile(arr, 1.0 - alpha / 2.0, axis=0),
        }

    return ci
