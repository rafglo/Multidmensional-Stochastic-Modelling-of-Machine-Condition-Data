import numpy as np
import pandas as pd

from estimation1d import estimate_1d_model_fixed_ar_order
from estimation2d import estimate_2d_model_fixed_var1
from simulation1d import simulate_1d_model
from simulation2d import simulate_2d_model


def run_mc_1d_for_alpha(
    alpha,
    base_params,
    n_replications=100,
    base_seed=10_000,
    trend_window=501,
    scale_window=151,
):
    """Runs the 1D simulation-estimation Monte Carlo loop for one alpha."""
    T_hats = []
    SC_hats = []
    phi_hats = []
    alpha_hats = []

    representative_sim = None
    representative_est = None

    for rep in range(n_replications):
        seed = base_seed + rep

        sim_rep = simulate_1d_model(
            **base_params,
            alpha=alpha,
            random_state=seed,
        )

        est_rep = estimate_1d_model_fixed_ar_order(
            sim_rep["S"],
            trend_window=trend_window,
            scale_window=scale_window,
            ar_order=1,
            h_max=30,
            compute_diagnostics=(rep == 0),
        )

        T_hats.append(est_rep["T_hat"])
        SC_hats.append(est_rep["SC_hat"])
        phi_hats.append(est_rep["phi_hat"][0])
        alpha_hats.append(est_rep["alpha_hat"])

        if rep == 0:
            representative_sim = sim_rep
            representative_est = est_rep

    return {
        "alpha": alpha,
        "t": representative_sim["t"],
        "true_T": representative_sim["T"],
        "true_SC": representative_sim["SC"],
        "T_hats": np.asarray(T_hats),
        "SC_hats": np.asarray(SC_hats),
        "phi_hats": np.asarray(phi_hats),
        "alpha_hats": np.asarray(alpha_hats),
        "representative_sim": representative_sim,
        "representative_est": representative_est,
    }


def run_mc_2d_for_alpha(
    alpha,
    base_params_2d,
    n_replications=100,
    base_seed=20_000,
    trend_window=501,
    scale_window=151,
    scale_multiplier=1.0,
    estimate_sigma=False,
):
    """Runs the 2D simulation-estimation Monte Carlo loop for one alpha."""
    T_hats = []
    SC_hats = []
    Theta_hats = []
    Sigma_hats = []
    alpha_hats = []
    alpha_1_hats = []
    alpha_2_hats = []

    representative_sim = None
    representative_est = None

    for rep in range(n_replications):
        seed = base_seed + rep

        sim_rep = simulate_2d_model(
            **base_params_2d,
            alpha=alpha,
            random_state=seed,
        )

        est_rep = estimate_2d_model_fixed_var1(
            sim_rep["S"],
            trend_window=trend_window,
            scale_window=scale_window,
            h_max=30,
            compute_diagnostics=(rep == 0),
            scale_multiplier=scale_multiplier,
            estimate_sigma=estimate_sigma,
            sigma_grid_size=9,
            sigma_grid_max=3.0,
            sigma_maxiter=300,
        )

        T_hats.append(est_rep["T_hat"])
        SC_hats.append(est_rep["SC_hat"])
        Theta_hats.append(est_rep["Theta_hat"])
        alpha_hats.append(est_rep["alpha_hat"])
        alpha_1_hats.append(est_rep["alpha_info"]["alpha_1"])
        alpha_2_hats.append(est_rep["alpha_info"]["alpha_2"])

        if estimate_sigma:
            Sigma_hats.append(est_rep["Sigma_hat"])

        if rep == 0:
            representative_sim = sim_rep
            representative_est = est_rep

    result = {
        "alpha": alpha,
        "t": representative_sim["t"],
        "true_T": representative_sim["T"],
        "true_SC": representative_sim["SC"],
        "true_Theta": representative_sim["Theta"],
        "true_Sigma": representative_sim["Sigma"],
        "T_hats": np.asarray(T_hats),
        "SC_hats": np.asarray(SC_hats),
        "Theta_hats": np.asarray(Theta_hats),
        "alpha_hats": np.asarray(alpha_hats),
        "alpha_1_hats": np.asarray(alpha_1_hats),
        "alpha_2_hats": np.asarray(alpha_2_hats),
        "representative_sim": representative_sim,
        "representative_est": representative_est,
    }

    if estimate_sigma:
        result["Sigma_hats"] = np.asarray(Sigma_hats)

    return result


def add_scale_adjusted_sigma(mc_2d_results, alphas_order, trim_edges=100):
    """
    Adds simulation-only scale-adjusted Sigma estimates to 2D Monte Carlo results.

    The true scale is available only in simulations. This diagnostic adjustment
    is therefore not used directly for real data.
    """
    for alpha in alphas_order:
        result = mc_2d_results[alpha]

        SC_hats = result["SC_hats"]
        true_SC = result["true_SC"]
        Sigma_hats = result["Sigma_hats"]

        ratios = SC_hats / true_SC[None, :, :]
        ratios_inner = ratios[:, trim_edges:-trim_edges, :] if trim_edges > 0 else ratios

        d1 = np.median(ratios_inner[:, :, 0])
        d2 = np.median(ratios_inner[:, :, 1])
        D = np.diag([d1, d2])

        result["Sigma_hats_adjusted"] = np.array([
            D @ Sigma_hat @ D
            for Sigma_hat in Sigma_hats
        ])
        result["sigma_scale_adjustment"] = {
            "d1": d1,
            "d2": d2,
            "D": D,
        }

    return mc_2d_results


def scale_ratio_2d_summary(mc_2d_results, alphas_order, trim_edges=100):
    """Summarizes the scale ratios used for simulation-only Sigma adjustment."""
    rows = []

    for alpha in alphas_order:
        result = mc_2d_results[alpha]

        SC_hats = result["SC_hats"]
        true_SC = result["true_SC"]
        ratios = SC_hats / true_SC[None, :, :]
        ratios_inner = ratios[:, trim_edges:-trim_edges, :] if trim_edges > 0 else ratios

        rows.append({
            "alpha": alpha,
            "mean_ratio_sc1": np.mean(ratios_inner[:, :, 0]),
            "median_ratio_sc1": np.median(ratios_inner[:, :, 0]),
            "mean_ratio_sc2": np.mean(ratios_inner[:, :, 1]),
            "median_ratio_sc2": np.median(ratios_inner[:, :, 1]),
            "mean_ratio_all": np.mean(ratios_inner),
            "median_ratio_all": np.median(ratios_inner),
        })

    return pd.DataFrame(rows)


def adjusted_sigma_summary(mc_2d_results, alphas_order, Sigma_true):
    """Builds the final adjusted Sigma summary table for the report."""
    rows = []

    for alpha in alphas_order:
        result = mc_2d_results[alpha]
        Sigma_hats_adj = result["Sigma_hats_adjusted"]

        rows.append({
            "true_alpha": alpha,
            "d1_scale_ratio": result["sigma_scale_adjustment"]["d1"],
            "d2_scale_ratio": result["sigma_scale_adjustment"]["d2"],
            "mean_sigma_11_adj": np.mean(Sigma_hats_adj[:, 0, 0]),
            "median_sigma_11_adj": np.median(Sigma_hats_adj[:, 0, 0]),
            "true_sigma_11": Sigma_true[0, 0],
            "mean_sigma_12_adj": np.mean(Sigma_hats_adj[:, 0, 1]),
            "median_sigma_12_adj": np.median(Sigma_hats_adj[:, 0, 1]),
            "true_sigma_12": Sigma_true[0, 1],
            "mean_sigma_21_adj": np.mean(Sigma_hats_adj[:, 1, 0]),
            "median_sigma_21_adj": np.median(Sigma_hats_adj[:, 1, 0]),
            "true_sigma_21": Sigma_true[1, 0],
            "mean_sigma_22_adj": np.mean(Sigma_hats_adj[:, 1, 1]),
            "median_sigma_22_adj": np.median(Sigma_hats_adj[:, 1, 1]),
            "true_sigma_22": Sigma_true[1, 1],
        })

    return pd.DataFrame(rows)
