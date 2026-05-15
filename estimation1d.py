# estimation1d.py

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# 1. Basic utilities
# ============================================================

def _as_1d_array(x):
    """
    Converts input data to a clean one-dimensional NumPy array.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)

    if arr.size == 0:
        raise ValueError("Input series is empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Input series contains NaN or infinite values.")

    return arr


def _make_odd(window):
    """
    Ensures that a rolling window length is odd.
    This makes centred rolling estimates easier to interpret.
    """
    window = int(window)

    if window < 3:
        raise ValueError("Window length must be at least 3.")

    if window % 2 == 0:
        window += 1

    return window


def _reflect_pad_1d(x, pad):
    """
    Reflect-pads a one-dimensional signal.

    For short series, NumPy reflect padding may fail if pad is too large,
    so we fall back to edge padding.
    """
    if pad == 0:
        return x

    mode = "reflect" if len(x) > pad else "edge"
    return np.pad(x, pad_width=pad, mode=mode)


# ============================================================
# 2. Trend estimation: Moving Median
# ============================================================

def moving_median(x, window=101):
    """
    Estimates the deterministic trend T(t) using a centred moving median.

    This corresponds to the robust trend extraction step in the
    one-dimensional estimation procedure.
    """
    x = _as_1d_array(x)
    window = _make_odd(window)
    pad = window // 2

    x_padded = _reflect_pad_1d(x, pad)
    windows = sliding_window_view(x_padded, window_shape=window)

    return np.median(windows, axis=1)


def remove_trend(S, trend_window=101):
    """
    Estimates and removes the deterministic component T(t).

    Returns
    -------
    R_hat : np.ndarray
        Estimated random component R(t) = S(t) - T_hat(t).
    T_hat : np.ndarray
        Estimated deterministic trend.
    """
    S = _as_1d_array(S)

    T_hat = moving_median(S, window=trend_window)
    R_hat = S - T_hat

    return R_hat, T_hat


# ============================================================
# 3. Robust scale estimator SC_x^Q
# ============================================================

def scq_scale(x, d=2.2219, order_fraction=0.20):
    """
    Robust scale estimator SC_x^Q used in the methodology.

    For a segment x = {x_1, ..., x_n}, it is based on the k-th order statistic
    of all pairwise absolute differences |x_i - x_j|, i < j.

    In the thesis methodology:
        d = 2.2219,
        k = binom(n, 2) / 5,
    so here order_fraction = 0.20.

    Parameters
    ----------
    x : array-like
        Data segment.
    d : float
        Multiplicative constant.
    order_fraction : float
        Fraction of pairwise distances used to define the order statistic.

    Returns
    -------
    float
        Robust scale estimate.
    """
    x = _as_1d_array(x)
    n = len(x)

    if n < 2:
        return np.nan

    diffs_matrix = np.abs(np.subtract.outer(x, x))
    diffs = diffs_matrix[np.triu_indices(n, k=1)]

    n_pairs = len(diffs)
    if n_pairs == 0:
        return np.nan

    k = int(np.floor(order_fraction * n_pairs))
    k = max(1, min(k, n_pairs))

    kth_statistic = np.partition(diffs, k - 1)[k - 1]

    return d * kth_statistic


def rolling_scq_scale(x, window=151, min_scale=1e-8):
    """
    Estimates the time-varying scale SC(t) by applying SC_x^Q
    in a centred moving window.

    Parameters
    ----------
    x : array-like
        Random component after trend removal.
    window : int
        Rolling window size.
    min_scale : float
        Lower bound preventing division by zero.

    Returns
    -------
    SC_hat : np.ndarray
        Estimated scale function.
    """
    x = _as_1d_array(x)
    window = _make_odd(window)
    pad = window // 2

    x_padded = _reflect_pad_1d(x, pad)
    windows = sliding_window_view(x_padded, window_shape=window)

    SC_hat = np.array([scq_scale(w) for w in windows], dtype=float)

    # Safety fixes for rare numerical edge cases
    if np.any(~np.isfinite(SC_hat)):
        finite_values = SC_hat[np.isfinite(SC_hat)]
        replacement = np.nanmedian(finite_values) if finite_values.size > 0 else 1.0
        SC_hat[~np.isfinite(SC_hat)] = replacement

    SC_hat = np.maximum(SC_hat, min_scale)

    return SC_hat


def normalize_random_component(R_hat, SC_hat, min_scale=1e-8):
    """
    Removes the time-varying scale:

        R2(t) = R(t) / SC(t).
    """
    R_hat = _as_1d_array(R_hat)
    SC_hat = _as_1d_array(SC_hat)

    if len(R_hat) != len(SC_hat):
        raise ValueError("R_hat and SC_hat must have the same length.")

    SC_safe = np.maximum(SC_hat, min_scale)

    return R_hat / SC_safe


# ============================================================
# 4. Robust ACF and ACVF
# ============================================================

def robust_acf(x, max_lags=30, center=True):
    """
    Robust autocorrelation function based on SC_x^Q.

    For lag h:
        u = first n-h observations,
        v = last n-h observations,

        rho_Q(h) =
        [SC_Q(u+v)^2 - SC_Q(u-v)^2] /
        [SC_Q(u+v)^2 + SC_Q(u-v)^2].

    Parameters
    ----------
    x : array-like
        One-dimensional time series.
    max_lags : int
        Maximum lag.
    center : bool
        If True, subtracts the median before computing ACF.

    Returns
    -------
    acf : np.ndarray
        Robust ACF values for lags 0, ..., max_lags.
    """
    x = _as_1d_array(x)

    if center:
        x = x - np.median(x)

    n = len(x)
    max_lags = int(max_lags)

    if max_lags >= n:
        raise ValueError("max_lags must be smaller than the length of the series.")

    acf = np.zeros(max_lags + 1, dtype=float)

    for h in range(max_lags + 1):
        u = x[:n - h]
        v = x[h:]

        sc_plus = scq_scale(u + v)
        sc_minus = scq_scale(u - v)

        denominator = sc_plus**2 + sc_minus**2

        if denominator <= 0 or not np.isfinite(denominator):
            acf[h] = 0.0
        else:
            acf[h] = (sc_plus**2 - sc_minus**2) / denominator

    # By definition, autocorrelation at lag 0 should be 1
    acf[0] = 1.0

    return acf


def robust_acvf(x, lag=0, center=True):
    """
    Robust autocovariance-like function based on SC_x^Q.

        gamma_Q(h) = 1/4 [SC_Q(u+v)^2 - SC_Q(u-v)^2].
    """
    x = _as_1d_array(x)

    if center:
        x = x - np.median(x)

    n = len(x)
    lag = int(lag)

    if lag < 0 or lag >= n:
        raise ValueError("lag must satisfy 0 <= lag < len(x).")

    u = x[:n - lag]
    v = x[lag:]

    sc_plus = scq_scale(u + v)
    sc_minus = scq_scale(u - v)

    return 0.25 * (sc_plus**2 - sc_minus**2)


# ============================================================
# 5. Robust Yule-Walker estimation for AR(p)
# ============================================================

def _acf_to_toeplitz(acf_values, p):
    """
    Builds the Yule-Walker Toeplitz matrix from ACF values.
    """
    R = np.empty((p, p), dtype=float)

    for i in range(p):
        for j in range(p):
            R[i, j] = acf_values[abs(i - j)]

    return R


def check_ar_stationarity(phi):
    """
    Checks the AR(p) stationarity condition.

    For the polynomial:
        phi(z) = 1 - phi_1 z - ... - phi_p z^p,
    all roots must lie outside the unit disk.
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)

    if len(phi) == 0:
        return True

    polynomial_coefficients = np.r_[-phi[::-1], 1.0]
    roots = np.roots(polynomial_coefficients)

    return np.all(np.abs(roots) > 1)


def estimate_ar_yule_walker_robust(x, p=1):
    """
    Estimates AR(p) coefficients using robust ACF and Yule-Walker equations.

    The series is centred by subtracting the median. An intercept is still returned,
    because in practice the estimated R2(t) may not be perfectly centred.

    Returns
    -------
    phi_hat : np.ndarray
        Estimated AR coefficients.
    intercept_hat : float
        Estimated intercept.
    info : dict
        Diagnostic information.
    """
    x = _as_1d_array(x)
    p = int(p)

    if p < 1:
        raise ValueError("p must be at least 1.")

    if len(x) <= p:
        raise ValueError("Series is too short for the requested AR order.")

    robust_location = np.median(x)
    x_centered = x - robust_location

    acf_values = robust_acf(x_centered, max_lags=p, center=False)

    R = _acf_to_toeplitz(acf_values, p)
    r = acf_values[1:p + 1]

    try:
        phi_hat = np.linalg.solve(R, r)
        used_pinv = False
    except np.linalg.LinAlgError:
        phi_hat = np.linalg.pinv(R) @ r
        used_pinv = True

    intercept_hat = robust_location * (1.0 - np.sum(phi_hat))

    info = {
        "robust_location": robust_location,
        "acf_used": acf_values,
        "used_pinv": used_pinv,
        "is_stationary": check_ar_stationarity(phi_hat),
    }

    return phi_hat, intercept_hat, info


def ar_residuals(x, phi, intercept=0.0):
    """
    Computes residuals from an AR(p) model:

        eps_t = x_t - intercept - phi_1 x_{t-1} - ... - phi_p x_{t-p}.

    Returns residuals starting from time index p.
    """
    x = _as_1d_array(x)
    phi = np.asarray(phi, dtype=float).reshape(-1)

    p = len(phi)

    if p == 0:
        return x - np.median(x)

    if len(x) <= p:
        raise ValueError("Series is too short for the AR order.")

    residuals = np.empty(len(x) - p, dtype=float)

    for t in range(p, len(x)):
        lagged_values = np.array([x[t - j - 1] for j in range(p)])
        residuals[t - p] = x[t] - intercept - np.dot(phi, lagged_values)

    return residuals


def select_ar_order_robust(
    x,
    p_max=5,
    h_max=30,
    include_zero_order=True,
    relative_tolerance=0.10,
):
    """
    Selects the AR order by minimizing:

        K_x(p) = max_{h=1,...,h_max} |rho_Q(h)|^2

    computed on residuals of the fitted AR(p) model.

    To avoid overfitting, we choose the smallest p whose K value is within
    relative_tolerance of the best K value.
    """
    x = _as_1d_array(x)
    p_max = int(p_max)
    h_max = int(h_max)

    if p_max < 0:
        raise ValueError("p_max must be non-negative.")

    candidates = range(0, p_max + 1) if include_zero_order else range(1, p_max + 1)

    results = []

    for p in candidates:
        try:
            if p == 0:
                residuals = x - np.median(x)
                phi_hat = np.array([], dtype=float)
                intercept_hat = np.median(x)
                ar_info = {
                    "is_stationary": True,
                    "used_pinv": False,
                    "robust_location": np.median(x),
                }
            else:
                phi_hat, intercept_hat, ar_info = estimate_ar_yule_walker_robust(x, p=p)
                residuals = ar_residuals(x, phi_hat, intercept=intercept_hat)

            max_lags_res = min(h_max, len(residuals) - 2)
            if max_lags_res < 1:
                raise ValueError("Residual series too short for ACF diagnostics.")

            acf_res = robust_acf(residuals, max_lags=max_lags_res, center=True)
            K = np.max(np.abs(acf_res[1:]) ** 2)

            results.append({
                "p": p,
                "K": K,
                "phi": phi_hat,
                "intercept": intercept_hat,
                "residuals": residuals,
                "acf_residuals": acf_res,
                "ar_info": ar_info,
            })

        except Exception as exc:
            results.append({
                "p": p,
                "K": np.inf,
                "phi": None,
                "intercept": None,
                "residuals": None,
                "acf_residuals": None,
                "ar_info": {"error": str(exc)},
            })

    finite_results = [item for item in results if np.isfinite(item["K"])]

    if len(finite_results) == 0:
        raise RuntimeError("AR order selection failed for all candidate orders.")

    min_K = min(item["K"] for item in finite_results)
    threshold = min_K * (1.0 + relative_tolerance)

    # Prefer the smallest p that is close enough to the best K value.
    acceptable_results = [item for item in finite_results if item["K"] <= threshold]
    best = min(acceptable_results, key=lambda item: item["p"])

    return best, results


# ============================================================
# 6. McCulloch alpha estimation
# ============================================================

def estimate_alpha_mcculloch(x, clip_to_table=True):
    """
    Estimates the stability index alpha using McCulloch's quantile method.

    The implementation uses the interpolation table for alpha as a function of:
        v_alpha = (x_0.95 - x_0.05) / (x_0.75 - x_0.25),
        v_beta  = (x_0.95 + x_0.05 - 2 x_0.50) / (x_0.95 - x_0.05).

    For the current project we mainly need alpha. The generated 1D noise is
    symmetric, so beta is not the main target here.
    """
    x = _as_1d_array(x)
    x = x - np.median(x)

    q05, q25, q50, q75, q95 = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])

    denom_alpha = q75 - q25
    denom_beta = q95 - q05

    if np.isclose(denom_alpha, 0.0) or np.isclose(denom_beta, 0.0):
        raise ValueError("Quantile denominators are too close to zero.")

    v_alpha_hat = (q95 - q05) / denom_alpha
    v_beta_hat = (q95 + q05 - 2.0 * q50) / denom_beta

    # McCulloch table used in the previous code.
    # Rows correspond to |v_beta| grid, columns to v_alpha grid.
    v_beta_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])

    v_alpha_grid = np.array([
        2.439, 2.5, 2.6, 2.7, 2.8,
        3.0, 3.2, 3.5, 4.0, 5.0,
        6.0, 8.0, 10.0, 15.0, 25.0
    ])

    alpha_values = np.array([
        [2.00, 1.916, 1.808, 1.729, 1.664, 1.563, 1.484, 1.391, 1.279, 1.128, 1.029, 0.896, 0.818, 0.698, 0.593],
        [2.00, 1.924, 1.813, 1.730, 1.663, 1.560, 1.480, 1.386, 1.273, 1.121, 1.021, 0.892, 0.812, 0.695, 0.590],
        [2.00, 1.924, 1.829, 1.737, 1.663, 1.553, 1.471, 1.378, 1.266, 1.114, 1.014, 0.887, 0.806, 0.692, 0.588],
        [2.00, 1.924, 1.829, 1.745, 1.668, 1.548, 1.460, 1.364, 1.250, 1.101, 1.004, 0.883, 0.801, 0.689, 0.586],
        [2.00, 1.924, 1.829, 1.745, 1.676, 1.547, 1.448, 1.337, 1.210, 1.067, 0.974, 0.855, 0.780, 0.676, 0.579],
        [2.00, 1.924, 1.829, 1.745, 1.676, 1.547, 1.438, 1.318, 1.184, 1.027, 0.935, 0.823, 0.756, 0.656, 0.563],
        [2.00, 1.924, 1.829, 1.745, 1.676, 1.547, 1.438, 1.318, 1.150, 0.973, 0.874, 0.769, 0.691, 0.595, 0.513],
    ])

    v_beta_abs = abs(v_beta_hat)

    clipped = False
    if clip_to_table:
        old_v_alpha = v_alpha_hat
        old_v_beta_abs = v_beta_abs

        v_alpha_hat = np.clip(v_alpha_hat, v_alpha_grid.min(), v_alpha_grid.max())
        v_beta_abs = np.clip(v_beta_abs, v_beta_grid.min(), v_beta_grid.max())

        clipped = (old_v_alpha != v_alpha_hat) or (old_v_beta_abs != v_beta_abs)

    interpolator = RegularGridInterpolator(
        (v_beta_grid, v_alpha_grid),
        alpha_values,
        bounds_error=False,
        fill_value=None,
    )

    alpha_hat = float(interpolator([[v_beta_abs, v_alpha_hat]])[0])

    diagnostics = {
        "q05": q05,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q95": q95,
        "v_alpha": v_alpha_hat,
        "v_beta": v_beta_hat,
        "v_beta_abs": v_beta_abs,
        "clipped_to_table": clipped,
    }

    return alpha_hat, diagnostics


# ============================================================
# 7. Full one-dimensional estimation pipeline
# ============================================================

def estimate_1d_model(
    S,
    trend_window=101,
    scale_window=151,
    p_max=5,
    h_max=30,
    include_zero_order=True,
    relative_tolerance=0.10,
):
    """
    Full estimation pipeline for the one-dimensional degradation model.

    Steps
    -----
    1. Estimate trend T(t) using moving median.
    2. Remove trend: R(t) = S(t) - T(t).
    3. Estimate time-varying scale SC(t) using robust SC_x^Q.
    4. Normalize: R2(t) = R(t) / SC(t).
    5. Fit AR(p) using robust Yule-Walker.
    6. Compute residuals R3(t).
    7. Estimate alpha using McCulloch's method.

    Returns
    -------
    result : dict
        Dictionary with estimated components and diagnostics.
    """
    S = _as_1d_array(S)

    # 1. Trend estimation and trend removal
    R_hat, T_hat = remove_trend(S, trend_window=trend_window)

    # 2. Scale estimation and normalization
    SC_hat = rolling_scq_scale(R_hat, window=scale_window)
    R2_hat = normalize_random_component(R_hat, SC_hat)

    # 3. Robust ACF of normalized component
    max_lags_R2 = min(h_max, len(R2_hat) - 2)
    acf_R2 = robust_acf(R2_hat, max_lags=max_lags_R2, center=True)

    # 4. AR order selection and residual computation
    best_ar, ar_candidates = select_ar_order_robust(
        R2_hat,
        p_max=p_max,
        h_max=h_max,
        include_zero_order=include_zero_order,
        relative_tolerance=relative_tolerance,
    )

    p_opt = best_ar["p"]
    phi_hat = best_ar["phi"]
    intercept_hat = best_ar["intercept"]
    residuals = best_ar["residuals"]
    acf_residuals = best_ar["acf_residuals"]

    if residuals is None:
        raise RuntimeError("AR order selection failed for all candidate orders.")

    # 5. McCulloch alpha estimation on residuals
    alpha_hat, alpha_diagnostics = estimate_alpha_mcculloch(residuals)

    return {
        "S": S,
        "T_hat": T_hat,
        "R_hat": R_hat,
        "SC_hat": SC_hat,
        "R2_hat": R2_hat,
        "p_opt": p_opt,
        "phi_hat": phi_hat,
        "intercept_hat": intercept_hat,
        "R3_hat": residuals,
        "alpha_hat": alpha_hat,
        "acf_R2": acf_R2,
        "acf_R3": acf_residuals,
        "ar_candidates": ar_candidates,
        "alpha_diagnostics": alpha_diagnostics,
        "settings": {
            "trend_window": trend_window,
            "scale_window": scale_window,
            "p_max": p_max,
            "h_max": h_max,
            "include_zero_order": include_zero_order,
        },
    }


# ============================================================
# 8. Optional helper for concise summary
# ============================================================

def summarize_1d_estimation(result):
    """
    Prints a concise summary of the one-dimensional estimation result.
    """
    print("One-dimensional estimation summary")
    print("---------------------------------")
    print(f"Estimated AR order p_opt: {result['p_opt']}")
    print(f"Estimated phi: {result['phi_hat']}")
    print(f"Estimated intercept: {result['intercept_hat']}")
    print(f"Estimated alpha: {result['alpha_hat']:.4f}")
    print()
    print("AR order candidates:")
    for item in result["ar_candidates"]:
        print(f"  p={item['p']}: K={item['K']:.6g}")