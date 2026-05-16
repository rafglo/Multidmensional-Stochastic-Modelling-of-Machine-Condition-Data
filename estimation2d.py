# estimation_2d.py

import numpy as np

from estimation1d import (
    remove_trend,
    rolling_scq_scale,
    normalize_random_component,
    robust_acf,
    scq_scale,
    estimate_alpha_mcculloch,
)


# ============================================================
# 1. Basic utilities
# ============================================================

def _as_2d_array(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("Input must be a two-dimensional array.")

    if X.shape[1] != 2:
        raise ValueError("For this project, the signal must have shape (N, 2).")

    return X


# ============================================================
# 2. Trend and scale estimation componentwise
# ============================================================

def remove_trend_2d(S, trend_window=501):
    """
    Estimates and removes the trend component from a two-dimensional signal.

    Each component is treated separately using the one-dimensional moving median
    procedure.

    Returns
    -------
    R_hat : np.ndarray, shape (N, 2)
        Estimated random component.
    T_hat : np.ndarray, shape (N, 2)
        Estimated trend vector.
    """
    S = _as_2d_array(S)

    R_hat = np.zeros_like(S, dtype=float)
    T_hat = np.zeros_like(S, dtype=float)

    for i in range(2):
        R_i, T_i = remove_trend(
            S[:, i],
            trend_window=trend_window,
        )

        R_hat[:, i] = R_i
        T_hat[:, i] = T_i

    return R_hat, T_hat


def estimate_scale_2d(R_hat, scale_window=151, scale_multiplier=1.0):
    """
    Estimates the scale vector componentwise using the robust SC_Q estimator.

    scale_multiplier is a calibration factor used only in the 2D model.
    """
    R_hat = _as_2d_array(R_hat)

    SC_hat = np.zeros_like(R_hat, dtype=float)

    for i in range(2):
        SC_hat[:, i] = scale_multiplier * rolling_scq_scale(
            R_hat[:, i],
            window=scale_window,
        )

    return SC_hat


def normalize_random_component_2d(R_hat, SC_hat):
    """
    Normalizes the random component componentwise:

        R2_i(t) = R_i(t) / SC_i(t).
    """
    R_hat = _as_2d_array(R_hat)
    SC_hat = _as_2d_array(SC_hat)

    return R_hat / SC_hat


# ============================================================
# 3. Robust NCV and VAR(1) estimation
# ============================================================

def normalized_covariation_matrix(X, lag=0):
    """
    Estimates the normalized covariation matrix NCV(lag).

    For lag >= 0, the (i,j)-th entry is approximately

        NCV_{i,j}(lag)
        = sum_t X_i(t) sign(X_j(t-lag)) / sum_t |X_j(t-lag)|.

    This is a robust dependence measure used for heavy-tailed data.

    Parameters
    ----------
    X : np.ndarray, shape (N, 2)
        Stationary two-dimensional signal.
    lag : int
        Non-negative lag.

    Returns
    -------
    NCV : np.ndarray, shape (2, 2)
    """
    X = _as_2d_array(X)

    if lag < 0:
        raise ValueError("This implementation expects lag >= 0.")

    N = X.shape[0]

    if lag >= N:
        raise ValueError("lag must be smaller than the number of observations.")

    if lag == 0:
        current = X
        past = X
    else:
        current = X[lag:]
        past = X[:-lag]

    NCV = np.zeros((2, 2), dtype=float)

    for i in range(2):
        for j in range(2):
            numerator = np.sum(current[:, i] * np.sign(past[:, j]))
            denominator = np.sum(np.abs(past[:, j]))

            if denominator == 0:
                NCV[i, j] = 0.0
            else:
                NCV[i, j] = numerator / denominator

    return NCV


def estimate_var1_ncv(R2_hat):
    """
    Estimates the VAR(1) matrix using normalized covariation.

    For VAR(1):

        R2(t) = Theta R2(t-1) + R3(t),

    the NCV-Yule-Walker-type equation is

        NCV(1) = Theta NCV(0),

    hence:

        Theta_hat = NCV(1) NCV(0)^{-1}.
    """
    R2_hat = _as_2d_array(R2_hat)

    NCV0 = normalized_covariation_matrix(R2_hat, lag=0)
    NCV1 = normalized_covariation_matrix(R2_hat, lag=1)

    try:
        Theta_hat = NCV1 @ np.linalg.inv(NCV0)
        used_pinv = False
    except np.linalg.LinAlgError:
        Theta_hat = NCV1 @ np.linalg.pinv(NCV0)
        used_pinv = True

    info = {
        "NCV0": NCV0,
        "NCV1": NCV1,
        "used_pinv": used_pinv,
        "eigenvalues": np.linalg.eigvals(Theta_hat),
    }

    return Theta_hat, info


def var1_residuals(R2_hat, Theta_hat):
    """
    Computes VAR(1) residuals:

        R3_hat(t) = R2_hat(t) - Theta_hat R2_hat(t-1).

    Returns residuals of length N-1.
    """
    R2_hat = _as_2d_array(R2_hat)
    Theta_hat = np.asarray(Theta_hat, dtype=float)

    if Theta_hat.shape != (2, 2):
        raise ValueError("Theta_hat must be a 2x2 matrix.")

    N = R2_hat.shape[0]

    residuals = np.zeros((N - 1, 2), dtype=float)

    for t in range(1, N):
        fitted = Theta_hat @ R2_hat[t - 1]
        residuals[t - 1] = R2_hat[t] - fitted

    return residuals


# ============================================================
# 4. Robust CCF
# ============================================================

def robust_ccf(x, y, max_lags=30, center=True):
    """
    Robust cross-correlation function based on the SC_Q scale estimator.

    For lag h >= 0, it compares x[h:] with y[:-h].

    Returns
    -------
    ccf : np.ndarray, length max_lags+1
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if center:
        x = x - np.median(x)
        y = y - np.median(y)

    N = len(x)
    max_lags = min(max_lags, N - 2)

    ccf_values = np.zeros(max_lags + 1, dtype=float)

    for h in range(max_lags + 1):
        if h == 0:
            u = x
            v = y
        else:
            u = x[h:]
            v = y[:-h]

        sc_plus = scq_scale(u + v)
        sc_minus = scq_scale(u - v)

        denominator = sc_plus**2 + sc_minus**2

        if denominator == 0:
            ccf_values[h] = 0.0
        else:
            ccf_values[h] = (sc_plus**2 - sc_minus**2) / denominator

    return ccf_values


def compute_2d_acf_ccf_diagnostics(X, max_lags=30):
    """
    Computes robust ACF for each component and robust CCF between components.

    Returns:
    - acf_1
    - acf_2
    - ccf_12
    - ccf_21
    """
    X = _as_2d_array(X)

    max_lags = min(max_lags, X.shape[0] - 2)

    acf_1 = robust_acf(X[:, 0], max_lags=max_lags, center=True)
    acf_2 = robust_acf(X[:, 1], max_lags=max_lags, center=True)

    ccf_12 = robust_ccf(X[:, 0], X[:, 1], max_lags=max_lags, center=True)
    ccf_21 = robust_ccf(X[:, 1], X[:, 0], max_lags=max_lags, center=True)

    return {
        "acf_1": acf_1,
        "acf_2": acf_2,
        "ccf_12": ccf_12,
        "ccf_21": ccf_21,
    }


# ============================================================
# 5. Alpha estimation
# ============================================================

def estimate_alpha_2d_from_residuals(R3_hat):
    """
    Estimates alpha componentwise using McCulloch's method and then averages.

    Returns
    -------
    alpha_hat : float
        Mean of componentwise alpha estimates.
    component_estimates : dict
        Alpha estimates and diagnostics for each component.
    """
    R3_hat = _as_2d_array(R3_hat)

    alpha_1, diag_1 = estimate_alpha_mcculloch(R3_hat[:, 0])
    alpha_2, diag_2 = estimate_alpha_mcculloch(R3_hat[:, 1])

    alpha_hat = float(np.mean([alpha_1, alpha_2]))

    component_estimates = {
        "alpha_1": alpha_1,
        "alpha_2": alpha_2,
        "diagnostics_1": diag_1,
        "diagnostics_2": diag_2,
    }

    return alpha_hat, component_estimates


# ============================================================
# 6. Full 2D estimation pipeline
# ============================================================

def estimate_2d_model_fixed_var1(
    S,
    trend_window=501,
    scale_window=151,
    h_max=30,
    compute_diagnostics=True,
    scale_multiplier=1.0,
):
    """
    Full two-dimensional estimation pipeline with fixed VAR(1) order.

    Steps:
    1. estimate trend vector componentwise,
    2. remove trend,
    3. estimate scale vector componentwise,
    4. normalize random component,
    5. estimate VAR(1) matrix using robust NCV,
    6. compute residuals,
    7. estimate alpha from residuals,
    8. optionally compute ACF/CCF diagnostics.
    """
    S = _as_2d_array(S)

    # 1. Trend
    R_hat, T_hat = remove_trend_2d(
        S,
        trend_window=trend_window,
    )

    # 2. Scale
    SC_hat = estimate_scale_2d(
        R_hat,
        scale_window=scale_window,
        scale_multiplier=scale_multiplier,
    )

    # 3. Normalization
    R2_hat = normalize_random_component_2d(
        R_hat,
        SC_hat,
    )

    # 4. VAR(1)
    Theta_hat, var_info = estimate_var1_ncv(R2_hat)

    # 5. Residuals
    R3_hat = var1_residuals(
        R2_hat,
        Theta_hat,
    )

    # 6. Alpha
    alpha_hat, alpha_info = estimate_alpha_2d_from_residuals(R3_hat)

    # 7. Diagnostics
    if compute_diagnostics:
        diagnostics_R2 = compute_2d_acf_ccf_diagnostics(
            R2_hat,
            max_lags=h_max,
        )

        diagnostics_R3 = compute_2d_acf_ccf_diagnostics(
            R3_hat,
            max_lags=h_max,
        )
    else:
        diagnostics_R2 = None
        diagnostics_R3 = None

    return {
        "S": S,
        "T_hat": T_hat,
        "R_hat": R_hat,
        "SC_hat": SC_hat,
        "R2_hat": R2_hat,
        "Theta_hat": Theta_hat,
        "R3_hat": R3_hat,
        "alpha_hat": alpha_hat,
        "alpha_info": alpha_info,
        "var_info": var_info,
        "diagnostics_R2": diagnostics_R2,
        "diagnostics_R3": diagnostics_R3,
        "settings": {
            "trend_window": trend_window,
            "scale_window": scale_window,
            "h_max": h_max,
            "compute_diagnostics": compute_diagnostics,
            "var_order": 1,
            "scale_multiplier": scale_multiplier,
        },
    }


def summarize_2d_estimation(est):
    """
    Prints a compact summary of the 2D estimation result.
    """
    print("Two-dimensional estimation summary")
    print("----------------------------------")
    print("Estimated Theta:")
    print(est["Theta_hat"])
    print()
    print(f"Estimated alpha: {est['alpha_hat']:.4f}")
    print()
    print("Componentwise alpha:")
    print(f"  alpha_1: {est['alpha_info']['alpha_1']:.4f}")
    print(f"  alpha_2: {est['alpha_info']['alpha_2']:.4f}")