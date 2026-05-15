# simulation1d.py

import numpy as np
from scipy.stats import levy_stable


def validate_1d_parameters(N, tau1, tau2, alpha, gamma):
    """
    Basic validation of one-dimensional model parameters.
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    if not (1 < tau1 < tau2 < N):
        raise ValueError("Regime change points must satisfy 1 < tau1 < tau2 < N.")

    if not (0 < alpha <= 2):
        raise ValueError("alpha must satisfy 0 < alpha <= 2.")

    if gamma <= 0:
        raise ValueError("gamma must be positive.")


def check_ar_stationarity(phi):
    """
    Checks stationarity condition for AR(p):
        phi(z) = 1 - phi_1 z - ... - phi_p z^p
    must have no roots in |z| <= 1.
    """
    phi = np.asarray(phi, dtype=float)

    if phi.ndim != 1 or len(phi) == 0:
        raise ValueError("phi must be a one-dimensional array of AR coefficients.")

    # Polynomial in z:
    # 1 - phi_1 z - ... - phi_p z^p
    # np.roots requires coefficients in descending powers:
    # -phi_p z^p - ... - phi_1 z + 1
    polynomial_coefficients = np.r_[-phi[::-1], 1.0]
    roots = np.roots(polynomial_coefficients)

    return np.all(np.abs(roots) > 1)


def compute_scale_parameters(N, tau1, tau2, sigma1, sigma2, sigma3, sigma4):
    """
    Computes parameters of the scale function SC(t):

    Regime 1: linear growth from sigma1 to sigma2
    Regime 2: linear growth from sigma2 to sigma3
    Regime 3: exponential growth from sigma3 to sigma4

    Conditions:
        SC(1) = sigma1
        SC(tau1) = sigma2
        SC(tau2) = sigma3
        SC(N) = sigma4
    """
    if min(sigma1, sigma2, sigma3, sigma4) <= 0:
        raise ValueError("All sigma values must be positive.")

    # First regime: a1 * t + b1
    a1 = (sigma2 - sigma1) / (tau1 - 1)
    b1 = sigma1 - a1 * 1

    # Second regime: a2 * t + b2
    a2 = (sigma3 - sigma2) / (tau2 - tau1)
    b2 = sigma2 - a2 * tau1

    # Third regime: a3 * exp(b3 * t)
    b3 = np.log(sigma4 / sigma3) / (N - tau2)
    a3 = sigma3 * np.exp(-b3 * tau2)

    return {
        "a1": a1,
        "b1": b1,
        "a2": a2,
        "b2": b2,
        "a3": a3,
        "b3": b3,
    }


def compute_trend_parameters(tau1, tau2, scale_params, c1):
    """
    Computes parameters of the trend function T(t):

    Regime 1: constant c1
    Regime 2: a2 * t + c2
    Regime 3: a3 * exp(b3 * t) + c3

    The parameters a2, a3, b3 are taken from the scale function,
    and c2, c3 are chosen to preserve continuity at tau1 and tau2.
    """
    a2 = scale_params["a2"]
    a3 = scale_params["a3"]
    b3 = scale_params["b3"]

    # Continuity at tau1:
    # c1 = a2 * tau1 + c2
    c2 = c1 - a2 * tau1

    # Value of the second-regime trend at tau2
    trend_tau2 = a2 * tau2 + c2

    # Continuity at tau2:
    # trend_tau2 = a3 * exp(b3 * tau2) + c3
    c3 = trend_tau2 - a3 * np.exp(b3 * tau2)

    return {
        "c1": c1,
        "a2": a2,
        "c2": c2,
        "a3": a3,
        "b3": b3,
        "c3": c3,
    }


def generate_scale(t, tau1, tau2, scale_params):
    """
    Generates values of the scale function SC(t).
    """
    t = np.asarray(t, dtype=float)
    SC = np.zeros_like(t, dtype=float)

    mask1 = (t > 0) & (t <= tau1)
    mask2 = (t > tau1) & (t <= tau2)
    mask3 = t > tau2

    SC[mask1] = scale_params["a1"] * t[mask1] + scale_params["b1"]
    SC[mask2] = scale_params["a2"] * t[mask2] + scale_params["b2"]
    SC[mask3] = scale_params["a3"] * np.exp(scale_params["b3"] * t[mask3])

    return SC


def generate_trend(t, tau1, tau2, trend_params):
    """
    Generates values of the trend function T(t).
    """
    t = np.asarray(t, dtype=float)
    T = np.zeros_like(t, dtype=float)

    mask1 = (t > 0) & (t <= tau1)
    mask2 = (t > tau1) & (t <= tau2)
    mask3 = t > tau2

    T[mask1] = trend_params["c1"]
    T[mask2] = trend_params["a2"] * t[mask2] + trend_params["c2"]
    T[mask3] = trend_params["a3"] * np.exp(trend_params["b3"] * t[mask3]) + trend_params["c3"]

    return T


def simulate_sas_noise(N, alpha, gamma=1.0, random_state=None):
    """
    Generates one-dimensional symmetric alpha-stable noise S_alpha(0, gamma, 0).

    In scipy's levy_stable distribution:
        alpha - stability parameter,
        beta=0 - symmetry,
        loc=0 - location,
        scale=gamma - scale.
    """
    rng = np.random.default_rng(random_state)

    return levy_stable.rvs(
        alpha=alpha,
        beta=0,
        loc=0,
        scale=gamma,
        size=N,
        random_state=rng,
    )


def simulate_ar_process(noise, phi, burn_in=0):
    """
    Simulates an AR(p) process:

        R2(t) = phi_1 R2(t-1) + ... + phi_p R2(t-p) + R3(t)

    Parameters
    ----------
    noise : array-like
        Noise sequence R3(t).
    phi : array-like
        AR coefficients [phi_1, ..., phi_p].
    burn_in : int
        Number of initial observations to discard.

    Returns
    -------
    R2 : np.ndarray
        Simulated AR(p) process after burn-in.
    """
    noise = np.asarray(noise, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p = len(phi)

    if not check_ar_stationarity(phi):
        raise ValueError("AR coefficients do not satisfy the stationarity condition.")

    total_N = len(noise)
    R2 = np.zeros(total_N, dtype=float)

    for t in range(p, total_N):
        lagged_values = np.array([R2[t - j - 1] for j in range(p)])
        R2[t] = np.dot(phi, lagged_values) + noise[t]

    if burn_in > 0:
        return R2[burn_in:]

    return R2


def simulate_1d_model(
    N=2000,
    tau1=None,
    tau2=None,
    phi=(0.5,),
    alpha=2.0,
    gamma=1.0,
    sigma1=1.0,
    sigma2=2.5,
    sigma3=5.0,
    sigma4=25.0,
    c1=10.0,
    burn_in=300,
    random_state=None,
):
    """
    Simulates data from the one-dimensional degradation model:

        S(t) = R(t) + T(t)
        R(t) = SC(t) * R2(t)

    where R2(t) is an AR(p) process with symmetric alpha-stable noise.

    Returns
    -------
    result : dict
        Dictionary containing:
        - t: time index
        - S: final signal
        - R: random component after scaling
        - R2: stationary AR(p) process
        - R3: alpha-stable noise
        - SC: scale function
        - T: trend function
        - scale_params
        - trend_params
        - params
    """
    if tau1 is None:
        tau1 = N / 3

    if tau2 is None:
        tau2 = 2 * N / 3

    validate_1d_parameters(N, tau1, tau2, alpha, gamma)

    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 1 or len(phi) == 0:
        raise ValueError("phi must be a one-dimensional array-like object.")

    # We generate more observations and discard burn-in from the AR process.
    total_N = N + burn_in

    R3_full = simulate_sas_noise(
        N=total_N,
        alpha=alpha,
        gamma=gamma,
        random_state=random_state,
    )

    R2_full = simulate_ar_process(
        noise=R3_full,
        phi=phi,
        burn_in=0,
    )

    R3 = R3_full[burn_in:]
    R2 = R2_full[burn_in:]

    t = np.arange(1, N + 1, dtype=float)

    scale_params = compute_scale_parameters(
        N=N,
        tau1=tau1,
        tau2=tau2,
        sigma1=sigma1,
        sigma2=sigma2,
        sigma3=sigma3,
        sigma4=sigma4,
    )

    trend_params = compute_trend_parameters(
        tau1=tau1,
        tau2=tau2,
        scale_params=scale_params,
        c1=c1,
    )

    SC = generate_scale(t, tau1, tau2, scale_params)
    T = generate_trend(t, tau1, tau2, trend_params)

    R = SC * R2
    S = R + T

    return {
        "t": t,
        "S": S,
        "R": R,
        "R2": R2,
        "R3": R3,
        "SC": SC,
        "T": T,
        "scale_params": scale_params,
        "trend_params": trend_params,
        "params": {
            "N": N,
            "tau1": tau1,
            "tau2": tau2,
            "phi": phi,
            "alpha": alpha,
            "gamma": gamma,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "sigma3": sigma3,
            "sigma4": sigma4,
            "c1": c1,
            "burn_in": burn_in,
        },
    }