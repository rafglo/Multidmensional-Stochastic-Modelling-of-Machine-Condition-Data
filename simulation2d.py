# simulation2d.py

import numpy as np


# ============================================================
# 1. Positive stable generator for sub-Gaussian construction
# ============================================================

def simulate_positive_stable_kanter(N, beta, random_state=None):
    """
    Generates positive beta-stable random variables using Kanter's algorithm.

    The generated variable A has Laplace transform:

        E exp(-s A) = exp(-s^beta),

    where 0 < beta <= 1.

    In the sub-Gaussian SαS construction we use beta = alpha / 2.
    For alpha = 2, beta = 1 and A is deterministic equal to 1.
    """
    if not (0 < beta <= 1):
        raise ValueError("beta must satisfy 0 < beta <= 1.")

    rng = np.random.default_rng(random_state)

    if np.isclose(beta, 1.0):
        return np.ones(N)

    U = rng.uniform(0.0, np.pi, size=N)
    W = rng.exponential(scale=1.0, size=N)

    A = (
        np.sin(beta * U) / (np.sin(U) ** (1.0 / beta))
        * (np.sin((1.0 - beta) * U) / W) ** ((1.0 - beta) / beta)
    )

    return A


def simulate_subgaussian_sas_noise_2d(N, alpha, Sigma, random_state=None):
    """
    Generates two-dimensional sub-Gaussian SαS noise.

    Construction:
        R3(t) = sqrt(A_t) * G_t,

    where:
        G_t ~ N(0, Sigma),
        A_t is positive alpha/2-stable.

    For alpha = 2, A_t = 1, so the noise is Gaussian.
    """
    if not (0 < alpha <= 2):
        raise ValueError("alpha must satisfy 0 < alpha <= 2.")

    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.shape != (2, 2):
        raise ValueError("For this project Sigma must be a 2x2 matrix.")

    rng = np.random.default_rng(random_state)

    beta = alpha / 2.0

    A = simulate_positive_stable_kanter(
        N=N,
        beta=beta,
        random_state=rng,
    )

    G = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=Sigma,
        size=N,
    )

    R3 = np.sqrt(A)[:, None] * G

    return R3


# ============================================================
# 2. VAR(1) simulation
# ============================================================

def simulate_var1(N, Theta, noise):
    """
    Simulates a VAR(1) process:

        R2(t) = Theta R2(t-1) + R3(t).

    Parameters
    ----------
    N : int
        Number of observations.
    Theta : array-like, shape (2, 2)
        Autoregression matrix.
    noise : array-like, shape (N, 2)
        Innovation vectors R3(t).

    Returns
    -------
    R2 : np.ndarray, shape (N, 2)
        Simulated VAR(1) process.
    """
    Theta = np.asarray(Theta, dtype=float)
    noise = np.asarray(noise, dtype=float)

    if Theta.shape != (2, 2):
        raise ValueError("Theta must be a 2x2 matrix.")

    if noise.shape != (N, 2):
        raise ValueError("noise must have shape (N, 2).")

    eigenvalues = np.linalg.eigvals(Theta)

    if np.any(np.abs(eigenvalues) >= 1):
        raise ValueError("VAR(1) is not stationary: eigenvalues must be inside the unit circle.")

    R2 = np.zeros((N, 2), dtype=float)

    for t in range(1, N):
        R2[t] = Theta @ R2[t - 1] + noise[t]

    return R2


# ============================================================
# 3. Trend and scale functions
# ============================================================

def _piecewise_scale_component(t, tau1, tau2, N, sigma1, sigma2, sigma3, sigma4):
    """
    Generates one component of the scale function:

    linear in regime 1,
    linear in regime 2,
    exponential in regime 3.
    """
    t = np.asarray(t, dtype=float)

    scale = np.zeros_like(t, dtype=float)

    # Regime 1: SC(1)=sigma1, SC(tau1)=sigma2
    a1 = (sigma2 - sigma1) / (tau1 - 1.0)
    b1 = sigma1 - a1 * 1.0

    # Regime 2: SC(tau1)=sigma2, SC(tau2)=sigma3
    a2 = (sigma3 - sigma2) / (tau2 - tau1)
    b2 = sigma2 - a2 * tau1

    # Regime 3: SC(tau2)=sigma3, SC(N)=sigma4
    b3 = np.log(sigma4 / sigma3) / (N - tau2)
    a3 = sigma3 * np.exp(-b3 * tau2)

    mask1 = (t > 0) & (t <= tau1)
    mask2 = (t > tau1) & (t <= tau2)
    mask3 = t > tau2

    scale[mask1] = a1 * t[mask1] + b1
    scale[mask2] = a2 * t[mask2] + b2
    scale[mask3] = a3 * np.exp(b3 * t[mask3])

    params = {
        "a1": a1,
        "b1": b1,
        "a2": a2,
        "b2": b2,
        "a3": a3,
        "b3": b3,
    }

    return scale, params


def _piecewise_trend_component(t, tau1, tau2, N, c1, scale_params):
    """
    Generates one component of the trend function.

    Trend:
    - constant in regime 1,
    - linear in regime 2,
    - exponential in regime 3,

    using the same a2, a3, b3 as the corresponding scale component,
    with vertical shifts chosen for continuity.
    """
    t = np.asarray(t, dtype=float)

    a2 = scale_params["a2"]
    a3 = scale_params["a3"]
    b3 = scale_params["b3"]

    trend = np.zeros_like(t, dtype=float)

    # Continuity at tau1:
    # c1 = a2*tau1 + c2
    c2 = c1 - a2 * tau1

    # Continuity at tau2:
    # a2*tau2 + c2 = a3*exp(b3*tau2) + c3
    c3 = a2 * tau2 + c2 - a3 * np.exp(b3 * tau2)

    mask1 = (t > 0) & (t <= tau1)
    mask2 = (t > tau1) & (t <= tau2)
    mask3 = t > tau2

    trend[mask1] = c1
    trend[mask2] = a2 * t[mask2] + c2
    trend[mask3] = a3 * np.exp(b3 * t[mask3]) + c3

    params = {
        "c1": c1,
        "a2": a2,
        "c2": c2,
        "a3": a3,
        "b3": b3,
        "c3": c3,
    }

    return trend, params


def generate_scale_2d(
    t,
    tau1,
    tau2,
    N,
    sigma_1,
    sigma_2,
    sigma_3,
    sigma_4,
):
    """
    Generates the two-dimensional scale vector SC(t).

    sigma_j should be length-2 arrays:
        sigma_1 = [sigma_1,1, sigma_1,2],
        ...
    """
    sigma_1 = np.asarray(sigma_1, dtype=float)
    sigma_2 = np.asarray(sigma_2, dtype=float)
    sigma_3 = np.asarray(sigma_3, dtype=float)
    sigma_4 = np.asarray(sigma_4, dtype=float)

    SC = np.zeros((len(t), 2), dtype=float)
    params = []

    for i in range(2):
        SC_i, params_i = _piecewise_scale_component(
            t=t,
            tau1=tau1,
            tau2=tau2,
            N=N,
            sigma1=sigma_1[i],
            sigma2=sigma_2[i],
            sigma3=sigma_3[i],
            sigma4=sigma_4[i],
        )

        SC[:, i] = SC_i
        params.append(params_i)

    return SC, params


def generate_trend_2d(
    t,
    tau1,
    tau2,
    N,
    c1,
    scale_params,
):
    """
    Generates the two-dimensional trend vector T(t).
    """
    c1 = np.asarray(c1, dtype=float)

    T = np.zeros((len(t), 2), dtype=float)
    params = []

    for i in range(2):
        T_i, params_i = _piecewise_trend_component(
            t=t,
            tau1=tau1,
            tau2=tau2,
            N=N,
            c1=c1[i],
            scale_params=scale_params[i],
        )

        T[:, i] = T_i
        params.append(params_i)

    return T, params


# ============================================================
# 4. Full 2D simulation
# ============================================================

def simulate_2d_model(
    N=2000,
    tau1=None,
    tau2=None,
    alpha=1.9,
    Theta=None,
    Sigma=None,
    sigma_1=None,
    sigma_2=None,
    sigma_3=None,
    sigma_4=None,
    c1=None,
    random_state=None,
):
    """
    Simulates the two-dimensional stochastic degradation model:

        S(t) = R(t) + T(t),

    where:
        R_i(t) = SC_i(t) R2_i(t),
        R2(t) follows VAR(1),
        R3(t) is sub-Gaussian SαS noise.
    """
    if tau1 is None:
        tau1 = N / 3

    if tau2 is None:
        tau2 = 2 * N / 3

    if Theta is None:
        Theta = np.array([
            [0.5, 0.2],
            [-0.1, 0.6],
        ], dtype=float)

    if Sigma is None:
        Sigma = np.array([
            [0.64, 0.32],
            [0.32, 0.64],
        ], dtype=float)

    if sigma_1 is None:
        sigma_1 = np.array([1.0, 0.8])

    if sigma_2 is None:
        sigma_2 = np.array([2.5, 1.5])

    if sigma_3 is None:
        sigma_3 = np.array([5.0, 2.0])

    if sigma_4 is None:
        sigma_4 = np.array([25.0, 10.0])

    if c1 is None:
        c1 = np.array([10.0, -5.0])

    t = np.arange(1, N + 1)

    R3 = simulate_subgaussian_sas_noise_2d(
        N=N,
        alpha=alpha,
        Sigma=Sigma,
        random_state=random_state,
    )

    R2 = simulate_var1(
        N=N,
        Theta=Theta,
        noise=R3,
    )

    SC, scale_params = generate_scale_2d(
        t=t,
        tau1=tau1,
        tau2=tau2,
        N=N,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        sigma_3=sigma_3,
        sigma_4=sigma_4,
    )

    T, trend_params = generate_trend_2d(
        t=t,
        tau1=tau1,
        tau2=tau2,
        N=N,
        c1=c1,
        scale_params=scale_params,
    )

    R = SC * R2
    S = T + R

    return {
        "t": t,
        "S": S,
        "R": R,
        "R2": R2,
        "R3": R3,
        "SC": SC,
        "T": T,
        "Theta": np.asarray(Theta, dtype=float),
        "Sigma": np.asarray(Sigma, dtype=float),
        "scale_params": scale_params,
        "trend_params": trend_params,
        "params": {
            "N": N,
            "tau1": tau1,
            "tau2": tau2,
            "alpha": alpha,
            "Theta": np.asarray(Theta, dtype=float),
            "Sigma": np.asarray(Sigma, dtype=float),
            "sigma_1": np.asarray(sigma_1, dtype=float),
            "sigma_2": np.asarray(sigma_2, dtype=float),
            "sigma_3": np.asarray(sigma_3, dtype=float),
            "sigma_4": np.asarray(sigma_4, dtype=float),
            "c1": np.asarray(c1, dtype=float),
        },
    }