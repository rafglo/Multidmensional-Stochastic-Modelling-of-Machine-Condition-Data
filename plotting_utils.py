import numpy as np
import matplotlib.pyplot as plt


def plot_mc_1d_estimation_results(result, true_phi=0.5):
    """Plots 1D Monte Carlo trend, scale, phi and alpha estimates."""
    alpha = result["alpha"]
    t = result["t"]

    T_hats = result["T_hats"]
    SC_hats = result["SC_hats"]
    phi_hats = result["phi_hats"]
    alpha_hats = result["alpha_hats"]

    true_T = result["true_T"]
    true_SC = result["true_SC"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    for T_hat_i in T_hats:
        ax.plot(t, T_hat_i, color="gray", alpha=0.18, linewidth=0.8)
    ax.plot(t, true_T, color="blue", linewidth=2.5, label="True trend")
    ax.set_title(rf"Estimated trend trajectories, $\alpha={alpha}$")
    ax.set_xlabel("t")
    ax.set_ylabel("T(t)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax = axes[0, 1]
    for SC_hat_i in SC_hats:
        ax.plot(t, SC_hat_i, color="gray", alpha=0.18, linewidth=0.8)
    ax.plot(t, true_SC, color="orange", linewidth=2.5, label="True scale")
    ax.set_title(rf"Estimated scale trajectories, $\alpha={alpha}$")
    ax.set_xlabel("t")
    ax.set_ylabel("SC(t)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax = axes[1, 0]
    ax.boxplot(phi_hats, tick_labels=[r"$\hat{\phi}$"])
    ax.axhline(true_phi, linestyle="--", color="red", label=rf"true $\phi={true_phi}$")
    ax.set_title("Distribution of AR(1) coefficient estimates")
    ax.set_ylabel(r"$\hat{\phi}$")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax = axes[1, 1]
    ax.boxplot(alpha_hats, tick_labels=[r"$\hat{\alpha}$"])
    ax.axhline(alpha, linestyle="--", color="red", label=rf"true $\alpha={alpha}$")
    ax.set_title("Distribution of stability index estimates")
    ax.set_ylabel(r"$\hat{\alpha}$")
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.suptitle(
        rf"Monte Carlo estimation results for the one-dimensional model, $\alpha={alpha}$",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_representative_acf_1d(result, ci_level=0.95):
    """Plots robust ACF before and after AR(1) for a representative replication."""
    alpha = result["alpha"]
    est_rep = result["representative_est"]

    acf_R2 = est_rep["acf_R2"]
    acf_R3 = est_rep["acf_R3"]
    R2_hat = est_rep["R2_hat"]
    R3_hat = est_rep["R3_hat"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    if ci_level == 0.95:
        z_value = 1.96
    elif ci_level == 0.90:
        z_value = 1.645
    else:
        raise ValueError("Use ci_level=0.95 or ci_level=0.90.")

    band_R2 = z_value / np.sqrt(len(R2_hat))
    band_R3 = z_value / np.sqrt(len(R3_hat))

    lags_R2 = np.arange(len(acf_R2))
    lags_R3 = np.arange(len(acf_R3))

    axes[0].stem(lags_R2, acf_R2)
    axes[0].axhline(0, linewidth=1)
    axes[0].axhline(band_R2, linestyle="--", label=f"{int(ci_level * 100)}% band")
    axes[0].axhline(-band_R2, linestyle="--")
    axes[0].set_title(rf"Before AR(1): robust ACF of $R2(t)$, $\alpha={alpha}$")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Robust ACF")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].stem(lags_R3, acf_R3)
    axes[1].axhline(0, linewidth=1)
    axes[1].axhline(band_R3, linestyle="--", label=f"{int(ci_level * 100)}% band")
    axes[1].axhline(-band_R3, linestyle="--")
    axes[1].set_title(rf"After AR(1): robust ACF of residuals, $\alpha={alpha}$")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Robust ACF")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_mc_1d_median_band(result):
    """Plots median and 5%-95% bands for 1D trend and scale estimates."""
    alpha = result["alpha"]
    t = result["t"]

    T_hats = result["T_hats"]
    SC_hats = result["SC_hats"]

    median_T_hat = np.median(T_hats, axis=0)
    q05_T = np.quantile(T_hats, 0.05, axis=0)
    q95_T = np.quantile(T_hats, 0.95, axis=0)

    median_SC_hat = np.median(SC_hats, axis=0)
    q05_SC = np.quantile(SC_hats, 0.05, axis=0)
    q95_SC = np.quantile(SC_hats, 0.95, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].fill_between(t, q05_T, q95_T, alpha=0.25, label="5%-95% band")
    axes[0].plot(t, result["true_T"], linewidth=2.5, linestyle="--", label="True trend")
    axes[0].plot(t, median_T_hat, linewidth=2, label="Median estimated trend")
    axes[0].set_title(rf"Median trend estimate, $\alpha={alpha}$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("T(t)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].fill_between(t, q05_SC, q95_SC, alpha=0.25, label="5%-95% band")
    axes[1].plot(t, result["true_SC"], linewidth=2.5, linestyle="--", label="True scale")
    axes[1].plot(t, median_SC_hat, linewidth=2, label="Median estimated scale")
    axes[1].set_title(rf"Median scale estimate, $\alpha={alpha}$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("SC(t)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_2d_acf_ccf_diagnostics(diagnostics, title_prefix=""):
    """Plots robust ACF/CCF diagnostics for a 2D normalized signal."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    entries = [
        ("acf_1", "robust ACF of component 1", axes[0, 0]),
        ("ccf_12", "robust CCF: component 1 vs 2", axes[0, 1]),
        ("ccf_21", "robust CCF: component 2 vs 1", axes[1, 0]),
        ("acf_2", "robust ACF of component 2", axes[1, 1]),
    ]

    for key, title, ax in entries:
        values = diagnostics[key]
        ax.stem(range(len(values)), values)
        ax.set_title(f"{title_prefix} {title}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("value")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_2d_acf_ccf_with_band(diagnostics, n_obs, title_prefix=""):
    """Plots robust ACF/CCF diagnostics with approximate 95% bands."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    entries = [
        ("acf_1", "ACF component 1", axes[0, 0]),
        ("ccf_12", "CCF component 1 vs 2", axes[0, 1]),
        ("ccf_21", "CCF component 2 vs 1", axes[1, 0]),
        ("acf_2", "ACF component 2", axes[1, 1]),
    ]

    band = 1.96 / np.sqrt(n_obs)

    for key, title, ax in entries:
        values = diagnostics[key]
        lags = np.arange(len(values))

        ax.stem(lags, values)
        ax.axhline(0, linewidth=1)
        ax.axhline(band, linestyle="--", label="approx. 95% band")
        ax.axhline(-band, linestyle="--")
        ax.set_title(f"{title_prefix} {title}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_mc_2d_part1(result):
    """Plots 2D Monte Carlo trend, scale and Theta estimates."""
    alpha = result["alpha"]
    t = result["t"]

    T_hats = result["T_hats"]
    SC_hats = result["SC_hats"]
    Theta_hats = result["Theta_hats"]

    true_T = result["true_T"]
    true_SC = result["true_SC"]
    true_Theta = result["true_Theta"]

    fig, axes = plt.subplots(4, 2, figsize=(13, 14))

    ax = axes[0, 0]
    for T_hat in T_hats:
        ax.plot(t, T_hat[:, 0], color="gray", alpha=0.15, linewidth=0.8)
    ax.plot(t, true_T[:, 0], color="blue", linewidth=2.5, label=r"True $T_1(t)$")
    ax.set_title(rf"Estimated trend trajectories $T_1(t)$, $\alpha={alpha}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$T_1(t)$")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    for T_hat in T_hats:
        ax.plot(t, T_hat[:, 1], color="gray", alpha=0.15, linewidth=0.8)
    ax.plot(t, true_T[:, 1], color="red", linewidth=2.5, label=r"True $T_2(t)$")
    ax.set_title(rf"Estimated trend trajectories $T_2(t)$, $\alpha={alpha}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$T_2(t)$")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 0]
    for SC_hat in SC_hats:
        ax.plot(t, SC_hat[:, 0], color="gray", alpha=0.15, linewidth=0.8)
    ax.plot(t, true_SC[:, 0], color="blue", linewidth=2.5, label=r"True $SC_1(t)$")
    ax.set_title(rf"Estimated scale trajectories $SC_1(t)$, $\alpha={alpha}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$SC_1(t)$")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 1]
    for SC_hat in SC_hats:
        ax.plot(t, SC_hat[:, 1], color="gray", alpha=0.15, linewidth=0.8)
    ax.plot(t, true_SC[:, 1], color="red", linewidth=2.5, label=r"True $SC_2(t)$")
    ax.set_title(rf"Estimated scale trajectories $SC_2(t)$, $\alpha={alpha}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$SC_2(t)$")
    ax.grid(True)
    ax.legend()

    theta_entries = [
        (0, 0, r"$\hat{\Theta}_{11}$", true_Theta[0, 0]),
        (0, 1, r"$\hat{\Theta}_{12}$", true_Theta[0, 1]),
        (1, 0, r"$\hat{\Theta}_{21}$", true_Theta[1, 0]),
        (1, 1, r"$\hat{\Theta}_{22}$", true_Theta[1, 1]),
    ]
    positions = [(2, 0), (2, 1), (3, 0), (3, 1)]

    for (i, j, label, true_value), (row, col) in zip(theta_entries, positions):
        ax = axes[row, col]
        values = Theta_hats[:, i, j]
        ax.boxplot(values, tick_labels=[label])
        ax.axhline(true_value, linestyle="--", color="red", label=rf"true = {true_value:.2f}")
        ax.set_title(rf"Distribution of {label}")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend()

    fig.suptitle(
        rf"Monte Carlo estimation results for the two-dimensional model, $\alpha={alpha}$ - part I",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def plot_mc_2d_part2(result):
    """Plots representative 2D ACF/CCF diagnostics and alpha estimates."""
    alpha = result["alpha"]
    est_rep = result["representative_est"]

    diagnostics_R2 = est_rep["diagnostics_R2"]
    diagnostics_R3 = est_rep["diagnostics_R3"]
    R2_hat = est_rep["R2_hat"]
    R3_hat = est_rep["R3_hat"]

    plot_2d_acf_ccf_with_band(
        diagnostics_R2,
        n_obs=len(R2_hat),
        title_prefix=rf"Before VAR(1), $\alpha={alpha}$:",
    )

    plot_2d_acf_ccf_with_band(
        diagnostics_R3,
        n_obs=len(R3_hat),
        title_prefix=rf"After VAR(1), $\alpha={alpha}$:",
    )

    plt.figure(figsize=(6, 5))
    plt.boxplot(result["alpha_hats"], tick_labels=[r"$\hat{\alpha}$"])
    plt.axhline(alpha, linestyle="--", color="red", label=rf"true $\alpha={alpha}$")
    plt.title(rf"Distribution of stability index estimates, $\alpha={alpha}$")
    plt.ylabel(r"$\hat{\alpha}$")
    plt.grid(True)
    plt.legend()
    plt.show()


def add_regime_annotations(ax, tau_values, n_obs):
    """Adds vertical regime-change lines and regime labels to an HI plot."""
    for idx, tau in enumerate(tau_values, start=1):
        ax.axvline(
            tau,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=rf"$\tau_{idx}={tau}$",
        )

    regimes = ["normal", "warning", "critical", "alarm"]
    bounds = [1] + list(tau_values) + [n_obs]
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.08 * (ymax - ymin)

    for label, left, right in zip(regimes, bounds[:-1], bounds[1:]):
        if right > left:
            ax.text(
                (left + right) / 2,
                y_text,
                label,
                ha="center",
                va="top",
                fontsize=9,
            )


def plot_sigma_boxplots_adjusted(result):
    """Plots scale-adjusted Sigma estimates for one 2D Monte Carlo result."""
    Sigma_hats = result["Sigma_hats_adjusted"]
    true_Sigma = result["true_Sigma"]
    alpha = result["alpha"]

    entries = [
        (0, 0, r"$\hat{\Sigma}_{11}^{adj}$", true_Sigma[0, 0]),
        (0, 1, r"$\hat{\Sigma}_{12}^{adj}$", true_Sigma[0, 1]),
        (1, 0, r"$\hat{\Sigma}_{21}^{adj}$", true_Sigma[1, 0]),
        (1, 1, r"$\hat{\Sigma}_{22}^{adj}$", true_Sigma[1, 1]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, (i, j, label, true_value) in zip(axes.ravel(), entries):
        values = Sigma_hats[:, i, j]
        ax.boxplot(values, tick_labels=[label])
        ax.axhline(
            true_value,
            linestyle="--",
            color="red",
            label=f"true = {true_value:.2f}",
        )
        ax.set_title(f"Distribution of {label}")
        ax.grid(True)
        ax.legend()

    fig.suptitle(rf"Scale-adjusted Sigma estimation, $\alpha={alpha}$")
    plt.tight_layout()
    plt.show()


def plot_real_acf_ccf_diagnostics(diagnostics, n_obs, title_prefix="", ci=None):
    """Plots robust ACF/CCF diagnostics with cached MBB or approximate bands."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    entries = [
        ("acf_1", "ACF component 1", axes[0, 0]),
        ("ccf_12", "CCF component 1 vs 2", axes[0, 1]),
        ("ccf_21", "CCF component 2 vs 1", axes[1, 0]),
        ("acf_2", "ACF component 2", axes[1, 1]),
    ]

    simple_band = 1.96 / np.sqrt(n_obs)

    for key, title, ax in entries:
        values = diagnostics[key]
        lags = np.arange(len(values))

        ax.stem(lags, values)
        ax.axhline(0, color="black", linewidth=1)

        if ci is not None and key in ci:
            lower = np.asarray(ci[key]["lower"])
            upper = np.asarray(ci[key]["upper"])
            ax.fill_between(lags, lower, upper, alpha=0.25, label="MBB 95% CI")
        else:
            ax.axhline(
                simple_band,
                color="red",
                linestyle="--",
                linewidth=1,
                label="approx. 95% band",
            )
            ax.axhline(-simple_band, color="red", linestyle="--", linewidth=1)

        ax.set_title(f"{title_prefix} {title}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
