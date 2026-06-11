# Modelling the Health Condition of Machines Using Sub-Gaussian Distributions

Final project for **Computer Modelling and Simulation of Stochastic Processes**.

The main report is `notebook.ipynb`. It contains the written project report,
source-code calls, simulation results, Monte Carlo summaries, diagnostics, and
the real-data analysis for the IMS Bearing Dataset.

## Project Structure

- `notebook.ipynb` - main final report notebook.
- `notebook-prezentacyjny.ipynb` - shorter presentation-oriented notebook.
- `simulation1d.py` - one-dimensional model simulation.
- `estimation1d.py` - one-dimensional trend, scale, AR and alpha estimation.
- `simulation2d.py` - two-dimensional model simulation and sub-Gaussian noise.
- `estimation2d.py` - two-dimensional trend, scale, VAR, diagnostics, alpha and
  Sigma estimation.
- `monte_carlo_utils.py` - reusable Monte Carlo runners and Sigma calibration
  summaries.
- `plotting_utils.py` - reusable plotting functions for report figures.
- `real_data_analysis.py` - IMS loading, RMS health-index construction,
  MIDAST fallback handling, Moving Block Bootstrap diagnostics, and
  quantile-band validation helpers.
- `references/` - seminar presentations, the reference thesis/report, and
  `dane_rzeczywiste.ipynb`, used as methodological reference material.

## Running the Report

Install the required packages:

```bash
pip install -r requirements.txt
```

Then open and run:

```bash
jupyter notebook notebook.ipynb
```

The notebook is intended to run from top to bottom. Some Monte Carlo cells are
computationally heavier because they reproduce the reported simulation study
with `M = 100` replications. They are kept visible so the results can be
recomputed, but they do not need to be rerun when only reading the report.

## IMS Bearing Dataset

The real-data section uses IMS Bearing Dataset, Set No. 1, Bearing 1. Place raw
IMS files in:

```text
ims/1st_test/1st_test
```

Each file should contain one second of vibration data with 20480 rows and
8 channels. Bearing 1 is represented by channels 0 and 1. The notebook computes
one RMS value per channel and per file, giving a two-dimensional health index.

To avoid re-reading all raw files on each run, the RMS health index is cached in:

```text
ims_set1_bearing1_rms_hi.csv
```

If the raw IMS folder is missing but the cache exists, the notebook uses the
cache. If both are missing, the real-data section prints a clear message instead
of failing.

## Real-Data Preprocessing and Change Points

The IMS analysis follows the reference notebook
`references/dane_rzeczywiste.ipynb`. The reported preprocessing uses:

```python
hi = np.array([rms_df["RMS Bearing 1"], rms_df["RMS Bearing 2"]])[:, 245:].T
```

The reference regime-change points are:

```python
tau_ref = [268, 1262, 1873]
```

These are labelled as reference values unless the MIDAST `ChangeDetector`
package is actually available and executed in the local workspace.

## Moving Block Bootstrap Cache

The real-data ACF/CCF confidence intervals can be expensive to recompute. The
default notebook settings are:

```python
RUN_MBB_BOOTSTRAP = False
USE_CACHED_MBB = True
MBB_REPS = 200
```

If available, the notebook loads:

```text
real_data_mbb_ci_VAR6_h20_B200.pkl
```

To recompute the bootstrap, set `RUN_MBB_BOOTSTRAP = True` in the corresponding
notebook cell. The new cache will be saved under the same filename pattern.

## Reproducibility Notes

The project uses robust estimators because alpha-stable and sub-Gaussian
innovations can be heavy-tailed. Sigma estimation is particularly sensitive to
scale normalization. In simulations, a scale-adjusted Sigma validation is
reported because the true scale is known; this adjustment is not directly
available for real data.
