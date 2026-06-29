# Modelling the Health Condition of Machines Using Sub-Gaussian Distributions

Final project for **Computer Modelling and Simulation of Stochastic Processes**.

`notebook.ipynb` is the final report notebook and should be opened from the
repository root. It contains the written report, simulations, estimation
results, Monte Carlo summaries, diagnostics, and the real-data analysis.

## Repository Structure

- `notebook.ipynb` - final report notebook.
- `ims_set1_bearing1_rms_hi.csv` - processed IMS Bearing Dataset RMS health
  index cache used by the real-data section.
- `real_data_mbb_ci_VAR6_h20_B200.pkl` - cached Moving Block Bootstrap ACF/CCF
  confidence intervals for the real-data VAR(6), h=20 analysis.
- `simulation1d.py` - one-dimensional alpha-stable model simulation.
- `estimation1d.py` - one-dimensional trend, scale, AR, residual, and alpha
  estimation utilities.
- `simulation2d.py` - two-dimensional model simulation and sub-Gaussian noise
  generation.
- `estimation2d.py` - two-dimensional trend, scale, VAR, diagnostic, alpha, and
  Sigma estimation utilities.
- `monte_carlo_utils.py` - reusable Monte Carlo runners and summary helpers.
- `plotting_utils.py` - reusable plotting functions for notebook figures.
- `real_data_analysis.py` - IMS cache loading, optional RMS recomputation,
  Moving Block Bootstrap diagnostics, VAR path simulation, and validation
  helpers.
- `archive/` - old, backup, and presentation-only notebooks. These files are
  preserved for reference but are not part of the final report.

## IMS Real Data

The raw IMS Bearing Dataset is not included in this repository. The notebook can
run the current real-data section using the processed cache file:

```text
ims_set1_bearing1_rms_hi.csv
```

The notebook path for this cache is:

```python
IMS_CACHE = Path("ims_set1_bearing1_rms_hi.csv")
```

If you want to recompute RMS values from the raw IMS files, place the raw files
in:

```text
ims/1st_test/1st_test
```

Otherwise, the cache CSV in the repository root is enough for the current
real-data section. The raw `ims/` folder is intentionally ignored by Git.

The real-data bootstrap confidence intervals are cached in:

```text
real_data_mbb_ci_VAR6_h20_B200.pkl
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

Open and run the final report notebook:

```bash
jupyter notebook notebook.ipynb
```

Run it from the repository root so the relative cache paths continue to work.
