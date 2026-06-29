# Modelling the Health Condition of Machines Using Sub-Gaussian Distributions

Final project for **Computer Modelling and Simulation of Stochastic Processes**.

The main report is contained in:

```text
notebook.ipynb
```

The notebook includes the theoretical description, simulation studies, Monte Carlo experiments, parameter estimation, residual diagnostics and real-data analysis.

## Repository Structure

* `notebook.ipynb` - final report notebook.
* `simulation1d.py` - simulation of the one-dimensional degradation model.
* `estimation1d.py` - estimation methods for the one-dimensional model.
* `simulation2d.py` - simulation of the multidimensional degradation model and sub-Gaussian noise.
* `estimation2d.py` - estimation methods for the multidimensional model.
* `monte_carlo_utils.py` - reusable Monte Carlo experiment utilities.
* `plotting_utils.py` - plotting functions used in the notebook.
* `real_data_analysis.py` - utilities for real-data loading, bootstrap diagnostics, model simulation and validation.
* `ims_set1_bearing1_rms_hi.csv` - processed RMS health index used in the real-data section.
* `real_data_mbb_ci_VAR6_h20_B200.pkl` - cached Moving Block Bootstrap confidence intervals used for ACF/CCF diagnostics in the real-data section.
* `requirements.txt` - Python dependencies.

## Real Data

The real-data section uses the processed IMS Bearing Dataset health index stored in:

```text
ims_set1_bearing1_rms_hi.csv
```

This file is included in the repository, so the notebook can be run without the raw IMS dataset.

The bootstrap confidence intervals for real-data ACF/CCF diagnostics are loaded from:

```text
real_data_mbb_ci_VAR6_h20_B200.pkl
```

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Running the Project

Open and run the final report notebook from the repository root:

```bash
jupyter notebook notebook.ipynb
```

The notebook should be run from the root folder so that the relative paths to the data and cache files work correctly.
