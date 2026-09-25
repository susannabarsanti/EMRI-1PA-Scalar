# EMRI-1PA-Scalar

Parameter-estimation pipeline and results for EMRI waveforms with scalar fields, incorporating first post-adiabatic (1PA) gravitational corrections and compared against 0PA / post-Newtonian / GR baselines.

If you use this repository, please cite [arXiv:2607.09310](https://arxiv.org/abs/2607.09310). 

The 1PA gravitational self-force flux data used in the trajectory modules (`scripts/shared_modules/data_few/QuasiCircularData_new.dat`) is taken from  [arXiv:2107.01298](https://arxiv.org/abs/2107.01298)
,  [arXiv:2112.12265](https://arxiv.org/abs/2112.12265) . 

The analytic PN formulas used for the 1PA gravitational terms above `r=30` (where no numerical 1PA data is available) are taken from [arXiv:2510.16112](https://arxiv.org/abs/2510.16112),  [arXiv:2510.16114](https://arxiv.org/abs/2510.16114). 

Parts of the analysis notebook are adapted from code originally written by Ollie Burke, see [EMRI_PE](https://github.com/OllieBurke/EMRI_PE).

## Repository structure

- `data/` — trimmed (post burn-in) MCMC posterior samples (`.h5`, eryn `HDFBackend` format) for every scenario in the paper (Cases 1-4, Appendix C1-3). Raw chains are not included here due to file size, email me if interested. 
- `figures/` — corner plots and waveform-comparison figures shown in the paper.
- `checks_pureGR/` — corner plots and waveform comparisons for a check against Burke et al. [arXiv:2310.08927](https://arxiv.org/abs/2310.08927) (Schwarzschild / Schwarzschild-bicubic / Kerr flux models).
- `notebooks/` — `notebook_1PAScalarVS0PAScalar.ipynb`, reproduces the paper's corner plots and bias computations from the files in `data/`. Cells that require the full (untrimmed) raw chains (only used for autocorrelation-time diagnostics, not for any figure) are commented out.
- `scripts/` — MCMC launch scripts for every scenario, plus `shared_modules/` (custom FEW-compatible trajectory ODE classes, including the flux-calibration data in `shared_modules/data_few/`). Scripts are self-contained: paths are resolved relative to the repository root, and running one will create local `psd_cache/` and `raw_runs/` folders (both git-ignored).

## Requirements

Tested with the following environment (Python 3.12):

- [`few`](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) == 2.0.0
- `eryn` == 1.1.14
- `cupy-cuda12x` == 13.4.1 (GPU runs; a CUDA toolkit, e.g. 12.6, must be available)
- `fastlisaresponse` == 1.0.11
- `lisaanalysistools` == 1.0.14
- `lisaconstants` == 1.3.6
- `numpy` == 2.2.6
- `scipy` == 1.15.3
- `h5py` == 3.14.0
- `corner` == 2.2.3
- `matplotlib` == 3.10.3

## Running a scenario

Each script under `scripts/<Case>/` can be run directly, e.g.:

```bash
python scripts/Case1_1PAScalarVS0PAScalar/1a_M1e6_mu50/MCMC_code_Case1.py
```

Output chains are written to `raw_runs/`, and PSD caching to `psd_cache/`, both created automatically at the repository root.
