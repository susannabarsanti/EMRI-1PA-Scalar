# Flux calibration data

Numerical flux tables used by the custom trajectory ODE classes in `../Post_Adiabatic_Trajectories.py` (plain whitespace-separated text, loaded via `np.loadtxt`). See the repository root `README.md` for the citation of the underlying 1PA gravitational flux data.

## Standard-range files (valid for `p` in ~[6.25, 30])

Used by every scenario **except** Case1d (`SchwarzCirc1PASpinScalar`, `SchwarzCirc1PAScalar`, `SchwarzCirc0PAScalarS`, `SchwarzCirc0PASpinScalar`):

- **`QuasiCircularData_new.dat`** (2375 rows, 3 cols: `p`, 0PA gravitational flux, 1PA gravitational flux).
- **`QuasiCircularDataSpin_restructured.dat`** (2375 rows, 2 cols: `p`, spin-correction flux term). Shares the same `p`-grid as `QuasiCircularData_new.dat` (row-for-row); used only by the two spin-scalar classes (`SchwarzCirc1PASpinScalar`, `SchwarzCirc0PASpinScalar`).
- **`pdotCircScal_y.dat`** (500 rows, 2 cols: `u = 1/sqrt(p - 0.9*6)`, scalar-charge flux). Covers `p` ≈ [6.05, 30.05].

## Extended-range files (valid up to `p` = 50-60), used only by the Hybrid Strong-Field-PN classes

Used only by `HybridSFPN_1PAScalar_DenominatorExpanded` / `HybridSFPN_0PAScalar_DenominatorExpanded` (scenario **1d**, `mu=40`), which integrate out to `p = 50` — beyond the range of the standard files above:

- **`Numerical_F1_F2_EFLx.dat`** (6000 rows, 4 cols: `p`, [unused column], 1PA gravitational flux, `dE1/dr_hat`). Valid `p`: [6.25, 30] only — no numerical 1PA data exists beyond `p=30`, so for `p>30` the 1PA piece switches to an analytic PN formula (`F2PN`/`E1PN_prime` in the code) instead.
- **`Numerical_Grav0pa_to50.dat`** (4375 rows, 2 cols: `p`, 0PA gravitational flux). Valid `p`: [6.25, 50] — used for the 0PA piece across the whole Hybrid range (both `p<=30` and `p>30`).
- **`pdotCircScal_y_ext.dat`** (532 rows, 2 cols: `u`, scalar-charge flux). Covers `p` ≈ [6.05, 60.3] — the extended-domain counterpart of `pdotCircScal_y.dat`, needed since the Hybrid class evaluates the scalar flux across its whole range up to `p=50`.
