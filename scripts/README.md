# MCMC run scripts

Environment: conda env "few" (see launchers/gpumcmc.sh, cpujob.sh for the exact
`conda activate` + `module load cuda/12.6` setup). An alternative env "few_kerr2"
(FEW Soton fork + lisa-on-gpu without EqualArms + custom FoM) is mentioned as a
commented-out alternative in gpumcmc.sh but was not the active one for these runs.

## Structure

Each scenario/case has its own folder (matching the naming used in GOOD_RESULTS/data/),
containing the MCMC_code_*.py script used AND a PE_EMRI_params.py with the exact
physical parameters for that scenario (M, mu, p0, SNR, angles, Lambda, spin),
reconstructed from the true_values embedded in the corresponding trimmed .h5 file.
This is necessary because PE_EMRI_params.py is normally edited in place before
each run (not passed via command-line arguments), so a single shared copy only
ever reflects the last run - historical per-scenario values would otherwise be
unrecoverable (as happened with Scenario 1d's raw chain search).

- shared_modules/  -> PSD_code.py, Post_Adiabatic_Trajectories.py (imported by all MCMC_code_*.py scripts, do not change between runs)
- launchers/       -> gpumcmc.sh, cpujob.sh (SLURM submission scripts, reused across runs by editing which `python <script>.py` line is active)

## Trajectory class audit (shared_modules/Post_Adiabatic_Trajectories.py)

Before any new GPU runs, all trajectory classes were reviewed against every other
copy of the same classes found across MCMC_scratch (Hybrid_SFPN_class.py,
Hybrid_SFPN_Scalar_class.py, Hybrid_SFPN.py, IMRI_multiband/Hybrid_SFPN.py,
IMRI_multiband/Hybrid_1PAPN.py, and the root Post_Adiabatic_Trajectories.py that
shared_modules' copy was sourced from), since several of these are older,
independently-edited duplicates rather than the current version.

Findings:
- **pdot_scal normalization - CONFIRMED CORRECT.** The current
  `pdot_scal = -Lambda*pdot_interp_scal(u_scal)/4.` (present in
  SchwarzCirc1PASpinScalar/1PAScalar/0PAScalarS/0PASpinScalar) differs from an
  older `Hybrid_SFPN_class.py` (2025-07-11) / `Hybrid_SFPN_Scalar_class.py`
  (2025-12-05) version that omits the `/4.`. Confirmed correct (with the `/4.`) by
  Susanna directly - do not remove it if you see the no-`/4.` version elsewhere.
- A `nu = eps/(1+eps)**2` (symmetric mass ratio) + overall `(nu/eps)` prefactor
  convention also differs between old/new versions, but is numerically negligible
  (<0.2%) for every mass ratio used in this project - not a live concern.
- **Interpolation bounds-check inconsistency - FIXED, all 4 classes synced.**
  Root's `Post_Adiabatic_Trajectories.py` (2026-04-17) already had a proper fix
  applied to `SchwarzCirc1PASpinScalar`/`SchwarzCirc1PAScalar`/`SchwarzCirc0PAScalarS`
  (but NOT `SchwarzCirc0PASpinScalar`), which shared_modules' older copy was
  missing entirely for all 4 classes. The fix does NOT uncomment the
  `if p < 6.25 or p > 30.: raise ValueError(...)` inside `interpolate_flux_grids`
  (that stays commented, even in the fixed root classes) - instead it uses FEW's
  proper architecture for this: `min_p()`/`max_p()` are overridden to return
  `6.0`/`30.0` (previously not overridden at all, so FEW's own default
  `max_p=+inf` applied - no upper-bound protection whatsoever), and
  `distance_to_outer_boundary(y)` returns `30.0 - p` (was commented out / using a
  different formula in some classes) so FEW's integrator can gracefully terminate
  the trajectory exactly at the grid edge, rather than raising mid-integration.
  `separatrix_buffer_dist` also increased from `0.1` to `0.25`.

  **Mechanism verified directly against FEW's own source**
  (`few/trajectory/integrate.py`), not just inferred from the `ODEBase`
  docstrings - the two bounds properties protect two *different* integration
  directions, independently:
  - **Forward integration** (injection + recovery waveform generation - i.e.
    almost every trajectory evaluation during MCMC) stops via
    `stop_integrate_check`: `if p - p_sep < self.separatrix_buffer_dist: stop`,
    where `p_sep = get_separatrix(a, e, x)` is the *true* relativistic separatrix
    (`=6.0` for circular Schwarzschild), **not** `min_p()`. So forward
    integration used to stop at `p = 6.0 + 0.1 = 6.1` - 0.15 *inside* the
    grid's invalid region (grid starts at 6.25) - on every single evaluation.
    With the fix, it stops at `p = 6.0 + 0.25 = 6.25`, exactly at the grid edge.
    `min_p()`/`max_p()` are used only once, at trajectory start, to validate p0
    itself (`isvalid_pex`) - they are not a continuous bound during integration.
  - **Backward integration** (used only for the p0-backward-solve step, e.g.
    `integrate_backwards=True`) stops via `distance_to_outer_boundary(y) < 0`,
    i.e. when `p > max_p` - this is where the `max_p=30.0` override and the
    `distance_to_outer_boundary` returning `30.0 - p` actually matter.

  So the real, live bug this fixes is specifically the **forward-integration**
  extrapolation (the common case), via `separatrix_buffer_dist`, not `min_p`
  directly - `min_p`'s role is a one-time p0 sanity check.

  `shared_modules/Post_Adiabatic_Trajectories.py` was synced to match root's fix
  for the 3 already-fixed classes verbatim, and the same fix was manually applied
  to `SchwarzCirc0PASpinScalar` (not yet fixed even in root) for consistency -
  confirmed with Susanna. The `pdot_scal`/`nu` formulas themselves were not
  touched by this sync (verified unchanged, still with `/4.` and `self.massratio`
  as confirmed correct above). Do NOT blanket-copy the `6.0`/`30.0` bounds into
  the Hybrid classes - they intentionally support p up to 50 (`_min_p=6.0`,
  `_max_p=50.0`, `separatrix_buffer_dist=0.251` -> forward integration stops at
  `p=6.251`, just above their own internal branch threshold at 6.25) via their
  own internal branching in `F0HybridExtended`/`F1HybridExtended`, and already
  have their own correct min_p/max_p/separatrix_buffer_dist - confirmed
  byte-identical to their root source (`Hybrid_SFPN.py`), no changes needed.
  `SchwarzCircPNScalar` (used by Case IV's `RECOVERY_MODEL="PN"`) was found to
  have the exact same unfixed bounds (`separatrix_buffer_dist=0.1`, no
  min_p/max_p override) - it still uses the same numerical 0PA grid for its
  `pdot_0pa` term (only `pdot_scal` is the fully analytic -1PN formula), so it
  has the same forward-integration extrapolation risk. Manually patched with the
  identical fix (confirmed with Susanna) - its `pdot_scal` analytic formula
  (`A=Lambda/12`) was not touched.
  Note: the root copies of `SchwarzCirc0PASpinScalar` and `SchwarzCircPNScalar`
  (i.e. outside GOOD_RESULTS, Susanna's own working copy) still have the old,
  unfixed version - only the GOOD_RESULTS/shared_modules copy was updated for
  both.

### Code-comment cleanup (shared_modules/Post_Adiabatic_Trajectories.py)

After the bounds-check sync above, a follow-up pass cleaned up comments across
all 6 classes (confirmed with Susanna before applying):
- Translated the recurring Italian comment ("lo sto definendo qui Edot_interp...")
  to English, in all 5 classes that had it.
- Removed a stale `##########CHECK OLD VS NEW ##############` TODO-style comment
  on `SchwarzCirc1PASpinScalar`'s data-loading line.
- Removed the unused `F1PN` method from both Hybrid classes (dead code -
  superseded by the extended numerical grid in `interpolate_flux_grav0pa`, valid
  up to p=50; only `F2PN` is actually used, for the O(nu) correction beyond
  p=30). The one dead reference to it (`#num = -self.F1PN(r0)`) was replaced
  with an explanatory comment instead of silently vanishing.
- Replaced the six vague `#CAREFUL HERE` comments in `F0HybridExtended`/
  `F1HybridExtended` (both Hybrid classes) with an explanation of the actual
  denominator-expansion logic (paper Eq. 18/30: `a(r_hat)=(dE0/dr_hat)^-1`;
  the O(nu) piece of the denominator is deliberately dropped from F0 and
  instead folded into F1 via `-F1_flux*term2/term1**2`, the correct first-order
  Taylor expansion of `1/(term1+nu*term2)` - this was verified against the
  paper, not a guess, see "Hybrid trajectory vs. paper" below).
- Added a comment clarifying that the local `F1`/`F2` variables inside
  `F0HybridExtended`/`F1HybridExtended` (from `interpolate_flux_grids`'s raw
  numerical-grid columns) are unrelated to the *methods* `F0HybridExtended`/
  `F1HybridExtended` themselves, or to `F1PN`/`F2PN` - three different things
  share confusingly similar names in this file.

### Hybrid trajectory vs. the paper (verified, not just structurally plausible)

`F0HybridExtended`/`F1HybridExtended` were checked term-by-term against the
paper's Eq. 18 and Eq. 30:
- `term1` in the code = `a(r_hat) = (dE0/dr_hat)^-1` (paper's Eq. 18 definition,
  the analytic Schwarzschild circular-orbit binding-energy-derivative inverse).
- `F0HybridExtended = num/term1` matches Eq. 18's `F0^A = a(r_hat)*script-F0^A`.
- `F1HybridExtended`'s two-term structure (`num/term1 - F1_flux*term2/term1**2`)
  matches Eq. 18's `F1^A = a(r_hat)*script-F1^A + a(r_hat)^2*(dE1^A/dr_hat)*script-F0^GSF`
  exactly, and is the literal Eq. 30 "denominator expansion" (expanding
  `[dE0/dr_hat + nu*dE1/dr_hat]^-1` to O(nu)) - hence the class name
  `DenominatorExpanded`.
- The paper's Eq. 18 second line (the F1 correction) is stated only for
  `A in {GSF, chi}`, NOT `Lambda` (scalar charge) - matching the code exactly:
  `pdot_scal` is always added as a plain leading-order term (never scaled by
  `nu`) in every class, Schwarz and Hybrid alike. This is consistent, not an
  oversight.
- One item not fully traced: Eq. 14 (`dr_hat/dt_hat = -nu*[F0+nu*F1]`) has an
  overall `-nu` multiplying the whole bracket, which isn't visible as a
  separate factor in `evaluate_rhs`'s `pdot = omegadot/domegadr`. Not flagged as
  a concern given scenario 1d's already-published posterior was independently
  reproduced from this exact code (see 1d's PROVENANCE.txt) - likely absorbed
  into a FEW-level normalization convention shared by all trajectory classes,
  not something specific to this one.
- `SchwarzCirc0PAScalarToFifty` exists in the root Post_Adiabatic_Trajectories.py
  but was not copied to shared_modules/ - not referenced by any current
  Case I-IV/Appendix script, but flagged in case it turns out to be needed.
- `IMRI_multiband/Hybrid_SFPN.py` and `IMRI_multiband/Hybrid_1PAPN.py` are the
  most recently modified files of this whole group (2026-04-30), postdating even
  1d's Hybrid run (March 2026) - they belong to separate, later IMRI-multiband
  work (note the class names lack "Scalar" - no scalar-charge term) and are
  unrelated to this paper's scenarios.

## Case I-IV: clean master scripts (rebuilt, confirmed physics)

Unlike the Appendix scripts below, Case1/Case2/Case3/Case4 have been rebuilt from scratch as
clean, self-consistent "master" scripts (rather than kept as one of the many
historical copy-pasted/manually-edited originals), after confirming the underlying
physics conventions directly with Susanna. All three share the same structural
fixes relative to the historical scripts:
- No stray `sys.path.append` to now-dead paths; PE_EMRI_params.py is imported from
  the script's own folder, Post_Adiabatic_Trajectories.py/PSD_code.py from
  `../shared_modules` (or `../../shared_modules` for Case I, which nests one level
  deeper under per-scenario subfolders).
- PSD file is named `tdi2_AE_w_background_T{T}.npy` (T-specific) instead of a fixed
  shared filename, to avoid a race condition if scripts with different T run
  concurrently (the PSD's galactic-foreground contribution genuinely depends on T
  via `stochastic_params=[T*YRSID_SI]`).
- Raw MCMC output (.h5 backend + diagnostic plot) is written to
  `/scratch/sbarsanti/MCMC_scratch/raw_runs/<case_folder_name>[/<scenario>]/`
  (auto-created, derived from the script's own location) instead of dumped flat
  into MCMC_scratch/ or run_direc, which is how that directory became a mess of
  hundreds of loose files historically. This is a fresh output area, separate from
  GOOD_RESULTS/ (the curated archive) - new run output must still be verified and
  promoted into GOOD_RESULTS/data/ manually, same as before.
- p0 is always computed via backward integration using `traj_0PASpin`
  (SchwarzCirc0PASpinScalar) at 0PA order, regardless of case or recovery order -
  called with the scenario's true Lambda and true Secondary_Spin (0 for the
  non-spin cases, since that's simply their physical truth). This is unlike
  Appendix C, where p0 is imported directly from PE_EMRI_params.py without
  recomputation - see the Appendix C section below.
- A failed MCMC point (waveform generation error) is appended to
  `<out_dir>/failed_points.log` via `log_failed_point()`, and `priors_<label>.json`
  / `run_config_<label>.json` are written to `<out_dir>` alongside the `.h5`
  output, recording the exact prior bounds and run parameters used (needed e.g.
  to compute Bayes factors later). Both features were found in the original
  Appendix C script (`MCMC_code_SPIN.py`) and have been generalized/added to
  Case I-IV too for consistency.

**Case1_1PAScalarVS0PAScalar/{1a,1b,1c,1d}/** -> `MCMC_code_Case1.py`. 1a/1b/1c
are identical except 1c's widened Lambda prior (see below); **1d is structurally
different** (see its own paragraph right after this one).

1a/1b/1c use the non-spin scalar-charge trajectories only (SchwarzCirc0PAScalarS,
SchwarzCirc1PAScalar). Injection is always 1PAScalar (Lambda free);
`RECOVERY_MODEL = "0PA"` or `"1PA"` selects which order's model is used for
recovery. Lambda prior is `uniform_dist(-0.6, 0.6)` for 1a/1b; widened to
`uniform_dist(-0.6, 5.0)` for 1c specifically, since 1c's true Lambda=1.0 falls
outside the standard range.

**1d_M1e5_mu40 uses a Hybrid Strong-Field-PN trajectory, not the plain
SchwarzCirc classes.** This was discovered after the fact (the other 3 scenarios'
scripts had already been rebuilt assuming the same non-spin classes throughout)
by tracing 1d's raw chain to `Hybrid1PAScalarVS0PAScalarResults/
DenominatorExpanded/M100000mu40SNR100/` (see its PROVENANCE.txt) and finding the
historical launch script (`MCMC_code_IMRI.py`, whose `output_dir` matches that
path exactly). 1d's `MCMC_code_Case1.py` therefore imports
`HybridSFPN_1PAScalar_DenominatorExpanded`/`HybridSFPN_0PAScalar_DenominatorExpanded`
(newly added to shared_modules/Post_Adiabatic_Trajectories.py - they were present
in the historical root copy of `Hybrid_SFPN.py` but not shared) instead of
SchwarzCirc1PAScalar/SchwarzCirc0PAScalarS. Structurally still the same as 1a/1b/1c
otherwise: `llike()` body unchanged (still just Lambda at params[9], no spin - the
Hybrid classes don't take a spin argument either), same RECOVERY_MODEL flag
convention, same Lambda prior `(-0.6, 0.6)`. One deliberate deviation from the
general p0-backward convention: 1d's p0 backward-integration uses
`traj_0PA_Hybrid` (the Hybrid class itself), NOT `traj_0PASpin` - because 1d's
whole generative model is Hybrid-based, using plain Schwarzschild physics for that
one step would be inconsistent with the actual injection/recovery model.
Output filename also differs (`MCMC_samps_IMRI_1PAdInj{RECOVERY_MODEL}dRec_hybrid_
DenominatorExpanded_...`, matching the historical pattern) rather than Case I's
usual `MCMC_samps_1PAdInj{RECOVERY_MODEL}dRec_...`.

Given this was only caught after 1a-1d had already been treated as uniform, it's
worth double-checking whether any *other* already-"confirmed" scenario/case
elsewhere in this project quietly relies on a Hybrid or otherwise-nonstandard
trajectory class before running new jobs from these clean scripts.

**Case2_1PAScalarVSGR/** -> `MCMC_code_Case2.py`. Same non-spin trajectories as
Case I. Single injection point (1PAScalar, Lambda free) tested against three
recovery variants, selected via two flags: `RECOVERY_ORDER` ("1PA"/"0PA") and
`FIX_LAMBDA_ZERO` (True/False). GR recovery = Lambda hardcoded to 0.0 in llike()
and its prior/start-vector entry simply omitted (ndim drops from 10 to 9), rather
than a different trajectory class - this is how "GR" is achieved throughout.
Output filename label is derived automatically (`1PAd`, `1PAGR`, `0PAGR`), matching
the three files already in GOOD_RESULTS/data/Case2_1PAScalarVSGR/.

**Case3_1PASpinScalarVS0PASpinScalar/** -> `MCMC_code_Case3.py`. Spin-scalar
trajectories (SchwarzCirc0PASpinScalar, SchwarzCirc1PASpinScalar); both Lambda
(index 9, prior `uniform_dist(-0.6, 0.6)`) and Secondary_Spin (index 10, prior
`uniform_dist(-1., 1.)`) are free sampled parameters. `RECOVERY_MODEL = "0PA"` or
`"1PA"` selects the recovery order, matching the two files in
GOOD_RESULTS/data/Case3_1PASpinScalarVS0PASpinScalar/. Note: the historical script
`MCMC_code_1pachargespinVS0pachargespin.py` (previously flagged here as an
"uncertain match" for this case) was found on closer inspection to hardcode
Lambda=0.0 and always use the 1PA-order recovery template - i.e. it never actually
performs a Lambda-free, order-comparison run at all. It was very likely an early
draft/relative of the Appendix C scripts (GR + free spin) rather than a genuine
Case III script, and has been removed; MCMC_code_Case3.py was built fresh from the
Case I template instead, using this old script only as a reference for the spin
prior range and trajectory call conventions.

**Case4_0PAScalarVS-1PNScalar/ (PNstudy)** -> `MCMC_code_Case4.py`. Injection is
always 0PAScalar (SchwarzCirc0PAScalarS, Lambda free, numerical flux). Recovery,
selected via `RECOVERY_MODEL`:
- `"PN"` -> leading-order (-1PN) analytic scalar trajectory (`SchwarzCircPNScalar`,
  newly added to shared_modules/Post_Adiabatic_Trajectories.py for this case - it
  was present in the historical root copy but missing from the shared_modules copy).
  This natively samples **"A"**, NOT Lambda directly: `A = Lambda/12` (see the
  hardcoded relation in `SchwarzCircPNScalar.interpolate_flux_grids`), a rescaling
  specific to how the leading-order PN flux formula expresses the scalar charge.
  The ×12 conversion back to Lambda units happens downstream, in the analysis
  notebook, when producing the published plot - confirmed by checking the trimmed
  data's stored samples (labelled "Λ", mean ≈0.00244, matching true Lambda=0.0025,
  not true A≈0.000208).
- `"0PA"` -> matching-order control recovery (`SchwarzCirc0PAScalarS`, same class as
  injection), which samples Lambda directly, same range as Case I. This is the
  second file present in GOOD_RESULTS/data/Case4_0PAScalarVS-1PNScalar/
  (`0PAdInj0PARec`, a sanity-check run against the PN-mismodeling test).

Two structural changes vs. the historical `MCMC_code_pn.py` (removed):
- Two extra trajectory objects that existed only to print a diagnostic dephasing
  value (never used in the actual injection/recovery likelihood) have been dropped.
  They depended on an external `flux.py` module (Kerr-flavored classes
  `KerrCircEqFluxScalar`, `SchwarzCircPNScalarKerr`) that was never integrated into
  shared_modules/ and is unrelated to the actual generative model.
- p0-backward integration now uses `traj_0PASpin` (spin=0), consistent with the
  convention established for Case I/II/III, instead of the historical script's use
  of the (Kerr, external-module) `KerrCircEqFluxScalar` for that specific step.

**pure_GR/AppendixC1/C2/C3_1PASpinGR/** -> `MCMC_code_AppendixC.py` (identical across all 3
folders except AppendixC3's `DISTANCE_PRIOR_STYLE`, see below). Same spin-scalar
trajectories as Case III. Injection is always 1PASpinScalar with the scenario's
true Lambda (0.0 for all three appendices - a real GR+spin system, no scalar
charge) and true Secondary_Spin (0.5). Recovery, selected via two flags:
- `RECOVERY_ORDER` ("1PA"/"0PA") - which order's spin-scalar trajectory to recover with.
- `FIX_SPIN_ZERO` (True/False) - whether Secondary_Spin is also fixed to 0.0 in
  recovery (pure Schwarzschild GR, testing spin-*neglect* bias) or left free
  (testing PA-*order* bias only). Lambda is *always* fixed to 0.0 in recovery -
  that is the "GR" throughout this whole appendix - using the same
  hardcode-rather-than-switch-class convention as Case II.

Output filename label is derived automatically (`1PASpinGR`, `0PASpinGR`, `1PAGR`,
`0PAGR`), matching the files in GOOD_RESULTS/data/pure_GR/AppendixC1/C2/C3. Each
folder's `PE_EMRI_params.py` must still be edited per scenario, and the two flags
above per recovery variant (AppendixC1 has 1 file, AppendixC2 has 2, AppendixC3
has 3 - see GOOD_RESULTS/data/pure_GR/ for the exact combinations).

Both `data/AppendixC1/C2/C3_1PASpinGR/` and `scripts/AppendixC1/C2/C3_1PASpinGR/`
were nested under a new `pure_GR/` parent folder (i.e.
`data/pure_GR/AppendixC1_1PASpinGR/`, `scripts/pure_GR/AppendixC1_1PASpinGR/`),
grouping all three appendices together as "pure-GR" scenarios, separate from
Case I-IV (which all involve a nonzero scalar charge).

`DISTANCE_PRIOR_STYLE` ("symmetric"/"upper_only") is the one setting that differs
by folder rather than by run: AppendixC3 uses `"upper_only"`
(`uniform_dist(0., dist + 25*Delta_theta_D)`), matching the
"DistanceUpperPrior25*deltaD" suffix found only in AppendixC3's raw filenames;
AppendixC1/C2 use the standard symmetric prior (no such suffix in their filenames).

Two structural notes vs. the historical `MCMC_code_SPIN.py` (removed, was
byte-identical across all 3 folders - i.e. it was just the last-edited state, not
necessarily what produced each folder's own data, same historical-editing issue
seen elsewhere in this project):
- Its `llike()` had a second, alternate implementation kept alive as a dead
  triple-quoted string (not really "commented out" via `#`) - a red herring at
  first glance, since Python doesn't execute string literals as code.
- Dropped an unused `Hybrid_SFPN` import and an unused `traj_GR` trajectory object
  (both dead code, never referenced by the actual generative model).
- p0 is imported directly from PE_EMRI_params.py and used as-is - NOT recomputed
  via backward integration, unlike Case I-IV. The appendix scenarios' p0 values
  were reconstructed from each historical run's true_values, so they are already
  the exact value used; this also matches what the historical script actually did
  (its own backward-integration step was commented out, using a hardcoded numeric
  p0 instead - recomputing it here would risk a slightly different value).
- Kept (and generalized to use `out_dir`) two genuinely useful features from the
  original: a `log_failed_point`/`ERROR_LOG_FILE` mechanism for logging MCMC
  points where waveform generation fails, and saving `priors_*.json`/
  `run_config_*.json` alongside each run's `.h5` output for reproducibility
  (e.g. computing Bayes factors later needs the exact prior bounds used). Both
  have since been generalized and added to Case I-IV too (see above).

**data/pure_GR/checks_Burke/** - a separate, standalone cross-check (not one of
the 3 numbered appendices, no corresponding clean script): for the AppendixC1
scenario (M=1e6, mu=100, Spin=0.5, T=1.5), Susanna had also run the same
injection/recovery using the plain "Schwarzschild" flux model
(FastSchwarzschildEccentricFlux) and a "SchwarzschildBicubic" variant
(FastSchwarzschildEccentricFluxBicubic), to check against Burke et al. - plain
Schwarzschild matched Burke et al., while SchwarzschildBicubic behaved like the
Kerr model instead. Both raw chains and the Schwarzschild corner plot were found
in MCMC_scratch and copied here (originals left in place); the equivalent
SchwarzschildBicubic corner plot could not be found (only a mismatched-mu
waveform-comparison plot, not an MCMC posterior). No launch script could be
recovered for either variant - see `checks_Burke/PROVENANCE.txt` for the full
file-by-file breakdown and caveats.
