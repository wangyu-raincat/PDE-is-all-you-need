# Retrospective weak-block development

All trials below used four already-viewed challenge conditions and their existing training/development roles only. No fresh training, development or final reference fields were generated or read by this subtask. Reference equations were used only to describe retrospective errors after fitting/forecast selection, never passed to `weak_block`, `fit_blocks` or `select_candidates`.

Three bounded sweeps of 12 fits each are retained:

1. `engine-v1.py`, `probe-v1-results.json`, `probe.log`: modes Nx/8 or Nx/4; quarter-time or 40%-time temporal support. Quarter-time, Nx/8 reduced Allen–Cahn 0.1% diffusion error from the earlier 49.53% to 1.49%; coarse clean Burgers/KS coefficient biases also reduced. At Allen–Cahn 1% all these variants omitted diffusion. Broad temporal support did not improve these selected outcomes.
2. `probe-low.py`, `probe-v2-results.json`, `probe-v2.log`: quarter-time modes Nx/32, Nx/16 and 40%-time Nx/32, minimum2. Very low modes retained Allen–Cahn diffusion under 1% noise, but at 59.4% coefficient error. For coarse KS, only2–4 modes caused poor fourth-derivative estimates, even though development forecast RMSE could prefer those models. This exposed a limitation of selecting solely by long-horizon forecast.
3. `probe-final.py`, `probe-v3-results.json`, `probe-v3.log`: final fixed menu compact/low/resolved at quarter-time support and `min(Nx//4, max(8, floor(fraction*Nx)))`, fractions 1/8,1/32,1/4. Minimum8 is applied to every family, with Nx//4 cap for small grids. It prevents the two-mode KS case without a family/label-dependent rule. These are the only final menu variants.

The caller pools candidate models from all three final variants and uses the unchanged development rule, rather than using any per-case oracle choice. The tables in these trial JSONs show each variant individually and are not claims about the pooled selector or unseen final data. This development inspected metrics and therefore is explicitly retrospective; full retrospective and fresh held-out assessments are separate root workflow steps.

No noise-variance correction was implemented. In particular no known corruption sigma, clean field or PDE coefficient reaches the block builder. Polynomial moments come from the observed trigonometric interpolant, with 2N padding to avoid folding high products into retained modes. Fourier spatial tests and compact temporal test derivatives avoid pointwise observed derivatives. Degree-nine composite Newton–Cotes uses existing samples only; it remains a finite-resolution quadrature, not exact integration of arbitrary signals. Independent fast-wave tests demonstrate refinement convergence and also show that a very coarse temporal grid can still be inaccurate.

`final-engine-tests.log`: 9 independent analytic/aliasing/input tests passed. No existing project modules were modified.
