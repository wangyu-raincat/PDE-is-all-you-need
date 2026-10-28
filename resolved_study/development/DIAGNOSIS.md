# Retrospective spatial-resolution diagnosis

This bounded diagnosis uses only already-viewed coarse KS fields from the prior weak-form round (`df9a7c3`). It is development evidence, not a new holdout. No new formal training, development or final initial condition was generated. Frozen old coefficients are not refitted. Reference coefficients appear only in explicitly labelled oracle controls; production `forecast()` accepts no reference equation or future field.

The main script fixes two conditions (clean and 1% noise, seed71), three old final initial conditions, three coefficient sources (reference, old adaptive, old weak), and internal spatial factors1/2/4. Six further clean-reference runs double temporal substeps by requesting twice as many output intervals and restricting back to original times. All60 results are retained in `resolution_diagnosis.json`; no failed or unfavorable result was removed.

Mean final relative L2, excluding the supplied initial frame, in percent:

| Condition and coefficients | Original grid (factor1) | factor2 | factor4 |
|---|---:|---:|---:|
| Clean, reference oracle | 116.1500 | 0.235879 | 0.000000894 |
| Clean, frozen adaptive | 48.4995 | 41.9905 | 41.9913 |
| Clean, frozen weak | 97.9751 | 9.92826 | 9.93170 |
| 1% noise, reference oracle | 115.0987 | 6.27795 | 6.27523 |
| 1% noise, frozen adaptive | 48.7785 | 42.3718 | 42.3725 |
| 1% noise, frozen weak | 87.9687 | 18.5232 | 18.5228 |

Doubling temporal substeps leaves the clean reference factor1 error at116.1500%; the factor4 error becomes0.000000494%. Thus time-step refinement does not explain away the large original-grid error in these three clean trajectories. At factor4, remaining frozen-weak error is materially larger than the reference-oracle error, so better spatial resolution does not erase coefficient error.

Changing internal resolution also changes the original solver's 2/3 initial projection. A separate12-run control first projects the observed initial frame with the original64-point2/3 cutoff, then advances at factor4 using reference or frozen-weak coefficients. `initial_projection_diagnosis.json` preserves all12 rows. For clean data, the projection changes the initial field only by4.73e−16 relative L2 and leaves the respective mean forecast errors at0.000000894% and9.93170%. For1% noise, initial projection changes it by0.0096368 relative L2; reference error is6.26542% and weak error18.51615%, very close to raw-initial refined values6.27523% and18.52280%.

The clean projection control isolates evolution on the coarse spatial grid as a dominant error source for these selected old trajectories. On the coarse grid the biased adaptive coefficients outperform the exact equation; after computational refinement, the ordering reverses. This is consistent with the biased coefficients partly compensating numerical resolution error in this diagnostic, not evidence that those coefficients represent the underlying PDE more accurately. It does not prove a universal explanation for other families or certify chaos.

The new forecast wrapper uses fixed factor4 universally, decided before new formal data preparation. Real periodic Fourier interpolation splits an even-grid Nyquist coefficient correctly, while odd grids keep their conjugate modes. The unchanged ETDRK4 solver then evolves the larger computational grid and output is directly restricted to the original sample locations. Extra grid points are computational degrees of freedom, never measurements or additional fit evidence. No family branch, learned factor, future field or true coefficient enters this rule. Internal grids over2048 points are rejected instead of silently lowering refinement; defaultfactor4 therefore supports at most512 observed points.

Nine independent manufactured tests passed (`forecast-tests.log`): exact high-mode advection/reaction/diffusion, odd/even Nyquist interpolation, nonlinear Burgers Cole–Hopf solution and refinement agreement, constant forcing, exact factor1 reproduction, observation-only interface and input immutability, invalid complex inputs, physical guards and explicit computational budgets. These are numerical/implementation checks, not independent physical experiments. The new round must still freeze its candidate models and evaluate fresh initial conditions.
