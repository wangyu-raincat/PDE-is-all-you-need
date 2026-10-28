# Prefix study design review

Reviewed on 2026-10-28 before formal data generation. Scope: `forecast.py`,
`engine.py`, `data.py`, `generators.py`, `run.py`, `protocol.json`, and
`tests/test_prefix_study.py`. This is a read-only scientific/code review; only
this review document was added. The reviewer previously authored the reused
single-frame filter, so review of that component is not fully independent.

**Decision: no blocking design or implementation issue found in the reviewed
workflow.** The protocol is deliberately `draft`; the formal `prepare` command
must remain blocked until its status is explicitly frozen. A validation-only
check with an in-memory status change passed every code/protocol/calibration
binding. There were zero formal clean arrays, no prepared receipt, and no
global model-freeze index at review time. No arrays or forecasts were generated
by this review. This is permission to proceed with the declared experiment,
not evidence that the 5% target will be achieved.

## Isolation and horizon

- The state estimator accepts only coordinates, candidate terms, and observed
  prefix frames. Each frame is filtered, propagated using the candidate PDE to
  the cutoff, and equally averaged there. Future values are absent from the
  prediction interface. The shared endpoint is one observed frame; propagated
  estimates are correctly described as correlated, not independent samples.
- Weak fitting receives only each training trajectory's post-cutoff segment.
  Candidate medians are training-only. Observed development futures select
  candidates, using the same predictor subsequently used for final evaluation.
  Final truth is used only by corruption/scoring, after model freeze.
- Generated timestamps span `0..1.1T`, with cutoff at `0.1T`. The predicted
  future still spans a full `T`: Allen–Cahn 0.5, Burgers 1.5, KS 20. Final
  scoring explicitly rejects a shortened horizon and excludes the initial
  forecast frame. Noise is added before the common history/future split and
  before nested spatial/temporal subsampling.
- Protocol and 22 source/dependency hashes bind preparation. The complete 120
  model identities, source/input hashes, and model digests are verified before
  any of the nine final reference trajectories are generated. There is no
  final-data selection or fallback to a better final model.

## Target, guards, and comparisons

- The target is every one of 30 condition means over three final trajectories
  being at most 5%; an individual trajectory may exceed 5%. Failed forecasts
  prevent a condition from passing. The code retains 157 checks: source
  quality, 30 times five condition checks, and six family coefficient-tail
  checks. Exact support, per-true-term coefficient accuracy, and coefficient
  regression checks against both controls remain necessary. Small prediction
  error cannot override a wrong PDE. A pass only permits review; regression
  tests and merge decisions remain separate.
- Within this study all four arms share training data, development/final
  observations, cutoff, and full future horizon. Prefix arms intentionally
  consume 11 full-grid or six coarse-grid historical frames; the controls use
  one cutoff frame. This is the user-approved conditional prediction task and
  must not be presented as an equal-information-budget or single-frame result.
- The forecast-start state has already evolved for `0.1T`. Although the
  generator's initial distribution at the *start of the prefix* is unchanged,
  the cutoff-state distribution differs from previous single-frame rounds.
  Attribute gains using the current paired controls; do not attribute an
  entire cross-round difference to the algorithm. Earlier viewed-data probes
  remain retrospective.

## Coverage and remaining limits

The 21 reviewed tests cover history boundaries, unchanged inputs, shared
endpoint and nested sampling, future-to-predictor isolation, fitting-input
separation, prepare/freeze ordering, inclusive condition-mean threshold, PDE
guards, unchanged future duration, and serial/parallel scoring agreement.
The parent reports 21 focused and 396 aggregate tests passing; this reviewer
did not rerun those suites. The independent validation-only binding check
described above passed.

The sparse-spectrum filter can remove genuine weak or broadband modes.
Candidate-dependent state propagation can introduce model/state compensation;
the retained PDE guards reduce but do not eliminate this limitation. The
results will apply only to the declared three known synthetic families and
their sampled conditions, not arbitrary PDEs or measured physical data. Final
trajectories and condition noise realizations are reused across comparisons;
360 forecasts are not 360 independent experiments.

One minor API issue is nonblocking for the fixed generator: prefix timestamps
are checked for order and cutoff, but uniform spacing is rejected later by
the underlying solver rather than immediately by `validate_prefix`. Generated
prefixes are uniformly spaced, so this does not affect the declared study.
