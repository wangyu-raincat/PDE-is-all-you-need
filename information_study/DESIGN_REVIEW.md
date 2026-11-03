# Pre-result design review: relative-coefficient experiment design

This review covers the independent `information_study` pipeline and its
connection to the new acquisition policies. It uses source inspection and
manufactured tests. The reviewer did not generate or inspect formal reference
fields. Passing these checks establishes implementation and protocol behavior,
not an improvement in PDE recovery.

## Scope and causal comparison

The existing four policies are retained: fixed rich excitation, random
selection, ensemble variance, and structural disagreement. Two additional
policies share exactly the fixed-rich first experiment, then use either
ensemble variance or the new relative-coefficient design proxy. This
`rich_then_variance` comparison helps separate the benefit of the common rich
opening from the benefit of the information-directed continuation.

All six policies use the unchanged `active_study` estimator, action space,
reference provider, and forecast implementation. The estimator still uses the
same weak formulation, adaptive L1 procedure, support alternatives, spectral
filtering, and eight trajectory-bootstrap draws. No old scientific file or
module global is changed by the new runtime. The new controller and runner
are independent modules with their own paths and lifecycle state.

The first chosen action, paired measurement noise, bootstrap seed, and
subsequent acquired training inputs are identical between fixed-rich and the
two rich-opening policies at budget six. Their selected models at that budget
should therefore match. The frozen artifacts must confirm that invariant;
later policy differences begin with the second additional experiment.

## What the new score means

The new policy conditions on the current full-data selected support and
coefficients. It constructs a relative-coefficient weak-feature Gram matrix
from acquired training trajectories and uses a predicted candidate experiment
to estimate the reduction in a regularized inverse-Gram trace. The selected
model is observation-fitted; the reference equation and future measurements
are not arguments to the acquisition function.

The observed feature scales, target scale, coefficient scale, and ridge are
fixed across candidate actions within a round. Candidate predictions use one
selected model, followed by the existing weak-feature transform. The projected
experiment uses 33 post-prefix frames over the full duration 1.0 as a declared
design approximation. Actual discovery and final validation retain the
original 101 future-window frames and 11 historical frames. Computation does
not create additional measured evidence.

This score is a local design proxy conditional on a possibly incorrect support
and coefficient estimate. Overlapping weak windows are correlated, the feature
matrix contains observation error, and filtering changes the noise structure.
Its inverse Gram is not a calibrated covariance, Fisher-information matrix,
posterior uncertainty, or guaranteed information gain. A positive score does
not demonstrate that the support is correct or that coefficient errors will
decrease. Empty, deficient, invalid, or uninformative geometry must remain
explicitly recorded when the policy falls back to fixed-rich excitation.

## New tasks and equal budgets

The formal task indices are 120 through 137 under the provider's formal role:
18 independent parameter draws, three per each of the six existing families.
The old active study used indices 0 through 11. Each new parameter task has
paired 1% and 3% noise conditions, giving 36 conditions, not 36 independent
parameter draws. The standard deviations are respectively 0.004 and 0.012,
because noise is scaled to the fixed initial RMS 0.4, not the evolving signal.

Development uses the provider's separate development role at indices 120
through 125. Its role-specific parameter and noise seeds differ from formal
generation even when the integer indices match. These six cases support
pipeline checks with two additions per policy; inspected development outcomes
are retrospective and cannot establish final performance.

The new initial-action, pool, learner, final-action, and final-noise seed bases
are 957000, 956000, 958000, 959000, and 960000. The actual parameter index is
used without rebasing to zero; final noise uses `960000 + 100*index + j` inside
the unchanged provider noise-seed convention. Same-action query noise remains
paired across policies.

Every policy pays for three initial training and two development trajectories,
then at most four additional queries. Each legal query reserves 111 distinct
time points at 64 sensors, or 7104 scalar measurements. The shared history
endpoint is counted once. Failed legal queries still consume an attempt;
invalid or repeated proposals are rejected before querying. Full budgets five
through nine are retained, including missing-checkpoint failure slots.

The complete formal comparison consists of 216 episodes, 1080 declared budget
slots, 54 clean final reference fields, and 3240 forecast assessments. Final
validation costs are additional offline costs. All actions have equal
measurement cost, so this experiment does not test heterogeneous prices or
sensor-location optimization.

## Freeze, validation, and reuse

The new runner binds its own controller, acquisition, runner, protocol, and
references together with every reused scientific dependency. The offline
`assess.py` implementation and `decision_protocol.json` are also included in
the scientific source freeze; the reporting gates cannot be changed after
looking at final results without invalidating that freeze. Prepared tasks
are checked against their noise level, action allocation, observation bytes,
provider receipts, and complete initial success/failure partition. The frozen
ledger reconstructs each checkpoint's training hashes and measurement counts
from its paid selected queries.

Every episode and budget declaration is frozen before any new final reference
field is generated. Final references are shared across all policies and
budgets for a task. Forecast functions receive only historical observations
ending at the prediction cutoff and then forecast the full subsequent duration
1.0. This remains prefix-conditioned prediction.

`information_study/data/final_release.json` is created exclusively before the
first final-generation attempt. It cannot be reset by changing output
directories. Its scientific-source and deterministic-task identity survives
a fresh preparation timestamp in another checkout, while retaining both the
original and current preparation hashes. Repeated or interrupted-release
attempts are conservatively labeled reproduction rather than unseen evidence.
The release marker must be retained when distributing this fixed benchmark.

The reused twelve-case source calibration checks the unchanged reference
provider on reserved cases. It does not validate the new information proxy,
establish performance, or guarantee numerical accuracy across the entire
continuous parameter/action domain.

## Predeclared outcome interpretation

The primary outcome is the budget-nine, 1%-noise mean across all 18 parameter
tasks of each task's maximum relative error over true coefficients. Missing
selected models remain failures, with missing true coefficients represented
in the recovery metrics; no task is dropped because forecasting or selection
failed. The 3% condition is a separately reported stress test.

Promotion requires at least a 10% primary improvement over both fixed-rich
and rich-then-variance, no worse performance than the other retained controls,
and the predeclared structure and forecasting guardrails. Those decisions are
offline report calculations, never inputs to online stopping or acquisition.
All exact-support, extra/missing-term, candidate-set coverage/size, wrong
singleton, forecast failure, and prediction-error results remain visible.

The previous 80% joint-success cost comparison and the 5% prediction threshold
are retained unchanged as secondary measurements. This new coefficient target
does not rewrite the old active study's no-saving conclusion or the earlier
KS 5.4104% result. A coefficient gain alone cannot excuse worse structural
recovery. No clear benefit means preserving the independent experiment without
merging it or retuning against these final fields.

Bootstrap inclusion frequencies remain descriptive selection stability, not
calibrated truth probabilities. The independent statistical unit is a
parameter task; two noise levels, six policies, five budgets, and repeated
final trajectories do not create independent tasks. Any task-resampling
intervals describe uncertainty on this small controlled synthetic benchmark.
No result establishes real-measurement validation, general PDE discovery, or
completed Huawei platform integration.

## Verification record

The independent static migration review found no old task-count, seed, path,
or nonzero development-index mismatch. The old `active_study` files remained
unchanged. New controller and pipeline tests retain the previous legality,
failure accounting, metadata isolation, snapshot, and release checks and add
six-policy matching, nonzero-index development, new design allocation, and
primary coefficient-field failure behavior. The affected suite passed:

```text
python -m unittest tests.test_information_controller tests.test_information_pipeline -v
Ran 26 tests in 59.013s
OK
```

The suite comprises 15 controller and 11 pipeline checks. Its complete
manufactured allocation includes 216 episodes and 1080 budget slots, with
210 initial-source-failure episodes retained. Formal provider generation is
mocked throughout. Compilation checks also passed. No blocking lifecycle
finding remains in the reviewed code. The formal result artifacts require a
separate audit after generation.
