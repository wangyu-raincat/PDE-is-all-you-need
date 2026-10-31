# Independent scientific design review

This review concerns the new `active_study` experiment only. It examines the
observation boundary, acquisition comparison, budget accounting, uncertainty
semantics, and final-validation lifecycle. The reviewer did not generate or
inspect formal trajectories or final performance results. Implementation tests
use manufactured observations and mocks; they are not experimental evidence
that an acquisition policy improves PDE recovery.

## What this experiment can establish

The study compares four initial-condition selection policies while holding the
discovery estimator, candidate library, preprocessing, bootstrap procedure,
initial observations, observation geometry, and available action pool fixed.
The three comparators are deterministic rich excitation, uniform random
selection without replacement, and an E-SINDy-inspired prediction-variance
policy. The proposed policy uses between-structure prediction disagreement
relative to within-structure coefficient variation and an observed noise
estimate. This quantity is a discrimination heuristic, not Bayesian information
gain. The E-SINDy-inspired comparator is not an exact reproduction of a
published implementation.

This design can identify differences caused by acquisition within the stated
synthetic setting. It cannot attribute the shared estimator's improvements to
active selection. It also cannot establish novelty merely from combining
ensembles, inclusion frequencies, and active experiments; related E-SINDy work
already combines those ideas.

The scope is six known families inside one constant-coefficient, scalar,
autonomous, periodic nine-term library. The initial measurements intentionally
use weak, single-mode excitation, which makes several spatial operators
observationally ambiguous. This is a useful controlled identifiability task;
it is not a representative random sample of all PDE-discovery problems.
Additional initial conditions can be informative, but boundary-condition and
sensor-location optimization are not implemented in this phase.

## Observation and truth separation

`provider.py` owns private coefficients and generates only the requested
observation. `engine.py`, `acquisition.py`, and `controller.py` do not import the
reference provider. The controller passes measured coordinates and fields to
discovery and acquired training observations to acquisition. Only the selected
action invokes the query callback. Candidate rollouts for every proposed action
are computations under discovered equations, not observations of the hidden
system.

The runner may retain private task specifications and reference fields for
generation and later scoring. Their presence in runner-owned receipts is not
permission to pass them to selection. Independent controller tests check that
extra family labels, true coefficients, noise labels, and clean fields supplied
alongside observations are removed from learning inputs.

The provider uses exact diagonal Fourier solutions for linear reference
families and a separately calibrated reference integrator for Burgers. Finer
spatial/time reference comparisons test selected reserved cases. These checks
do not establish a uniform error bound for every continuous parameter or action
in the allowed domain.

## Fairness and experiment cost

Each policy starts with the same three training and two development
trajectories. All five count toward its acquisition budget. The common initial
fit is copied for each policy; subsequent bootstrap seeds use the same
task-and-round convention. Noise for an identical chosen action is paired
between policies. The action pool and its bounds are common to every policy.

An experiment contains 64 spatial measurements at 111 distinct time points.
The 11-frame history and the 101-frame forecast window share one endpoint:
`(11 + 101 - 1) * 64 = 7104` acquired scalar observations. Thus budgets five
through nine correspond to 35,520 through 63,936 reserved measurements. Failed
legal queries consume an attempt and the reserved measurement budget; their
returned-data count is separately zero. Invalid or repeated actions are
rejected before a provider call, and their selection failure is retained.

Every legal action has the same geometry and duration. Dividing acquisition
scores by cost therefore does not change the ranking in this experiment. The
study evaluates constrained experiment counts, not heterogeneous-cost or
sensor-count optimization. Reference calibration, development probes, and
final validation are additional offline costs and must remain separately
visible; they are not free physical experiments.

Four phase-distinct, broad-spectrum fixed actions form a strong predetermined
baseline. If this baseline resolves ambiguity with one additional experiment,
the active policy may have no data-saving advantage. A negative result must
remain in the report. Savings are meaningful only at the same predeclared
recovery target; a policy that never reaches that target cannot be assigned a
fictitious crossing budget by dropping failures.

## Uncertainty interpretation

The engine resamples whole training trajectories and reruns sparse structure
selection for every distinct bootstrap multiplicity vector. Exact-repeat
resamples can reuse a deterministic fit. Repeated lambda paths and identical
coefficient vectors do not acquire extra committee votes. Correlated-support
alternatives are retained explicitly instead of equating one selected support
with unique identifiability.

Eight bootstrap draws provide coarse empirical selection frequencies. With
three initial training trajectories there are only ten possible unordered
bootstrap multiplicity vectors. Frequencies are not calibrated posterior
probabilities, and correlated observations from one trajectory are not treated
as independent bootstrap experiments. Failed bootstrap refits are recorded;
unconditional frequencies are unavailable when failures would make their
denominator misleading, with separate conditional diagnostics and bounds.

Near-optimal alternatives are a set compatible with the current observations,
library, fitting procedure, and tolerance. They are not a proof of global PDE
equivalence. A singleton committee remains a candidate, not a claim that the
physical equation is unique. Omitted library terms remain possible. The
controller and acquisition explicitly preserve this distinction when falling
back to coverage exploration after an empty, singleton, or zero-disagreement
committee.

Candidate-set coverage must be reported together with its size. Singleton
wrong answers, missing true terms, and extra terms remain important even if
forecast error is small. Positive- and negative-term Brier scores computed from
raw frequencies are descriptive frequency-quality diagnostics on this
benchmark; they do not certify probability calibration.

The shared spectral filter assumes white measurement noise and weak physical
energy in its upper-band estimator. It can suppress weak physical modes and
can mistake genuine high-frequency structure for noise. That assumption and
the retained-mode diagnostics must remain visible. The stated 1% noise is a
standard deviation of 1% of the fixed initial RMS 0.4, namely 0.004; it is not
1% of each evolving trajectory's RMS.

## Validation and reporting constraints

There are 24 conditions but only 12 parameter tasks: six families, two
parameter realizations each, and paired clean/noisy conditions. Policies,
noise conditions, budgets, and final trajectories reuse those tasks. They must
not be presented as independent samples for significance claims.

All policy episodes and their declared budget slots must be frozen before
generation of the first final reference field. Source/protocol/input hashes
and episode hashes bind the evaluation to that freeze. Failed initial
generation or acquisition must remain visible in all declared denominators;
missing budget states count as failures rather than disappearing. The first
successful budget computed using the known reference is an offline evaluation
statistic, not a deployable stopping rule.

The three final trajectories for each parameter task are shared between all
policies and budgets. Forecasts receive only the observed history ending at
the cutoff and retain the full subsequent duration 1.0. The comparison is
therefore prefix-conditioned forecasting, with 11 observed historical frames;
it is not a single-frame or unconstrained forecast. Fine computational grids
and interpolated states are not additional measured evidence.

Exact support, extra/missing terms, and true-coefficient errors are primary.
Prediction error is auxiliary. The predeclared compound success threshold is
exact support, maximum true-coefficient relative error at most 10%, and mean
relative forecast error over three held-out trajectories at most 5%. Reporting
the separate metrics prevents that compound threshold from hiding a tradeoff.

Development cases already inspected during implementation are retrospective.
Once the fixed final seeds have been evaluated, later use of their data must
also be labeled reproduction or retrospective. A new output directory does
not make a previously released final task unseen again.

No result from these synthetic tasks demonstrates real-world measurement
validation, completed Huawei integration, general unknown-PDE recovery, or
resolution of the earlier KS 5.4104% prediction result. The existing baseline
remains a separate rollback reference. No automatic merge is justified by
passing implementation tests.

## Independent test record and resolved review findings

The independently written `tests/test_active_controller.py` covers cost at
every budget, started-query failures, action legality and uniqueness, metadata
isolation, shared-fit copy isolation, paired deterministic seeds, changed
measurement geometry, invalid history endpoints, failed discovery, and callback
mutation isolation. An added real-valued-input test reproduced a silent
complex-to-real conversion at the controller boundary. The controller now
rejects complex coordinates, fields, and historical observations before any
conversion. All 12 controller tests pass.

The runner review requested and implemented three release checks before formal
data generation:

1. Prepared tasks now bind noise conditions, declared action pools, initial
   action allocation, ordered initial observations, raw measurement receipts,
   and the complete five-attempt success/failure partition to the registered
   deterministic design.
2. Every frozen episode declares five budget slots. Missing checkpoints are
   explicit failure slots. The freeze verifier checks query actions against
   proposals, charges every legal attempt, checks each returned observation's
   raw bytes and geometry, and reconstructs the training hashes and available
   measurement counts at every checkpoint.
3. A study-wide `data/final_release.json` is created exclusively before the
   first final-reference generation attempt. It survives interrupted attempts
   and changes of output directory. Its source and deterministic-task binding
   also survives a new preparation timestamp in a fresh checkout. The first
   and current preparation hashes remain separately traceable; a repeated
   release is labeled retrospective reproduction. This marker must be retained
   with the study when distributing its fixed final tasks.

`tests/test_active_pipeline.py` adds eight manufactured-fixture checks. They
exercise a complete 96-episode allocation with 92 initial-failure episodes,
all 480 budget declarations, altered inputs and charges despite refreshed
container hashes, release-before-reference ordering, interrupted-release
handling, new-output reproduction, and fresh-preparation reproduction. Hidden
reference generation is mocked throughout; these are protocol tests, not
formal data. The final affected-test run was:

```text
python -m unittest tests.test_active_controller tests.test_active_pipeline -v
Ran 20 tests in 25.707s
OK
```

No blocking design finding remains in this reviewed scope. This conclusion
does not replace the aggregate regression checks or the independent audit of
the subsequently frozen experiment artifacts.

This document is a pre-result review. Aggregate test counts, numerical results,
artifact-audit findings, and any integration decision belong to the final
run's audit and report.
