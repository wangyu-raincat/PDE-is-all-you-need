# Independent pre-result scientific review

Review date: 2026-11-03 UTC. Scope: the proposed information-acquisition
strategy and its preregistered comparison, inspected before formal data
generation. This reviewer inspected source and existing manufactured-test
definitions; no formal trajectory, recovery result, or final forecast was
generated or inspected, and no acquisition/estimation code was edited.

## Conclusion before results

The local relative-coefficient design calculation is internally consistent.
The retained controls and prospective coefficient endpoint avoid claiming a
cost saving below the previous fixed policy's one-query lower bound. The
design can answer whether this complete acquisition policy improves
same-measurement-budget coefficient recovery on new draws from the six
existing synthetic families. It cannot establish calibrated information
gain, support correctness, or improvement on real measurements.

At the first inspection, `assess.py` and the final frozen `protocol.json`
were not yet present. The assessor subsequently became available for
inspection, and the implementation findings below were fixed before formal
generation. No unresolved mathematical or scientific-design blocker was
found in the reviewed final code. The coordinating agent must still finish
the manufactured checks and freeze the completed protocol before generation.

## Parameter scaling and information accumulation

`acquisition.py:136` computes all scales from currently acquired training
observations. With selected coefficients `c`, common observed target scale
`q`, and feature scales `s`, its construction is

```text
theta_j = abs(c_j) * s_j / q
Z_ij = (X_ij / s_j) * theta_j = X_ij * abs(c_j) / q
G = sum_i Z_i.T @ Z_i / rows_i
G_new = G + Z_a.T @ Z_a / rows_a
```

This is the local design matrix for coordinate changes `delta c / abs(c)`
in the response scaled by `q`. Using the same scales, selected coefficients,
and ridge for every candidate in a round makes their scores comparable.
Feature rescaling accompanied by inverse coefficient rescaling leaves this
construction unchanged. The intermediate `s` does not secretly normalize
each candidate independently.

Each equal-cost trajectory contributes one mean-row Gram. Contributions are
summed, rather than renormalizing all old data and the new experiment to the
same total evidence. Duplicating acquired trajectories therefore doubles
the observed Gram. This is a consistent equal-trajectory design convention;
overlapping weak rows are not being counted as independent experiments.

The stable trace-reduction identity at `acquisition.py:90` is correct. For
`A = G + ridge I`, `C = A^(-1/2) G_a A^(-1/2)`, and eigenpairs `(mu,v)` of
`C`, the code sums `mu/(1+mu) * norm(A^(-1/2)v)^2`. This equals
`trace(A^-1) - trace((A+G_a)^-1)` while avoiding subtraction cancellation.
Finite/PSD checks, positive shared ridge, and monotonicity checks are
appropriate. Ridge cannot establish identifiability; the separate observed
support-rank check is retained.

The inverse trace is an L2-style local relative-parameter proxy. The formal
endpoint is the mean of taskwise maximum true-term errors, an L-infinity
quantity. Optimizing the former does not guarantee improving the latter.
This mismatch is a limitation to test prospectively, not a justification
for changing the primary metric after seeing results.

## Forecast and weak-window approximations

The two uniform integrations at `acquisition.py:316`, first `[0, .1]` and
then 33 frames over `[.1, 1.1]`, respect the unchanged solver's uniform-grid
contract. The first interval uses the public controlled initial condition;
it does not obtain a hidden measured state. The candidate score receives no
future measurement.

The predicted weak Gram is an approximation to the measured weak Gram in
more ways than a smaller frame count. Under the unchanged resolved weak
transform (`weak_study/engine.py:57`), 101 frames produce 24-interval windows
of duration 0.24 with stride 0.06. The 33-frame prediction produces
8-interval windows of duration 0.25 with stride 0.0625. The last window also
ends at a different time. Equal weak-row counts do not make the quadrature
and window geometry identical. Noise-free predicted fields and noisy
observed fields can also receive different spectral filtering.

These choices are prospectively declared and do not invalidate a policy
comparison. They prohibit interpreting the score as the exact information
of the forthcoming 101-frame observation. Do not change resolution or
window geometry after formal results to rescue a failed policy. Any later
refinement requires another independent study.

## What the ablation does and does not isolate

`rich_then_variance` and `rich_then_information` exactly match fixed-rich's
first additional action. Shared observation noise and fitting seeds should
make their budget-six selected models identical; artifact audits should
verify this, rather than relying only on action names.

The ablation controls the rich opening. It does not isolate the mathematical
A-opt score alone: the information policy also uses a single selected
anchor and a longer prediction horizon, whereas the original ensemble
criterion uses a committee and its original short horizon. Any benefit is
attributable to the complete continuation policy. No claim that the score
alone caused the benefit is supported by these six arms.

All arms have equal measurement budgets. Their simulation effort is not
equal. Forecast counts and compute time remain separate diagnostics; a
measurement saving or equal measurement cost is not a CPU/FLOP saving.

## Failure behavior and support risk

Absent/invalid anchors, deficient observed support, zero excitation,
nonfinite calculations, and all-invalid or zero-score actions explicitly
fall back to the original unused fixed-rich action. Individual failed
predictions stay in the action ledger; the policy does not silently replace
the anchor to turn a failed candidate into a successful one.

A well-conditioned Gram only describes the selected support. It can be
well-conditioned for a wrong equation, and a missing true term has no
relative-parameter direction in this score. An extra term can absorb design
effort. The unchanged discovery engine may correct structure after another
query, but the score offers no guarantee that it will. A singleton support
set and bootstrap selection frequency remain stability diagnostics, not
truth probabilities or identified physical laws.

Legal queries consume budget even if acquisition or observation later
fails. Missing budget checkpoints retain their slots, missing true
coefficients retain 100% error, and forecast failures cannot be deleted to
improve a mean or pass promotion. All fallback and failed-query counts
should remain in the final report.

## Data roles and prospective decisions

The provider's separate development role distinguishes inspected pilot
parameters from formal indices 120--137 even where integer indices match.
Formal acquisition consumes observation-only views; reference terms are
used only by the offline scorer. All six-policy episodes and budget slots
must freeze before final fields are generated. The global release receipt
must remain intact after interruption or reproduction; a new output folder
does not create unseen evidence.

There are 18 independent parameter draws, three in each of six families.
Paired 1% and 3% noise levels produce 36 conditions, not 36 independent
physical systems. The primary budget-nine 1% mean is distinct from the 3%
stress test. The predeclared task-stratified resampling preserves pairing,
but three tasks per family supports only a descriptive small-benchmark
interval, not calibrated population guarantees.

The reused source-calibration file is byte-identical to the prior active
study (`SHA256 9850c7d6b39ae4ad37d1aeb9ed8c1f4dfa4340c013cba7cb0c1c64b32dc87c17`).
All three bound calibration dependencies (actions, provider, and independent
coefficient generator) still match their declared source hashes. The twelve
reserved calibration comparisons passed; their largest relative difference
was `1.657236120407634e-12` and largest absolute difference was
`5.341727060681478e-12`. Reusing those checks for unchanged code is appropriate.
They cover one reserved parameter draw per family and two initial conditions,
not every new formal parameter/action combination or the continuous domain.

Five of the six families are linear; the reaction-diffusion family here
contains `u` and `u_xx`. Burgers is the sole nonlinear family. Linear
references use analytic Fourier evolution and the nonlinear provider uses
the separately implemented refined integrator. Source calibration does not
validate the acquisition predictor or turn same-family synthetic parameter
draws into evidence for unknown equation families, nonperiodic boundaries,
higher dimensions, correlated/heteroscedastic noise, or sparse sensors.

`decision_protocol.json` requires a primary reduction of at least 10%
versus both fixed-rich and rich-then-variance, no worse primary performance
than random or either original active control, structure guards at budgets
six and nine for both noises, terminal coefficient-tail and stress guards,
and available predictions below 5% for every terminal condition. All gates
are conjunctive; passing grants review eligibility, not automatic merge.
The old 80% joint-success cost target and its no-saving conclusion stay
unchanged. These are credible prospective standards for the stated scope.

## Completion check

The draft `assess.py` was subsequently inspected. It independently
recalculates term errors, retains all 1080 declared identities, uses the
18-task primary denominator, separates stress noise, implements each
structure/tail/forecast comparison, and keeps paired resampling descriptive.
The source-freeze inventory already includes both the assessor and decision
document. No formal preparation or final-release receipt existed at this
inspection.

Resolved finding: the initial CLI bound the prior audit to comparison bytes,
but did not bind the currently loaded assessor/decision document to
`freeze_index.json`. Changing criteria after a successful audit could reuse
the old audit to evaluate new thresholds. The revised CLI reads the existing
freeze receipt, verifies its hash against the comparison, requires the
current assessor and criteria hashes to match its `sources` mapping, and
checks all bytes again before publication. The source-binding gate fails
closed without matching evidence. Manufactured tests now cover post-audit
criteria, assessor, and freeze-receipt changes. The static fix is correct and
requires no field or training module access.

Resolved boundary finding: the coordinating agent separately identified
that a required-improvement control with mean at or below numerical
tolerance cannot support a claimed 10% gain. The revised assessor fails that
gate and reports relative improvement as null for both zero and positive
near-zero means. Equality at zero is not promoted as improvement.

This reviewer did not run the manufactured tests and does not claim their
execution as independent evidence. Their implementation agent supplies the
run results; the coordinating agent owns aggregate regression checks and
the final freeze. At this review snapshot the three inspected source hashes
were:

```text
information_study/acquisition.py
a84cab76019a86f382f6d0a92848b8590ae7e3744dfb801be5df9750b8026a68
information_study/assess.py
d610038a8381a66f09dc1507d94348d6a379c9c0d1c91e1cc507865c1f99edd5
information_study/decision_protocol.json
3d1c430e01f137a4638961aa0d1e150cdb6af1d22430ba6ab97c7b7a5da5658e
```

No result-dependent change to the prospective standards is proposed.
