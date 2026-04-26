# Method2 A/B/C Experimental Foundations

## Current cleaned boundary

After the diagnostics cleanup, the Method2-related code is split into three layers:

- `src/faithc_infra/services/uv/method2_pipeline.py`
  - Main production algorithm path.
  - This is the place for changes that alter runtime UV behavior.
- `tools/diagnostics/audit_method2_internal_core.py`
  - Stable audit infra.
  - Mesh loading, metric computation, context construction, report generation, and CLI entry logic.
- `tools/diagnostics/audit_method2_internal_experiments.py`
  - Experiment matrix and one-off diagnostic probes.
  - Field projection variants, residual-space variants, patch tests, repartition tests, etc.
- `tools/diagnostics/run_method2_internal_experiments.py`
  - Cross-case comparison runner.
- `tools/diagnostics/run_method2_massive_diagnostic_suite.py`
  - Massive-focused audit suite runner.

This split is intentional:

- production algorithm stays in `src/`
- reusable audit infra stays in `audit_method2_internal_core.py`
- speculative experiment logic stays in `audit_method2_internal_experiments.py`

That separation should be preserved for A/B/C work.

## Shared requirements before A/B/C diverge

All three routes need a common experimental base. Without this, comparisons will keep drifting.

### 1. Stable benchmark protocol

Need a fixed benchmark set with at least:

- `massive` as the stress case
- `corgi` as the small sanity case
- one medium case with moderate curvature and less extreme topology

Each benchmark should have:

- fixed high mesh
- fixed low mesh
- fixed sanitization policy
- fixed seam strategy
- fixed reporting schema

### 2. Unified evaluation contract

Every route should output the same minimum metrics:

- stretch high quantiles
- out-of-bounds ratio
- sample reprojection error
- Jacobian error against current target field
- Jacobian error against route-specific target field
- flip ratio
- per-island summary
- runtime and memory

For route A, route-specific losses are allowed, but they cannot replace these metrics.

### 3. Comparable artifact export

Each run should export:

- final UV mesh
- structured JSON metrics
- per-face diagnostic arrays when cheap enough
- optional sidecar numpy arrays for deeper postmortem

Without persistent artifacts, cross-route debugging will regress into screenshots and anecdotes.

### 4. Variant registry

Need one explicit place to register experiment variants and their knobs.

Recommended shape:

- route id: `A`, `B`, `C`
- variant name
- config overrides
- output tag
- expected prerequisites

This avoids hard-coded ad hoc branches inside runners.

## Route A: Differentiable / DR track

Route A is not just “another Method2 tweak”. It is a different research mode.

It changes the workflow from:

- hand-designed field proxy
- then solve
- then inspect metrics

into:

- define final UV-space losses
- backprop through solver path
- optimize upstream field or confidence modules directly

### What must exist before Route A experiments are meaningful

#### A1. Differentiable solve path

Need a torch-native or otherwise differentiable implementation of the relevant solve path.

Minimum viable scope:

- differentiable assembly of target field parameters
- differentiable Poisson or Poisson-like solve step
- differentiable post-processing if it materially affects final metrics

If post-align, clipping, or island stitching remains outside the gradient path, Route A results will be misleading.

#### A2. Learnable control surface

Need to decide what Route A is actually optimizing.

Reasonable first candidates:

- per-face confidence weights
- residual injection weights
- smoothness weights
- sample acceptance weights

Do not start by making the entire field directly free unless the baseline is already stable. That makes attribution much harder.

#### A3. Final-loss definition

Need explicit solve-space objectives, not field-space proxies.

Minimum loss bundle:

- stretch penalty
- OOB penalty
- sample reprojection penalty
- Jacobian penalty
- regularization term to prevent degenerate learned masks

This is the whole point of Route A: stop optimizing surrogate field metrics in isolation.

#### A4. Training / optimization harness

Need a dedicated experiment harness, not just the current one-shot audit runner.

Required components:

- step loop
- optimizer config
- checkpointing
- early stopping
- seed control
- loss curve export
- best-checkpoint selection

This should live outside `method2_pipeline.py`.

#### A5. Reproducibility guardrails

Need tighter controls than B/C:

- deterministic seed handling
- explicit mixed precision policy
- per-iteration logging
- gradient norm logging
- NaN / divergence watchdogs

Without this, Route A will be very hard to trust.

### Route A code boundary recommendation

Recommended new locations:

- `src/faithc_infra/services/uv/method2_diff/`
  - differentiable assembly / solve blocks
- `tools/diagnostics/method2_diff_experiments/`
  - training harness and reports

Do not mix this directly into the current production `method2_pipeline.py` until it proves value.

## Route B: Solve-space hard feasibility track

Route B keeps the current Method2 philosophy, but stops relying on soft hints alone.

Its target is clear:

- prevent OOB and large affine drift
- optionally limit area explosion or collapse
- keep internal detail injection compatible with explicit solve constraints

### What must exist before Route B experiments are meaningful

#### B1. Constraint-capable solver interface

Current code can solve Poisson-style systems, but Route B needs a cleaner solver abstraction for constraints.

Need an interface that can express:

- free solve
- fixed boundary solve
- soft boundary penalties
- box-like penalties or barrier terms
- optional per-island constraint policies

This should not be hidden inside experiment-only code.

#### B2. Boundary policy abstraction

Need an explicit boundary policy object or config block.

Examples:

- keep boundary on initial anchor hull
- map boundary to safe box in `[0,1]`
- freeze selected anchor vertices
- penalize only risky boundary segments

Right now these choices are too implicit.

#### B3. Feasibility metrics

Route B is about hard feasibility, so diagnostics must grow accordingly.

Need additional reporting for:

- boundary displacement
- boundary stretch
- area ratio near boundary
- fraction of vertices hitting box limits or barrier-active region
- per-island translation / scale drift proxies

#### B4. Failure-mode comparison harness

Need side-by-side comparison between:

- unconstrained baseline
- soft constrained variants
- hard constrained variants

The main question is not just “did OOB drop”, but “what failure replaced it”.

### Route B code boundary recommendation

Recommended new locations:

- `src/faithc_infra/services/uv/solve_constraints.py`
  - reusable constrained solve assembly
- `tools/diagnostics/audit_method2_internal_experiments.py`
  - early route-B probes

If Route B succeeds, only then should its stable pieces move into `method2_pipeline.py`.

## Route C: Topology intervention / dynamic seam track

Route C also stays in the current Method2 family, but changes topology instead of only changing numbers.

Its purpose is different from B:

- B constrains the solve in the current domain
- C changes the domain itself

This is the only route that directly addresses cases where the current low-mesh UV domain may be too restrictive to carry the injected detail.

### What must exist before Route C experiments are meaningful

#### C1. Seam candidate scoring infra

Need a way to score edges or edge chains as potential release cuts.

Candidate signals:

- residual energy concentration
- persistent edge jump after projection
- stretch concentration after solve
- conflict persistence across adjacent faces

This must be exported in a form that can be inspected, not only used internally.

#### C2. Safe topology edit path

Need a controlled mechanism to:

- cut candidate seams
- rebuild islands
- re-run solve on edited topology
- map diagnostics back to the original face ids when possible

This is nontrivial. Without bookkeeping, Route C becomes impossible to compare fairly.

#### C3. Iterative audit loop

Route C is unlikely to be one-shot.

Need a loop of:

- solve
- locate stress
- cut
- re-solve
- compare improvement vs cut cost

This implies a higher-level experiment controller than the current static variant matrix.

#### C4. Cut budget accounting

Need explicit control of how aggressive topology intervention is.

Metrics should include:

- added seam length
- number of new islands
- UV discontinuity budget
- improvement per unit cut

Without this, Route C can trivially win by over-cutting.

### Route C code boundary recommendation

Recommended new locations:

- `src/faithc_infra/services/uv/seam_optimization.py`
  - seam proposal and topology edit utilities
- `tools/diagnostics/method2_topology_experiments/`
  - iterative route-C controller and reports

Do not hide topology mutation inside field-projector experiments.

## Dependency relationship between A, B, C

A is a different research branch.

- It should not be treated as “one more projector variant”.
- It deserves its own harness and its own experiment lifecycle.

B and C are still descendants of the current Method2 logic.

- B strengthens solve-space feasibility.
- C strengthens topology-space feasibility.

A can eventually consume ideas from B/C, but it should not be blocked by them.

## Recommended build order

If the goal is disciplined experimentation rather than immediate productization, the clean order is:

1. Shared benchmark and artifact infra
2. Route B minimal hard-feasibility prototype
3. Route C seam-candidate audit only, without cutting yet
4. Route A differentiable sandbox on a reduced problem

Reason:

- B has the best chance to clarify whether OOB and drift are mostly solve-feasibility problems.
- C can then test whether unresolved failures are actually topology-limited.
- A should start only after the problem definition is tighter, otherwise it will learn around unclear objectives.

## Bottom line

What is needed next is not more projector variants inside the current monolith.

What is needed is:

- a stable benchmark/evaluation substrate shared by all routes
- a separate differentiable research branch for A
- explicit solve-feasibility infrastructure for B
- explicit topology-edit infrastructure for C

Those are three different experiment programs, not one blended patch series.
