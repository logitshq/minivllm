# Design

## Calibration-only / Full-width
Survival estimates use only frozen calibration from full-width-8 DFlash drafts. Verified rounds that were width4 are censored for suffix positions and must not be used to estimate suffix survival.

## Frozen eval
`survival` is frozen empirical unconditional P(A>=j) for j=1..7. `choose_width` is pure greedy rate max `expected/cost`. No online learning, no confidence head.

## Same DFlash block 8
Draft is always 8 tokens unchanged. Policy only chooses verification width 4 or 8 after draft.

## Generation cost vs sunk draft
`round_cost_ns` includes draft8+verify+selection total generation cost. Draft cost is not sunk; both widths incur draft8. Distinction from sunk-draft view documented.

## Future C1/C8 separate calibration
Future widths (C1/C8) require separate calibration due to different verification cost/acceptance distribution; cannot reuse C4/C8 survival directly.

## Formula
expected = 1 + sum(survival[:min(width-1, remaining-1)]); rate=expected/cost; tie->4.

## Validation
Inputs validated: len7, finite 0..1 nonincreasing, remaining positive int non-bool, costs keys exactly {4,8} positive finite non-bool.
