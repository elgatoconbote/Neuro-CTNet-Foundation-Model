# Implementation contract

This repository must never become a decorative wrapper around a Transformer.

## Rule 1: every organ must act

Each architectural organ must affect at least one of:

- forward transition
- internal state
- loss
- diagnostics
- ablation result

If a component only computes a number that is never used, it is not implemented.

## Rule 2: coherence is causal

The coherence tensor must return mass or drive that enters the next state.

Valid:

```text
Z_next = Z_base + mass * coherence_drive
```

Invalid:

```text
metric = coherence(Z)
log(metric)
```

## Rule 3: memory is pre-output

Memory must act before readout. A vector store or retrieval stage attached only to logits is not Neuro-CTNet memory.

## Rule 4: u/p is reciprocal

The action part and inertia part must reconstruct or constrain each other.

## Rule 5: admissibility legalizes transitions

The admissibility gate must alter the proposed state, not only measure it.

## Rule 6: residue must reenter

Residue must contribute to loss, debt, regime switching or diagnostics. It cannot be ignored.

## Rule 7: multicard readout is projection

The output is a projection from internal state, not the identity of the model.

## Rule 8: v0.2 split must preserve behavior

When `core.py` is split into separate modules, public tests must pass before and after the split. New modules may import from the old core during transition, but the final state should move implementation into module files.

## Rule 9: no claims without ablations

A component is not considered useful until an ablation changes behavior and degrades at least one structural task.
