# Synthetic structural tasks

The first proof of Neuro-CTNet should not be generic language modeling. It should be a controlled battery where each organ has a reason to exist.

## 1. Persistent entity task

A sequence defines entities and later queries their attributes after distractors.

Expected pressure:

- folded memory
- regime switching
- residue when distractors conflict

## 2. Rule switching task

The sequence changes rule midway. The model must detect the active rule and apply only the current regime.

Expected pressure:

- regime controller
- multicard selector
- admissibility gate

## 3. Inversion and reciprocity task

The input requires forward and inverse consistency.

Expected pressure:

- u/p reciprocity
- reversible coupling
- reversibility loss

## 4. Contradiction task

The sequence states facts and later gives conflicting statements. The model must avoid treating both as equally admissible.

Expected pressure:

- admissibility
- residue
- coherence mass

## 5. Long delayed copy task

The model must preserve structure over delay without relying only on adjacent tokens.

Expected pressure:

- topological memory
- relation bank

## 6. Multi-hop relation task

The answer requires composing multiple relations.

Expected pressure:

- relation bank
- memory read
- coherence tensor

## Acceptance

For each task, run full model and ablations. The full model should outperform the corresponding ablated model after matched training budget.
