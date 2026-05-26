# Roadmap

## v0.1

Implemented as consolidated core:

- package metadata
- public exports
- debug and tiny LM configs
- u/p split
- reversible additive coupling
- u/p reciprocity
- topological memory
- relation bank
- regime controller
- admissibility gate
- residue projector
- causal coherence tensor
- dynamic mass
- coherence drive
- multicard readout
- tiny language model
- synthetic dataset
- trainer
- CLI
- smoke tests

## v0.2

Split the consolidated implementation into modules while preserving behavior:

- config
- state
- tensors
- reversible
- up
- memory
- relations
- regime
- admissibility
- coherence
- residual
- multicard
- blocks
- language model
- trainer
- CLI

Acceptance: tests pass before and after the split.

## v0.3

Add real ablation switches for coherence, u/p, memory, relations, admissibility, regime, cards and residue. Each ablation must change logits and degrade at least one synthetic structural task.

## v0.4

Add active-chart attention and explicit constant-cost contracts. Full attention, local attention, sparse attention and active-chart attention must be clearly separated.

## v0.5

Add tiny image prototype: patch embedding, NCT core, latent denoising head, synthetic shape dataset and coherence/residue maps.

## v1.0

A release counts only if the structural components affect transition, ablations degrade measurable tasks, coherence mass stays healthy, selectors avoid collapse, memory matters, and residue correlates with unresolved structure.
