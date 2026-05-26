# Neuro-CTNet Architecture

This document fixes the v0.1 engineering contract.

## State

The model state is not a flat hidden stream. It is an operational tuple:

```text
Omega_l = (Z_l, M_l, R_l, rho_l, A_l, C_l, nu_l)
```

where:

- `Z_l` is the distributed hidden state.
- `M_l` is folded topological memory.
- `R_l` is the reified relation bank.
- `rho_l` is the slow regime distribution.
- `A_l` is admissibility.
- `C_l` is causal coherence.
- `nu_l` is operative residue.

## Transition

```text
Z_l -> [u_l, p_l]
[u_l, p_l] -> reversible additive coupling
Z, M, R -> regime
Z, M, R, regime -> admissibility
Z, u/p, M, R, admissibility, residue -> coherence tensor
coherence -> mass + drive
state -> multicard readout
```

## Non-decoration rule

A component is valid only if it is used by at least one of:

```text
forward transition
loss
state update
diagnostics
ablation
```

A module that only logs values is not part of Neuro-CTNet.

## v0.1 status

`src/nctnet/core.py` contains the consolidated implementation. The public modules expose stable names and allow later internal separation without breaking imports.
