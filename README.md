# Neuro-CTNet Foundation Model

Neuro-CTNet is not a Transformer with decorative losses. It is a foundation-model architecture whose transition state explicitly contains action/inertia partitioning (`u/p`), folded topological memory, reified relations, slow regime, admissibility, causal coherence tensor, dynamic mass, residue and multicard projection.

Core flow:

```text
input -> distributed state Z
Z -> [u,p]
[u,p] -> reversible/semi-reversible transition
Z,M,R -> regime + admissibility
Z,M,R,A,rho,residue -> causal coherence tensor
coherence -> mass + drive
state -> multicard projected output
residue -> structural loss and diagnostics
```

Minimal run:

```bash
python -m pip install -e .[dev]
pytest -q
python -m nctnet.cli train-lm --config configs/debug.yaml
python -m nctnet.cli inspect --checkpoint runs/debug/best.pt --prompt "1 2 3 4"
```

A component only counts as implemented if it is used in forward, loss/drive, diagnostics and ablation.
