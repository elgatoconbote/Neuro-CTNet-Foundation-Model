# Ablations

Neuro-CTNet components must not be decorative. Each organ must be removable, and removal should change the model trajectory or logits.

## Supported v0.3 ablations

```text
no_coherence
no_memory
no_relations
no_admissibility
no_regime
single_card
no_residue
```

## Run

```bash
python -m nctnet.cli train-lm --config configs/debug.yaml

python -m nctnet.cli ablate --checkpoint runs/debug/best.pt --ablation no_memory
python -m nctnet.cli ablate --checkpoint runs/debug/best.pt --ablation no_relations
python -m nctnet.cli ablate --checkpoint runs/debug/best.pt --ablation no_admissibility
python -m nctnet.cli ablate --checkpoint runs/debug/best.pt --ablation no_regime
python -m nctnet.cli ablate --checkpoint runs/debug/best.pt --ablation single_card
python -m nctnet.cli ablate --checkpoint runs/debug/best.pt --ablation no_residue
python -m nctnet.cli ablate --checkpoint runs/debug/best.pt --ablation no_coherence
```

## Interpretation

`mean_abs_delta` measures the mean absolute difference between baseline logits and ablated logits for a fixed probe sequence.

A nonzero delta proves the organ affects behavior. It does not yet prove that the organ improves task quality. Improvement requires matched training and task-level evaluation.

## Next step

v0.4 should add task-level ablation evaluation over the synthetic structural tasks.
