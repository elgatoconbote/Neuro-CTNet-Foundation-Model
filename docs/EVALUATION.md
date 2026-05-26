# Synthetic evaluation

The v0.3 evaluation harness measures task loss and token accuracy for the full model and each structural ablation.

This is different from `ablate`, which only measures whether logits change for one probe sequence.

## Run

```bash
python -m nctnet.cli train-lm --config configs/debug.yaml
python -m nctnet.cli eval-synthetic --checkpoint runs/debug/best.pt --seq-len 16 --size 64 --batch-size 4
```

## Output

The table contains:

```text
ablation
task_loss
accuracy
delta_loss
delta_accuracy
tokens
```

`delta_loss` and `delta_accuracy` are measured against the full model row named `none`.

## Interpretation

A structural organ passes the first non-decoration test if disabling it changes logits.

It passes the stronger task-level test only if disabling it worsens task loss or accuracy on at least one synthetic task family after matched training.

v0.3 provides the harness. v0.4 should split the synthetic dataset into named task families and report per-family degradation.
