# Reports

`train-eval-report` runs the complete v0.5 smoke pipeline:

1. train a tiny Neuro-CTNet language model
2. evaluate by synthetic task family
3. evaluate full model against structural ablations
4. write TSV and JSON artifacts

## Run

```bash
python -m nctnet.cli train-eval-report --config configs/debug.yaml --seq-len 16 --size 64 --batch-size 4
```

## Artifacts

The command writes:

```text
runs/debug/best.pt
runs/debug/eval_family.tsv
runs/debug/eval_family.json
```

## Why this matters

`ablate` only proves that an organ changes logits.

`eval-synthetic --by-family` proves that a checkpoint can be evaluated per task family.

`train-eval-report` makes the whole experiment reproducible and leaves artifacts that can be compared across commits.

## Next step

Add a baseline Transformer and write `baseline_eval_family.tsv` next to the Neuro-CTNet report.
