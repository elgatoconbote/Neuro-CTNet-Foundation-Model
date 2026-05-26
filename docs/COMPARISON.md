# Comparing Neuro-CTNet with the Transformer baseline

v0.7 adds report comparison.

## Full smoke pipeline

```bash
python -m nctnet.cli train-eval-report --config configs/debug.yaml --seq-len 16 --size 64 --batch-size 4
python -m nctnet.cli train-baseline-report --config configs/debug.yaml --seq-len 16 --size 64 --batch-size 4
python -m nctnet.cli compare-reports \
  --nct runs/debug/eval_family.json \
  --baseline runs/debug/baseline_eval_family.json \
  --out-dir runs/debug
```

## Outputs

```text
runs/debug/comparison.tsv
runs/debug/comparison.json
```

## Table columns

```text
family
nct_loss
baseline_loss
delta_loss
nct_accuracy
baseline_accuracy
delta_accuracy
nct_tokens
baseline_tokens
```

Negative `delta_loss` means Neuro-CTNet has lower loss than the baseline for that family.

Positive `delta_accuracy` means Neuro-CTNet has higher token accuracy than the baseline for that family.

## Important

The debug configuration is a smoke test, not a scientific benchmark. Real comparison requires matched training budgets, repeated seeds, larger synthetic sizes and confidence intervals.
