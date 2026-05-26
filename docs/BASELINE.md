# Baseline Transformer

v0.6 adds a small conventional causal Transformer baseline.

It intentionally does not include:

```text
u/p
folded memory
relation bank
regime controller
admissibility gate
coherence tensor
residue
multicard readout
```

## Train and report Neuro-CTNet

```bash
python -m nctnet.cli train-eval-report --config configs/debug.yaml --seq-len 16 --size 64 --batch-size 4
```

Writes:

```text
runs/debug/best.pt
runs/debug/eval_family.tsv
runs/debug/eval_family.json
```

## Train and report baseline

```bash
python -m nctnet.cli train-baseline-report --config configs/debug.yaml --seq-len 16 --size 64 --batch-size 4
```

Writes:

```text
runs/debug/baseline.pt
runs/debug/baseline_eval_family.tsv
runs/debug/baseline_eval_family.json
```

## Interpretation

This baseline is not intended to be a final competitive Transformer. It is a same-config smoke baseline to prevent Neuro-CTNet from only comparing against its own ablations.

The next step is a combined comparison command that prints both tables and deltas in one report.
