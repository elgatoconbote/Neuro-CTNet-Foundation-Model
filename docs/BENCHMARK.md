# Multi-seed benchmark

v0.8 adds repeated-seed benchmarking.

The benchmark trains Neuro-CTNet and the Transformer baseline for each seed, evaluates both by synthetic family, compares reports and aggregates mean, standard deviation and win rates.

## Run

```bash
python -m nctnet.cli run-benchmark \
  --config configs/debug.yaml \
  --seeds 0,1,2 \
  --seq-len 16 \
  --size 64 \
  --batch-size 4 \
  --out-dir runs/benchmark
```

## Outputs

```text
runs/benchmark/benchmark.tsv
runs/benchmark/benchmark.json
runs/benchmark/seed_0/nct/eval_family.json
runs/benchmark/seed_0/baseline/baseline_eval_family.json
...
```

## Columns

```text
family
runs
nct_loss_mean
baseline_loss_mean
delta_loss_mean
delta_loss_std
nct_accuracy_mean
baseline_accuracy_mean
delta_accuracy_mean
delta_accuracy_std
win_rate_loss
win_rate_accuracy
```

Negative `delta_loss_mean` favours Neuro-CTNet.

Positive `delta_accuracy_mean` favours Neuro-CTNet.

`win_rate_loss` is the fraction of seeds where Neuro-CTNet loss is lower than baseline.

`win_rate_accuracy` is the fraction of seeds where Neuro-CTNet accuracy is higher than baseline.

## Warning

The debug config is still a smoke benchmark. A serious benchmark requires more steps, larger task size, more seeds and fixed compute budgets.
