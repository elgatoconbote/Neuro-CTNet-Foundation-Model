# Benchmark markdown report

v0.9 adds automatic Markdown reports from `benchmark.json`.

## Run full benchmark

```bash
python -m nctnet.cli run-benchmark \
  --config configs/debug.yaml \
  --seeds 0,1,2 \
  --seq-len 16 \
  --size 64 \
  --batch-size 4 \
  --out-dir runs/benchmark
```

## Generate Markdown report

```bash
python -m nctnet.cli benchmark-report \
  --benchmark runs/benchmark/benchmark.json \
  --out runs/benchmark/BENCHMARK_REPORT.md \
  --min-win-rate 0.5
```

## Verdicts

```text
NCT_STRONG
NCT_LOSS_ADVANTAGE
NCT_ACCURACY_ADVANTAGE
BASELINE_ADVANTAGE
MIXED_OR_INCONCLUSIVE
```

## Rules

- Negative delta loss favours Neuro-CTNet.
- Positive delta accuracy favours Neuro-CTNet.
- Win-rate checks are computed across seeds.
- Debug configs are smoke tests, not scientific claims.
