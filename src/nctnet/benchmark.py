from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .baseline import train_baseline_report
from .compare import compare_reports
from .core import NCTConfig, TrainConfig
from .report import train_eval_report


@dataclass
class BenchmarkRow:
    family: str
    runs: int
    nct_loss_mean: float
    baseline_loss_mean: float
    delta_loss_mean: float
    delta_loss_std: float
    nct_accuracy_mean: float
    baseline_accuracy_mean: float
    delta_accuracy_mean: float
    delta_accuracy_std: float
    win_rate_loss: float
    win_rate_accuracy: float


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _aggregate(comparisons_by_seed: list[list]) -> list[BenchmarkRow]:
    by_family: dict[str, list] = {}
    for rows in comparisons_by_seed:
        for row in rows:
            by_family.setdefault(row.family, []).append(row)

    out: list[BenchmarkRow] = []
    for family, rows in sorted(by_family.items()):
        delta_loss = [r.delta_loss for r in rows]
        delta_acc = [r.delta_accuracy for r in rows]
        out.append(
            BenchmarkRow(
                family=family,
                runs=len(rows),
                nct_loss_mean=_mean([r.nct_loss for r in rows]),
                baseline_loss_mean=_mean([r.baseline_loss for r in rows]),
                delta_loss_mean=_mean(delta_loss),
                delta_loss_std=_std(delta_loss),
                nct_accuracy_mean=_mean([r.nct_accuracy for r in rows]),
                baseline_accuracy_mean=_mean([r.baseline_accuracy for r in rows]),
                delta_accuracy_mean=_mean(delta_acc),
                delta_accuracy_std=_std(delta_acc),
                win_rate_loss=sum(1 for x in delta_loss if x < 0) / max(len(delta_loss), 1),
                win_rate_accuracy=sum(1 for x in delta_acc if x > 0) / max(len(delta_acc), 1),
            )
        )
    return out


def format_benchmark_table(rows: list[BenchmarkRow]) -> str:
    lines = [
        "family\truns\tnct_loss_mean\tbaseline_loss_mean\tdelta_loss_mean\tdelta_loss_std\t"
        "nct_accuracy_mean\tbaseline_accuracy_mean\tdelta_accuracy_mean\tdelta_accuracy_std\twin_rate_loss\twin_rate_accuracy"
    ]
    for r in rows:
        lines.append(
            f"{r.family}\t{r.runs}\t{r.nct_loss_mean:.6f}\t{r.baseline_loss_mean:.6f}\t"
            f"{r.delta_loss_mean:+.6f}\t{r.delta_loss_std:.6f}\t"
            f"{r.nct_accuracy_mean:.6f}\t{r.baseline_accuracy_mean:.6f}\t"
            f"{r.delta_accuracy_mean:+.6f}\t{r.delta_accuracy_std:.6f}\t"
            f"{r.win_rate_loss:.6f}\t{r.win_rate_accuracy:.6f}"
        )
    return "\n".join(lines)


def run_benchmark(
    model_cfg: NCTConfig,
    train_cfg: TrainConfig,
    seeds: list[int],
    seq_len: int = 16,
    size: int = 64,
    batch_size: int = 4,
    out_dir: str | Path = "runs/benchmark",
) -> tuple[list[BenchmarkRow], str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparisons_by_seed = []

    for seed in seeds:
        seed_dir = out / f"seed_{seed}"
        nct_train = TrainConfig(**{**train_cfg.__dict__, "seed": seed, "run_dir": str(seed_dir / "nct")})
        base_train = TrainConfig(**{**train_cfg.__dict__, "seed": seed, "run_dir": str(seed_dir / "baseline")})

        nct_paths, _ = train_eval_report(model_cfg, nct_train, seq_len=seq_len, size=size, batch_size=batch_size)
        base_paths, _ = train_baseline_report(model_cfg, base_train, seq_len=seq_len, size=size, batch_size=batch_size)
        comparisons_by_seed.append(compare_reports(nct_paths.json, base_paths.json))

    rows = _aggregate(comparisons_by_seed)
    tsv = out / "benchmark.tsv"
    js = out / "benchmark.json"
    tsv.write_text(format_benchmark_table(rows) + "\n", encoding="utf-8")
    js.write_text(json.dumps([r.__dict__ for r in rows], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, str(tsv), str(js)
