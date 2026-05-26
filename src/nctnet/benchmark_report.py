from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkVerdict:
    family: str
    status: str
    reason: str


def load_benchmark(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"benchmark must be a list: {path}")
    return data


def verdict_for_row(row: dict, min_win_rate: float = 0.5) -> BenchmarkVerdict:
    loss_win = float(row.get("win_rate_loss", 0.0)) >= min_win_rate
    acc_win = float(row.get("win_rate_accuracy", 0.0)) >= min_win_rate
    dloss = float(row.get("delta_loss_mean", 0.0))
    dacc = float(row.get("delta_accuracy_mean", 0.0))

    if dloss < 0 and dacc > 0 and loss_win and acc_win:
        return BenchmarkVerdict(row["family"], "NCT_STRONG", "lower loss and higher accuracy with sufficient win-rate")
    if dloss < 0 and loss_win:
        return BenchmarkVerdict(row["family"], "NCT_LOSS_ADVANTAGE", "lower loss with sufficient win-rate")
    if dacc > 0 and acc_win:
        return BenchmarkVerdict(row["family"], "NCT_ACCURACY_ADVANTAGE", "higher accuracy with sufficient win-rate")
    if dloss > 0 and dacc < 0:
        return BenchmarkVerdict(row["family"], "BASELINE_ADVANTAGE", "baseline has lower loss and higher accuracy")
    return BenchmarkVerdict(row["family"], "MIXED_OR_INCONCLUSIVE", "mixed metrics or insufficient win-rate")


def make_markdown_report(
    rows: list[dict],
    title: str = "Neuro-CTNet Benchmark Report",
    min_win_rate: float = 0.5,
) -> str:
    verdicts = [verdict_for_row(row, min_win_rate=min_win_rate) for row in rows]
    counts = {}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1

    lines = [
        f"# {title}",
        "",
        "This report is generated from `benchmark.json`.",
        "",
        "## Summary",
        "",
    ]
    for status in sorted(counts):
        lines.append(f"- {status}: {counts[status]}")

    lines += [
        "",
        "## Family table",
        "",
        "| family | delta loss | delta accuracy | win loss | win accuracy | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    verdict_by_family = {v.family: v for v in verdicts}
    for row in rows:
        v = verdict_by_family[row["family"]]
        lines.append(
            f"| {row['family']} | {float(row['delta_loss_mean']):+.6f} | "
            f"{float(row['delta_accuracy_mean']):+.6f} | "
            f"{float(row['win_rate_loss']):.3f} | {float(row['win_rate_accuracy']):.3f} | {v.status} |"
        )

    lines += [
        "",
        "## Verdict details",
        "",
    ]
    for v in verdicts:
        lines.append(f"- **{v.family}**: `{v.status}` — {v.reason}.")

    lines += [
        "",
        "## Reading rules",
        "",
        "- Negative delta loss favours Neuro-CTNet.",
        "- Positive delta accuracy favours Neuro-CTNet.",
        "- Win rates are computed across seeds.",
        "- Debug configs are smoke tests, not scientific claims.",
    ]
    return "\n".join(lines) + "\n"


def write_markdown_report(
    benchmark_json: str | Path,
    out_path: str | Path,
    title: str = "Neuro-CTNet Benchmark Report",
    min_win_rate: float = 0.5,
) -> str:
    rows = load_benchmark(benchmark_json)
    report = make_markdown_report(rows, title=title, min_win_rate=min_win_rate)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return str(out)
