from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComparisonRow:
    family: str
    nct_loss: float
    baseline_loss: float
    delta_loss: float
    nct_accuracy: float
    baseline_accuracy: float
    delta_accuracy: float
    nct_tokens: int
    baseline_tokens: int


def _load_report(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"report must be a list: {path}")
    return data


def _index_full_rows(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        if row.get("ablation") == "none":
            out[row["family"]] = row
    return out


def compare_reports(nct_json: str | Path, baseline_json: str | Path) -> list[ComparisonRow]:
    nct = _index_full_rows(_load_report(nct_json))
    base = _index_full_rows(_load_report(baseline_json))
    families = sorted(set(nct) & set(base))
    rows: list[ComparisonRow] = []
    for family in families:
        n = nct[family]
        b = base[family]
        rows.append(
            ComparisonRow(
                family=family,
                nct_loss=float(n["task_loss"]),
                baseline_loss=float(b["task_loss"]),
                delta_loss=float(n["task_loss"]) - float(b["task_loss"]),
                nct_accuracy=float(n["accuracy"]),
                baseline_accuracy=float(b["accuracy"]),
                delta_accuracy=float(n["accuracy"]) - float(b["accuracy"]),
                nct_tokens=int(n["tokens"]),
                baseline_tokens=int(b["tokens"]),
            )
        )
    return rows


def format_comparison_table(rows: list[ComparisonRow]) -> str:
    lines = [
        "family\tnct_loss\tbaseline_loss\tdelta_loss\tnct_accuracy\tbaseline_accuracy\tdelta_accuracy\tnct_tokens\tbaseline_tokens"
    ]
    for row in rows:
        lines.append(
            f"{row.family}\t{row.nct_loss:.6f}\t{row.baseline_loss:.6f}\t{row.delta_loss:+.6f}\t"
            f"{row.nct_accuracy:.6f}\t{row.baseline_accuracy:.6f}\t{row.delta_accuracy:+.6f}\t"
            f"{row.nct_tokens}\t{row.baseline_tokens}"
        )
    return "\n".join(lines)


def write_comparison(rows: list[ComparisonRow], out_dir: str | Path, stem: str = "comparison") -> tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / f"{stem}.tsv"
    js = out / f"{stem}.json"
    tsv.write_text(format_comparison_table(rows) + "\n", encoding="utf-8")
    js.write_text(
        json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(tsv), str(js)
