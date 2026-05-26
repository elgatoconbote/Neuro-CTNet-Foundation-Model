from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .core import NCTConfig, NCTLanguageModel, TrainConfig, train_tiny_lm
from .eval import EvalResult, evaluate_synthetic_families, format_eval_table


@dataclass
class ReportPaths:
    checkpoint: str
    tsv: str
    json: str


def _result_to_dict(result: EvalResult) -> dict:
    return {
        "family": result.family,
        "ablation": result.name,
        "task_loss": result.task_loss,
        "accuracy": result.accuracy,
        "tokens": result.tokens,
    }


def write_eval_report(results: list[EvalResult], run_dir: str | Path, stem: str = "eval_family") -> ReportPaths:
    """Write TSV and JSON evaluation artifacts for a run directory."""
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    tsv_path = run / f"{stem}.tsv"
    json_path = run / f"{stem}.json"

    tsv_path.write_text(format_eval_table(results) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps([_result_to_dict(r) for r in results], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ReportPaths(checkpoint=str(run / "best.pt"), tsv=str(tsv_path), json=str(json_path))


def train_eval_report(
    model_cfg: NCTConfig,
    train_cfg: TrainConfig,
    seq_len: int = 16,
    size: int = 64,
    batch_size: int = 4,
) -> tuple[ReportPaths, list[EvalResult]]:
    """Train a tiny Neuro-CTNet LM and emit by-family evaluation artifacts."""
    checkpoint = train_tiny_lm(model_cfg, train_cfg)
    saved = torch.load(checkpoint, map_location="cpu")
    cfg = NCTConfig(**saved["config"])
    model = NCTLanguageModel(cfg)
    model.load_state_dict(saved["model"])
    model.eval()

    results = evaluate_synthetic_families(
        model,
        seq_len=seq_len,
        size=size,
        batch_size=batch_size,
    )
    paths = write_eval_report(results, train_cfg.run_dir)
    return paths, results
