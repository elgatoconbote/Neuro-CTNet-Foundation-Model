from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .ablations import SUPPORTED_ABLATIONS, temporary_ablation
from .core import SyntheticLMDataset, collate


@dataclass
class EvalResult:
    name: str
    task_loss: float
    accuracy: float
    tokens: int


def _run_eval(model, loader, ablation: str = "none") -> EvalResult:
    if ablation not in SUPPORTED_ABLATIONS:
        raise ValueError(f"unsupported ablation: {ablation}")

    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad(), temporary_ablation(model, ablation):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(ids)
            logits = out.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
            pred = logits.argmax(dim=-1)
            total_loss += float(loss.cpu())
            total_correct += int((pred == labels).sum().cpu())
            total_tokens += int(labels.numel())

    return EvalResult(
        name=ablation,
        task_loss=total_loss / max(total_tokens, 1),
        accuracy=total_correct / max(total_tokens, 1),
        tokens=total_tokens,
    )


def evaluate_synthetic(
    model,
    seq_len: int = 16,
    size: int = 64,
    batch_size: int = 4,
    ablations: Iterable[str] | None = None,
) -> list[EvalResult]:
    """Evaluate the model and selected ablations on the built-in synthetic LM battery.

    This is deliberately task-loss based. Structural losses are excluded so the
    table measures output degradation, not whether an ablation zeroed its own
    regularizer.
    """
    if ablations is None:
        ablations = [
            "none",
            "no_coherence",
            "no_memory",
            "no_relations",
            "no_admissibility",
            "no_regime",
            "single_card",
            "no_residue",
        ]
    dataset = SyntheticLMDataset(
        vocab_size=min(model.cfg.vocab_size, 256),
        seq_len=seq_len,
        size=size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return [_run_eval(model, loader, ablation=name) for name in ablations]


def format_eval_table(results: list[EvalResult]) -> str:
    if not results:
        return ""
    base = next((r for r in results if r.name == "none"), results[0])
    lines = [
        "ablation\ttask_loss\taccuracy\tdelta_loss\tdelta_accuracy\ttokens"
    ]
    for r in results:
        lines.append(
            f"{r.name}\t{r.task_loss:.6f}\t{r.accuracy:.6f}\t"
            f"{(r.task_loss - base.task_loss):+.6f}\t{(r.accuracy - base.accuracy):+.6f}\t{r.tokens}"
        )
    return "\n".join(lines)
