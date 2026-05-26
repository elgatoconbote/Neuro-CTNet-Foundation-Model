from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .ablations import SUPPORTED_ABLATIONS, temporary_ablation
from .core import SyntheticLMDataset, collate
from .task_families import TASK_FAMILIES, SyntheticTaskFamilyDataset


@dataclass
class EvalResult:
    name: str
    task_loss: float
    accuracy: float
    tokens: int
    family: str = "mixed"


DEFAULT_ABLATIONS = [
    "none",
    "no_coherence",
    "no_memory",
    "no_relations",
    "no_admissibility",
    "no_regime",
    "single_card",
    "no_residue",
]


def _run_eval(model, loader, ablation: str = "none", family: str = "mixed") -> EvalResult:
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
        family=family,
    )


def _normalise_ablations(ablations: Iterable[str] | None) -> list[str]:
    return list(DEFAULT_ABLATIONS if ablations is None else ablations)


def evaluate_synthetic(
    model,
    seq_len: int = 16,
    size: int = 64,
    batch_size: int = 4,
    ablations: Iterable[str] | None = None,
) -> list[EvalResult]:
    """Evaluate the model and selected ablations on the mixed synthetic battery."""
    ablations = _normalise_ablations(ablations)
    dataset = SyntheticLMDataset(
        vocab_size=min(model.cfg.vocab_size, 256),
        seq_len=seq_len,
        size=size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return [_run_eval(model, loader, ablation=name, family="mixed") for name in ablations]


def evaluate_synthetic_families(
    model,
    seq_len: int = 16,
    size: int = 64,
    batch_size: int = 4,
    ablations: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
) -> list[EvalResult]:
    """Evaluate full vs ablated model per named synthetic task family."""
    ablations = _normalise_ablations(ablations)
    families = list(TASK_FAMILIES if families is None else families)
    results: list[EvalResult] = []
    for family in families:
        dataset = SyntheticTaskFamilyDataset(
            family=family,
            vocab_size=min(model.cfg.vocab_size, 256),
            seq_len=seq_len,
            size=size,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
        for ablation in ablations:
            results.append(_run_eval(model, loader, ablation=ablation, family=family))
    return results


def format_eval_table(results: list[EvalResult]) -> str:
    if not results:
        return ""
    base_by_family = {}
    for r in results:
        if r.name == "none":
            base_by_family[r.family] = r
    lines = [
        "family\tablation\ttask_loss\taccuracy\tdelta_loss\tdelta_accuracy\ttokens"
    ]
    for r in results:
        base = base_by_family.get(r.family, r)
        lines.append(
            f"{r.family}\t{r.name}\t{r.task_loss:.6f}\t{r.accuracy:.6f}\t"
            f"{(r.task_loss - base.task_loss):+.6f}\t{(r.accuracy - base.accuracy):+.6f}\t{r.tokens}"
        )
    return "\n".join(lines)
