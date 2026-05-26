from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset


TASK_FAMILIES = (
    "persistent_entity",
    "rule_switching",
    "inversion_reciprocity",
    "contradiction",
    "delayed_copy",
    "multi_hop_relation",
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    expected_organs: tuple[str, ...]
    description: str


TASK_SPECS = {
    "persistent_entity": TaskSpec(
        name="persistent_entity",
        expected_organs=("memory", "regime", "residue"),
        description="Entities and attributes must persist across distractors.",
    ),
    "rule_switching": TaskSpec(
        name="rule_switching",
        expected_organs=("regime", "multicard", "admissibility"),
        description="The active transformation rule changes inside the sequence.",
    ),
    "inversion_reciprocity": TaskSpec(
        name="inversion_reciprocity",
        expected_organs=("u/p", "reversibility"),
        description="Forward and inverse structure must remain mutually consistent.",
    ),
    "contradiction": TaskSpec(
        name="contradiction",
        expected_organs=("admissibility", "residue", "coherence"),
        description="Conflicting assertions create pressure on admissibility and residue.",
    ),
    "delayed_copy": TaskSpec(
        name="delayed_copy",
        expected_organs=("memory", "relations"),
        description="Earlier symbols must be recovered after a long distractor span.",
    ),
    "multi_hop_relation": TaskSpec(
        name="multi_hop_relation",
        expected_organs=("relations", "memory", "coherence"),
        description="The answer depends on composing multiple symbolic relations.",
    ),
}


def _clip_token(x: int, vocab_size: int) -> int:
    return int(x % vocab_size)


def _persistent_entity(idx: int, seq_len: int, vocab_size: int) -> list[int]:
    entity = 11 + (idx % 7)
    attr = 40 + ((idx * 3) % 13)
    distractor = [70 + ((idx + i * 5) % 31) for i in range(max(seq_len - 6, 1))]
    seq = [3, entity, 4, attr] + distractor + [5, entity, attr]
    return [_clip_token(x, vocab_size) for x in seq[: seq_len + 1]]


def _rule_switching(idx: int, seq_len: int, vocab_size: int) -> list[int]:
    base = 20 + (idx % 17)
    seq = [7, base]
    x = base
    while len(seq) < seq_len + 1:
        if len(seq) == max(4, seq_len // 2):
            seq.append(8)
        if 8 in seq:
            x = x - 1
        else:
            x = x + 2
        seq.append(x)
    return [_clip_token(x, vocab_size) for x in seq[: seq_len + 1]]


def _inversion_reciprocity(idx: int, seq_len: int, vocab_size: int) -> list[int]:
    a = 30 + (idx % 11)
    b = 30 + ((idx + 3) % 11)
    pattern = [9, a, b, 10, b, a]
    seq = (pattern * ((seq_len + len(pattern)) // len(pattern)))[: seq_len + 1]
    return [_clip_token(x, vocab_size) for x in seq]


def _contradiction(idx: int, seq_len: int, vocab_size: int) -> list[int]:
    entity = 50 + (idx % 9)
    true_attr = 90 + (idx % 17)
    false_attr = 120 + (idx % 17)
    filler = [140 + ((idx + i) % 23) for i in range(max(seq_len - 8, 1))]
    seq = [12, entity, true_attr] + filler[: len(filler) // 2] + [13, entity, false_attr] + filler[len(filler) // 2 :] + [14, entity, true_attr]
    return [_clip_token(x, vocab_size) for x in seq[: seq_len + 1]]


def _delayed_copy(idx: int, seq_len: int, vocab_size: int) -> list[int]:
    key = 160 + (idx % 19)
    gap = [180 + ((idx * 7 + i) % 41) for i in range(max(seq_len - 3, 1))]
    seq = [15, key] + gap + [key]
    return [_clip_token(x, vocab_size) for x in seq[: seq_len + 1]]


def _multi_hop_relation(idx: int, seq_len: int, vocab_size: int) -> list[int]:
    a = 30 + (idx % 10)
    b = 60 + (idx % 10)
    c = 90 + (idx % 10)
    d = 120 + (idx % 10)
    chain = [16, a, b, 16, b, c, 16, c, d, 17, a, d]
    filler = [200 + ((idx + i * 3) % 37) for i in range(max(seq_len + 1 - len(chain), 0))]
    seq = chain[:-2] + filler + chain[-2:]
    return [_clip_token(x, vocab_size) for x in seq[: seq_len + 1]]


GENERATORS: dict[str, Callable[[int, int, int], list[int]]] = {
    "persistent_entity": _persistent_entity,
    "rule_switching": _rule_switching,
    "inversion_reciprocity": _inversion_reciprocity,
    "contradiction": _contradiction,
    "delayed_copy": _delayed_copy,
    "multi_hop_relation": _multi_hop_relation,
}


class SyntheticTaskFamilyDataset(Dataset):
    def __init__(self, family: str, vocab_size: int = 256, seq_len: int = 32, size: int = 1024):
        if family not in GENERATORS:
            raise ValueError(f"unknown synthetic task family: {family}")
        self.family = family
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
        self.generator = GENERATORS[family]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        seq = self.generator(idx, self.seq_len, self.vocab_size)
        if len(seq) < self.seq_len + 1:
            seq = seq + [0] * (self.seq_len + 1 - len(seq))
        ids = torch.tensor(seq[: self.seq_len], dtype=torch.long)
        labels = torch.tensor(seq[1 : self.seq_len + 1], dtype=torch.long)
        return {"input_ids": ids, "labels": labels, "family": self.family}
