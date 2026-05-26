from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .core import NCTConfig, TrainConfig, SyntheticLMDataset, collate
from .eval import evaluate_synthetic_families
from .report import ReportPaths, write_eval_report


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    """Small conventional causal Transformer baseline.

    It intentionally has no u/p, folded memory, relation bank, regime,
    admissibility, coherence tensor, residue or multicard projection.
    """

    def __init__(self, cfg: NCTConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids, labels=None):
        batch, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        h = self.tok(input_ids) + self.pos(pos)
        mask = self._causal_mask(seq_len, input_ids.device)
        for block in self.blocks:
            h = block(h, mask=mask)
        logits = self.head(self.norm(h))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return type("TransformerLMOutput", (), {"loss": loss, "logits": logits})()


def train_baseline_lm(model_cfg: NCTConfig, train_cfg: TrainConfig):
    torch.manual_seed(train_cfg.seed)
    Path(train_cfg.run_dir).mkdir(parents=True, exist_ok=True)
    model = TinyTransformerLM(model_cfg)
    dataset = SyntheticLMDataset(
        vocab_size=min(model_cfg.vocab_size, 256),
        seq_len=train_cfg.seq_len,
        size=max(32, train_cfg.steps * train_cfg.batch_size),
    )
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    iterator = iter(loader)
    for _ in tqdm(range(train_cfg.steps), desc="baseline-train", leave=False):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        out = model(batch["input_ids"], labels=batch["labels"])
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    path = Path(train_cfg.run_dir) / "baseline.pt"
    torch.save({"model": model.state_dict(), "config": model_cfg.__dict__, "kind": "tiny_transformer_baseline"}, path)
    return path


def train_baseline_report(
    model_cfg: NCTConfig,
    train_cfg: TrainConfig,
    seq_len: int = 16,
    size: int = 64,
    batch_size: int = 4,
) -> tuple[ReportPaths, list]:
    checkpoint = train_baseline_lm(model_cfg, train_cfg)
    saved = torch.load(checkpoint, map_location="cpu")
    cfg = NCTConfig(**saved["config"])
    model = TinyTransformerLM(cfg)
    model.load_state_dict(saved["model"])
    model.eval()
    results = evaluate_synthetic_families(
        model,
        seq_len=seq_len,
        size=size,
        batch_size=batch_size,
    )
    paths = write_eval_report(results, train_cfg.run_dir, stem="baseline_eval_family")
    return ReportPaths(checkpoint=str(checkpoint), tsv=paths.tsv, json=paths.json), results
