from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


SUPPORTED_ABLATIONS = {
    "none",
    "no_coherence",
    "no_memory",
    "no_relations",
    "no_admissibility",
    "no_regime",
    "single_card",
    "no_residue",
}


def _zero_like_drive(z):
    return torch.zeros_like(z)


@contextmanager
def temporary_ablation(model, name: str) -> Iterator[None]:
    """Temporarily ablate one Neuro-CTNet organ.

    The ablation is intentionally done through runtime patching so v0.1 can test
    non-decoration without rewriting the consolidated core. v0.2 should move
    these switches into first-class module flags.
    """
    if name not in SUPPORTED_ABLATIONS:
        raise ValueError(f"unsupported ablation: {name}")
    if name == "none":
        yield
        return

    patches = []
    old_gain = getattr(model.cfg, "coh_gain", 0.0)

    def patch(obj, attr, value):
        patches.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    try:
        if name == "no_coherence":
            model.set_coh_gain(0.0)

        for block in getattr(model, "blocks", []):
            if name == "no_memory":
                original = block.memory.forward

                def mem_forward(z, M, _original=original):
                    read, summary, next_memory, drive, energy = _original(z, M)
                    return torch.zeros_like(read), torch.zeros_like(summary), M, torch.zeros_like(drive), energy * 0.0

                patch(block.memory, "forward", mem_forward)

            elif name == "no_relations":
                original = block.relations.forward

                def rel_forward(z, R, mem_read, _original=original):
                    read, summary, next_relations, drive, energy = _original(z, R, mem_read)
                    return torch.zeros_like(read), torch.zeros_like(summary), R, torch.zeros_like(drive), energy * 0.0

                patch(block.relations, "forward", rel_forward)

            elif name == "no_admissibility":
                original = block.adm.forward

                def adm_forward(z, ctx, mem, rel, probs, _original=original):
                    za, gate, drive, energy = _original(z, ctx, mem, rel, probs)
                    return z, torch.ones_like(gate), torch.zeros_like(drive), energy * 0.0

                patch(block.adm, "forward", adm_forward)

            elif name == "no_regime":
                original = block.regime.forward

                def regime_forward(zm, ms, rs, prev=None, task=None, _original=original):
                    ctx, probs, ent, sw = _original(zm, ms, rs, prev, task)
                    uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
                    zero_ctx = torch.zeros_like(ctx)
                    return zero_ctx, uniform, ent * 0.0, sw * 0.0

                patch(block.regime, "forward", regime_forward)

            elif name == "no_residue":
                original = block.res.forward

                def res_forward(z, _original=original):
                    nu, energy = _original(z)
                    return torch.zeros_like(nu), energy * 0.0

                patch(block.res, "forward", res_forward)

        if name == "single_card":
            original = model.readout.forward

            def readout_forward(z, ctx, mem, _original=original):
                mixed, weights, energy, entropy = _original(z, ctx, mem)
                one = torch.zeros_like(weights)
                one[:, 0] = 1.0
                cards = torch.stack([head(z) for head in model.readout.heads], 2)
                forced = cards[:, :, 0, :]
                return forced, one, energy * 0.0, entropy * 0.0

            patch(model.readout, "forward", readout_forward)

        yield
    finally:
        model.set_coh_gain(old_gain)
        for obj, attr, value in reversed(patches):
            setattr(obj, attr, value)


def logits_delta_under_ablation(model, input_ids, name: str) -> float:
    with torch.no_grad():
        base = model(input_ids).logits
        with temporary_ablation(model, name):
            ablated = model(input_ids).logits
    return float((base - ablated).abs().mean())
