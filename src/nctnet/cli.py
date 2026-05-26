from __future__ import annotations

import argparse
import torch

from .core import NCTConfig, NCTLanguageModel, TrainConfig, _load_cfg, train_tiny_lm
from .ablations import SUPPORTED_ABLATIONS, logits_delta_under_ablation


def main():
    parser = argparse.ArgumentParser("nctnet")
    sub = parser.add_subparsers(required=True)

    train = sub.add_parser("train-lm")
    train.add_argument("--config", required=True)

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--prompt", default="1 2 3 4")

    ablate = sub.add_parser("ablate")
    ablate.add_argument("--checkpoint", required=True)
    ablate.add_argument("--ablation", default="no_coherence", choices=sorted(SUPPORTED_ABLATIONS - {"none"}))

    args = parser.parse_args()

    if hasattr(args, "config"):
        raw = _load_cfg(args.config)
        checkpoint = train_tiny_lm(NCTConfig(**raw.get("model", {})), TrainConfig(**raw.get("train", {})))
        print(f"saved={checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = NCTConfig(**checkpoint["config"])
    model = NCTLanguageModel(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if hasattr(args, "prompt"):
        toks = [int(x) % cfg.vocab_size for x in args.prompt.split() if x.lstrip("-").isdigit()] or [1, 2, 3, 4]
        out = model(torch.tensor(toks).unsqueeze(0))
        print("logits_shape", tuple(out.logits.shape))
        print("top_next", torch.topk(out.logits[0, -1], min(5, cfg.vocab_size)).indices.tolist())
        for key, value in out.metrics.items():
            print(f"{key}: {value:.6f}")
        return

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    delta = logits_delta_under_ablation(model, ids, args.ablation)
    print("ablation", args.ablation)
    print("mean_abs_delta", delta)


if __name__ == "__main__":
    main()
