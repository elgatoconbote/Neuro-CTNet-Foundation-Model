from __future__ import annotations

import argparse
import torch

from .core import NCTConfig, NCTLanguageModel, TrainConfig, _load_cfg, train_tiny_lm
from .ablations import SUPPORTED_ABLATIONS, logits_delta_under_ablation
from .eval import evaluate_synthetic, evaluate_synthetic_families, format_eval_table


def _load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = NCTConfig(**checkpoint["config"])
    model = NCTLanguageModel(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg


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

    ev = sub.add_parser("eval-synthetic")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--seq-len", type=int, default=16)
    ev.add_argument("--size", type=int, default=64)
    ev.add_argument("--batch-size", type=int, default=4)
    ev.add_argument("--by-family", action="store_true")

    args = parser.parse_args()

    if hasattr(args, "config"):
        raw = _load_cfg(args.config)
        checkpoint = train_tiny_lm(NCTConfig(**raw.get("model", {})), TrainConfig(**raw.get("train", {})))
        print(f"saved={checkpoint}")
        return

    model, cfg = _load_model(args.checkpoint)

    if hasattr(args, "prompt"):
        toks = [int(x) % cfg.vocab_size for x in args.prompt.split() if x.lstrip("-").isdigit()] or [1, 2, 3, 4]
        out = model(torch.tensor(toks).unsqueeze(0))
        print("logits_shape", tuple(out.logits.shape))
        print("top_next", torch.topk(out.logits[0, -1], min(5, cfg.vocab_size)).indices.tolist())
        for key, value in out.metrics.items():
            print(f"{key}: {value:.6f}")
        return

    if hasattr(args, "seq_len"):
        if args.by_family:
            results = evaluate_synthetic_families(
                model,
                seq_len=args.seq_len,
                size=args.size,
                batch_size=args.batch_size,
            )
        else:
            results = evaluate_synthetic(
                model,
                seq_len=args.seq_len,
                size=args.size,
                batch_size=args.batch_size,
            )
        print(format_eval_table(results))
        return

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    delta = logits_delta_under_ablation(model, ids, args.ablation)
    print("ablation", args.ablation)
    print("mean_abs_delta", delta)


if __name__ == "__main__":
    main()
