from __future__ import annotations

import argparse
import torch

from .core import NCTConfig, NCTLanguageModel, TrainConfig, _load_cfg, train_tiny_lm
from .ablations import SUPPORTED_ABLATIONS, logits_delta_under_ablation
from .eval import evaluate_synthetic, evaluate_synthetic_families, format_eval_table
from .report import train_eval_report
from .baseline import train_baseline_report
from .compare import compare_reports, format_comparison_table, write_comparison
from .benchmark import run_benchmark, format_benchmark_table
from .benchmark_report import write_markdown_report


def _load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = NCTConfig(**checkpoint["config"])
    model = NCTLanguageModel(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg


def _parse_seeds(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser("nctnet")
    sub = parser.add_subparsers(required=True)

    train = sub.add_parser("train-lm")
    train.set_defaults(command="train_lm")
    train.add_argument("--config", required=True)

    report = sub.add_parser("train-eval-report")
    report.set_defaults(command="train_eval_report")
    report.add_argument("--config", required=True)
    report.add_argument("--seq-len", type=int, default=16)
    report.add_argument("--size", type=int, default=64)
    report.add_argument("--batch-size", type=int, default=4)

    baseline = sub.add_parser("train-baseline-report")
    baseline.set_defaults(command="train_baseline_report")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--seq-len", type=int, default=16)
    baseline.add_argument("--size", type=int, default=64)
    baseline.add_argument("--batch-size", type=int, default=4)

    compare = sub.add_parser("compare-reports")
    compare.set_defaults(command="compare_reports")
    compare.add_argument("--nct", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--out-dir", default="runs/debug")

    bench = sub.add_parser("run-benchmark")
    bench.set_defaults(command="run_benchmark")
    bench.add_argument("--config", required=True)
    bench.add_argument("--seeds", default="0,1,2")
    bench.add_argument("--seq-len", type=int, default=16)
    bench.add_argument("--size", type=int, default=64)
    bench.add_argument("--batch-size", type=int, default=4)
    bench.add_argument("--out-dir", default="runs/benchmark")

    bench_report = sub.add_parser("benchmark-report")
    bench_report.set_defaults(command="benchmark_report")
    bench_report.add_argument("--benchmark", required=True)
    bench_report.add_argument("--out", default="runs/benchmark/BENCHMARK_REPORT.md")
    bench_report.add_argument("--min-win-rate", type=float, default=0.5)

    inspect = sub.add_parser("inspect")
    inspect.set_defaults(command="inspect")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--prompt", default="1 2 3 4")

    ablate = sub.add_parser("ablate")
    ablate.set_defaults(command="ablate")
    ablate.add_argument("--checkpoint", required=True)
    ablate.add_argument("--ablation", default="no_coherence", choices=sorted(SUPPORTED_ABLATIONS - {"none"}))

    ev = sub.add_parser("eval-synthetic")
    ev.set_defaults(command="eval_synthetic")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--seq-len", type=int, default=16)
    ev.add_argument("--size", type=int, default=64)
    ev.add_argument("--batch-size", type=int, default=4)
    ev.add_argument("--by-family", action="store_true")

    args = parser.parse_args()

    if args.command == "train_lm":
        raw = _load_cfg(args.config)
        checkpoint = train_tiny_lm(NCTConfig(**raw.get("model", {})), TrainConfig(**raw.get("train", {})))
        print(f"saved={checkpoint}")
        return

    if args.command == "train_eval_report":
        raw = _load_cfg(args.config)
        paths, results = train_eval_report(
            NCTConfig(**raw.get("model", {})),
            TrainConfig(**raw.get("train", {})),
            seq_len=args.seq_len,
            size=args.size,
            batch_size=args.batch_size,
        )
        print(format_eval_table(results))
        print(f"checkpoint={paths.checkpoint}")
        print(f"tsv={paths.tsv}")
        print(f"json={paths.json}")
        return

    if args.command == "train_baseline_report":
        raw = _load_cfg(args.config)
        paths, results = train_baseline_report(
            NCTConfig(**raw.get("model", {})),
            TrainConfig(**raw.get("train", {})),
            seq_len=args.seq_len,
            size=args.size,
            batch_size=args.batch_size,
        )
        print(format_eval_table(results))
        print(f"checkpoint={paths.checkpoint}")
        print(f"tsv={paths.tsv}")
        print(f"json={paths.json}")
        return

    if args.command == "compare_reports":
        rows = compare_reports(args.nct, args.baseline)
        tsv, js = write_comparison(rows, args.out_dir)
        print(format_comparison_table(rows))
        print(f"tsv={tsv}")
        print(f"json={js}")
        return

    if args.command == "run_benchmark":
        raw = _load_cfg(args.config)
        rows, tsv, js = run_benchmark(
            NCTConfig(**raw.get("model", {})),
            TrainConfig(**raw.get("train", {})),
            seeds=_parse_seeds(args.seeds),
            seq_len=args.seq_len,
            size=args.size,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
        )
        print(format_benchmark_table(rows))
        print(f"tsv={tsv}")
        print(f"json={js}")
        return

    if args.command == "benchmark_report":
        path = write_markdown_report(
            args.benchmark,
            args.out,
            min_win_rate=args.min_win_rate,
        )
        print(f"markdown={path}")
        return

    model, cfg = _load_model(args.checkpoint)

    if args.command == "inspect":
        toks = [int(x) % cfg.vocab_size for x in args.prompt.split() if x.lstrip("-").isdigit()] or [1, 2, 3, 4]
        out = model(torch.tensor(toks).unsqueeze(0))
        print("logits_shape", tuple(out.logits.shape))
        print("top_next", torch.topk(out.logits[0, -1], min(5, cfg.vocab_size)).indices.tolist())
        for key, value in out.metrics.items():
            print(f"{key}: {value:.6f}")
        return

    if args.command == "eval_synthetic":
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

    if args.command == "ablate":
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        delta = logits_delta_under_ablation(model, ids, args.ablation)
        print("ablation", args.ablation)
        print("mean_abs_delta", delta)
        return


if __name__ == "__main__":
    main()
