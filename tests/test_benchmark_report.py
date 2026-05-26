import json

from nctnet.benchmark_report import make_markdown_report, verdict_for_row, write_markdown_report


def row(**kw):
    base = {
        "family": "delayed_copy",
        "delta_loss_mean": -0.1,
        "delta_accuracy_mean": 0.2,
        "win_rate_loss": 1.0,
        "win_rate_accuracy": 1.0,
    }
    base.update(kw)
    return base


def test_verdict_strong_when_loss_and_accuracy_win():
    verdict = verdict_for_row(row())
    assert verdict.status == "NCT_STRONG"


def test_verdict_baseline_when_both_metrics_lose():
    verdict = verdict_for_row(row(delta_loss_mean=0.2, delta_accuracy_mean=-0.1, win_rate_loss=0.0, win_rate_accuracy=0.0))
    assert verdict.status == "BASELINE_ADVANTAGE"


def test_make_markdown_report_contains_table():
    md = make_markdown_report([row()])
    assert "Neuro-CTNet Benchmark Report" in md
    assert "| family |" in md
    assert "delayed_copy" in md
    assert "NCT_STRONG" in md


def test_write_markdown_report(tmp_path):
    bench = tmp_path / "benchmark.json"
    out = tmp_path / "BENCHMARK_REPORT.md"
    bench.write_text(json.dumps([row()]), encoding="utf-8")
    written = write_markdown_report(bench, out)
    assert written == str(out)
    assert out.exists()
    assert "NCT_STRONG" in out.read_text()
