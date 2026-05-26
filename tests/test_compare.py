import json

from nctnet.compare import compare_reports, format_comparison_table, write_comparison


def test_compare_reports_and_write_outputs(tmp_path):
    nct = [
        {"family": "delayed_copy", "ablation": "none", "task_loss": 1.0, "accuracy": 0.5, "tokens": 10},
        {"family": "delayed_copy", "ablation": "no_memory", "task_loss": 1.2, "accuracy": 0.4, "tokens": 10},
    ]
    baseline = [
        {"family": "delayed_copy", "ablation": "none", "task_loss": 1.5, "accuracy": 0.3, "tokens": 10},
    ]
    nct_path = tmp_path / "eval_family.json"
    base_path = tmp_path / "baseline_eval_family.json"
    nct_path.write_text(json.dumps(nct), encoding="utf-8")
    base_path.write_text(json.dumps(baseline), encoding="utf-8")

    rows = compare_reports(nct_path, base_path)
    assert len(rows) == 1
    assert rows[0].family == "delayed_copy"
    assert rows[0].delta_loss == -0.5
    assert rows[0].delta_accuracy == 0.2

    table = format_comparison_table(rows)
    assert "nct_loss" in table
    assert "baseline_loss" in table
    assert "delayed_copy" in table

    tsv, js = write_comparison(rows, tmp_path)
    assert (tmp_path / "comparison.tsv").exists()
    assert (tmp_path / "comparison.json").exists()
    assert tsv.endswith("comparison.tsv")
    assert js.endswith("comparison.json")
