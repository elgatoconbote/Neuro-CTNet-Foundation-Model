from nctnet.benchmark import BenchmarkRow, format_benchmark_table


def test_format_benchmark_table_contains_expected_columns():
    rows = [
        BenchmarkRow(
            family="delayed_copy",
            runs=3,
            nct_loss_mean=1.0,
            baseline_loss_mean=1.2,
            delta_loss_mean=-0.2,
            delta_loss_std=0.1,
            nct_accuracy_mean=0.5,
            baseline_accuracy_mean=0.4,
            delta_accuracy_mean=0.1,
            delta_accuracy_std=0.05,
            win_rate_loss=1.0,
            win_rate_accuracy=0.666666,
        )
    ]
    table = format_benchmark_table(rows)
    assert "family" in table
    assert "win_rate_loss" in table
    assert "delayed_copy" in table
    assert "-0.200000" in table
