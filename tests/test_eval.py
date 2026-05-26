import torch

from nctnet import NCTConfig, NCTLanguageModel
from nctnet.eval import evaluate_synthetic, format_eval_table


def cfg():
    return NCTConfig(
        vocab_size=128,
        max_seq_len=16,
        d_model=64,
        n_layers=1,
        d_ff=128,
        memory_slots=8,
        memory_dim=64,
        relation_slots=8,
        relation_dim=64,
        n_regimes=4,
        n_cards=4,
        coherence_rank=4,
        residual_bottleneck=16,
        coh_gain=0.05,
    )


def test_evaluate_synthetic_returns_full_and_ablation_rows():
    torch.manual_seed(3)
    model = NCTLanguageModel(cfg())
    results = evaluate_synthetic(
        model,
        seq_len=8,
        size=8,
        batch_size=2,
        ablations=["none", "no_memory", "single_card"],
    )
    assert [r.name for r in results] == ["none", "no_memory", "single_card"]
    assert all(r.tokens > 0 for r in results)
    assert all(r.task_loss > 0 for r in results)
    assert all(0.0 <= r.accuracy <= 1.0 for r in results)


def test_format_eval_table_contains_deltas():
    model = NCTLanguageModel(cfg())
    results = evaluate_synthetic(
        model,
        seq_len=8,
        size=8,
        batch_size=2,
        ablations=["none", "no_relations"],
    )
    table = format_eval_table(results)
    assert "ablation" in table
    assert "delta_loss" in table
    assert "no_relations" in table
