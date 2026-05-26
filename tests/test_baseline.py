import torch

from nctnet import NCTConfig
from nctnet.baseline import TinyTransformerLM, train_baseline_lm, train_baseline_report
from nctnet.core import TrainConfig


def cfg():
    return NCTConfig(
        vocab_size=128,
        max_seq_len=16,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        memory_slots=8,
        memory_dim=64,
        relation_slots=8,
        relation_dim=64,
        n_regimes=4,
        n_cards=4,
        coherence_rank=4,
        residual_bottleneck=16,
    )


def test_tiny_transformer_forward_shapes():
    c = cfg()
    model = TinyTransformerLM(c)
    ids = torch.randint(0, c.vocab_size, (2, 8))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 8, c.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_train_baseline_lm_smoke(tmp_path):
    c = cfg()
    checkpoint = train_baseline_lm(c, TrainConfig(run_dir=str(tmp_path), steps=1, batch_size=2, seq_len=8))
    assert checkpoint.exists()


def test_train_baseline_report_smoke(tmp_path):
    c = cfg()
    paths, results = train_baseline_report(
        c,
        TrainConfig(run_dir=str(tmp_path), steps=1, batch_size=2, seq_len=8),
        seq_len=8,
        size=4,
        batch_size=2,
    )
    assert results
    assert (tmp_path / "baseline.pt").exists()
    assert (tmp_path / "baseline_eval_family.tsv").exists()
    assert (tmp_path / "baseline_eval_family.json").exists()
    assert paths.checkpoint.endswith("baseline.pt")
