import torch

from nctnet import NCTConfig, NCTLanguageModel
from nctnet.ablations import SUPPORTED_ABLATIONS, logits_delta_under_ablation, temporary_ablation


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


def test_supported_ablations_declared():
    expected = {
        "none",
        "no_coherence",
        "no_memory",
        "no_relations",
        "no_admissibility",
        "no_regime",
        "single_card",
        "no_residue",
    }
    assert expected.issubset(SUPPORTED_ABLATIONS)


def test_temporary_ablation_restores_coherence_gain():
    model = NCTLanguageModel(cfg())
    old = model.cfg.coh_gain
    with temporary_ablation(model, "no_coherence"):
        assert model.cfg.coh_gain == 0.0
    assert model.cfg.coh_gain == old


def test_structural_ablations_change_logits():
    torch.manual_seed(7)
    model = NCTLanguageModel(cfg())
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    for name in sorted(SUPPORTED_ABLATIONS - {"none"}):
        delta = logits_delta_under_ablation(model, ids, name)
        assert delta >= 0.0
        if name in {"no_memory", "no_relations", "no_admissibility", "no_regime", "single_card"}:
            assert delta > 1e-8, name
