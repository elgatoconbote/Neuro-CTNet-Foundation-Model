import torch
from nctnet import NCTConfig, NCTLanguageModel
from nctnet.core import AdditiveCouplingBlock, TopologicalMemory, CausalCoherenceTensor, split_up, merge_up, train_tiny_lm, TrainConfig


def cfg():
    return NCTConfig(vocab_size=128, max_seq_len=16, d_model=64, n_layers=1, d_ff=128, memory_slots=8, memory_dim=64, relation_slots=8, relation_dim=64, n_regimes=4, n_cards=4, coherence_rank=4, residual_bottleneck=16)


def test_split_merge_identity():
    x = torch.randn(2, 5, 64)
    u, p = split_up(x)
    assert torch.allclose(merge_up(u, p), x)


def test_reversible_block_inverse():
    block = AdditiveCouplingBlock(32, 64)
    x = torch.randn(2, 5, 64)
    y = block(x)
    x2 = block.inverse(y)
    assert (x - x2).abs().mean().item() < 1e-5


def test_memory_shape_fixed():
    mem = TopologicalMemory(8, 64)
    z = torch.randn(2, 16, 64)
    state = mem.initial_state(2)
    read, summary, next_memory, drive, energy = mem(z, state)
    assert next_memory.shape == state.shape
    assert read.shape == (2, 64)
    assert torch.isfinite(energy)


def test_coherence_mass_positive_and_clamped():
    coh = CausalCoherenceTensor(64, 4, 4)
    z = torch.randn(2, 8, 64) * 1000
    probs = torch.softmax(torch.randn(2, 4), -1)
    e = torch.tensor(0.1)
    mass, info, energy, idiag, ilow = coh(z, e, e, e, e, e, probs)
    assert torch.isfinite(mass).all()
    assert (mass > 0).all()
    assert mass.max() <= torch.exp(torch.tensor(0.5 * 5.0)) + 1e-5


def test_lm_forward_and_state_contract():
    c = cfg()
    model = NCTLanguageModel(c)
    ids = torch.randint(0, c.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 16, c.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.state.memory.shape[1] == c.memory_slots
    assert out.state.relations.shape[1] == c.relation_slots
    assert 'mass' in out.metrics


def test_tiny_train_smoke(tmp_path):
    c = cfg()
    ckpt = train_tiny_lm(c, TrainConfig(run_dir=str(tmp_path), steps=1, batch_size=2, seq_len=8))
    assert ckpt.exists()
