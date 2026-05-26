import json

from nctnet import NCTConfig, NCTLanguageModel
from nctnet.eval import evaluate_synthetic_families
from nctnet.report import write_eval_report


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
    )


def test_write_eval_report_creates_tsv_and_json(tmp_path):
    model = NCTLanguageModel(cfg())
    results = evaluate_synthetic_families(
        model,
        seq_len=8,
        size=4,
        batch_size=2,
        ablations=["none", "no_memory"],
        families=["persistent_entity"],
    )
    paths = write_eval_report(results, tmp_path)
    tsv = tmp_path / "eval_family.tsv"
    js = tmp_path / "eval_family.json"
    assert tsv.exists()
    assert js.exists()
    assert "persistent_entity" in tsv.read_text()
    data = json.loads(js.read_text())
    assert data[0]["family"] == "persistent_entity"
    assert paths.tsv == str(tsv)
    assert paths.json == str(js)
