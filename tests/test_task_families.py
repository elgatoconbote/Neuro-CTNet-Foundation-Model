from nctnet.task_families import TASK_FAMILIES, TASK_SPECS, SyntheticTaskFamilyDataset


def test_all_task_families_have_specs():
    assert TASK_FAMILIES
    for family in TASK_FAMILIES:
        assert family in TASK_SPECS
        assert TASK_SPECS[family].expected_organs


def test_family_datasets_have_fixed_shapes():
    for family in TASK_FAMILIES:
        ds = SyntheticTaskFamilyDataset(family=family, vocab_size=256, seq_len=12, size=3)
        item = ds[0]
        assert item["input_ids"].shape[0] == 12
        assert item["labels"].shape[0] == 12
        assert item["family"] == family
        assert int(item["input_ids"].max()) < 256
        assert int(item["labels"].max()) < 256
