from nctnet.tensors import split_up, merge_up
from nctnet.reversible import AdditiveCouplingBlock
from nctnet.up import UPReciprocity
from nctnet.memory import TopologicalMemory
from nctnet import NCTConfig, NCTLanguageModel, NCTBlock, NCTState


def test_public_module_api_imports():
    assert split_up is not None
    assert merge_up is not None
    assert AdditiveCouplingBlock is not None
    assert UPReciprocity is not None
    assert TopologicalMemory is not None
    assert NCTConfig is not None
    assert NCTLanguageModel is not None
    assert NCTBlock is not None
    assert NCTState is not None
