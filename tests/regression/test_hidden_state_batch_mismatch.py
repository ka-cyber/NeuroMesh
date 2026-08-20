"""
Regression test for a real bug found during real-data evaluation.

NeuroMeshUNet's controller carries a recurrent hidden state across batches.
evaluate_modality_dropout (neuromesh/evaluation/metrics.py) threads that
hidden state through every batch in a sweep. If a dataset's size isn't
evenly divisible by the batch size -- true of essentially all real data, and
NOT true of this repo's synthetic defaults (32 samples / batch 4), which is
exactly why this bug went undetected until real data was used -- the final
batch is smaller than the rest, and GRUCell throws a batch-size mismatch.

Fixed by resetting the hidden state whenever the incoming batch size changes
(see the `if h_state is not None and h_state.size(0) != x.size(0)` guards in
both evaluate_modality_dropout and neuromesh/training/train_real.py's
training loop). This test exists so that fix can never silently regress.
"""
import torch
from torch.utils.data import DataLoader

from neuromesh.models import NeuroMeshUNet
from neuromesh.data import MockBraTSDataset
from neuromesh.evaluation import evaluate_modality_dropout


def test_evaluate_modality_dropout_survives_non_divisible_batch_size():
    """17 samples / batch size 5 => batches of [5, 5, 5, 2] -- the final
    batch changes size mid-sweep. Before the fix, this raised a RuntimeError
    from GRUCell on the 4th batch. It must now complete without error."""
    ds = MockBraTSDataset(n_samples=17, image_size=(32, 32), num_classes=4, seed=0)
    loader = DataLoader(ds, batch_size=5, shuffle=False)

    model = NeuroMeshUNet(in_channels=4, num_classes=4, base_ch=4)
    model.eval()

    dropout_configs = [{"name": "clean", "modality_idx": None, "drop_prob": 0.0}]
    # Must not raise. This is the actual regression -- prior to the fix, this
    # call raised RuntimeError: Expected hidden size (5, ...), got (2, ...).
    results = evaluate_modality_dropout(model, loader, num_classes=4,
                                         device=torch.device("cpu"),
                                         dropout_configs=dropout_configs)
    assert "clean" in results
    assert results["clean"]["mean_dice"] >= 0.0


def test_hidden_state_reset_on_batch_size_change_directly():
    """More targeted: directly exercise the batch-size-mismatch guard by
    feeding decreasing batch sizes through the same model call sequence,
    the way a real final-batch-of-an-epoch does."""
    model = NeuroMeshUNet(in_channels=4, num_classes=4, base_ch=4)
    model.eval()

    h_state = None
    with torch.no_grad():
        for batch_size in [5, 5, 5, 2]:  # mimics 17 samples / batch 5
            x = torch.randn(batch_size, 4, 32, 32)
            if h_state is not None and h_state.size(0) != x.size(0):
                h_state = None
            logits, h_state, mask = model(x, h_state)
            assert logits.shape == (batch_size, 4, 32, 32)
            assert h_state.shape[0] == batch_size
