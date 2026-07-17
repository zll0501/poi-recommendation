"""验证独立多输入 SASRec Trainer 的训练与闭集预测。"""

import torch
from torch import nn

from src.sasrec_trainer import SASRecTrainer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4, padding_idx=0)
        self.output = nn.Linear(4, 8)

    def forward(self, poi_sequence, attention_mask, time_sequence=None):
        del time_sequence
        lengths = attention_mask.long().sum(dim=1).sub(1)
        starts = poi_sequence.size(1) - attention_mask.long().sum(dim=1)
        positions = starts + lengths
        hidden = self.embedding(poi_sequence)
        return self.output(hidden[torch.arange(len(hidden)), positions])


def _batch():
    return {
        "inputs": {
            "poi_sequence": torch.tensor([[0, 2, 3], [0, 4, 5]]),
            "time_sequence": torch.tensor([[0, 1, 2], [0, 3, 4]]),
            "attention_mask": torch.tensor(
                [[False, True, True], [False, True, True]]
            ),
        },
        "target": torch.tensor([4, 6]),
        "event_id": torch.tensor([10, 11]),
    }


def test_trainer_supports_nested_multi_input_batches_and_prediction():
    model = _TinyModel()
    trainer = SASRecTrainer(
        model,
        torch.optim.Adam(model.parameters(), lr=0.01),
        nn.CrossEntropyLoss(),
        "cpu",
    )

    history = trainer.fit([_batch()], [_batch()], epochs=2, patience=1)
    predictions = trainer.predict([_batch()], [2, 3, 4, 5, 6, 7], top_k=3)

    assert len(history["train_loss"]) >= 1
    assert len(history["valid_loss"]) >= 1
    assert set(predictions) == {10, 11}
    assert all(len(values) == 3 for values in predictions.values())
    assert all(set(values) <= {2, 3, 4, 5, 6, 7} for values in predictions.values())
