"""PyTorch model definition for the Titanic classifier.

A small Multi-Layer Perceptron (fully-connected feed-forward network) is
the right default for tabular data — there's no spatial or sequential
structure to exploit, so we don't need convolutions or recurrence. Keeping
the model shallow (2 hidden layers) and narrow (32→16 units) matches the
small dataset: ~700 training rows cannot support a larger network without
overfitting.
"""

import torch
from torch import nn


class TitanicMLP(nn.Module):
    """Binary classifier for Titanic survival.

    Architecture:
        Linear(input_features → 32) → ReLU → Dropout(0.3)
        Linear(32 → 16)             → ReLU → Dropout(0.3)
        Linear(16 → 1)                              # raw logit

    The final layer outputs a single raw score (logit), not a probability.
    Training uses `BCEWithLogitsLoss`, which fuses sigmoid + BCE in a
    numerically stable way. At inference, apply `torch.sigmoid` to recover
    the probability of survival, or just threshold the logit at 0.

    Dropout at 0.3 is relatively aggressive — justified by the small
    dataset, where regularization matters more than raw capacity.

    Args:
        input_features: Number of columns in the preprocessed feature
            matrix (24 with the current `preprocess.py`).
    """

    def __init__(self, input_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            inputs: Float tensor of shape (batch_size, input_features).

        Returns:
            Float tensor of shape (batch_size,) containing one logit per
            passenger. The trailing dim-1 from the final Linear is squeezed
            away so the output lines up with `BCEWithLogitsLoss`'s expected
            target shape.
        """
        return self.network(inputs).squeeze(-1)
