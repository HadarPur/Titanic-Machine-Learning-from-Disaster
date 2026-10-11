"""Evaluation utilities for the Titanic classifier.

Shared between training (per-epoch validation) and any future inference
or scoring script. Kept separate from `train.py` so it can be imported
without executing the training loop.
"""

import torch


def evaluate(model, data_loader, loss_function, device):
    """Compute average loss and accuracy over a DataLoader.

    The model is put into `eval()` mode (disables dropout) and gradient
    tracking is suppressed with `torch.no_grad()` for speed and memory.

    Args:
        model:         A `torch.nn.Module` producing one logit per row.
        data_loader:   Yields `(features, labels)` batches with float
                       labels in {0.0, 1.0}.
        loss_function: Any `nn.Module` compatible with raw logits
                       (e.g. `BCEWithLogitsLoss`).
        device:        Target device for batches.

    Returns:
        `(average_loss, accuracy)` — both floats in [0, 1-ish for loss].
    """
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for feature_batch, label_batch in data_loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(feature_batch)
            running_loss += loss_function(logits, label_batch).item() * label_batch.size(0)
            predictions = (logits > 0).float()
            correct_predictions += (predictions == label_batch).sum().item()
            total_samples += label_batch.size(0)
    average_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy
