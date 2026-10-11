import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from artifacts import save_artifacts
from config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    SEED,
    VAL_FRACTION,
    WEIGHT_DECAY,
)
from evaluate import evaluate
from model import TitanicMLP
from preprocess import preprocess

# Load the data
train_df = pd.read_csv('./data/train.csv')

# preprocess
X_train, y_train, imputers = preprocess(train_df)

# start training
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_numpy = X_train.to_numpy(dtype=np.float32)
y_train_numpy = y_train.to_numpy(dtype=np.float32)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_numpy,
    y_train_numpy,
    test_size=VAL_FRACTION,
    stratify=y_train_numpy,
    random_state=SEED,
)

feature_mean = X_train_split.mean(axis=0)
feature_std = X_train_split.std(axis=0)
feature_std[feature_std == 0] = 1.0
X_train_split = (X_train_split - feature_mean) / feature_std
X_val_split = (X_val_split - feature_mean) / feature_std

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train_split), torch.from_numpy(y_train_split)),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val_split), torch.from_numpy(y_val_split)),
    batch_size=BATCH_SIZE,
)

model = TitanicMLP(X_train_numpy.shape[1]).to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

best_val_accuracy = 0.0
best_model_state = None
epochs_without_improvement = 0

print("\n--------------training:--------------")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    for feature_batch, label_batch in train_loader:
        feature_batch = feature_batch.to(device)
        label_batch = label_batch.to(device)
        optimizer.zero_grad()
        batch_loss = loss_function(model(feature_batch), label_batch)
        batch_loss.backward()
        optimizer.step()

    train_loss, train_accuracy = evaluate(model, train_loader, loss_function, device)
    val_loss, val_accuracy = evaluate(model, val_loader, loss_function, device)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    print(
        f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_accuracy:.4f} "
        f"| val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        + (" *" if epochs_without_improvement == 0 else "")
    )

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"early stop at epoch {epoch} (best val accuracy {best_val_accuracy:.4f})")
        break

save_artifacts(best_model_state, feature_mean, feature_std, X_train.columns, imputers)
print(f"\nsaved to {ARTIFACTS_DIR}/ — best val accuracy {best_val_accuracy:.4f}")
