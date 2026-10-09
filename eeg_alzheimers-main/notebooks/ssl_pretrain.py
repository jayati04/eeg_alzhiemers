import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight

# ======================================================
# CONFIG
# ======================================================
USE_PRETRAINED_SSL = True
SSL_EPOCHS = 20
CLS_EPOCHS = 50
BATCH_SIZE = 64
FREEZE_ENCODER_AFTER = 20
FREEZE_TRANSFORMER_AFTER = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# LOAD DATA
# ======================================================
data = np.load("data/alzheimer_eeg.npz", allow_pickle=True)
X = data["X_raw"]
y_raw = data["y_labels"]

# ======================================================
# NORMALIZE + RESHAPE
# ======================================================
def normalize_eeg(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mean) / std

X = normalize_eeg(X)
X = np.transpose(X, (0, 2, 1)).astype(np.float32)

y = np.array([1 if "AD" in str(lbl) else 0 for lbl in y_raw], dtype=int)
print("Class distribution:", np.bincount(y))

# ======================================================
# TRAIN–TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ======================================================
# SSL DATASET
# ======================================================
class SSLDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        noise = np.random.normal(0, 0.03, x.shape)
        shift = np.roll(x, shift=5, axis=-1)
        return torch.tensor(x + noise).float(), torch.tensor(shift).float()

# ======================================================
# CNN ENCODER
# ======================================================
class EEGEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )

    def forward(self, x):
        return self.net(x)

# ======================================================
# SSL PRETRAINING
# ======================================================
encoder = EEGEncoder(channels=X.shape[1]).to(DEVICE)

if USE_PRETRAINED_SSL and os.path.exists("ssl_encoder.pth"):
    print("Loading pretrained SSL encoder...")
    encoder.load_state_dict(torch.load("ssl_encoder.pth"))
else:
    ssl_loader = DataLoader(SSLDataset(X_train), BATCH_SIZE, shuffle=True)
    optimizer_ssl = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    for epoch in range(SSL_EPOCHS):
        encoder.train()
        total_loss = 0

        for x1, x2 in ssl_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

            z1 = encoder(x1).mean(dim=2)
            z2 = encoder(x2).mean(dim=2)

            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            loss = -torch.mean(torch.sum(z1 * z2, dim=1))

            optimizer_ssl.zero_grad()
            loss.backward()
            optimizer_ssl.step()

            total_loss += loss.item()

        print(f"SSL Epoch {epoch+1}/{SSL_EPOCHS} | Loss {total_loss/len(ssl_loader):.4f}")

    torch.save(encoder.state_dict(), "ssl_encoder.pth")

# ======================================================
# TRANSFORMER
# ======================================================
class EEGTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, nhead=4, dropout=0.25, batch_first=True
            ),
            num_layers=2
        )

    def forward(self, x):
        return self.encoder(x.permute(0, 2, 1)).mean(dim=1)

# ======================================================
# FINAL MODEL
# ======================================================
class AlzheimerModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.transformer = EEGTransformer()
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        return self.fc(self.transformer(self.encoder(x)))

model = AlzheimerModel(encoder).to(DEVICE)

# ======================================================
# DATASET
# ======================================================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

train_loader = DataLoader(EEGDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(EEGDataset(X_test, y_test), BATCH_SIZE)

# ======================================================
# LOSS + OPTIMIZER
# ======================================================
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights).float().to(DEVICE),
    label_smoothing=0.03
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

# ======================================================
# SUPERVISED TRAINING
# ======================================================
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(CLS_EPOCHS):
    model.train()

    if epoch == FREEZE_ENCODER_AFTER:
        print("Freezing CNN encoder")
        for p in model.encoder.parameters():
            p.requires_grad = False

    if epoch == FREEZE_TRANSFORMER_AFTER:
        print("Freezing Transformer")
        for p in model.transformer.parameters():
            p.requires_grad = False

    if epoch == 20:
        for g in optimizer.param_groups:
            g['lr'] *= 0.5

    loss_sum, correct, total = 0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = criterion(out, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += loss.item()
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)

    train_loss = loss_sum / len(train_loader)
    train_acc = correct / total

    model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss_sum += criterion(out, yb).item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

    test_loss = loss_sum / len(test_loader)
    test_acc = correct / total

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{CLS_EPOCHS} | "
          f"Train Loss {train_loss:.4f} | Test Loss {test_loss:.4f} | "
          f"Train Acc {train_acc:.4f} | Test Acc {test_acc:.4f}")

torch.save(model.state_dict(), "best_model.pth")

# ======================================================
# SAVE METRICS
# ======================================================
np.save("train_losses.npy", train_losses)
np.save("test_losses.npy", test_losses)
np.save("train_accs.npy", train_accs)
np.save("test_accs.npy", test_accs)

# ======================================================
# TRAIN vs TEST PLOTS
# ======================================================
epochs = range(1, CLS_EPOCHS + 1)

plt.figure()
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, test_losses, label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs, train_accs, label="Training Accuracy")
plt.plot(epochs, test_accs, label="Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ======================================================
# ROC CURVE
# ======================================================
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        y_true.extend(yb.numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_prob.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

print(classification_report(y_true, y_pred, target_names=["Non-AD","AD"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("ROC-AUC:", roc_auc)
print("MCC:", matthews_corrcoef(y_true, y_pred))
