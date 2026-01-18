import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model definition (must be at top level for multiprocessing)
class MLP(nn.Module):
    def __init__(self, in_dim=21, h1=64, h2=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1)
        )
    
    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # Load data
    h5_path = Path("B:\\download\\TruthSeeker-app\\training_features.h5")

    print("Loading features from HDF5...")
    with h5py.File(h5_path, "r") as f:
        X = f["features"][:].astype(np.float32)
        y = f["label"][:].astype(np.int64)
        paths = f["path"][:].astype(str)
        
    print(f"Loaded {len(X)} samples from {h5_path}")
    print(f"Real videos: {(y==1).sum()}, Fake videos: {(y==0).sum()}")

    # Balance classes (equal priors)
    idx_real = np.where(y == 1)[0]
    idx_fake = np.where(y == 0)[0]

    n_real = len(idx_real)
    n_fake = len(idx_fake)
    n_balanced = min(n_real, n_fake)

    # Randomly sample equal numbers from each class
    rng = np.random.default_rng(seed=42)
    sel_real = rng.choice(idx_real, n_balanced, replace=False)
    sel_fake = rng.choice(idx_fake, n_balanced, replace=False)

    balanced_idx = np.concatenate([sel_real, sel_fake])
    assert len(balanced_idx) == len(np.unique(balanced_idx)), "Duplicate indices in balanced dataset!"
    rng.shuffle(balanced_idx)

    X_bal = X[balanced_idx]
    y_bal = y[balanced_idx]
    paths_bal = paths[balanced_idx]

    print(f"\nBalanced dataset size: {len(X_bal)} (real={n_balanced}, fake={n_balanced})")

    # Normalize
    mean = X_bal.mean(axis=0, keepdims=True)
    std = X_bal.std(axis=0, keepdims=True) + 1e-8
    X_bal = (X_bal - mean) / std

    print(f"Feature normalization - Mean: {mean.mean():.4f}, Std: {std.mean():.4f}")

    # Stratified 50/50 train/test split
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X_bal,
        y_bal,
        paths_bal,
        test_size=0.5,
        stratify=y_bal,
        random_state=42
    )

    # Convert to tensors and datasets
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # Create DataLoaders - SET num_workers=0 for Windows!
    batch_size = 64

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
    )

    print(f"\nTrain: {len(train_ds)}  Test: {len(test_ds)}  (each balanced 50/50)")

    # Sanity check
    xb, yb = next(iter(train_loader))
    print(f"Example batch: {xb.shape}, {yb.shape}")
    print(f"Label distribution (train): {torch.bincount(torch.from_numpy(y_train))}")
    print(f"Label distribution (test): {torch.bincount(torch.from_numpy(y_test))}")

    # Setup device and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = MLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Training
    epochs = 20
    best_f1 = 0.0
    train_losses = []
    val_f1_scores = []

    print(f"\nStarting training for {epochs} epochs...")
    print("="*60)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
                val_logits.append(out)
                val_labels.append(yb.numpy())
        
        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels)
        val_preds = (val_logits >= 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_preds)
        val_f1_scores.append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  → New best F1: {best_f1:.4f} (model saved)")

    print("="*60)
    print("Training complete!\n")

    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    # Optimize threshold τ* on training set (F1)
    print("Optimizing classification threshold on training set...")
    train_logits, train_labels = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            train_logits.append(out)
            train_labels.append(yb.numpy())

    train_logits = np.concatenate(train_logits)
    train_labels = np.concatenate(train_labels)

    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1_threshold, best_tau = 0.0, 0.5
    for t in thresholds:
        preds = (train_logits >= t).astype(int)
        f1 = f1_score(train_labels, preds)
        if f1 > best_f1_threshold:
            best_f1_threshold, best_tau = f1, t

    print(f"Optimal threshold τ* = {best_tau:.3f} (Train F1 = {best_f1_threshold:.3f})")

    # Evaluate on test set with τ*
    print("\nEvaluating on test set...")
    test_logits, test_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            test_logits.append(out)
            test_labels.append(yb.numpy())

    test_logits = np.concatenate(test_logits)
    test_labels = np.concatenate(test_labels)
    test_preds = (test_logits >= best_tau).astype(int)

    # Save predictions
    df = pd.DataFrame({
        "path": paths_test,
        "true_label": test_labels,
        "pred_label": test_preds,
        "prob_real": test_logits,
    })
    df.to_csv("test_predictions.csv", index=False)
    print(f"Saved predictions to test_predictions.csv")

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary"
    )
    auc = roc_auc_score(test_labels, test_logits)
    acc = accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    # Print results
    print("\n" + "="*60)
    print("FINAL TEST PERFORMANCE")
    print("="*60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"\nConfusion Matrix (rows=true [Fake, Real], cols=pred [Fake, Real]):")
    print(cm)
    print(f"\nTrue Negatives  (Fake → Fake): {cm[0,0]}")
    print(f"False Positives (Fake → Real): {cm[0,1]}")
    print(f"False Negatives (Real → Fake): {cm[1,0]}")
    print(f"True Positives  (Real → Real): {cm[1,1]}")
    print("="*60)

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(range(1, epochs+1), val_f1_scores, marker='o', color='green', label='Validation F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Score Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved training curves to training_curves.png")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Test Set)\nF1={f1:.3f}, AUC={auc:.3f}')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to confusion_matrix.png")

    # Save model and normalization parameters
    torch.save(model.state_dict(), "model.pt")
    np.save("mean.npy", mean)
    np.save("std.npy", std)
    np.save("best_tau.npy", best_tau)

    print("\n✓ Saved model artifacts:")
    print("  - model.pt (trained model)")
    print("  - best_model.pt (best validation model)")
    print("  - mean.npy (normalization mean)")
    print("  - std.npy (normalization std)")
    print("  - best_tau.npy (optimal threshold)")
    print("\nTraining complete! 🎉")

print("Loading features from HDF5...")
with h5py.File(h5_path, "r") as f:
    X = f["features"][:].astype(np.float32)
    y = f["label"][:].astype(np.int64)
    paths = f["path"][:].astype(str)
    
print(f"Loaded {len(X)} samples from {h5_path}")
print(f"Real videos: {(y==1).sum()}, Fake videos: {(y==0).sum()}")

# Balance classes (equal priors)
idx_real = np.where(y == 1)[0]
idx_fake = np.where(y == 0)[0]

n_real = len(idx_real)
n_fake = len(idx_fake)
n_balanced = min(n_real, n_fake)

# Randomly sample equal numbers from each class
rng = np.random.default_rng(seed=42)
sel_real = rng.choice(idx_real, n_balanced, replace=False)
sel_fake = rng.choice(idx_fake, n_balanced, replace=False)

balanced_idx = np.concatenate([sel_real, sel_fake])
assert len(balanced_idx) == len(np.unique(balanced_idx)), "Duplicate indices in balanced dataset!"
rng.shuffle(balanced_idx)

X_bal = X[balanced_idx]
y_bal = y[balanced_idx]
paths_bal = paths[balanced_idx]

print(f"\nBalanced dataset size: {len(X_bal)} (real={n_balanced}, fake={n_balanced})")

# Normalize
mean = X_bal.mean(axis=0, keepdims=True)
std = X_bal.std(axis=0, keepdims=True) + 1e-8
X_bal = (X_bal - mean) / std

print(f"Feature normalization - Mean: {mean.mean():.4f}, Std: {std.mean():.4f}")

# Stratified 50/50 train/test split
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X_bal,
    y_bal,
    paths_bal,
    test_size=0.5,
    stratify=y_bal,
    random_state=42
)

# Convert to tensors and datasets
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

# Create DataLoaders
batch_size = 64  # Increased for faster training

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"\nTrain: {len(train_ds)}  Test: {len(test_ds)}  (each balanced 50/50)")

# Sanity check
xb, yb = next(iter(train_loader))
print(f"Example batch: {xb.shape}, {yb.shape}")
print(f"Label distribution (train): {torch.bincount(torch.from_numpy(y_train))}")
print(f"Label distribution (test): {torch.bincount(torch.from_numpy(y_test))}")

# Model definition
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

class MLP(nn.Module):
    def __init__(self, in_dim=21, h1=64, h2=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print(f"\nModel architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Training
epochs = 20
best_f1 = 0.0
train_losses = []
val_f1_scores = []

print(f"\nStarting training for {epochs} epochs...")
print("="*60)

for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        xb, yb = xb.to(device), yb.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    
    avg_loss = running_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    
    # Validation phase
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            val_logits.append(out)
            val_labels.append(yb.numpy())
    
    val_logits = np.concatenate(val_logits)
    val_labels = np.concatenate(val_labels)
    val_preds = (val_logits >= 0.5).astype(int)
    val_f1 = f1_score(val_labels, val_preds)
    val_f1_scores.append(val_f1)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  → New best F1: {best_f1:.4f} (model saved)")

print("="*60)
print("Training complete!\n")

# Load best model
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Optimize threshold τ* on training set (F1)
print("Optimizing classification threshold on training set...")
train_logits, train_labels = [], []
with torch.no_grad():
    for xb, yb in train_loader:
        xb = xb.to(device)
        out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
        train_logits.append(out)
        train_labels.append(yb.numpy())

train_logits = np.concatenate(train_logits)
train_labels = np.concatenate(train_labels)

thresholds = np.linspace(0.1, 0.9, 81)
best_f1_threshold, best_tau = 0.0, 0.5
for t in thresholds:
    preds = (train_logits >= t).astype(int)
    f1 = f1_score(train_labels, preds)
    if f1 > best_f1_threshold:
        best_f1_threshold, best_tau = f1, t

print(f"Optimal threshold τ* = {best_tau:.3f} (Train F1 = {best_f1_threshold:.3f})")

# Evaluate on test set with τ*
print("\nEvaluating on test set...")
test_logits, test_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
        test_logits.append(out)
        test_labels.append(yb.numpy())

test_logits = np.concatenate(test_logits)
test_labels = np.concatenate(test_labels)
test_preds = (test_logits >= best_tau).astype(int)

# Save predictions
df = pd.DataFrame({
    "path": paths_test,
    "true_label": test_labels,
    "pred_label": test_preds,
    "prob_real": test_logits,
})
df.to_csv("test_predictions.csv", index=False)
print(f"Saved predictions to test_predictions.csv")

# Calculate metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, test_preds, average="binary"
)
auc = roc_auc_score(test_labels, test_logits)
acc = accuracy_score(test_labels, test_preds)
cm = confusion_matrix(test_labels, test_preds)

# Print results
print("\n" + "="*60)
print("FINAL TEST PERFORMANCE")
print("="*60)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")
print(f"\nConfusion Matrix (rows=true [Fake, Real], cols=pred [Fake, Real]):")
print(cm)
print(f"\nTrue Negatives  (Fake → Fake): {cm[0,0]}")
print(f"False Positives (Fake → Real): {cm[0,1]}")
print(f"False Negatives (Real → Fake): {cm[1,0]}")
print(f"True Positives  (Real → Real): {cm[1,1]}")
print("="*60)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Over Time')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(range(1, epochs+1), val_f1_scores, marker='o', color='green', label='Validation F1')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.set_title('Validation F1 Score Over Time')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print(f"\nSaved training curves to training_curves.png")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix (Test Set)\nF1={f1:.3f}, AUC={auc:.3f}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Saved confusion matrix to confusion_matrix.png")

# Save model and normalization parameters
torch.save(model.state_dict(), "model.pt")
np.save("mean.npy", mean)
np.save("std.npy", std)
np.save("best_tau.npy", best_tau)

print("\n✓ Saved model artifacts:")
print("  - model.pt (trained model)")
print("  - best_model.pt (best validation model)")
print("  - mean.npy (normalization mean)")
print("  - std.npy (normalization std)")
print("  - best_tau.npy (optimal threshold)")
print("\nTraining complete! 🎉")