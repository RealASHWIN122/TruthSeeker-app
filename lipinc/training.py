import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import get_color_structure_frames
from model import LIPINC_V2

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
N_FRAMES = 5
BATCH_SIZE = 2        # keep small (heavy model)
EPOCHS = 30
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------


def load_dataset():
    X_color, X_residue, y = [], [], []

    for label_name, label_vec in [("real", [0, 1]), ("fake", [1, 0])]:
        folder = os.path.join(DATASET_DIR, label_name)
        videos = os.listdir(folder)

        print(f"Loading {label_name} videos:", len(videos))

        for vid in tqdm(videos):
            video_path = os.path.join(folder, vid)

            try:
                _, _, combined, residue, _, _ = get_color_structure_frames(
                    N_FRAMES, video_path
                )

                # Shape checks (important)
                if combined.shape != (8, 64, 144, 3):
                    continue
                if residue.shape != (7, 64, 144, 3):
                    continue

                X_color.append(combined)
                X_residue.append(residue)
                y.append(label_vec)

            except Exception as e:
                # skip bad videos
                continue

    return (
        np.asarray(X_color, dtype=np.float32) / 255.0,
        np.asarray(X_residue, dtype=np.float32) / 255.0,
        np.asarray(y, dtype=np.float32),
    )


# ---------------- LOAD DATA ----------------
X_color, X_residue, y = load_dataset()

print("Dataset shapes:")
print("Color:", X_color.shape)
print("Residue:", X_residue.shape)
print("Labels:", y.shape)

Xc_train, Xc_val, Xr_train, Xr_val, y_train, y_val = train_test_split(
    X_color, X_residue, y, test_size=0.2, random_state=42, shuffle=True
)

# ---------------- MODEL ----------------
model = LIPINC_V2()
model.summary()

# ---------------- CALLBACKS ----------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "Best_Weights.hdf5"),
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

# ---------------- TRAIN ----------------
history = model.fit(
    [Xc_train, Xr_train],
    y_train,
    validation_data=([Xc_val, Xr_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_cb, early_stop],
    shuffle=True,
)

print("Training complete. Best model saved.")
