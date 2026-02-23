"""
metrics.py — Stream C: Professional Validation Metrics & Visualisations
=======================================================================
Provides:
  • compute_metrics()  — full sklearn metric suite from logits / labels
  • plot_roc_curve()   — ROC + AUC figure saved to disk
  • plot_confusion_matrix() — seaborn heatmap CM saved to disk
  • evaluate_epoch()   — one-stop validation loop returning a MetricBundle

Author : Principal AI Research Engineer
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

logger = logging.getLogger("StreamC")


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MetricBundle:
    """All scalar metrics + paths to saved figures for one epoch."""
    epoch:     int
    val_loss:  float

    # Classification metrics
    accuracy:  float = 0.0
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0
    auc:       float = 0.0

    # Figure paths
    roc_path: str = ""
    cm_path:  str = ""

    # Raw arrays (for downstream analysis)
    labels:  List[int]   = field(default_factory=list)
    probs:   List[float] = field(default_factory=list)

    def log_summary(self) -> None:
        # CHANGED: Replaced box drawing characters with standard dashes
        logger.info(
            "--- Validation Results ---------------------------------------\n"
            f"  Loss      : {self.val_loss:.4f}\n"
            f"  AUC       : {self.auc:.4f}   <- primary metric for checkpoint\n"
            f"  Accuracy  : {self.accuracy:.4f}\n"
            f"  Precision : {self.precision:.4f}\n"
            f"  Recall    : {self.recall:.4f}\n"
            f"  F1-Score  : {self.f1:.4f}\n"
            "--------------------------------------------------------------"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    labels: np.ndarray,
    probs:  np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute a full suite of binary classification metrics.

    Parameters
    ----------
    labels    : ground-truth integer labels (0 = real, 1 = fake)
    probs     : predicted probabilities P(fake) ∈ [0, 1]
    threshold : decision threshold (default 0.5)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc
    """
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc":       float(roc_auc_score(labels, probs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(
    labels:    np.ndarray,
    probs:     np.ndarray,
    auc_score: float,
    epoch:     int,
    log_dir:   str,
) -> str:
    """
    Plot and save the ROC curve for one epoch.

    Returns
    -------
    str : path to the saved PNG file
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    optimal_idx  = np.argmax(tpr - fpr)
    optimal_thr  = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # ROC curve
    ax.plot(
        fpr, tpr,
        color="#00e5ff", linewidth=2.5,
        label=f"AUC = {auc_score:.4f}",
    )
    # Diagonal (chance)
    ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=1.2, label="Chance")
    # Optimal operating point
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        s=100, color="#ff6b6b", zorder=5,
        label=f"Optimal threshold = {optimal_thr:.3f}",
    )

    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate",  color="white", fontsize=12)
    ax.set_title(
        f"ROC Curve — Epoch {epoch:02d}  |  Stream C Frequency Model",
        color="white", fontsize=13, fontweight="bold",
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    ax.legend(
        facecolor="#1c1c2e", edgecolor="#555555",
        labelcolor="white", fontsize=10,
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, color="#1e2030", linewidth=0.5)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(log_dir) / f"epoch_{epoch:02d}_roc.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # CHANGED: Arrow symbol removed
    logger.info(f"  ROC curve saved -> {save_path}")
    return save_path


def plot_confusion_matrix(
    labels:      np.ndarray,
    probs:       np.ndarray,
    class_names: List[str],
    epoch:       int,
    log_dir:     str,
    threshold:   float = 0.5,
) -> str:
    """
    Plot and save a seaborn confusion matrix heatmap.

    Returns
    -------
    str : path to the saved PNG file
    """
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)

    # Normalised version (row-wise recall) for the annotation overlay
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    annot = np.array(
        [[f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)" for j in range(cm.shape[1])]
         for i in range(cm.shape[0])]
    )

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="#333333",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label",  color="white", fontsize=12)
    ax.set_ylabel("True Label",       color="white", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — Epoch {epoch:02d}  |  Stream C Frequency Model",
        color="white", fontsize=13, fontweight="bold",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.figure.axes[-1].tick_params(colors="white")   # colorbar ticks

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(log_dir) / f"epoch_{epoch:02d}_cm.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # CHANGED: Arrow symbol removed
    logger.info(f"  Confusion matrix saved -> {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Full validation loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    epoch:       int,
    log_dir:     str,
    class_names: List[str],
) -> MetricBundle:
    """
    Run a full validation pass and produce all metrics + figures.

    Parameters
    ----------
    model       : FrequencyModel in eval() mode (caller's responsibility)
    loader      : validation DataLoader
    criterion   : loss function (BCEWithLogitsLoss)
    device      : torch.device
    epoch       : current epoch number (1-indexed for display)
    log_dir     : directory where PNG files are saved
    class_names : list of class label strings (from dataset.classes)

    Returns
    -------
    MetricBundle with all scalars and figure paths populated
    """
    all_labels: List[int]   = []
    all_probs:  List[float] = []
    total_loss  = 0.0
    n_batches   = 0

    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        # AMP inference
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        n_batches  += 1

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_labels.extend(labels.squeeze(1).long().cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    labels_arr = np.array(all_labels, dtype=int)
    probs_arr  = np.array(all_probs,  dtype=float)
    val_loss   = total_loss / max(n_batches, 1)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(labels_arr, probs_arr)

    # ── Figures ───────────────────────────────────────────────────────────────
    roc_path = plot_roc_curve(
        labels_arr, probs_arr, metrics["auc"], epoch, log_dir
    )
    cm_path = plot_confusion_matrix(
        labels_arr, probs_arr, class_names, epoch, log_dir
    )

    bundle = MetricBundle(
        epoch     = epoch,
        val_loss  = val_loss,
        accuracy  = metrics["accuracy"],
        precision = metrics["precision"],
        recall    = metrics["recall"],
        f1        = metrics["f1"],
        auc       = metrics["auc"],
        roc_path  = roc_path,
        cm_path   = cm_path,
        labels    = all_labels,
        probs     = all_probs,
    )
    bundle.log_summary()
    return bundle