"""
train_stage1_artifact.py — Stream C: Frequency Analysis Pre-Training
=====================================================================
Entry point for Stream C of the Deepfake Video Detection System.

Pre-trains a frequency-domain ConvNeXt-Nano on the ArtiFact Dataset to
detect 2026-era Latent Diffusion artefacts (Sora, Veo, Kling) by
analysing high-frequency anomalies in the Fourier magnitude spectrum.

Hardware target : NVIDIA RTX 3060 12 GB
Mixed precision : torch.amp.autocast + GradScaler  (mandatory)
Auto-resume     : reads checkpoint_stage1_last.pth if present
Graceful exit   : SIGINT / SIGTERM -> emergency checkpoint

Usage
-----
    # Train from scratch
    python train_stage1_artifact.py

    # Override data directories
    python train_stage1_artifact.py --train-dir /data/artifact/train \
                                    --val-dir   /data/artifact/val

Author : Principal AI Research Engineer
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# -- Internal modules ----------------------------------------------------------
from model   import FrequencyModel
from dataset import build_dataloaders
from metrics import evaluate_epoch
from utils   import (
    AverageMeter,
    CheckpointManager,
    EarlyStopping,
    GracefulExitHandler,
    set_seed,
    setup_logging,
)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream C — Frequency Domain Deepfake Detector (Stage 1 Pre-training)"
    )
    # -- Paths --------------------------------------------------------------
    p.add_argument("--train-dir",  default="B:\\download\\TruthSeeker-app\\frequency\\ArtiFact_240K\\train",
                   help="ImageFolder root for training data")
    p.add_argument("--val-dir",    default="B:\\download\\TruthSeeker-app\\frequency\\ArtiFact_240K\\validation",
                   help="ImageFolder root for validation data")
    p.add_argument("--log-dir",    default="logs",
                   help="Directory for logs and figure outputs")
    p.add_argument("--ckpt-dir",   default=".",
                   help="Directory for checkpoint files")

    # -- Training -----------------------------------------------------------
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight-decay",type=float, default=1e-2)
    p.add_argument("--image-size",  type=int,   default=224)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--save-every",  type=int,   default=5,
                   help="Save milestone checkpoint every N epochs")

    # -- Augmentation -------------------------------------------------------
    p.add_argument("--jpeg-low",    type=int,   default=50,
                   help="JPEG compression quality lower bound")
    p.add_argument("--jpeg-high",   type=int,   default=90,
                   help="JPEG compression quality upper bound")

    # -- Model --------------------------------------------------------------
    p.add_argument("--no-pretrained",     action="store_true",
                   help="Disable ImageNet pre-trained weights")
    p.add_argument("--freeze-epochs",     type=int, default=0,
                   help="Freeze ConvNeXt backbone for this many epochs")
    p.add_argument("--early-stop-patience", type=int, default=10,
                   help="EarlyStopping patience (0 to disable)")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Training step
# -----------------------------------------------------------------------------
def train_one_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device:    torch.device,
    epoch:     int,
    logger:    logging.Logger,
    exit_handler: GracefulExitHandler,
) -> Tuple[float, float]:
    """
    Run one full training epoch with AMP.

    Returns
    -------
    (train_loss, train_acc) : Tuple of floats
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    acc_meter  = AverageMeter("train_acc")
    
    n_batches  = len(loader)
    log_every  = max(1, n_batches // 10)     # log ~10 times per epoch

    for batch_idx, (images, labels) in enumerate(loader):

        # -- Interrupt check (between batches) ---------------------------------
        if exit_handler.should_exit:
            logger.warning("Exit flag detected - breaking training loop early.")
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        # -- Forward pass (AMP) ------------------------------------------------
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        # -- Backward + gradient clipping + step -------------------------------
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # -- Metrics -----------------------------------------------------------
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc   = (preds == labels).float().mean()

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(acc.item(),   n=images.size(0))

        # -- Progress logging --------------------------------------------------
        if (batch_idx + 1) % log_every == 0 or batch_idx == n_batches - 1:
            pct = 100.0 * (batch_idx + 1) / n_batches
            logger.info(
                f"  Epoch [{epoch:02d}] "
                f"[{batch_idx+1:4d}/{n_batches}] ({pct:5.1f}%)  "
                f"Loss: {loss_meter.avg:.4f}  "
                f"Acc: {acc_meter.avg:.4f}  "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    # Scheduler steps once per epoch
    scheduler.step()

    return loss_meter.avg, acc_meter.avg


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args   = parse_args()
    logger = setup_logging(log_dir=args.log_dir, log_file="training.log")

    # -- Header ---------------------------------------------------------------
    logger.info("=" * 68)
    logger.info("  Stream C — Frequency Domain Deepfake Detector  |  Stage 1")
    logger.info("=" * 68)
    logger.info(f"  Config: {vars(args)}")

    # -- Reproducibility -------------------------------------------------------
    set_seed(args.seed)

    # -- Device ---------------------------------------------------------------
    if not torch.cuda.is_available():
        logger.error("CUDA is required for AMP training. No GPU detected — aborting.")
        raise SystemExit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem  = torch.cuda.get_device_properties(device).total_memory / 1e9
    logger.info(f"  GPU : {gpu_name}  ({gpu_mem:.1f} GB)")

    # -- Data -----------------------------------------------------------------
    logger.info("Building data loaders ...")
    train_loader, val_loader, class_info = build_dataloaders(
        train_dir         = args.train_dir,
        val_dir           = args.val_dir,
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
        image_size        = args.image_size,
        jpeg_quality_low  = args.jpeg_low,
        jpeg_quality_high = args.jpeg_high,
    )
    logger.info(
        f"  Classes     : {class_info['classes']}\n"
        f"  Train       : {class_info['train_samples']:,} samples  "
        f"(per class: {class_info['class_counts']})\n"
        f"  Validation  : {class_info['val_samples']:,} samples"
    )

    # -- Model -----------------------------------------------------------------
    logger.info("Instantiating FrequencyModel (ConvNeXt-Nano backbone) ...")
    model = FrequencyModel(
        pretrained            = not args.no_pretrained,
        freeze_backbone_epochs = args.freeze_epochs,
    ).to(device)

    param_info = model.count_parameters()
    logger.info(
        f"  Parameters: {param_info['total']:,} total | "
        f"{param_info['trainable']:,} trainable"
    )

    # -- Loss, Optimizer, Scheduler, Scaler ------------------------------------
    pos_count = class_info["class_counts"][1] if len(class_info["class_counts"]) > 1 else 1
    neg_count = class_info["class_counts"][0]
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"  BCEWithLogitsLoss pos_weight = {pos_weight.item():.3f}")

    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = args.epochs,
        eta_min = args.lr * 0.01,
    )

    scaler = torch.amp.GradScaler("cuda")

    # -- Infrastructure --------------------------------------------------------
    ckpt_manager = CheckpointManager(
        ckpt_dir     = args.ckpt_dir,
        save_every_n = args.save_every,
    )
    exit_handler = GracefulExitHandler()
    early_stop   = (
        EarlyStopping(patience=args.early_stop_patience)
        if args.early_stop_patience > 0
        else None
    )

    # -- Auto-resume -----------------------------------------------------------
    start_epoch = ckpt_manager.load_if_exists(
        model, optimizer, scheduler, scaler, device
    )

    # -- Training loop ---------------------------------------------------------
    logger.info("=" * 68)
    logger.info(f"  Starting training from epoch {start_epoch + 1} / {args.epochs}")
    logger.info("=" * 68)

    train_history = []

    for epoch in range(start_epoch + 1, args.epochs + 1):

        # -- Interrupt check ---------------------------------------------------
        if exit_handler.should_exit:
            exit_handler.do_exit(
                epoch - 1, model, optimizer, scheduler, scaler, ckpt_manager
            )

        epoch_start = time.time()

        # -- Freeze / unfreeze schedule ----------------------------------------
        model.maybe_unfreeze(epoch - 1)

        # -- Train -------------------------------------------------------------
        logger.info(f"\n{'='*68}")
        logger.info(f"  EPOCH {epoch:02d}/{args.epochs}  [TRAIN]")
        logger.info(f"{'-'*68}")

        train_loss, train_acc = train_one_epoch(
            model        = model,
            loader       = train_loader,
            criterion    = criterion,
            optimizer    = optimizer,
            scaler       = scaler,
            scheduler    = scheduler,
            device       = device,
            epoch        = epoch,
            logger       = logger,
            exit_handler = exit_handler,
        )

        # -- Validate ----------------------------------------------------------
        logger.info(f"\n  EPOCH {epoch:02d}/{args.epochs}  [VALIDATE]")

        bundle = evaluate_epoch(
            model        = model,
            loader       = val_loader,
            criterion    = criterion,
            device       = device,
            epoch        = epoch,
            log_dir      = args.log_dir,
            class_names  = class_info["classes"],
        )

        elapsed = time.time() - epoch_start
        logger.info(
            f"\n  Epoch {epoch:02d} result:"
            f"\n  --------------------------------------------------"
            f"\n  |  Train Loss : {train_loss:.4f}  |  Train Acc : {train_acc:.4f}"
            f"\n  |  Val Loss   : {bundle.val_loss:.4f}  |  Val Acc   : {bundle.accuracy:.4f}"
            f"\n  |  Val AUC    : {bundle.auc:.4f}"
            f"\n  --------------------------------------------------"
            f"\n  Time: {elapsed:.1f}s"
        )

        # -- Checkpoint --------------------------------------------------------
        ckpt_manager.save(
            epoch     = epoch,
            model     = model,
            optimizer = optimizer,
            scheduler = scheduler,
            scaler    = scaler,
            val_auc   = bundle.auc,
            extra     = {
                "train_loss": train_loss,
                "train_acc" : train_acc,
                "val_loss"  : bundle.val_loss,
                "val_acc"   : bundle.accuracy,
                "val_f1"    : bundle.f1,
            },
        )

        # -- History -----------------------------------------------------------
        train_history.append({
            "epoch"     : epoch,
            "train_loss": train_loss,
            "train_acc" : train_acc,
            "val_loss"  : bundle.val_loss,
            "val_acc"   : bundle.accuracy,
            "val_auc"   : bundle.auc,
        })

        # -- Early stopping ----------------------------------------------------
        if early_stop is not None and early_stop.step(bundle.auc):
            logger.info(
                f"  Early stopping at epoch {epoch} "
                f"(best AUC = {early_stop.best:.4f})"
            )
            break

    # -- Training summary ------------------------------------------------------
    logger.info("\n" + "=" * 68)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 68)
    if train_history:
        best = max(train_history, key=lambda x: x["val_auc"])
        logger.info(
            f"  Best epoch   : {best['epoch']:02d}\n"
            f"  Best Val AUC : {best['val_auc']:.4f}\n"
            f"  Best Val Acc : {best['val_acc']:.4f}\n"
            f"  Best Checkpoint -> {ckpt_manager.best_path}"
        )
    logger.info("=" * 68)

    # -- Plotting --------------------------------------------------------------
    _save_training_curve(train_history, args.log_dir)


# -----------------------------------------------------------------------------
# Training curve summary plot
# -----------------------------------------------------------------------------
def _save_training_curve(history: list, log_dir: str) -> None:
    """Save a 3-panel plot: Loss, Accuracy, and AUC."""
    if not history:
        return

    import matplotlib.pyplot as plt

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc  = [h["train_acc"]  for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_acc    = [h["val_acc"]    for h in history]
    val_auc    = [h["val_auc"]    for h in history]

    # Create 3 subplots side-by-side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333333")
        ax.grid(True, color="#1e2030", linewidth=0.5)
        ax.set_xlabel("Epoch", color="white")

    # 1. Loss
    ax1.plot(epochs, train_loss, color="#00e5ff", lw=2, label="Train")
    ax1.plot(epochs, val_loss,   color="#ff6b6b", lw=2, label="Val")
    ax1.set_ylabel("Loss", color="white")
    ax1.set_title("Loss", color="white", fontweight="bold")
    ax1.legend(facecolor="#1c1c2e", edgecolor="#555", labelcolor="white")

    # 2. Accuracy
    ax2.plot(epochs, train_acc, color="#00e5ff", lw=2, label="Train")
    ax2.plot(epochs, val_acc,   color="#ff6b6b", lw=2, label="Val")
    ax2.set_ylabel("Accuracy", color="white")
    ax2.set_title("Accuracy", color="white", fontweight="bold")
    ax2.legend(facecolor="#1c1c2e", edgecolor="#555", labelcolor="white")
    ax2.set_ylim([0, 1.05])

    # 3. AUC (Val only)
    ax3.plot(epochs, val_auc, color="#a8ff78", lw=2.5, label="Val AUC")
    best_auc = max(val_auc)
    ax3.axhline(best_auc, color="#ff6b6b", ls="--", lw=1, label=f"Best={best_auc:.4f}")
    ax3.set_ylabel("AUC", color="white")
    ax3.set_title("Validation AUC", color="white", fontweight="bold")
    ax3.legend(facecolor="#1c1c2e", edgecolor="#555", labelcolor="white")
    ax3.set_ylim([0, 1.05])

    fig.suptitle(
        "Stream C — Frequency Domain Model | Training Summary",
        color="white", fontsize=14, fontweight="bold",
    )
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(log_dir) / "training_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logging.getLogger("StreamC").info(f"  Training curve saved -> {path}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()