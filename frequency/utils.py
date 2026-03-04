"""
utils.py — Stream C: Training Infrastructure Utilities
======================================================
Provides:
  • setup_logging()        — dual console + file logger
  • CheckpointManager      — save / load / auto-resume logic
  • GracefulExitHandler    — SIGINT / SIGTERM → checkpoint before quit
  • set_seed()             — global reproducibility seed
  • AverageMeter           — running average for loss/metric tracking
  • EarlyStopping          — optional patience-based stopping

Author : Principal AI Research Engineer
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slower but reproducible; comment out for max speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_dir: str = "logs", log_file: str = "training.log") -> logging.Logger:
    """
    Configure the 'StreamC' logger to write to both stdout and a rotating
    file.  Safe to call multiple times (idempotent after first call).

    Returns
    -------
    logging.Logger  — the configured logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("StreamC")
    if logger.handlers:          # already configured — skip
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler ──────────────────────────────────────────────────────────
    fh = logging.FileHandler(os.path.join(log_dir, log_file), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────────────────────────────────────
class CheckpointManager:
    """
    Handles all checkpoint I/O in a fault-tolerant manner.

    Saving strategy
    ───────────────
    • last.pth    — overwritten every epoch (fast restart)
    • best.pth    — overwritten when val AUC improves
    • epoch_N.pth — written every `save_every_n` epochs (milestone backup)

    Args
    ----
    ckpt_dir      : directory to store all .pth files
    save_every_n  : milestone checkpoint cadence (epochs)
    """

    def __init__(
        self,
        ckpt_dir:     str = ".",
        save_every_n: int = 5,
    ):
        self.ckpt_dir     = Path(ckpt_dir)
        self.save_every_n = save_every_n
        self.best_auc     = 0.0
        self.logger       = logging.getLogger("StreamC")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── File path helpers ─────────────────────────────────────────────────────
    @property
    def last_path(self) -> Path:
        return self.ckpt_dir / "checkpoint_stage1_last.pth"

    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "checkpoint_stage1_best.pth"

    def milestone_path(self, epoch: int) -> Path:
        return self.ckpt_dir / f"checkpoint_epoch_{epoch:02d}.pth"

    # ── Core I/O ──────────────────────────────────────────────────────────────
    def _build_state(
        self,
        epoch:     int,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler:    torch.amp.GradScaler,
        best_auc:  float,
        extra:     Optional[Dict] = None,
    ) -> dict:
        state = {
            "epoch"          : epoch,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state"   : scaler.state_dict(),
            "best_auc"       : best_auc,
            "timestamp"      : time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            state.update(extra)
        return state

    def save(
        self,
        epoch:     int,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler:    torch.amp.GradScaler,
        val_auc:   float,
        extra:     Optional[Dict] = None,
    ) -> None:
        """
        Execute the full save strategy for the completed epoch.
        """
        state = self._build_state(
            epoch, model, optimizer, scheduler, scaler, self.best_auc, extra
        )

        # ── 1. Always write 'last' checkpoint ─────────────────────────────────
        torch.save(state, self.last_path)
        # CHANGED: Checkmark and arrow removed
        self.logger.info(f"  [OK] Saved last checkpoint -> {self.last_path}")

        # ── 2. Write 'best' checkpoint if AUC improved ────────────────────────
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            torch.save(state, self.best_path)
            # CHANGED: Star and arrow removed
            self.logger.info(
                f"  [*] New best AUC {val_auc:.4f}! "
                f"Saved best checkpoint -> {self.best_path}"
            )

        # ── 3. Write milestone checkpoint every N epochs ──────────────────────
        if epoch % self.save_every_n == 0:
            mp = self.milestone_path(epoch)
            torch.save(state, mp)
            # CHANGED: Block symbol and arrow removed
            self.logger.info(f"  [M] Milestone checkpoint saved -> {mp}")

    def save_emergency(
        self,
        epoch:     int,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler:    torch.amp.GradScaler,
    ) -> None:
        """Called by the interrupt handler; saves an emergency checkpoint."""
        path = self.ckpt_dir / f"checkpoint_stage1_interrupt_ep{epoch:02d}.pth"
        state = self._build_state(
            epoch, model, optimizer, scheduler, scaler, self.best_auc
        )
        torch.save(state, path)
        self.logger.warning(f"  [!] Emergency checkpoint saved -> {path}")

    # ── Auto-resume ───────────────────────────────────────────────────────────
    def load_if_exists(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler:    torch.amp.GradScaler,
        device:    torch.device,
    ) -> int:
        """
        If `checkpoint_stage1_last.pth` exists, load all states and return
        the epoch to resume from.  Otherwise, return 0 (train from scratch).

        Returns
        -------
        int  — start_epoch (0 = fresh run, N = resume from epoch N+1)
        """
        if not self.last_path.exists():
            self.logger.info("No checkpoint found — starting from scratch.")
            return 0

        self.logger.info(f"Auto-resuming from {self.last_path} ...")
        ckpt = torch.load(self.last_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        self.best_auc = ckpt.get("best_auc", 0.0)

        start_epoch = ckpt["epoch"]
        self.logger.info(
            f"  Resumed from epoch {start_epoch} | "
            f"Best AUC so far: {self.best_auc:.4f} | "
            f"Saved at: {ckpt.get('timestamp', 'unknown')}"
        )
        return start_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Graceful Exit Handler
# ─────────────────────────────────────────────────────────────────────────────
class GracefulExitHandler:
    """
    Catches SIGINT (Ctrl+C) and SIGTERM and sets an internal flag.
    The training loop polls `.should_exit` and saves a checkpoint before
    terminating — preventing loss of the current epoch's progress.

    Usage
    -----
        handler = GracefulExitHandler()
        for epoch in range(...):
            if handler.should_exit:
                handler.do_exit(model, ...)
                break
            ...
    """

    def __init__(self):
        self.should_exit = False
        self._logger     = logging.getLogger("StreamC")
        signal.signal(signal.SIGINT,  self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        sig_name = signal.Signals(signum).name
        self._logger.warning(
            f"\n{'='*60}\n"
            f"  {sig_name} received — training will stop after this batch.\n"
            f"  An emergency checkpoint will be saved.\n"
            f"{'='*60}"
        )
        self.should_exit = True

    def do_exit(
        self,
        epoch:        int,
        model:        nn.Module,
        optimizer:    torch.optim.Optimizer,
        scheduler:    Any,
        scaler:       torch.amp.GradScaler,
        ckpt_manager: CheckpointManager,
    ) -> None:
        ckpt_manager.save_emergency(epoch, model, optimizer, scheduler, scaler)
        self._logger.info("Graceful exit complete. Goodbye.")
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Running average meter
# ─────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    """Track a running mean of scalar values (e.g. per-batch loss)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum   += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.4f}, n={self.count})"


# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Raises the internal flag when the monitored metric has not improved
    for `patience` consecutive evaluations.

    Higher-is-better mode (AUC, accuracy, F1, etc.)

    Args
    ----
    patience  : epochs to wait before triggering stop
    min_delta : minimum change to count as an improvement
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best       = 0.0
        self.stop       = False
        self._logger    = logging.getLogger("StreamC")

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            self._logger.info(
                f"  EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self._logger.warning(
                    "  EarlyStopping triggered — no improvement in AUC."
                )
                self.stop = True
        return self.stop