"""
run_experiment.py
=================
Orchestration script for Stream C: Frequency Domain Deepfake Detector.

This script wraps 'train_stage1_artifact.py' to easily manage paths
(based on your local ArtiFact_240K structure) and hyperparameters.
"""

import os
import sys
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. USER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Update this path to match the parent folder seen in your screenshot
# Example: r"B:\download\TruthSeeker-app\frequency\ArtiFact_240K"
DATASET_ROOT = r"B:\download\TruthSeeker-app\frequency\ArtiFact_240K"

# Directory where logs and checkpoints will be saved
OUTPUT_DIR   = "experiments/run_01"

# Hyperparameters
BATCH_SIZE   = 32      # Default: 32
EPOCHS       = 20      # Default: 30
LEARNING_RATE= 1e-3    # Default: 1e-4
NUM_WORKERS  = 4       # Adjust based on CPU cores

# Advanced Flags
FREEZE_EPOCHS = 0      # Set > 0 to freeze backbone initially
RESUME_CHECKPOINT = False # If True, the script will look for existing .pth files

# ─────────────────────────────────────────────────────────────────────────────
# 2. PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Convert to Path objects for cross-platform safety
    root_path = Path(DATASET_ROOT)
    
    # Based on your screenshot, the structure is root/train and root/validation
    train_path = root_path / "train"
    val_path   = root_path / "validation"
    
    # Verify paths exist before starting
    if not train_path.exists():
        print(f"❌ Error: Train directory not found at: {train_path}")
        print("   Please check the DATASET_ROOT variable in run_experiment.py")
        sys.exit(1)
        
    if not val_path.exists():
        print(f"❌ Error: Validation directory not found at: {val_path}")
        sys.exit(1)

    # Create output directories
    log_dir  = Path(OUTPUT_DIR) / "logs"
    ckpt_dir = Path(OUTPUT_DIR) / "checkpoints"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ Data found at: {root_path}")
    print(f"📂 Outputting to: {Path(OUTPUT_DIR).absolute()}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. BUILD COMMAND
    # ─────────────────────────────────────────────────────────────────────────
    
    # The command corresponds to the arguments in train_stage1_artifact.py
    cmd = [
        sys.executable, "B:\\download\\TruthSeeker-app\\frequency\\train_stage1_artifact.py",
        "--train-dir", str(train_path),
        "--val-dir",   str(val_path),
        "--log-dir",   str(log_dir),
        "--ckpt-dir",  str(ckpt_dir),
        "--epochs",     str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--lr",         str(LEARNING_RATE),
        "--num-workers",str(NUM_WORKERS),
        "--freeze-epochs", str(FREEZE_EPOCHS),
        "--no-pretrained", # Uncomment if you want to train from scratch
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # 4. EXECUTE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🚀 Starting Training Pipeline...")
    print(f"   Command: {' '.join(cmd)}\n")
    
    try:
        # Check if requirements are installed (optional sanity check)
        # This mirrors requirements.txt
        import torch
        if not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA not detected. Training will be extremely slow.")
            
        # Run the training script
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()