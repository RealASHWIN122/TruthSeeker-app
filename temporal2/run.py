import os
import sys
import subprocess

# ==============================================================================
# 🚀 TRUTHSEEKER LAUNCHPAD: STREAM B (TEMPORAL PHYSICS ENGINE)
# ==============================================================================

# 📂 1. DIRECTORY CONFIGURATION
# Ensure this points exactly to the folder containing your 'train' and 'val' folders
DATA_DIR = r"B:\download\TruthSeeker-app\temporal_dataset"

# ⚙️ 2. VRAM SURVIVAL HYPERPARAMETERS (Optimized for RTX 3060)
BATCH_SIZE = "4"           # Physical batch size (Keep it at 4 to avoid CUDA Out of Memory)
ACCUMULATE_STEPS = "8"     # Virtual batch size multiplier (4 * 8 = Effective batch of 32)
NUM_FRAMES = "16"          # How many frames the 3D tensor extracts across time
EPOCHS = "15"              # Total training epochs
LEARNING_RATE = "0.0001"   # 1e-4 is standard for fine-tuning Vision Transformers
PATIENCE = "5"             # Stop early if validation AUC doesn't improve for 5 epochs

def launch_training():
    print(f"🔥 IGNITING TRUTHSEEKER LAUNCHPAD 🔥")
    print(f"Checking dataset paths...")

    # Pre-flight safety check
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ FATAL ERROR: Could not find the dataset at: {DATA_DIR}")
        print("Please check the path and make sure your 'train' and 'val' folders are inside it.")
        sys.exit(1)

    train_path = os.path.join(DATA_DIR, "train")
    if not os.path.exists(train_path):
        print(f"\n❌ FATAL ERROR: Could not find the 'train' folder inside {DATA_DIR}")
        sys.exit(1)

    print("✅ All directories verified. Initiating Temporal Training Sequence...\n")

    # Build the terminal command mathematically
    command = [
        sys.executable, "B:\\download\\TruthSeeker-app\\temporal\\train.py", # sys.executable ensures it uses your active Python/Conda environment
        "--data_dir", DATA_DIR,
        "--batch_size", BATCH_SIZE,
        "--accumulate_steps", ACCUMULATE_STEPS,
        "--num_frames", NUM_FRAMES,
        "--epochs", EPOCHS,
        "--lr", LEARNING_RATE,
        "--patience", PATIENCE
    ]

    # Optional: If you have a specific checkpoint you want to force-resume from, 
    # you can uncomment the lines below and add the path.
    # RESUME_CHECKPOINT = r"B:\download\TruthSeeker-app\temporal_dataset\checkpoints_temporal\temporal_last_checkpoint.pth"
    # command.extend(["--resume", RESUME_CHECKPOINT])

    try:
        # Launch the training script and stream the output to the console
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Training process crashed or was interrupted. Exit code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 Manual abort triggered from launchpad.")

if __name__ == "__main__":
    launch_training()