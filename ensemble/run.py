import os

# Configuration
# Note: DATA_DIR is less relevant now as paths in txt files are absolute, 
# but we keep it for argument consistency
DATA_DIR = r"B:\download\TruthSeeker-app\ensemble\preprocessed_dataset"
TRAIN_LIST = r"B:\download\TruthSeeker-app\ensemble\final_balanced_dataset_train.txt"
VAL_LIST = r"B:\download\TruthSeeker-app\ensemble\final_balanced_dataset_val.txt"

# Optimized for RTX 3060 (You can likely increase Batch Size to 32 now)
BATCH_SIZE = 24 
EPOCHS = 50

# The Command
command = f"python B:\\download\\TruthSeeker-app\\ensemble\\train_spatial.py --data-dir {DATA_DIR} --train-list {TRAIN_LIST} --val-list {VAL_LIST} --batch-size {BATCH_SIZE} --epochs {EPOCHS} --patience 15"

print("🚀 Starting Training (Optimized Pipeline)...")
print(f"Executing: {command}")
print("-" * 50)

os.system(command)