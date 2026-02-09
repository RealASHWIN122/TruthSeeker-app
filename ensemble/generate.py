import os
import random
from pathlib import Path

# --- CONFIGURATION (UPDATED) ---
# Point to the NEW folder created by extract_frames.py
DATASET_ROOT = r"B:\download\TruthSeeker-app\ensemble\preprocessed_dataset"
REAL_FOLDERS = ["Celeb-real", "original"]
FAKE_FOLDERS = ["Celeb-synthesis"]
OUTPUT_NAME = "final_balanced_dataset"
VAL_SPLIT = 0.2
# -------------------------------

def get_video_folders(root, folder_names):
    video_folders = []
    for folder in folder_names:
        path = Path(root) / folder
        if path.exists():
            print(f"  Scanning {folder}...")
            # In the preprocessed dataset, every subfolder is a "video"
            # e.g. Celeb-real/id0_0000/
            subfolders = [f for f in path.iterdir() if f.is_dir()]
            video_folders.extend([str(f) for f in subfolders])
        else:
            print(f"⚠️ Warning: Folder '{folder}' not found in {root}!")
    return video_folders

def main():
    print(f"Scanning preprocessed data at {DATASET_ROOT}...")
    
    real_videos = get_video_folders(DATASET_ROOT, REAL_FOLDERS)
    fake_videos = get_video_folders(DATASET_ROOT, FAKE_FOLDERS)
    
    # Remove duplicates
    real_videos = list(set(real_videos))
    fake_videos = list(set(fake_videos))

    print(f"\n--- Found ---")
    print(f"Real Samples: {len(real_videos)}")
    print(f"Fake Samples: {len(fake_videos)}")

    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("❌ Error: No data found. Did you run extract_frames.py?")
        return

    # Split Train/Val
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    real_split = int(len(real_videos) * (1 - VAL_SPLIT))
    train_real = real_videos[:real_split]
    val_real = real_videos[real_split:]

    fake_split = int(len(fake_videos) * (1 - VAL_SPLIT))
    train_fake = fake_videos[:fake_split]
    val_fake = fake_videos[fake_split:]

    # Balance Training Data
    target_count = len(train_fake)
    current_count = len(train_real)
    balanced_train_real = []
    
    if target_count > current_count:
        ratio = target_count / current_count
        print(f"Balancing: Repeating Reals {ratio:.2f}x")
        full_repeats = int(ratio)
        for _ in range(full_repeats):
            balanced_train_real.extend(train_real)
        remainder = target_count - len(balanced_train_real)
        balanced_train_real.extend(random.sample(train_real, remainder))
    else:
        balanced_train_real = random.sample(train_real, target_count)

    # Create Lines
    train_lines = [f"{v} 0" for v in balanced_train_real] + [f"{v} 1" for v in train_fake]
    val_lines = [f"{v} 0" for v in val_real] + [f"{v} 1" for v in val_fake]

    random.shuffle(train_lines)
    random.shuffle(val_lines)

    with open(f"{OUTPUT_NAME}_train.txt", "w") as f:
        f.write("\n".join(train_lines))
    
    with open(f"{OUTPUT_NAME}_val.txt", "w") as f:
        f.write("\n".join(val_lines))

    print(f"\n✅ Success! Lists created.")

if __name__ == "__main__":
    main()