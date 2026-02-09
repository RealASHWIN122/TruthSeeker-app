"""
Pre-processing script: Converts video dataset into image dataset.
Run this ONCE before training.
"""
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Your raw video dataset location
SOURCE_ROOT = Path(r"B:\download\TruthSeeker-app\ensemble\datasets")
# Where to save the images (New folder)
OUTPUT_ROOT = Path(r"B:\download\TruthSeeker-app\ensemble\preprocessed_dataset")

# Folders to process (Must match your folders in 'datasets')
TARGET_FOLDERS = ["Celeb-real", "Celeb-synthesis", "original"]
# ---------------------

def extract_frames(video_path, output_folder, num_frames=20):
    """Extracts frames from a video and saves them as JPGs."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return

    # Create a specific folder for this video's frames
    # e.g., preprocessed/Celeb-real/id0_0000
    video_name = video_path.stem
    save_dir = output_folder / video_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Uniform sampling: Get frames spread evenly across the video
    if total_frames > num_frames:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    else:
        indices = range(total_frames)

    current_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_idx in indices:
            # Save as high-quality JPG (Quality 95 is near-lossless but fast)
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(save_dir / frame_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1
            
        current_idx += 1
        
    cap.release()

def main():
    print(f"🚀 Starting Extraction...")
    print(f"Source: {SOURCE_ROOT}")
    print(f"Destination: {OUTPUT_ROOT}")
    
    for folder_name in TARGET_FOLDERS:
        source_folder = SOURCE_ROOT / folder_name
        dest_folder = OUTPUT_ROOT / folder_name
        
        if not source_folder.exists():
            print(f"⚠️  Skipping {folder_name} (Not found)")
            continue
            
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        videos = list(source_folder.glob("*.mp4")) + list(source_folder.glob("*.avi")) + list(source_folder.glob("*.mov"))
        print(f"\nProcessing {folder_name} ({len(videos)} videos)...")
        
        # Process with progress bar
        for video in tqdm(videos):
            extract_frames(video, dest_folder)

    print("\n✅ Extraction Complete!")

if __name__ == "__main__":
    main()