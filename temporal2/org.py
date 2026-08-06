import os
import shutil
from tqdm import tqdm

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Update this to exactly where your extracted train/val folders live
DATA_DIR = r"B:\download\TruthSeeker-app\dataset" 
# ==========================================

def organize_loose_frames(directory_path):
    """Sweeps up loose _frameX.jpg files and groups them into video subfolders."""
    if not os.path.exists(directory_path):
        print(f"⚠️ Skipping {directory_path} (Not found)")
        return

    # Grab only the loose image files (ignores folders if any already exist)
    all_files = [
        f for f in os.listdir(directory_path) 
        if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not all_files:
        print(f"✅ {directory_path} is already organized (0 loose files found).")
        return

    print(f"\n🚀 Organizing {len(all_files)} files in: {directory_path}")

    for filename in tqdm(all_files, desc="Boxing frames", unit="file"):
        # THE MAGIC CUT: 
        # "0001_HOpensora_frame4.jpg" -> Splits into ["0001_HOpensora", "4.jpg"]
        # We grab the first part [0] to use as the folder name.
        if "_frame" not in filename:
            continue # Skip files that don't match the pattern just in case
            
        video_id = filename.rsplit('_frame', 1)[0]
        
        # Create the specific subfolder for this video
        folder_path = os.path.join(directory_path, video_id)
        os.makedirs(folder_path, exist_ok=True)
        
        # Move the image into its new home
        src_path = os.path.join(directory_path, filename)
        dst_path = os.path.join(folder_path, filename)
        shutil.move(src_path, dst_path)

if __name__ == "__main__":
    print("🧹 Initiating Pre-Extracted Frame Organizer...")
    
    # Run the sweeper on all 4 possible directories
    organize_loose_frames(os.path.join(DATA_DIR, "train", "real"))
    organize_loose_frames(os.path.join(DATA_DIR, "train", "fake"))
    organize_loose_frames(os.path.join(DATA_DIR, "val", "real"))
    organize_loose_frames(os.path.join(DATA_DIR, "val", "fake"))
    
    print("\n✅ SUCCESS! All frames are perfectly boxed up and ready for the 3D Temporal Dataloader.")