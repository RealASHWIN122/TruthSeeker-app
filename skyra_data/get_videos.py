import os
import zipfile
from huggingface_hub import hf_hub_download

print("⏳ Initiating 20GB video download from Hugging Face...")
print("   (Depending on your server's connection, this could take a few minutes)")

try:
    # Download the massive zip file
    file_path = hf_hub_download(
        repo_id="JoeLeelyf/ViF-CoT-4K", 
        filename="source_videos.zip", 
        repo_type="dataset",
        local_dir="."
    )

    print("📦 Download complete! Extracting the 300-video training subset...")
    
    os.makedirs("./videos", exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        video_files = [f for f in zip_ref.namelist() if f.endswith('.mp4')]
        subset = video_files[:300]
        
        for i, file in enumerate(subset):
            zip_ref.extract(file, "./videos")
            if i % 50 == 0 and i > 0:
                print(f"  ...extracted {i}/300 videos")

    # Clean up to save 20GB of disk space
    print("🧹 Deleting the original 20GB zip file...")
    os.remove(file_path)
    print("✅ DATASET IS FULLY LOADED AND READY!")

except Exception as e:
    print(f"❌ Error: {e}")
