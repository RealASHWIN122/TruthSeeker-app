import os
import cv2
import subprocess
import argparse
from tqdm import tqdm

def get_video_frame_count(video_path):
    """Quickly grab the total frames using cv2 metadata."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def extract_frames_with_gpu(video_path, output_folder, num_frames=16):
    """Uses FFmpeg and the NVIDIA NVDEC chip to extract frames."""
    total_frames = get_video_frame_count(video_path)
    if total_frames <= 0:
        return False

    os.makedirs(output_folder, exist_ok=True)
    step = max(total_frames // num_frames, 1)
    
    output_pattern = os.path.join(output_folder, "frame_%02d.jpg")
    
    # The magic FFmpeg command
    # -hwaccel cuda: Forces the work onto the RTX 3060
    # -vf select: Tells it exactly which frames to grab to ensure uniform spacing
    cmd = [
        "ffmpeg", 
        "-hwaccel", "cuda",      
        "-v", "error",           
        "-i", video_path, 
        "-vf", f"select='not(mod(n\,{step}))'", 
        "-vframes", str(num_frames),
        "-q:v", "2",             # High quality JPEG
        "-y",                    # Overwrite if exists
        output_pattern
    ]
    
    try:
        # Run the command silently
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except FileNotFoundError:
        print("\n❌ ERROR: FFmpeg is not installed or not in your system PATH.")
        print("Please run 'winget install ffmpeg' in your terminal and restart it.")
        return False
    except subprocess.CalledProcessError:
        return False

def process_dataset_gpu(input_dir, output_dir, num_frames=16):
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            video_dir = os.path.join(input_dir, split, label)
            save_dir = os.path.join(output_dir, split, label)
            
            if not os.path.exists(video_dir):
                continue
                
            videos = [v for v in os.listdir(video_dir) if v.endswith(('.mp4', '.avi'))]
            print(f"Extracting {len(videos)} videos from {split}/{label} via GPU...")
            
            for video_name in tqdm(videos):
                video_path = os.path.join(video_dir, video_name)
                video_folder_name = video_name.split('.')[0]
                output_folder = os.path.join(save_dir, video_folder_name)
                
                if os.path.exists(output_folder) and len(os.listdir(output_folder)) == num_frames:
                    continue
                    
                success = extract_frames_with_gpu(video_path, output_folder, num_frames)
                if not success:
                    break # Stop if FFmpeg isn't installed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save frames")
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()
    
    process_dataset_gpu(args.input_dir, args.output_dir, args.num_frames)