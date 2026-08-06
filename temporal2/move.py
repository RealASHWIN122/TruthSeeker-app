import os
import cv2
import argparse
from tqdm import tqdm

def extract_and_save_frames(video_path, output_folder, num_frames=16):
    """Extracts uniform frames from a video and saves them as JPEGs."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return False

    os.makedirs(output_folder, exist_ok=True)
    step = max(total_frames // num_frames, 1)
    
    for i in range(num_frames):
        frame_idx = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Save frame directly to disk
            frame_path = os.path.join(output_folder, f"frame_{i:02d}.jpg")
            cv2.imwrite(frame_path, frame)
            
    cap.release()
    return True

def process_dataset(input_dir, output_dir, num_frames=16):
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            video_dir = os.path.join(input_dir, split, label)
            save_dir = os.path.join(output_dir, split, label)
            
            if not os.path.exists(video_dir):
                continue
                
            videos = [v for v in os.listdir(video_dir) if v.endswith(('.mp4', '.avi'))]
            print(f"Extracting {len(videos)} videos from {split}/{label}...")
            
            for video_name in tqdm(videos):
                video_path = os.path.join(video_dir, video_name)
                # Create a subfolder for this specific video's frames
                video_folder_name = video_name.split('.')[0]
                output_folder = os.path.join(save_dir, video_folder_name)
                
                # Skip if already extracted
                if os.path.exists(output_folder) and len(os.listdir(output_folder)) == num_frames:
                    continue
                    
                extract_and_save_frames(video_path, output_folder, num_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save frames")
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, args.num_frames)