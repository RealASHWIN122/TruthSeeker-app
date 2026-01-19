import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import dinov2_features as d2

# OPTIMIZED SETTINGS FOR RTX 3060 (12GB VRAM)
batch_size = 64
device = "cuda:0" if torch.cuda.is_available() else "cpu"
T = 16
window_sec = 1.5
num_workers = 8  # Parallel video decoding threads

real_root = Path("B:\\download\\TruthSeeker-app\\restrav\\DATA\\TRAINING_DATA\\REAL")
fake_root = Path("B:\\download\\TruthSeeker-app\\restrav\\DATA\\TRAINING_DATA\\FAKE")
output_h5 = Path("training_features.h5")

real_videos = sorted(real_root.rglob("*.mp4"))
fake_videos = sorted(fake_root.rglob("*.mp4"))
all_videos = [(str(p), 1) for p in real_videos] + [(str(p), 0) for p in fake_videos]

print(f"Found {len(real_videos)} real and {len(fake_videos)} fake videos.")
print(f"Device: {device}")
print(f"Batch size: {batch_size}")
print(f"Parallel workers: {num_workers}")

# Pre-load model to GPU once
print("Loading DINOv2 model to GPU...")
d2.dinov2 = d2.dinov2.to(device)


def decode_video_batch(video_paths):
    """Decode a batch of videos in parallel"""
    outs = []
    
    def decode_single(p):
        try:
            clip = d2.decode_clip(p, T=T, window_sec=window_sec)
            return d2.preprocess(clip)
        except Exception as e:
            print(f"Error decoding {p}: {e}")
            return torch.zeros(T, 3, 224, 224)
    
    # Parallel decoding
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        outs = list(executor.map(decode_single, video_paths))
    
    return outs


def process_batch_on_gpu(preprocessed_clips):
    """Process preprocessed clips on GPU"""
    if len(preprocessed_clips) == 0:
        raise ValueError("No clips to process")
    
    # Stack and move to GPU
    batch = torch.cat(preprocessed_clips, dim=0).to(device)
    
    with torch.no_grad():
        feats = d2.dinov2.forward_features(batch)
        cls = feats["x_norm_clstoken"].unsqueeze(1)
        patches = feats["x_norm_patchtokens"]
        tokens = torch.cat([cls, patches], dim=1)
        Z_flat = tokens.flatten(1)
        Z = Z_flat.view(len(preprocessed_clips), T, -1)
        
        # Compute features
        features = d2.features_from_Z(Z).cpu().numpy()
    
    return features


# Main processing loop with pipelining
with h5py.File(output_h5, "w") as h5f:
    dt = h5py.special_dtype(vlen=str)
    path_ds = h5f.create_dataset("path", (len(all_videos),), dtype=dt)
    label_ds = h5f.create_dataset("label", (len(all_videos),), dtype="i")
    feat_ds = h5f.create_dataset("features", (len(all_videos), 21), dtype="f")

    failed_count = 0
    
    # Create a queue for pipelining CPU and GPU work
    decode_queue = queue.Queue(maxsize=2)  # Pre-decode 2 batches ahead
    
    def decoder_thread():
        """Background thread for video decoding"""
        for idx in range(0, len(all_videos), batch_size):
            batch_items = all_videos[idx:idx+batch_size]
            batch_paths = [p for p, _ in batch_items]
            batch_labels = [l for _, l in batch_items]
            
            preprocessed = decode_video_batch(batch_paths)
            decode_queue.put((idx, batch_paths, batch_labels, preprocessed))
        
        decode_queue.put(None)  # Signal completion
    
    # Start decoder thread
    decoder = threading.Thread(target=decoder_thread, daemon=True)
    decoder.start()
    
    # Process batches as they become available
    pbar = tqdm(total=len(all_videos), desc="Extracting features")
    
    while True:
        item = decode_queue.get()
        if item is None:
            break
        
        idx, batch_paths, batch_labels, preprocessed = item
        
        try:
            # GPU processing
            feats = process_batch_on_gpu(preprocessed)
            
            # Save to HDF5
            for j, (path, label, f) in enumerate(zip(batch_paths, batch_labels, feats)):
                pos = idx + j
                path_ds[pos] = path
                label_ds[pos] = label
                feat_ds[pos, :] = f
                
        except Exception as e:
            print(f"\nError in batch {idx}: {e}")
            failed_count += len(batch_paths)
            # Save dummy features for failed batch
            for j, (path, label) in enumerate(zip(batch_paths, batch_labels)):
                pos = idx + j
                path_ds[pos] = path
                label_ds[pos] = label
                feat_ds[pos, :] = np.zeros(21, dtype=np.float32)
        
        pbar.update(len(batch_paths))
    
    pbar.close()
    decoder.join()

print(f"\nSaved features for {len(all_videos)} videos to {output_h5}")
if failed_count > 0:
    print(f"Warning: {failed_count} videos failed to process")