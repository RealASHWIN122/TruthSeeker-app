import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class PreExtractedTemporalDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None, is_training=True):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.is_training = is_training
        
        self.samples = []
        for label_name, label_val in zip(['real', 'fake'], [0, 1]):
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(class_dir): continue
            
            for video_folder in os.listdir(class_dir):
                folder_path = os.path.join(class_dir, video_folder)
                if os.path.isdir(folder_path):
                    self.samples.append((folder_path, label_val))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        # Get all JPEGs and sort them to maintain temporal order
        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        tensor_frames = []
        
        # --- THE FIX: ROBUST PADDING ---
        for i in range(self.num_frames):
            if i < len(frame_files):
                img_path = os.path.join(folder_path, frame_files[i])
                img = Image.open(img_path).convert('RGB')
            else:
                # If we run out of frames, repeat the very last one available
                # This prevents the 'resize storage' crash
                img_path = os.path.join(folder_path, frame_files[-1])
                img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            tensor_frames.append(img)
            
        video_tensor = torch.stack(tensor_frames) # Always (16, 3, 224, 224)
        return video_tensor, label