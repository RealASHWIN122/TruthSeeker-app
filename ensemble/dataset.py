import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    """
    Fast Dataset for loading pre-extracted deepfake frames.
    """
    
    def __init__(self,
                 video_paths: List[str], # These are now paths to FOLDERS containing jpgs
                 labels: List[int],
                 num_frames: int = 16,
                 frame_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 compression_augment: bool = True):
        
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment
        self.compression_augment = compression_augment
        
        # Spatial augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.Resize(height=frame_size[0], width=frame_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=frame_size[0], width=frame_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def apply_compression_augment(self, frame: np.ndarray) -> np.ndarray:
        if not self.compression_augment or np.random.rand() > 0.5:
            return frame
        quality = np.random.randint(30, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    def load_frames(self, folder_path: str) -> torch.Tensor:
        """Load frames from a pre-extracted folder (Super Fast)"""
        path = Path(folder_path)
        
        # Find all jpgs
        frame_files = sorted(list(path.glob("*.jpg")))
        
        frames = []
        
        # Handle empty or missing folders (Fallback)
        if len(frame_files) == 0:
            dummy = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            transformed = self.transform(image=dummy)['image']
            return transformed.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # Sampling Logic
        if len(frame_files) >= self.num_frames:
            # Uniformly select N frames
            indices = np.linspace(0, len(frame_files) - 1, self.num_frames, dtype=int)
            selected_files = [frame_files[i] for i in indices]
        else:
            # Loop if we don't have enough frames
            selected_files = []
            while len(selected_files) < self.num_frames:
                selected_files.extend(frame_files)
            selected_files = selected_files[:self.num_frames]
        
        # Load Images
        for img_path in selected_files:
            frame = cv2.imread(str(img_path))
            if frame is None: continue # Skip corrupt files
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame = self.apply_compression_augment(frame)
            transformed = self.transform(image=frame)
            frames.append(transformed['image'])
        
        if len(frames) == 0: # Double check safety
             dummy = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
             transformed = self.transform(image=dummy)['image']
             return transformed.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        return torch.stack(frames, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frames = self.load_frames(self.video_paths[idx])
        return {
            'frames': frames,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn(batch):
    frames = torch.stack([item['frames'] for item in batch], dim=0)
    labels = torch.stack([item['label'] for item in batch], dim=0)
    return {'frames': frames, 'labels': labels}

def create_dataloaders(train_paths, train_labels, val_paths, val_labels, 
                       batch_size=32, num_workers=4, num_frames=16):
    
    train_ds = DeepfakeDataset(train_paths, train_labels, num_frames=num_frames, augment=True)
    val_ds = DeepfakeDataset(val_paths, val_labels, num_frames=num_frames, augment=False)
    
    # ADD persistent_workers=True
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn, 
                              pin_memory=True, drop_last=True, 
                              persistent_workers=True) # <--- ADD THIS
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn, 
                            pin_memory=True, 
                            persistent_workers=True) # <--- ADD THIS
    
    return train_loader, val_loader