import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from decord import VideoReader, cpu, gpu
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message="xFormers is available.*",
    category=UserWarning,
    module=r"dinov2\.layers\..*",
)

dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()

def preprocess(frames):
    """
    Preprocess frames for DINOv2
    Args:
        frames: Tensor of shape (T, C, H, W) with values in [0, 255]
    Returns:
        Tensor of shape (T, 3, 224, 224) normalized to [0, 1]
    """
    frames = frames.float() / 255.0
    resized = []
    for i in range(frames.shape[0]):
        frame = frames[i]
        resized_frame = TF.resize(frame, [224, 224], InterpolationMode.BICUBIC, antialias=True)
        resized.append(resized_frame)
    return torch.stack(resized)

def decode_clip(video_path, T=24, window_sec=2.0, center_time=None, use_gpu=False):
    """
    Extract T frames from video using decord
    Args:
        use_gpu: If True, use GPU for video decoding (faster but uses VRAM)
    """
    ctx = cpu(0) if not use_gpu else gpu(0)
    vr = VideoReader(str(video_path), ctx=ctx)
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    dur = total_frames / fps if fps > 0 else 0.0
    
    if dur <= 0:
        raise ValueError(f"Could not read video duration: {video_path}")
    
    if center_time is None:
        center_time = dur / 2.0

    half = window_sec / 2
    start, end = max(0.0, center_time - half), min(dur, center_time + half)

    start_frame = int(start * fps)
    end_frame = min(int(end * fps), total_frames - 1)

    if T > 1 and end_frame > start_frame:
        frame_indices = np.linspace(start_frame, end_frame, T, dtype=int)
    else:
        center_frame = int((start_frame + end_frame) / 2)
        frame_indices = np.full(T, center_frame, dtype=int)

    frame_indices = np.clip(frame_indices, 0, total_frames - 1)

    # Get frames as numpy array (T, H, W, C) in RGB
    frames = vr.get_batch(frame_indices.tolist()).asnumpy()
    
    # Convert to tensor (T, C, H, W)
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    
    return frames_tensor

def extract_pixel_embeddings(video_paths, T=24, window_sec=1.0):
    outs = []
    for i, p in enumerate(video_paths):
        try:
            clip = decode_clip(p, T=T, window_sec=window_sec)
            preprocess_clip = preprocess(clip)
            outs.append(preprocess_clip.flatten(1))
        except Exception as e:
            print(f"Error decoding clip: {e}")
    return torch.stack(outs, dim=0)

def extract_dinov2_embeddings(video_paths, device=None, T=24, window_sec=2.0):
    device = device or torch.device("cpu")
    model = dinov2.to(device)
    
    outs = []
    valid_indices = []
    
    # Decode all videos first (CPU operation)
    for i, p in enumerate(video_paths):
        try:
            clip = decode_clip(p, T=T, window_sec=window_sec)
            preprocess_clip = preprocess(clip)
            outs.append(preprocess_clip)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error decoding {p}: {e}")
            # Add dummy for failed videos
            outs.append(torch.zeros(T, 3, 224, 224))
            valid_indices.append(i)

    if len(outs) == 0:
        raise ValueError("No videos successfully decoded")

    # Process all frames in one GPU batch
    batch = torch.cat(outs, dim=0).to(device)
    
    with torch.no_grad():
        feats = model.forward_features(batch)
        cls = feats["x_norm_clstoken"].unsqueeze(1)
        patches = feats["x_norm_patchtokens"]
        tokens = torch.cat([cls, patches], dim=1)
        Z_flat = tokens.flatten(1)
        Z = Z_flat.view(len(outs), T, -1)
        return Z

def compute_temporal_geometry(Z):
    delta = Z[:, 1:, :] - Z[:, :-1, :]                                 
    d = delta.norm(dim=-1)                                    
    cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:,  :], dim=-1)  
    theta = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))      
    return d, theta

def moment4(x):
    mu  = x.mean(dim=-1)                   
    mn  = x.amin(dim=-1)                   
    mx  = x.amax(dim=-1)                   
    var = x.var(dim=-1, unbiased=False)    
    return mu, mn, mx, var

def features_from_Z(Z):
    d, t = compute_temporal_geometry(Z)
    # Adjust based on T (number of frames)
    # For T frames: d has T-1 values, t has T-2 values
    num_d = min(7, d.shape[1])  # Take up to 7 distances
    num_t = min(6, t.shape[1])  # Take up to 6 angles
    
    d_subset = d[:, :num_d]
    t_subset = t[:, :num_t]
    
    # Pad if needed to keep feature size consistent at 21
    if num_d < 7:
        d_subset = F.pad(d_subset, (0, 7 - num_d), value=0)
    if num_t < 6:
        t_subset = F.pad(t_subset, (0, 6 - num_t), value=0)
    
    mu_d, mn_d, mx_d, var_d = moment4(d)
    mu_t, mn_t, mx_t, var_t = moment4(t)
    stats = torch.stack([mu_d, mn_d, mx_d, var_d, mu_t, mn_t, mx_t, var_t], dim=1) 
    return torch.cat([d_subset, t_subset, stats], dim=1)