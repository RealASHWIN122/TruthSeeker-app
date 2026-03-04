"""
app.py — Stream C: Deepfake Detection Inference App
===================================================
A Streamlit dashboard to test the Frequency Domain Model on video files.

Features:
  • Video upload & playback
  • Frame-by-frame frequency analysis
  • Real-time inference using your trained checkpoint
  • Frame-level probability plotting

Usage:
    streamlit run app.py
"""

import av
import cv2
import numpy as np
import streamlit as st
import tempfile
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

# Import your model architecture
# Ensure model.py is in the same directory as app.py
from model import FrequencyModel

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "experiments/run_01/checkpoints/checkpoint_stage1_best.pth"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE      = 224
FRAMES_TO_TEST  = 30    # Number of frames to sample (evenly spaced) to speed up inference

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path):
    """
    Loads the FrequencyModel and weights from the specified checkpoint.
    Cached to prevent reloading on every interaction.
    """
    try:
        # Instantiate model (must match training config: no pretrained weights needed for inference)
        model = FrequencyModel(pretrained=False)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Handle cases where the checkpoint saves 'model_state' or just the state_dict
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"❌ Checkpoint not found at: {checkpoint_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def get_inference_transforms():
    """
    Matches the Validation transforms from dataset.py (NO Normalization).
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Force resize to square
        T.ToTensor(),
        # CRITICAL: No T.Normalize, matching your latest training fix
    ])

def extract_frames(video_path, max_frames=30):
    """
    Extracts evenly spaced frames from the video file.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    
    # Calculate stride to get exactly max_frames
    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = np.arange(total_frames)

    frames = []
    original_images = [] # Keep PIL images for display
    
    # Seek and retrieve specific frames
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            img = frame.to_image()  # Convert to PIL
            original_images.append(img)
            frames.append(img)
            
        if i > indices[-1]:
            break
            
    return frames, original_images

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inference Logic
# ─────────────────────────────────────────────────────────────────────────────
def predict_video(model, frames):
    """
    Runs the model on a list of PIL frames.
    Returns: average probability, list of per-frame probabilities
    """
    transform = get_inference_transforms()
    
    # Stack frames into a single tensor batch [B, C, H, W]
    batch = torch.stack([transform(f) for f in frames]).to(DEVICE)
    
    with torch.no_grad():
        # Get logits
        logits = model(batch)
        # Apply sigmoid to get probabilities [0.0 = Real, 1.0 = Fake]
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
    avg_prob = np.mean(probs)
    return avg_prob, probs

# ─────────────────────────────────────────────────────────────────────────────
# 4. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TruthSeeker | Frequency Analysis", layout="wide")

st.title("🕵️ TruthSeeker: Frequency Domain Analysis")
st.markdown(
    """
    **Stream C Artifact Detector** Analyzing the invisible frequency spectrum fingerprints of video generators.
    """
)

# Sidebar for status
with st.sidebar:
    st.header("System Status")
    st.info(f"Using Device: **{DEVICE.upper()}**")
    
    if st.button("Reload Model"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.write("Current Checkpoint:")
    st.code(CHECKPOINT_PATH, language="bash")

# Main Interface
uploaded_file = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # 1. Save temp file (OpenCV/PyAV need a path)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns([1, 1])

    with col1:
        st.video(video_path)
        analyze_btn = st.button("🔍 Run Frequency Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        # Load Model
        with st.spinner("Loading Frequency Model..."):
            model = load_model(CHECKPOINT_PATH)

        # Process Frames
        with st.spinner(f"Extracting {FRAMES_TO_TEST} frames & analyzing spectrum..."):
            pil_frames, original_imgs = extract_frames(video_path, max_frames=FRAMES_TO_TEST)
            
            if not pil_frames:
                st.error("Could not extract frames from video.")
                st.stop()

            # Run Inference
            avg_score, frame_scores = predict_video(model, pil_frames)

        # Display Results
        with col2:
            st.markdown("### Diagnosis")
            
            # Logic: > 0.5 is Fake, < 0.5 is Real
            if avg_score > 0.5:
                verdict = "FAKE / GENERATED"
                color = "red"
                confidence = avg_score * 100
                st.error(f"## 🚨 {verdict}")
            else:
                verdict = "REAL / AUTHENTIC"
                color = "green"
                confidence = (1 - avg_score) * 100
                st.success(f"## ✅ {verdict}")

            st.metric("Confidence Score", f"{confidence:.2f}%")
            st.progress(float(avg_score))
            st.caption(f"Raw Frequency Anomaly Score: {avg_score:.4f} (0=Real, 1=Fake)")

        # Detailed Analysis
        st.divider()
        st.subheader("Frame-by-Frame Frequency Analysis")
        
        # Plot chart
        chart_data = {"Frame": range(len(frame_scores)), "Anomaly Score": frame_scores}
        st.line_chart(chart_data, x="Frame", y="Anomaly Score")

        # Show top outlier frames
        st.subheader("Suspicious Frames (Highest Artifact Scores)")
        
        # Get indices of top 3 highest scores
        top_indices = np.argsort(frame_scores)[-4:][::-1]
        
        cols = st.columns(4)
        for idx, col in zip(top_indices, cols):
            with col:
                st.image(original_imgs[idx], use_container_width=True)
                st.caption(f"Frame {idx}: **{frame_scores[idx]:.2f}**")