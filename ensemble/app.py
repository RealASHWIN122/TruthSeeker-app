import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your model architecture
from spatial_model import SpatialModel

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "B:\download\TruthSeeker-app\checkpoints\spatial_model_epoch_15.pth"  # <--- UPDATE THIS PATH
FRAME_SIZE = (224, 224)

# --- UI SETTINGS ---
st.set_page_config(
    page_title="TruthSeeker AI",
    page_icon="👁️",
    layout="centered"
)

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    """
    Loads the trained SpatialModel.
    Cached so it doesn't reload on every interaction.
    """
    try:
        model = SpatialModel(num_classes=2, embedding_dim=512)
        
        # Load weights
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            # Handle both full checkpoint dicts and direct state_dicts
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            st.error(f"Model not found at {MODEL_PATH}. Please check the path.")
            return None

        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. PREPROCESSING ---
def get_transform():
    """
    Exact same preprocessing as training (normalization).
    """
    return A.Compose([
        A.Resize(height=FRAME_SIZE[0], width=FRAME_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def preprocess_image(image_rgb):
    """Prepares a single frame for the model."""
    transform = get_transform()
    augmented = transform(image=image_rgb)
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    return augmented['image'].unsqueeze(0)

# --- 3. INFERENCE LOGIC ---
def predict_frame(model, frame_tensor):
    """Runs inference on a single frame tensor."""
    with torch.no_grad():
        frame_tensor = frame_tensor.to(DEVICE)
        output = model(frame_tensor)
        
        # Extract results
        probs = output['probabilities'][0] # [Real, Fake]
        fake_prob = probs[1].item()
        reliability = output['reliability'][0].item()
        
        return fake_prob, reliability

def process_video(video_path, model, num_frames=16):
    """
    Samples frames from video and averages predictions.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return 0.0, 0.0, []

    # Sample frames uniformly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frame_scores = []
    reliability_scores = []
    preview_frames = []
    
    progress_bar = st.progress(0)
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # Convert BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Keep a few frames for visualization
        if i % 4 == 0:
            preview_frames.append(frame_rgb)
            
        # Preprocess
        tensor = preprocess_image(frame_rgb)
        
        # Predict
        fake_prob, reliability = predict_frame(model, tensor)
        frame_scores.append(fake_prob)
        reliability_scores.append(reliability)
        
        progress_bar.progress((i + 1) / len(indices))
        
    cap.release()
    progress_bar.empty()
    
    # Aggregate results
    if not frame_scores: return 0.0, 0.0, []
    
    avg_fake_prob = np.mean(frame_scores)
    avg_reliability = np.mean(reliability_scores)
    
    return avg_fake_prob, avg_reliability, preview_frames

# --- 4. MAIN APP INTERFACE ---
def main():
    st.title("👁️ TruthSeeker: Deepfake Detector")
    st.write("Upload an image or video to analyze it for deepfake artifacts.")
    
    # Load Model
    with st.spinner("Loading AI Model..."):
        model = load_model()
        
    if model is None:
        st.stop()
        
    # Sidebar
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider("Fake Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Input
    uploaded_file = st.file_uploader("Choose a file...", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        # --- IMAGE HANDLING ---
        if file_type == 'image':
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    # Convert PIL to Numpy
                    image_np = np.array(image)
                    tensor = preprocess_image(image_np)
                    
                    fake_prob, reliability = predict_frame(model, tensor)
                    
                    display_results(fake_prob, reliability, conf_threshold)

        # --- VIDEO HANDLING ---
        elif file_type == 'video':
            # Save to temp file because OpenCV needs a path
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            st.video(tfile.name)
            
            if st.button("Analyze Video"):
                with st.spinner("Scanning video frames..."):
                    fake_prob, reliability, previews = process_video(tfile.name, model)
                    
                    display_results(fake_prob, reliability, conf_threshold)
                    
                    # Show analyzed frames
                    if previews:
                        st.subheader("Analyzed Frames")
                        st.image(previews, width=150, caption=[f"Frame {i+1}" for i in range(len(previews))])
            
            tfile.close()

def display_results(fake_prob, reliability, threshold):
    """Beautifully displays the prediction results."""
    st.divider()
    
    col1, col2 = st.columns(2)
    
    is_fake = fake_prob >= threshold
    
    with col1:
        if is_fake:
            st.error("🚨 RESULT: FAKE")
        else:
            st.success("✅ RESULT: REAL")
            
    with col2:
        st.metric("Deepfake Probability", f"{fake_prob:.1%}")
        st.metric("Model Reliability", f"{reliability:.1%}", 
                 help="How confident the model is in its own analysis (texture clarity, etc.)")

    # Visual Bar
    st.write("### Probability Meter")
    bar_color = "red" if is_fake else "green"
    st.progress(fake_prob)
    if is_fake:
        st.caption("High probability of manipulation detected.")
    else:
        st.caption("Content appears authentic.")

if __name__ == "__main__":
    main()