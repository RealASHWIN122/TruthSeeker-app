import streamlit as st
import os
import cv2
import torch
import tempfile
from PIL import Image
from torchvision import transforms

# Import your custom 3D architecture
from tmodel import TemporalPhysicsEngine

# ==========================================
# 1. MODEL CACHING (Loads into VRAM exactly once)
# ==========================================
@st.cache_resource
def load_model(checkpoint_path, num_frames=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the architecture without pre-trained weights (since we load our own)
    model = TemporalPhysicsEngine(pretrained=False).to(device)
    
    # Load the TruthSeeker weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, device

# ==========================================
# 2. FRAME EXTRACTION
# ==========================================
def extract_frames_in_memory(video_path, num_frames=16):
    """Extracts exactly 16 frames from the video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None

    step = max(total_frames // num_frames, 1)
    frames = []
    
    for i in range(num_frames):
        frame_idx = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # OpenCV uses BGR, Convert to RGB for PyTorch
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)
            
    cap.release()
    
    # Pad with last frame if the video is too short
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
        
    return frames

# ==========================================
# 3. STREAMLIT UI & LOGIC
# ==========================================
def main():
    st.set_page_config(page_title="TruthSeeker AI", page_icon="🕵️‍♂️", layout="wide")
    
    # Sidebar Configuration
    st.sidebar.title("⚙️ System Config")
    st.sidebar.markdown("Configure the detection parameters.")
    
    # Default path pointing to your temporal checkpoint
    default_ckpt = r"/home/me/Videos/final year project/TruthSeeker-app/factcheck/models/temporal_checkpoint3.pth"
    checkpoint_path = st.sidebar.text_input("Checkpoint Path", value=default_ckpt)
    
    # Attempt to load model
    try:
        model, device = load_model(checkpoint_path)
        st.sidebar.success(f"✅ Model loaded on {device.type.upper()}")
    except Exception as e:
        st.sidebar.error("❌ Model failed to load. Check the file path.")
        st.sidebar.exception(e)
        st.stop()

    # Main UI
    st.title("🕵️‍♂️ TruthSeeker: Temporal Deepfake Detector")
    st.markdown("Upload a video to analyze its physical consistency and temporal coherence.")

    uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Display the video player
        st.video(uploaded_video)
        
        if st.button("🔍 Analyze Temporal Physics", use_container_width=True):
            
            # Create a temporary file to allow OpenCV to read the video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            tfile.close() # Must close so cv2 can open it on Windows

            with st.spinner("Extracting spatial-temporal frames..."):
                pil_frames = extract_frames_in_memory(video_path, num_frames=16)
                
            if not pil_frames:
                st.error("Failed to read video frames.")
                os.unlink(video_path) # Clean up temp file
                st.stop()

            # --- OPTIONAL: Display the extracted frames so you can see what the model sees ---
            with st.expander("👁️ View Extracted Frame Sequence"):
                cols = st.columns(8)
                for i in range(8):
                    cols[i].image(pil_frames[i], use_container_width=True)
                cols2 = st.columns(8)
                for i in range(8, 16):
                    cols2[i - 8].image(pil_frames[i], use_container_width=True)

            with st.spinner("Running Divided Space-Time Attention Analysis..."):
                # Standard Spatial Transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Stack and format for TimeSformer
                tensor_frames = [transform(frame) for frame in pil_frames]
                video_tensor = torch.stack(tensor_frames) # Shape: (16, 3, 224, 224)
                video_tensor = video_tensor.unsqueeze(0).to(device) # Add Batch Dim: (1, 16, 3, 224, 224)

                # Inference
                with torch.no_grad():
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        output = model(video_tensor)
                        probability = torch.sigmoid(output).item()

            # Clean up the temporary file
            os.unlink(video_path)

            # ==========================================
            # 4. RESULTS DASHBOARD
            # ==========================================
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            fake_percent = probability * 100
            real_percent = (1.0 - probability) * 100

            col1, col2 = st.columns(2)
            
            if probability > 0.5:
                st.error("🚨 CLASSIFICATION: AI GENERATED (FAKE)")
                col1.metric(label="AI Probability", value=f"{fake_percent:.2f}%", delta="High Risk", delta_color="inverse")
                col2.metric(label="Authentic Probability", value=f"{real_percent:.2f}%")
                st.progress(probability)
                st.warning("Temporal inconsistencies and physics violations detected in the frame sequence.")
            else:
                st.success("✅ CLASSIFICATION: AUTHENTIC (REAL)")
                col1.metric(label="Authentic Probability", value=f"{real_percent:.2f}%", delta="Safe")
                col2.metric(label="AI Probability", value=f"{fake_percent:.2f}%")
                st.progress(probability) # Progress bar shows the "Fake" percentage
                st.info("Video sequence maintains natural temporal coherence and physics.")

if __name__ == "__main__":
    main()