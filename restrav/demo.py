import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import yt_dlp
import tempfile
import os
from pathlib import Path
import dinov2_features as d2

# Page config
st.set_page_config(
    page_title="ReStrav - Deepfake Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-real {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .prediction-fake {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Model definition
class MLP(nn.Module):
    def __init__(self, in_dim=21, h1=64, h2=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1)
        )
    
    def forward(self, x):
        return self.net(x)


@st.cache_resource
def load_model():
    """Load model and parameters (cached)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP()
    model.load_state_dict(torch.load("B:\\download\\TruthSeeker-app\\best_model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    mean = np.load("B:\\download\\TruthSeeker-app\\mean.npy")
    std = np.load("B:\\download\\TruthSeeker-app\\std.npy")
    best_tau = float(np.load("B:\\download\\TruthSeeker-app\\best_tau.npy"))

    return model, mean, std, best_tau, device


def download_video(url, progress_bar):
    """Download video using yt-dlp"""
    try:
        temp_dir = tempfile.mkdtemp()
        outfile = os.path.join(temp_dir, "video.mp4")
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                try:
                    percent = d.get('_percent_str', '0%').replace('%', '')
                    progress_bar.progress(float(percent) / 100)
                except:
                    pass
        
        ydl_opts = {
            "outtmpl": outfile,
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [progress_hook],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Find the downloaded file
        if os.path.exists(outfile):
            return outfile
        
        for ext in ['.mp4', '.webm', '.mkv']:
            potential_file = outfile.replace('.mp4', ext)
            if os.path.exists(potential_file):
                return potential_file
        
        raise FileNotFoundError("Downloaded file not found")
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return None


def classify_video(video_path, model, mean, std, tau, device):
    """Classify video as real or fake"""
    try:
        with st.spinner("🔍 Analyzing video..."):
            # Extract features
            progress_text = st.empty()
            progress_text.text("Extracting DINOv2 embeddings...")
            
            Z = d2.extract_dinov2_embeddings([video_path], device=device, T=16, window_sec=1.5)
            features = d2.features_from_Z(Z)
            
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            
            progress_text.text("Computing prediction...")
            
            # Normalize
            features_norm = (features - mean) / std
            features_norm = features_norm.astype(np.float32)
            
            # Predict
            x_tensor = torch.from_numpy(features_norm).to(device)
            
            with torch.no_grad():
                logits = model(x_tensor)
                prob_real = torch.sigmoid(logits).cpu().numpy().item()
            
            prob_fake = 1 - prob_real
            prediction = "REAL" if prob_real >= tau else "FAKE"
            confidence = max(prob_real, prob_fake)
            
            progress_text.empty()
            
            return {
                'prediction': prediction,
                'prob_real': prob_real,
                'prob_fake': prob_fake,
                'confidence': confidence,
                'success': True
            }
    
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return {'success': False, 'error': str(e)}


def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 ReStrav Deepfake Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered video authenticity analysis</p>', unsafe_allow_html=True)
    
    # Load model
    model, mean, std, tau, device = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info(f"**Device:** {device}")
        st.info(f"**Threshold:** {tau:.3f}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a trained deep learning model to detect AI-generated (deepfake) videos.
        
        **How it works:**
        1. Extracts DINOv2 visual embeddings
        2. Computes temporal geometry features
        3. Classifies using trained MLP
        
        **Model Performance:**
        - F1 Score: 89.3%
        - AUC-ROC: 95.6%
        - Accuracy: 88.9%
        """)
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.write(f"**Parameters:** 3,521")
        st.write(f"**Architecture:** MLP (64→32→1)")
        st.write(f"**Training samples:** 100,000")
    
    # Main content
    tab1, tab2 = st.tabs(["📤 Upload Video", "🔗 URL Download"])
    
    with tab1:
        st.markdown("### Upload a video file")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload a video file to analyze"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            # Display video
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.video(temp_path)
            
            with col2:
                if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                    result = classify_video(temp_path, model, mean, std, tau, device)
                    
                    if result['success']:
                        # Display prediction
                        if result['prediction'] == "REAL":
                            st.markdown(f"""
                            <div class="prediction-real">
                                <h2>✅ REAL Video</h2>
                                <p>This video appears to be authentic.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-fake">
                                <h2>⚠️ FAKE Video (Deepfake)</h2>
                                <p>This video appears to be AI-generated.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Metrics
                        st.markdown("### 📊 Detailed Results")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                        with metric_col2:
                            st.metric("Real Probability", f"{result['prob_real']:.1%}")
                        
                        with metric_col3:
                            st.metric("Fake Probability", f"{result['prob_fake']:.1%}")
                        
                        # Progress bar visualization
                        st.markdown("### 📈 Probability Distribution")
                        st.progress(result['prob_real'], text=f"Real: {result['prob_real']:.1%}")
                        st.progress(result['prob_fake'], text=f"Fake: {result['prob_fake']:.1%}")
    
    with tab2:
        st.markdown("### Download video from URL")
        st.markdown("Supports YouTube, Twitter, TikTok, and many other platforms")
        
        url = st.text_input(
            "Enter video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a video URL from any supported platform"
        )
        
        if st.button("⬇️ Download Video", type="secondary"):
            if url:
                progress_bar = st.progress(0)
                with st.status("Downloading video...", expanded=True) as status:
                    st.write("Fetching video from URL...")
                    video_path = download_video(url, progress_bar)
                    
                    if video_path:
                        status.update(label="✅ Download complete!", state="complete")
                        st.session_state['downloaded_video'] = video_path
                    else:
                        status.update(label="❌ Download failed", state="error")
            else:
                st.warning("Please enter a URL")
        
        # Analyze downloaded video
        if 'downloaded_video' in st.session_state:
            video_path = st.session_state['downloaded_video']
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if os.path.exists(video_path):
                    st.video(video_path)
                else:
                    st.error("Video file not found")
            
            with col2:
                if st.button("🔍 Analyze Downloaded Video", type="primary", use_container_width=True):
                    result = classify_video(video_path, model, mean, std, tau, device)
                    
                    if result['success']:
                        # Display prediction
                        if result['prediction'] == "REAL":
                            st.markdown(f"""
                            <div class="prediction-real">
                                <h2>✅ REAL Video</h2>
                                <p>This video appears to be authentic.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-fake">
                                <h2>⚠️ FAKE Video (Deepfake)</h2>
                                <p>This video appears to be AI-generated.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Metrics
                        st.markdown("### 📊 Detailed Results")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                        with metric_col2:
                            st.metric("Real Probability", f"{result['prob_real']:.1%}")
                        
                        with metric_col3:
                            st.metric("Fake Probability", f"{result['prob_fake']:.1%}")
                        
                        # Progress bar visualization
                        st.markdown("### 📈 Probability Distribution")
                        st.progress(result['prob_real'], text=f"Real: {result['prob_real']:.1%}")
                        st.progress(result['prob_fake'], text=f"Fake: {result['prob_fake']:.1%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Powered by DINOv2 & PyTorch</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()