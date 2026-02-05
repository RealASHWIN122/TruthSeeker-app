import streamlit as st
import os
import tempfile
import cv2
from datetime import datetime

# import your functions
from demo import get_result, create_demo_video, get_result_description

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="LIPINC-V2 Deepfake Detector",
    layout="wide"
)

st.title("🧠 LIPINC-V2 Lip-Sync Deepfake Detector")

st.markdown(
    "Detect AI-generated lip-synced deepfake videos using spatio-temporal transformers."
)

# ------------------ ENV SETUP (FIXED) ------------------
# Get the absolute path of the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script location
# This works on both Linux and Windows automatically
DLIB_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "Best_Weights.hdf5")

# Verify files exist to avoid silent failures later
if not os.path.exists(DLIB_PATH):
    st.error(f"⚠️ Error: Could not find dlib predictor at: {DLIB_PATH}")
if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"⚠️ Error: Could not find weights at: {CHECKPOINT_PATH}")

os.environ["DLIB_LANDMARK_PATH"] = DLIB_PATH

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Settings")
device = st.sidebar.selectbox("Device", ["cuda", "cpu"])
show_demo = st.sidebar.checkbox("Generate Demo Video", value=True)

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:
    st.video(uploaded_file)

    if st.button("🚀 Run Detection"):
        with st.spinner("Analyzing video... This may take a minute."):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                input_video_path = tmp.name

            advanced_folder = os.path.join(OUTPUT_DIR, "Advanced_results")
            os.makedirs(advanced_folder, exist_ok=True)

            # -------- RUN INFERENCE --------
            # NOTE: If this crashes with 'AttributeError: module mediapipe...',
            # the issue is inside demo.py, not here.
            try:
                result, combined_frames, residue_frames, l_id, g_id, adv_path, runtime = get_result(
                    input_video_path,
                    advanced_folder,
                    CHECKPOINT_PATH
                )

                st.success("Analysis complete")

                # -------- RESULTS --------
                if 0 <= result <= 1:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Real Probability", f"{result:.3f}")

                    with col2:
                        st.metric("Fake Probability", f"{1 - result:.3f}")

                    st.info(get_result_description(result))
                    st.caption(f"⏱ Runtime: {runtime} seconds")

                else:
                    st.error("Face or lips not detected reliably.")

                # -------- DEMO VIDEO --------
                if show_demo and 0 <= result <= 1:
                    color = (0, 255, 0) if result > 0.5 else (0, 0, 255)

                    video_des = {
                        "Task": "Lip-synced Deepfake Detection",
                        "Input File": uploaded_file.name,
                        "Analytic Name": "LIPINC-V2",
                        "Analysis Date": str(datetime.now()),
                        "Result": {
                            "Real Probability": result,
                            "Fake Probability": round(1 - result, 3),
                        },
                        "Result Description": get_result_description(result),
                    }

                    frames = create_demo_video(
                        input_video_path,
                        color,
                        adv_path,
                        g_id,
                        video_des
                    )

                    output_path = os.path.join(
                        OUTPUT_DIR, f"{uploaded_file.name}_demo.mp4"
                    )

                    h, w = frames[0].shape[:2]
                    out = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        21,
                        (w, h)
                    )

                    for f in frames:
                        out.write(f)
                    out.release()

                    st.video(output_path)

                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Demo Video",
                            f,
                            file_name="lipinc_demo.mp4",
                            mime="video/mp4"
                        )
            
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.write("Check the terminal logs for more details.")