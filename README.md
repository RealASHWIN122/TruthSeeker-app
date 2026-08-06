<div align="center">
  <img src="./kotlearn/app/src/main/res/drawable/cyverlogo.png" alt="TruthSeeker Logo" width="200" height="200" onerror="this.src='https://img.icons8.com/color/200/000000/artificial-intelligence.png';">
  <h1>👁️ TruthSeeker</h1>
  <p><b>Advanced Deepfake Detection & AI-Powered Fact-Checking Ecosystem</b></p>
  <p>
    <img src="https://img.shields.io/badge/Platform-Android-3DDC84?style=flat-square&logo=android" alt="Android">
    <img src="https://img.shields.io/badge/Language-Kotlin-7F52FF?style=flat-square&logo=kotlin" alt="Kotlin">
    <img src="https://img.shields.io/badge/Backend-Python-3776AB?style=flat-square&logo=python" alt="Python">
    <img src="https://img.shields.io/badge/AI-PyTorch%20%7C%20TensorFlow-EE4C2C?style=flat-square&logo=pytorch" alt="AI">
  </p>
</div>

---

## 📖 Overview

**TruthSeeker** is a comprehensive, multi-modal application designed to detect synthetic media (deepfakes) and verify the authenticity of digital rumors. It leverages a hybrid **Edge + Cloud AI architecture** inspired by the state-of-the-art **Skyra** methodology. 

By utilizing local optimized models for rapid inference and powerful cloud Vision-Language Models (VLMs) like Qwen2.5-VL for explainable "Chain of Thought" reasoning, TruthSeeker provides users with fast, reliable, and interpretable forensic analysis of modern media.

---

## 📂 Project Structure

| Directory / File | Description |
| :--- | :--- |
| 📱 `kotlearn/` | The Android Studio project containing the Kotlin application. |
| 🚀 `skyra_data/` | The AMD MI300X cloud backend (Training, Consolidation, and `skyra_api.py` Deployment). |
| 🕵️ `factcheck/` | Python FastAPI backend utilizing `Phi-3-mini` for local fact-checking. |
| 🧠 `temporal2/` & `modelly/` | Scripts for training the TimeSformer engine and converting it to Edge TFLite format. |
| ☁️ `audio_api.py` | Google Colab deployment script for advanced Audio XAI and TimeSformer Attention Maps. |

---

## 🚀 How to Run the Ecosystem

The TruthSeeker ecosystem is powered by a network of distributed microservices. Here is how to boot the entire system:

### 1. The Mobile App (Android Client)
1. Open the `kotlearn` folder in **Android Studio**.
2. **Local Model Dependency:** Ensure that the pre-trained `lip_flex.tflite` model is inside `kotlearn/app/src/main/assets/`. *(Note: Due to size constraints, you must compile this yourself using the `temporal2` and `modelly` scripts if it is missing).*
3. **Build & Run:** Click the "Run" button to emulate it on an API 26+ device, or click `Build > Build Bundle(s) / APK(s) > Build APK(s)` to generate an APK for your physical phone.

### 2. The Cloud Deep Forensics Server (AMD MI300X)
This server handles heavy Chain-of-Thought deepfake analysis using Qwen2.5-VL.

**A. Training Skyra from Scratch**
SSH into your AMD server, navigate to the folder, and run the pipeline:
```bash
cd skyra_data
# 1. Download the 20GB dataset from Hugging Face
python3 get_videos.py

# 2. Consolidate the annotations into a master JSON
python3 consolidate.py

# 3. Train the model using LoRA and ROCm survival flags
python3 train_skyra.py
```

**B. Deploying the Skyra API**
Once the model is trained, boot the inference server:
```bash
cd skyra_data
python3 skyra_api.py
```
*The API will start securely on `0.0.0.0:8000` with Uvicorn signal bypasses enabled.*

### 3. The Colab XAI Server (Audio & TimeSformer)
This endpoint handles audio deepfake detection and generates visual forensic grids using Integrated Gradients and Attention Maps.
1. Upload `audio_api.py` (or `audio_api(1).ipynb`) to Google Colab.
2. Ensure you have the `Wav2Vec2` and `temporal_checkpoint3.pth` model weights in your connected Google Drive.
3. Run the notebook/script. It will start a FastAPI server and expose it to the internet using **localtunnel**.
4. **Important:** Copy the generated `https://xxxx.loca.lt` URL and update the `COLAB_AUDIO_URL` constant inside `kotlearn/app/src/main/java/com/example/kotlearn/FactCheckApi.kt`, then rebuild the Android app.

### 4. The Local Fact-Checker (Laptop/Local Network)
This acts as a fast, offline LLM agent to verify text claims.
1. Download the [Phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf) model and place it in the `factcheck/models/` directory.
2. Start the local server:
```bash
cd factcheck
python -m venv venv2
source venv2/bin/activate  # (or venv2\Scripts\activate on Windows)
pip install fastapi uvicorn llama-cpp-python duckduckgo-search

python app1.py
```
*The API will start on `0.0.0.0:8000`. Ensure your Android phone is on the same Wi-Fi network as the computer running this server.*

---
<div align="center">
  <i>Built to ensure truth and transparency in the era of Generative AI.</i>
</div>
