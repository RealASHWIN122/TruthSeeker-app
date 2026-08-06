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

The AI boom of the 2020s has brought powerful generative tools, making it easier than ever to create synthetic media capable of deceiving the average person. **TruthSeeker** is a comprehensive, multi-modal application designed to combat this by detecting deepfakes and verifying the authenticity of digital rumors. 

TruthSeeker prioritizes three core tenets:
1. **Privacy**: Protect user data by utilizing localized edge AI where possible.
2. **Cost-Efficiency**: Optimize models to run on affordable or existing hardware.
3. **Explainability**: Remove the "black box" of AI detection by providing users with clear, evidence-based reasoning for every classification.

---

## 🧠 System Architecture & Methodology

TruthSeeker employs a hybrid **Edge + Cloud** pipeline, combining the speed of mobile processing with the heavy reasoning capabilities of advanced cloud nodes.

### 📱 Stage 1: Edge-Based Processing
Designed for resource-constrained environments, this stage runs entirely on the user's mobile device for fast, private, binary classification (REAL or FAKE).
* **Video Detection:** Uses a compressed `TimeSformer` (converted to TFLite via INT8 quantization), capable of catching temporal inconsistencies like eye-flickering and unnatural movements.
* **Audio Detection:** Employs a lightweight footprint designed to flag anomalies before data is sent to the cloud.

### ☁️ Stage 2: Cloud-Based Processing
When media is complex or users demand an explanation, the data is pushed to our dedicated inference servers for Explainable AI (XAI) analysis.
* **Skyra (The Texture Inspector):** A Vision-Language Multimodal Model (Qwen2.5-VL-7B) trained on the ViF-CoT-4K dataset. It provides deep **Chain-of-Thought (CoT)** reasoning, identifying physical law violations, object inconsistencies, and spatial artifacts.
* **Wav2Vec 2.0 (The Digital Forensics):** A powerful transformer-based audio classifier that ingests raw 1D waveforms. It preserves raw signals to detect subtle phase anomalies and structural glitches introduced by AI voice cloning platforms (e.g., Suno, Udio).

### 🕵️ Fact-Checking via RAG
Information overload and paywalls make manual fact-checking difficult. TruthSeeker features an autonomous backend utilizing **Phi-3** (a Small Language Model).
* **Retrieval-Augmented Generation (RAG):** The model performs live web searches (via DuckDuckGo) to fetch external context before answering.
* **Grounded Facts:** By forcing the LLM to analyze retrieved evidence rather than its internal parametric memory, we dramatically reduce hallucination risks and provide source-attributed results.

---

## 📂 Project Structure

| Directory | Description |
| :--- | :--- |
| `kotlearn/` | The Android Studio project containing the Kotlin application. |
| `skyra_data/` | The AMD MI300X cloud backend (Contains dataset scripts, training logic, and the `skyra_api.py` deployment). |
| `colab_backend/` | Google Colab deployment scripts (`audio_api.py`) for Wav2Vec Audio XAI and TimeSformer Attention Maps. |
| `factcheck/` | Python FastAPI backend utilizing `Phi-3-mini` with RAG for local fact-checking. |
| `docs/` | Contains the complete `TruthSeeker_Report.md`, system methodology PDFs, and formatting tools. |
| `temporal2/` & `modelly/` | Scripts for training the TimeSformer engine and compiling it to Edge TFLite format. |

---

## 🚀 Getting Started

The TruthSeeker ecosystem is a network of distributed microservices. Here is how to boot the entire system:

### 1. The Mobile App (Android Client)
1. Open the `kotlearn` folder in **Android Studio**.
2. **Local Model Dependency:** Ensure that the pre-trained `lip_flex.tflite` model is inside `kotlearn/app/src/main/assets/`. *(Note: Due to size constraints, you must compile this yourself using the `temporal2` and `modelly` scripts if it is missing).*
3. **Build & Run:** Click the "Run" button to emulate it on an API 26+ device, or click `Build > Build Bundle(s) / APK(s) > Build APK(s)` to generate an APK for your physical Android phone.

### 2. The Cloud Deep Forensics Server (AMD MI300X)
This server handles the heavy Chain-of-Thought deepfake analysis via Skyra.
**A. Training Skyra from Scratch**
SSH into your AMD server and run the pipeline:
```bash
cd skyra_data
python3 get_videos.py        # Downloads the 20GB dataset from Hugging Face
python3 consolidate.py       # Consolidates the annotations into a master JSON
python3 train_skyra.py       # Trains the model using LoRA and ROCm survival flags
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
1. Upload the files in `colab_backend/` to Google Colab.
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
