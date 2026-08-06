<div align="center">
  <img src="./kotlearn/app/src/main/res/drawable/cyverlogo.png" alt="TruthSeeker Logo" width="200" height="200" onerror="this.src='https://img.icons8.com/color/200/000000/artificial-intelligence.png';">
  <h1>👁️ TruthSeeker</h1>
  <p><b>Advanced Deepfake Detection & AI-Powered Fact-Checking Ecosystem</b></p>
  <p>
    <img src="https://img.shields.io/badge/Platform-Android-3DDC84?style=flat-square&logo=android" alt="Android">
    <img src="https://img.shields.io/badge/Language-Kotlin-7F52FF?style=flat-square&logo=kotlin" alt="Kotlin">
    <img src="https://img.shields.io/badge/Backend-Python-3776AB?style=flat-square&logo=python" alt="Python">
    <img src="https://img.shields.io/badge/AI-PyTorch%20%7C%20TensorFlow-EE4C2C?style=flat-square&logo=pytorch" alt="AI">
    <img src="https://img.shields.io/badge/Hardware-AMD%20MI300X-black?style=flat-square&logo=amd" alt="AMD">
  </p>
</div>

---

## 📖 Executive Overview

The AI boom of the 2020s has brought powerful generative tools, making it easier than ever to create synthetic media capable of deceiving the average person. From state-of-the-art video generators (Sora, Runway) to voice cloning (Suno, Udio), hyper-realistic deepfakes pose a profound threat to digital trust, entertainment, and journalism.

**TruthSeeker** is a comprehensive, multi-modal application designed to combat this by detecting deepfakes and verifying the authenticity of digital rumors. Built on a hybrid architecture, it achieves rapid analysis while maintaining privacy, cost-efficiency, and—most importantly—**Explainability (XAI)**.

---

## 🎯 The Problem & Our Solution

### The Gaps in Current Detectors
1. **The Black Box**: Most existing models just output "Fake" or "Real" without providing any rationale.
2. **Specialized but Limited**: Detectors often focus on *just* audio or *just* video. 
3. **Factually Blind**: Detecting a deepfake doesn't solve misinformation if the context or claims surrounding the media are also fabricated.
4. **Dataset Dependency**: Detectors trained on old GANs fail instantly against modern Diffusion models.

### The TruthSeeker Solution
1. **Privacy & Edge Execution**: Keep data safe by scanning media locally on the user's mobile device first.
2. **Explainable AI (XAI)**: We don't just output labels. We output heatmaps, timestamps, and Chain-of-Thought reasoning to explain *why* a video is fake.
3. **Multi-Modal**: A unified ecosystem handling Audio, Video, and Semantic Fact-Checking.

---

## 🧠 Core Methodology & Architecture

TruthSeeker employs a **Two-Stage Hybrid Architecture**, combining the speed of mobile processing with the heavy reasoning capabilities of advanced cloud nodes.

### 📱 Stage 1: Edge-Based Processing (Local App)
Designed for resource-constrained environments, this stage runs entirely on the user's mobile device for fast, private classification.
* **TFLite Optimization**: We utilize INT8 Quantization to shrink massive models (4x smaller, 3x faster) with less than 1% accuracy loss.
* **Hardware Acceleration**: Taps into the Android NNAPI to bypass the CPU.
* **Function**: Provides an immediate "Real / Fake" triage. If the input is too complex or an explanation is required, the data is escalated to the cloud.

### ☁️ Stage 2: Cloud-Based Processing (Deep Forensics)
When media is complex or users demand an explanation, the data is pushed to our dedicated inference servers for Explainable AI (XAI) analysis.

#### 1. Video Analysis: Skyra (The Texture & Physics Inspector)
* **Model:** A Vision-Language Multimodal Model (Qwen2.5-VL-7B) trained using LoRA and Reinforcement Learning on our custom `ViF-CoT-4K` dataset.
* **Mechanism:** Leverages **Chain-of-Thought (CoT)** reasoning. Instead of generic labels, Skyra identifies *human-perceivable artifacts*.
* **What it Detects:**
  - *Low-Level Forgery:* Texture anomalies, lighting inconsistencies.
  - *Violation of Laws:* Shape distortion, abnormal object disappearance, impossible rigid-body crossing.
* **Hardware:** Optimized to run on AMD MI300X accelerators with custom ROCm survival flags.

#### 2. Temporal Analysis: TimeSformer (The Physics Watcher)
* **Model:** A purely attention-based architecture by Meta designed to understand how features change over time, without 3D convolutions.
* **Mechanism:** Uses *Divided Space-Time Attention* to compare spatial patches across different time frames over a 16-frame sequence.
* **What it Detects:** Unnatural movements, eye-flickering, temporal jitter.

#### 3. Audio Analysis: Wav2Vec 2.0 (The Digital Forensics)
* **Model:** A transformer context network paired with a CNN feature extractor.
* **Mechanism:** Ingests raw 1D audio waveforms directly, bypassing traditional Mel-spectrograms.
* **What it Detects:** Preserves raw signals to detect subtle phase anomalies, unnatural phonetic transitions, and structural glitches introduced by AI voice cloning platforms like Suno and Udio.
* **XAI Integration:** Employs Grad-CAM to highlight anomalous frequency bands and visualize inconsistencies over time.

---

## 🕵️ Semantic Fact Verification (Phi-3 RAG)

Information overload makes manual fact-checking difficult. TruthSeeker features an autonomous backend utilizing **Phi-3** (a Small Language Model) connected to a **Retrieval-Augmented Generation (RAG)** pipeline.

**The Three-Phase Workflow:**
1. **Active Retrieval:** The backend acts as an autonomous agent, fetching external data via live web search (DuckDuckGo).
2. **Context Augmentation:** The raw evidence is stitched into the LLM's memory buffer alongside strict system instructions.
3. **Grounded Generation:** The LLM's token probabilities shift dramatically. It is forced to rely *only* on the injected evidence, acting as a reasoning engine rather than an error-prone encyclopedia.
**Result:** Source-attributed, grounded facts with near-zero hallucination risk.

---

## 📂 Project Structure & Navigation

| Directory | Description |
| :--- | :--- |
| 📱 `kotlearn/` | The Android Studio project containing the Kotlin application. |
| 🚀 `skyra_data/` | The AMD MI300X cloud backend. Contains the `train_skyra.py` pipeline and `skyra_api.py` deployment. |
| ☁️ `colab_backend/` | Google Colab deployment scripts (`audio_api.py`) for Wav2Vec Audio XAI and TimeSformer Attention Maps. |
| 🕵️ `factcheck/` | Python FastAPI backend utilizing `Phi-3-mini` with RAG for local fact-checking. |
| 📄 `docs/` | Contains the complete `TruthSeeker_Report.pdf`, methodology presentations, and system design files. |
| 🧠 `temporal2/` & `modelly/` | Scripts for training the TimeSformer engine and compiling it to Edge TFLite format. |
| 🎬 `media/videos/` | Contains the `truthdemo.mp4` and `truthdemovideo.mp4` demonstration files. |

---

## 🛠️ Comprehensive Setup Guide

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

## 🚧 Challenges & Mitigation Strategies

| Challenge | Approach |
| :--- | :--- |
| **Video models sensitive to compressed media** | Using extensive Data Augmentation (Gaussian blur, invariant scaling) to create distorted frames so models generalize well. |
| **Models too large to run locally** | Using Cloud services like AMD Developer Cloud to host massive models (Skyra), while using **Quantization** (FP32 -> INT8) to shrink models for edge devices. |
| **Lack of detailed datasets** | Utilized the `ViF-CoT-4K` dataset with rich, human-annotated Chain-of-Thought narratives. |

---

## 📊 Model Evaluation Metrics

Our models achieve state-of-the-art results across benchmarks:
* **TimeSformer (Temporal Video):** Accuracy: 95.2% | Loss: 0.1281
* **Skyra (Visual Artifact Reasoning):** Loss: 0.8952 (Significant outperformance over binary classifiers)
* **Wav2Vec 2.0 (Audio):** Accuracy: 99.73% | Loss: 0.0141

---

## 🔮 Future Scope

* **Multi-Platform Native Support:** Deploy applications for iOS and other OS environments using cross-platform frameworks.
* **Real-Time Detection:** Integrate real-time hooks to flag deepfakes live during video watching and streaming media consumption.
* **Screen Recording Analysis:** Expand the Android floating widget to automatically capture and analyze content beyond directly uploaded files.

---
<div align="center">
  <i>Built to ensure truth and transparency in the era of Generative AI.</i>
</div>
