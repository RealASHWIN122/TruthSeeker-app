import os

# --- 1. THE MI300X SHIELD (MUST BE AT THE TOP) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:128"

import torch, cv2, uuid, re, gc
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Disable internal driver optimization that causes GPFs on MI300X
torch.backends.cudnn.enabled = False

app = FastAPI()

# --- 2. GLOBAL MODEL LOADING ---
print("\n🚀 STAGE 1: Anchoring Qwen2.5-VL in VRAM...")
MODEL_PATH = "./skyra_standalone"

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0", 
    attn_implementation="eager"
).eval()
print("✅ STAGE 2: GPU Context Stable.")

def extract_markers(text, cols=16):
    """Your precise coordinate-to-grid mapping logic"""
    found_markers = []
    pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
    matches = re.findall(pattern, text)
    
    for i, match in enumerate(matches):
        ymin, xmin, ymax, xmax = map(int, match)
        start_col, end_col = int(xmin * cols / 1000), int(xmax * cols / 1000)
        start_row, end_row = int(ymin * cols / 1000), int(ymax * cols / 1000)
        
        indices = [r * cols + c for r in range(start_row, min(end_row + 1, cols)) 
                                for c in range(start_col, min(end_col + 1, cols))]
        
        found_markers.append({
            "timestampMs": (i + 1) * 1000, 
            "summary": f"Artifact {i+1} detected",
            "gridIndices": indices
        })
    return found_markers

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    temp_name = f"tmp_{uuid.uuid4()}.mp4"
    with open(temp_name, "wb") as f:
        f.write(await video.read())

    try:
        cap = cv2.VideoCapture(temp_name)
        frames = []
        # Your 16-frame deep forensic logic
        for _ in range(16):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (448, 448))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        os.remove(temp_name)

        if not frames:
            return {"status": "error", "message": "Failed to extract frames."}

        prompt = "Perform deepfake forensic analysis. Locate all artifacts and provide their [ymin, xmin, ymax, xmax] coordinates."
        messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}]}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        analysis = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "status": "success", 
            "analysis": analysis.strip(),
            "markers": extract_markers(analysis)
        }
    except Exception as e:
        print(f"Inference Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # --- STAGE 3: THE SIGNAL BYPASS STARTUP ---
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, access_log=True)
    server = uvicorn.Server(config)
    
    # We manually disable the signal handlers that cause the Segfault
    server.install_signal_handlers = lambda: None 
    
    print("🛰️ Launching Skyra FastAPI on Port 8000 (Signal Bypass Active)...")
    server.run()
