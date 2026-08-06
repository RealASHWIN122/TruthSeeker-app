import os
import gc

# 1. THE AMD SURVIVAL FLAGS
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import json
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import torch.optim as optim

# 2. THE VISION CRASH FIX
torch.backends.cudnn.enabled = False

print("🖥️ Starting Skyra Final Training Pipeline...")
print(f"🔥 Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 3. Load Model & Processor
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("🚀 Loading Model in BF16 (Native Precision)...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    device_map="cuda", 
    trust_remote_code=True,
    attn_implementation="eager"
)

# 4. LoRA Setup
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
))
model.train()

# 5. Data Setup
with open("skyra_master.json", "r") as f:
    raw_data = json.load(f)

video_map = {}
for root, _, files in os.walk("."):
    for f in files:
        if f.endswith(".mp4"):
            video_map[f.replace(".mp4", "")] = os.path.abspath(os.path.join(root, f))

matched_data = [ex for ex in raw_data if ex['video_id'] in video_map]
print(f"✅ Successfully matched {len(matched_data)} videos for training.")

# 6. Optimization Config
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
accumulation_steps = 16
current_loss = 0.0

# 7. Training Loop
print("🔥 TRAINING STARTED...")
for i, item in enumerate(matched_data):
    vid_path = video_map[item['video_id']]
    
    try:
        # Extract Frames Safely on CPU
        cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
        frames = []
        for _ in range(8):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (224, 224))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        if not frames:
            continue

        messages = [
            {"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": "Analyze the deepfake artifacts in this video."}]},
            {"role": "assistant", "content": [{"type": "text", "text": item['cot_response']}]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], videos=[frames], padding=True, return_tensors="pt")

        prompt_text = processor.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(text=[prompt_text], videos=[frames], padding=False, return_tensors="pt")
        prompt_length = prompt_inputs["input_ids"].shape[1]

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.contiguous()
                if v.is_floating_point():
                    inputs[k] = v.to("cuda", dtype=torch.bfloat16)
                else:
                    inputs[k] = v.to("cuda")

        labels = inputs["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[0, :prompt_length] = -100

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        current_loss += loss.item()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            print(f"✅ Step {(i + 1) // accumulation_steps} | Videos Processed: {i + 1} | Loss: {current_loss * accumulation_steps:.4f}")
            current_loss = 0.0

        del inputs, prompt_inputs, outputs, loss, frames, labels
        if (i + 1) % 4 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"⚠️ Error on {os.path.basename(vid_path)}: {e}")
        continue

print("💾 Saving Model Weights...")
model.save_pretrained("./skyra_final_weights")
print("🏁 Training Complete!")
