import torch
import torchaudio
from df_arena.modeling_antispoofing import DF_Arena_1B_Antispoofing
from df_arena.configuration_antispoofing import DF_Arena_1B_Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "pytorch_model.bin"   


config = DF_Arena_1B_Config()
model = DF_Arena_1B_Antispoofing(config)

state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()

# 2️⃣ Load MP3 correctly
def load_mp3(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)  # MP3 decoding
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    waveform = waveform.mean(dim=0)  # stereo → mono
    return waveform.numpy()

# 3️⃣ Detection
def detect_spoof_mp3(mp3_path):
    audio = load_mp3(mp3_path)

    with torch.no_grad():
        inputs = model.feature_extractor(audio)["input_values"]
        inputs = inputs.unsqueeze(0).to(DEVICE)

        outputs = model(input_values=inputs)
        logits = outputs["logits"]

        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    return {
        "label": config.id2label[pred],
        "confidence": probs[0][pred].item(),
        "bonafide": probs[0][1].item(),
        "spoof": probs[0][0].item()
    }

# 4️⃣ Run
if __name__ == "__main__":
    result = detect_spoof_mp3("real1.mp3")
    print(result)
  


