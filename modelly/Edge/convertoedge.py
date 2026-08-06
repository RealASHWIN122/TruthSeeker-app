import torch
# Install this via: pip install ai-edge-torch
import litert_torch
from tmodel import TemporalPhysicsEngine

def convert_to_edge():
    print("🧊 Loading TruthSeeker TimeSformer...")
    
    # 1. Initialize the model structure
    model = TemporalPhysicsEngine(pretrained=False).eval()
    
    # 2. Load your best trained weights
    # (Ensure this path points to where your train.py saved the file)
    checkpoint = torch.load(
        "/home/me/Videos/final year project/TruthSeeker-app/modelly/Edge/temporal_best_model.pth", 
        map_location="cpu"
    )
    model.load_state_dict(checkpoint['state_dict'])
    print("✅ Weights loaded successfully.")

    # 3. Create a Dummy Tensor that perfectly matches the Android App's camera feed
    # Shape: [Batch=1, Channels=3, Frames=8, Height=224, Width=224]
    dummy_input = torch.randn(1, 3, 8, 224, 224)

    # 4. Convert directly to Google's Edge format
    print("🔄 Compiling PyTorch Graph to LiteRT Engine. This may take a few minutes...")
    edge_model = litert_torch.convert(model, (dummy_input,))
    
    # 5. Export the raw file
    output_name = "truthseeker_timesformer_raw.tflite"
    edge_model.export(output_name)
    print(f"✅ Conversion Complete! Saved as {output_name}")

if __name__ == "__main__":
    convert_to_edge()