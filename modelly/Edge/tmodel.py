import torch
import torch.nn as nn
from transformers import TimesformerModel

class TemporalPhysicsEngine(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        print("🧊 Initializing Meta TimeSformer (Divided Space-Time Attention)...")
        
        # 1. LOAD THE CORE BRAIN
        # We use the base model pre-trained on Kinetics-400 (Microsoft's standard physics dataset).
        # It expects exactly 8 frames and 224x224 resolution, which perfectly matches our dataloader!
        if pretrained:
            self.backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400", 
                num_frames=8, 
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import TimesformerConfig
            config = TimesformerConfig(num_frames=8)
            self.backbone = TimesformerModel(config)

        # 2. THE TITANIUM ARMOR (FREEZING)
        # TimeSformer has 12 attention blocks. We freeze the embeddings and the first 6 blocks.
        # This forces the model to retain basic real-world physics and saves massive VRAM.
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
            
        for layer in self.backbone.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

        # 3. THE CUSTOM TRUTHSEEKER HEAD
        # TimeSformer outputs a hidden state of size 768. 
        # We attach a dropout layer to prevent memorization, and compress it down to 1 output (Real/Fake).
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1) # Single output for Binary Cross Entropy Loss
        )
        
        print("✅ Custom TimeSformer Deepfake Binary Head Attached.")

    def forward(self, x):
        """
        The Dataloader provides: [Batch, Channels, Time, Height, Width]
        TimeSformer requires:    [Batch, Time, Channels, Height, Width]
        """
        # 1. Rotate the 4D tensor dynamically
        x = x.permute(0, 2, 1, 3, 4) 
        
        # 2. Pass through the TimeSformer backbone
        outputs = self.backbone(pixel_values=x)
        
        # 3. Extract the [CLS] token (Class Token)
        # The CLS token acts as the aggregate summary of the entire video sequence.
        # It is always located at index 0 of the sequence output.
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # 4. Pass the summary token through our custom Judge head
        logits = self.classifier(cls_token)
        
        return logits

# ==========================================
# 🧪 QUICK TENSOR SHAPE TEST
# ==========================================
if __name__ == "__main__":
    # Simulating what your dataloader outputs: Batch of 2, 3 Channels (RGB), 8 Frames, 224x224 resolution
    dummy_input = torch.randn(2, 3, 8, 224, 224)
    model = TruthSeekerTemporal(pretrained=False) # False just for the rapid local test
    
    print("\n🚀 Testing TimeSformer Forward Pass...")
    output = model(dummy_input)
    
    print(f"🎯 Output Shape: {output.shape}")
    if list(output.shape) == [2, 1]:
        print("✅ SUCCESS! The architecture swap is structurally perfect.")