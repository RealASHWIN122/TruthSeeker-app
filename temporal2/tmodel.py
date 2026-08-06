import torch
import torch.nn as nn
from transformers import TimesformerModel, TimesformerConfig

class TemporalPhysicsEngine(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        print("🧊 Initializing Meta TimeSformer (Divided Space-Time Attention)...")
        
        # 1. LOAD THE CORE BRAIN
        if pretrained:
            self.backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400", 
                num_frames=16, 
                ignore_mismatched_sizes=True # This handles the 8-frame to 16-frame warning automatically
            )
        else:
            config = TimesformerConfig(num_frames=16)
            self.backbone = TimesformerModel(config)

        # 2. THE TITANIUM ARMOR (FREEZING)
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
            
        for layer in self.backbone.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

        # 3. THE CUSTOM TRUTHSEEKER HEAD
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1) 
        )
        
        print("✅ Custom TimeSformer Deepfake Binary Head Attached.")

    def forward(self, x):
        """
        Input x shape: (Batch, Frames, Channels, Height, Width) -> (B, 16, 3, 224, 224)
        """
        # --- DO NOT PERMUTE ---
        # HuggingFace expects (Batch, Num_Frames, Channels, Height, Width)
        
        # Pass through the backbone
        outputs = self.backbone(pixel_values=x)
        
        # Grab the CLS token (index 0) which represents the entire video summary
        # outputs.last_hidden_state shape: (Batch, Seq_Len, 768)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through your custom classification head
        logits = self.classifier(sequence_output)
        
        return logits