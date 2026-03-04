"""
model.py — Stream C: Frequency Domain ConvNeXt Architecture
============================================================
Implements the full frequency-domain pipeline:
    RGB image  ->  FrequencyInputLayer  ->  ConvNeXt-Nano  ->  binary logit

Author : Principal AI Research Engineer
"""

import timm
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# FrequencyInputLayer
# -----------------------------------------------------------------------------
class FrequencyInputLayer(nn.Module):
    """
    Converts a normalized RGB tensor into a 3-channel log-magnitude Fourier
    spectrum that ConvNeXt can consume.

    Changes from v1:
    - Replaced MinMax normalization with BatchNorm2d.
    - This prevents the massive DC component (center star) from squashing
      high-frequency artifacts to zero.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Fixed BT.601 luma weights
        self.register_buffer(
            "rgb_weights",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
        )
        # Learnable normalization to handle high dynamic range of FFT
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -- 1. Grayscale ------------------------------------------------------
        gray = (x * self.rgb_weights).sum(dim=1, keepdim=True)  # [B,1,H,W]

        # -- 2. 2D FFT + shift -------------------------------------------------
        # FFT is computed on the last two dimensions (-2, -1)
        fft_out = torch.fft.fft2(gray)
        fft_out = torch.fft.fftshift(fft_out, dim=(-2, -1))     # [B,1,H,W] complex

        # -- 3. Log-magnitude spectrum -----------------------------------------
        # log(1 + |X|) handles the massive dynamic range better than log(|X|)
        log_mag = torch.log(torch.abs(fft_out) + 1.0)           # [B,1,H,W]

        # -- 4. Learnable Normalization ----------------------------------------
        # Replaces rigid MinMax. BatchNorm centers the spectrum data.
        norm_mag = self.bn(log_mag)

        # -- 5. Replicate -> 3-channel tensor ----------------------------------
        return norm_mag.repeat(1, 3, 1, 1)                      # [B,3,H,W]


# -----------------------------------------------------------------------------
# FrequencyModel  (full pipeline)
# -----------------------------------------------------------------------------
class FrequencyModel(nn.Module):
    """
    Full frequency-domain deepfake detector.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone_epochs: int = 0,
    ):
        super().__init__()
        self.freeze_backbone_epochs = freeze_backbone_epochs

        self.freq_layer = FrequencyInputLayer()
        self.backbone   = timm.create_model(
            "convnext_nano",
            pretrained=pretrained,
            num_classes=1,
        )

    # -- Backbone freeze / unfreeze helpers -----------------------------------
    def freeze_backbone(self) -> None:
        """Freeze all ConvNeXt parameters except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "head" not in name:
                param.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze entire backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    def maybe_unfreeze(self, epoch: int) -> None:
        """Called at the start of each epoch to manage freeze schedule."""
        if epoch == 0 and self.freeze_backbone_epochs > 0:
            self.freeze_backbone()
        elif epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
            self.unfreeze_backbone()

    # -- Forward --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq_layer(x)    # RGB -> Normalized Log-Spectrum
        return self.backbone(x)   # -> Logit

    # -- Convenience ----------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(fake) in [0, 1]."""
        return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}