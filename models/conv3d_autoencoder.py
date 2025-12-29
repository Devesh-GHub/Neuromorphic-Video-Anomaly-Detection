import torch
import torch.nn as nn

class Conv3DAutoencoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128):
        super().__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, hidden_dim, kernel_size=3, stride=2, padding=1),  # 128→64
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
        )

        # ---------- Bottleneck (KEY FOR AUC) ----------
        self.bottleneck = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU()
        )

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # (B,T,C,H,W) → (B,C,T,H,W)
        x = x.permute(0, 2, 1, 3, 4)
        z = self.encoder(x)
        z = self.bottleneck(z)
        out = self.decoder(z)
        return out.permute(0, 2, 1, 3, 4)  # back to (B,T,C,H,W)
