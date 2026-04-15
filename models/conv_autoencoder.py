import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # =====================
        # Encoder
        # =====================
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Keep values stable during training 
            nn.ReLU()
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

        # =====================
        # Bottleneck   --> Bottleneck is like saturation point where we cannot generate any more features
        # =====================
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU()
        )

        # =====================
        # Decoder
        # =====================
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Stride controls upsampling 16×16 → 32×32
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # =====================
        # Output layer
        # =====================
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):   # forward pass means how data flows through the network
        # Encoder
        e1 = self.enc1(x)              # (B, 32, 128, 128)
        p1 = self.pool(e1)             # (B, 32, 64, 64)

        e2 = self.enc2(p1)             # (B, 64, 64, 64)
        p2 = self.pool(e2)             # (B, 64, 32, 32)

        e3 = self.enc3(p2)             # (B, 128, 32, 32)
        p3 = self.pool(e3)             # (B, 128, 16, 16)

        # Bottleneck
        b = self.bottleneck(p3)        # (B, 256, 16, 16)

        # Decoder + Skip connections (U-Net style concatenation of encoder features)
        d3 = self.up3(b)               # (B, 128, 32, 32)
        d3 = torch.cat([d3, e3], dim=1)   # Concatenate because unsampling alone cannot recover lost info so we add encoder features + decoder features
        d3 = self.dec3(d3)

        d2 = self.up2(d3)              # (B, 64, 64, 64)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)              # (B, 32, 128, 128)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)           # (B, 3, 128, 128)
        return out
