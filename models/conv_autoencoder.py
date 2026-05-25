import torch.nn as nn


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
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # =====================
        # Decoder (no skip connections — forces bottleneck compression)
        # =====================
        self.up3 = nn.ConvTranspose2d(32, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
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

    def forward(self, x):
        # Encoder
        p1 = self.pool(self.enc1(x))   # (B, 32, 64, 64)
        p2 = self.pool(self.enc2(p1))  # (B, 64, 32, 32)
        p3 = self.pool(self.enc3(p2))  # (B, 128, 16, 16)

        # Bottleneck
        b = self.bottleneck(p3)        # (B, 32, 16, 16)

        # Decoder (bottleneck only — no skip connections)
        d3 = self.up3(b)               # (B, 128, 32, 32)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)              # (B, 64, 64, 64)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)              # (B, 32, 128, 128)
        d1 = self.dec1(d1)

        out = self.final(d1)           # (B, 3, 128, 128)
        return out
