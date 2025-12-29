# Defines the convolutional autoencoder architecture.
# - Encoder compresses input frames into a latent representation.
# - Decoder reconstructs frames using learned upsampling.
# - Designed to learn normal appearance patterns in video data.


import torch  # import the main PyTorch library
import torch.nn as nn   # import the neural network module from PyTorch

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder,self).__init__()  # initialize the parent class constructor()

        # --------------------
        # Encoder
        # Input: (B, 3, 128, 128) // batch size B, 3 color channels RGB, 128x128 image
        # --------------------
        self.encoder = nn.Sequential(
            # 3 = input channels (RGB), 32 = output channels (learn 32 feature maps ,it is 32 kernels with size 3x3)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (B, 32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # (B, 32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # (B, 64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B, 128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # (B, 128, 16, 16)
        )

        # --------------------
        # Decoder
        # Output: (B, 3, 128, 128)
        # --------------------
        self.decoder = nn.Sequential(
            # 128 = input channels, 64 = output channels, stride = 2 means divide 
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # (B, 64, 32, 32)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # (B, 32, 64, 64)
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),    # (B, 3, 128, 128)
            nn.Sigmoid()  # Output in range [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
