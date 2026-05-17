import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)

        i, f, o, g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

# 2D Convolutional LSTM Autoencoder
class ConvLSTMAutoencoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64):
        super().__init__()

        self.encoder = ConvLSTMCell(
            input_dim=input_channels,
            hidden_dim=hidden_dim
        )

        self.decoder = ConvLSTMCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

        self.output_layer = nn.Conv2d(
            hidden_dim,
            input_channels,
            kernel_size=1
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        hidden_dim = self.encoder.hidden_dim

        h, c = self._init_hidden(B, H, W, x.device)

        # Pre-allocate — avoids T list appends and a torch.stack call
        enc_h = torch.zeros(B, T, hidden_dim, H, W, device=x.device)
        output = torch.zeros(B, T, C, H, W, device=x.device)

        for t in range(T):
            h, c = self.encoder(x[:, t], h, c)
            enc_h[:, t] = h

        for t in range(T):
            h, c = self.decoder(enc_h[:, t], h, c)
            output[:, t] = self.output_layer(h)

        return output

    def _init_hidden(self, B, H, W, device):
        h = torch.zeros(B, self.encoder.hidden_dim, H, W, device=device)
        c = torch.zeros(B, self.encoder.hidden_dim, H, W, device=device)
        return h, c
