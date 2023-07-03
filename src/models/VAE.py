import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCVAE(nn.Module):
    def __init__(self, input_channels, latent_size, out_channels1, out_channels2, kernel_size, stride, padding):
        super(SpectrogramCVAE, self).__init__()

        # Encoder
        # The encoder is made up of convolutional layers followed by fully connected layers.
        # It's responsible for encoding the input spectrograms into a latent representation.
        self.encoder = nn.Sequential(
            # First convolutional layer with ReLU activation
            # input_channels specifies the number of channels in the input spectrogram.
            # The output has 16 channels.
            nn.Conv2d(input_channels, out_channels1, kernel_size, stride, padding),
            nn.ReLU(),
            # Second convolutional layer with ReLU activation
            # Takes the 16 channels from the previous layer and outputs 32 channels.
            nn.Conv2d(out_channels1, out_channels2, kernel_size, padding),
            nn.ReLU(),
            # Flatten the output of the convolutions to feed into a fully connected layer
            nn.Flatten(),
            # Fully connected layer to produce the parameters of the latent distribution
            # 32*H*W should be adjusted based on the size of your spectrograms and the architecture.
            nn.Linear(32 * H * W, latent_size*2)  # We'll split this into mu and logvar later
        )

        # Decoder
        # The decoder consists of fully connected layers followed by transposed convolutional layers.
        # It takes the sampled latent variables and decodes them back into a reconstructed spectrogram.
        self.decoder = nn.Sequential(
            # Fully connected layer to upsample the latent representation
            nn.Linear(latent_size, 32 * H * W),
            nn.ReLU(),
            # Reshape the output back into a feature map
            nn.Unflatten(1, (out_channels2, H, W)),
            # First transposed convolutional layer with ReLU activation
            # Upsamples the feature map and produces 16 channels.
            nn.ConvTranspose2d(out_channels2, out_channels1, 3, stride=2, padding=1),
            nn.ReLU(),
            # Second transposed convolutional layer
            # Further upsamples the feature map to the original size and produces output with the same number of channels as the input.
            nn.ConvTranspose2d(out_channels1, input_channels, 3, stride=2, padding=1, output_padding=1),
            # Sigmoid activation to make the output values in the range [0, 1].
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decorder(z)

    def reparameterize(self, mu, logvar):
        # This function applies the reparameterization trick:
        # z = mu + epsilon * sigma, where epsilon is sampled from N(0, 1).
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # The forward method is called when you pass data through the model.
        # Encode the input data into mu and logvar
        h = self.encode(x)
        # Split the output of the encoder into mu and logvar
        mu, logvar = h.split(h.size(1)//2, dim=1)
        # Sample a latent vector using reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Decode the latent vector into a reconstructed spectrogram
        return self.decode(z), mu, logvar
