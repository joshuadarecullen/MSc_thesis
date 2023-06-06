class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Define the architecture for the encoder and decoder networks.
        # The encoder network takes the input data and outputs two things: mu and logvar.
        # The decoder network takes a point in the latent space and outputs the reconstructed input.

        # For the encoder:
        # fc1 transforms the input data to the hidden space
        # fc21 and fc22 transform the hidden space to the parameters of the latent space (mu and logvar)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        # For the decoder:
        # fc3 transforms the latent space back to the hidden space
        # fc4 transforms the hidden space to the output data
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        # This method passes the input through the encoder.
        # It first applies the fc1 layer and then applies a ReLU activation function.
        # Then it applies fc21 and fc22 to output mu and logvar.
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # This method applies the "reparameterization trick" to allow for gradient descent.
        # It generates a random tensor of the same shape as the standard deviation (which is computed as the exponential of half the log variance), drawn from a standard normal distribution.
        # This random tensor is then scaled by the standard deviation and shifted by the mean.
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # This method passes the latent representation through the decoder.
        # It first applies the fc3 layer and then applies a ReLU activation function.
        # Then it applies fc4 and applies a sigmoid activation function, which ensures the output values are between 0 and 1.
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # This method defines the forward pass of the VAE.
        # It first applies the encoder to the input x to get mu and logvar.
        # Then it reparameterizes mu and logvar to get a sample from the latent space.
        # Then it decodes this sample to get the reconstructed input.
        # It returns both the reconstructed input and the parameters of the latent space distribution.
        mu, logvar = self.encode(x.view(-1, input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
