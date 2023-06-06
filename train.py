import torch
import torch.nn.functional as F
from VAE import VAE as vae

# train loop
def train(model, dataloader, epochs, optimiser, device):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)  # Send the data to the device (GPU or CPU)
            optimiser.zero_grad()  # Zero out any gradients from the last step
            recon_batch, mu, logvar = model(data)  # Forward pass through the model
            loss = loss_function(recon_batch, data, mu, logvar)  # Calculate the loss
            loss.backward()  # Backpropagation
            train_loss += loss.item()  # Accumulate the train loss
            optimiser.step()  # Update the weights

        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')


def loss_function(recon_x, x, mu, logvar):
    # The loss function for a VAE consists of two parts:
    # 1) a reconstruction term (the binary cross entropy, BCE) and
    # 2) a regularisation term that pushes the learned distribution to be close to a standard normal distribution (the Kullback-Leibler divergence, KLD).

    # The binary cross entropy (BCE) is used as the reconstruction loss here.
    # It measures how well the reconstructed output (recon_x) matches the original input data (x).
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # The Kullback-Leibler divergence (KLD) measures how much one probability distribution diverges from a second, expected probability distribution.
    # In this case, it measures the divergence of the learned distribution (defined by mu and logvar) from a standard normal distribution.

    # It's computed directly from mu and logvar, based on the mathematical form of the KL divergence for two Gaussians.
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # The total loss is the sum of the BCE and the KLD.
    # The BCE encourages the VAE to reconstruct the input data well, while the KLD encourages the VAE to produce a latent space distribution that's close to a standard normal distribution.
    return BCE + KLD


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.set_default_device('cuda')
    else:
        device = torch.set_default_device('cpu')

    model = vae()
    epochs = 10
    optimiser = torch.optim.Adam(lr=0.001)

    '''
    datalader = 
    '''
