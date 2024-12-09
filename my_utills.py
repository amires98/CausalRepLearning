import numpy as np
import torch.nn as nn
import torch

def generate_data(num_segs, samples_per_seg):

    # Total number of samples
    N = num_segs * samples_per_seg

    # Segment labels (auxiliary variable)
    u = np.repeat(np.arange(num_segs), samples_per_seg)

    # Latent dimension
    latent_dim = 2

    # Initialize arrays to store means and variances
    mu = np.zeros((num_segs, latent_dim))
    var = np.zeros((num_segs, latent_dim))

    # Randomly generate segment-specific means and variances
    for m in range(num_segs):
        mu[m] = np.random.randn(latent_dim)*1.5
        var[m] =( -1 if m%2 == 0 else 1 ) * (np.random.rand(latent_dim))  # Variances between -1 and 1

    # Sample latent variables z
    z = np.zeros((N, latent_dim))
    for m in range(num_segs):
        idx = u == m
        z[idx] = np.random.randn(samples_per_seg, latent_dim) * var[m] + mu[m]



    # Define the mixing function f as a simple MLP
    class MixingFunction(nn.Module):
        def __init__(self, n, d):
            super(MixingFunction, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(n, 100),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(100, d)
            )

        def forward(self, z):
            return self.net(z)

    # Observation dimension
    d = 2  # Same as latent dimension for the 2D example

    # Instantiate the mixing function
    f = MixingFunction(latent_dim, d)

    # Convert z to torch tensor
    z_tensor = torch.from_numpy(z).float()

    # Apply the mixing function
    with torch.no_grad():
        x = f(z_tensor).numpy()

    # Add small Gaussian noise
    epsilon = np.random.randn(N, d) * 0.1  # Adjust variance as needed
    x += epsilon
    return x, z, u

