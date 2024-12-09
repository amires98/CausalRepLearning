from myModels import *
from my_utills import generate_data
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
from models import iVAE

# Training settings
batch_size = 1000
epochs = 70
learning_rate = 1e-3


# Data loading
num_segs = 5
samples_per_seg = 4000
N = num_segs * samples_per_seg
x ,z ,u = generate_data(num_segs, samples_per_seg)

u_onehot = np.zeros((N, num_segs))
u_onehot[np.arange(N), u] = 1

x_tensor = torch.from_numpy(x).float()
u_tensor = torch.from_numpy(u_onehot).float()

# Create dataset and dataloader
dataset = data.TensorDataset(x_tensor, u_tensor)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Model, optimizer, device
# Check if MPS is available and set the device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = iVAE_anneal(latent_dim=2, data_dim=2, aux_dim=5, hidden_dim=50, device = device, anneal=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
it = 0
for epoch in range(epochs):
    for batch_x, batch_u in dataloader:
        it +=1
        model.anneal(N, 1000, it)
        batch_x = batch_x.to(device)
        batch_u = batch_u.to(device)
        optimizer.zero_grad()
        loss, _ = model.elbo(batch_x, batch_u)
        loss.mul(-1).backward()
        # Backpropagation
        optimizer.step()

    print(f'MY iVAE version Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')


model_code = iVAE(2,2,5,device = device, anneal=True).to(device)
optimizer_sec = optim.Adam(model_code.parameters(), lr=learning_rate)
it = 0
for epoch in range(epochs):
    for batch_x, batch_u in dataloader:
        it +=1
        model_code.anneal(N, 1000, it)
        batch_x = batch_x.to(device)
        batch_u = batch_u.to(device)
        optimizer_sec.zero_grad()
        loss, _ = model_code.elbo(batch_x, batch_u)
        loss.mul(-1).backward()
        # Backpropagation
        optimizer_sec.step()

    print(f'Paper iVAE version Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
model_code.eval()



input_dim = 2
latent_dim = 10
model_vae = MyVAE(latent_dim,input_dim, None, None, True).to(device)
optimizer_vae = optim.Adam(model_vae.parameters(), lr=learning_rate)

epochs= 20 if epochs>20 else epochs
# Training loop
for epoch in range(epochs):
    model_vae.train()
    train_loss = 0
    for batch_x, batch_u in dataloader:
        data = batch_x.to(device)
        optimizer_vae.zero_grad()
        x_prime, z_, mu, logvar = model_vae(data)
        loss = model_vae.elbo(x_prime, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer_vae.step()
        
    print(f'VAE Epoch {epoch + 1}, Loss: {train_loss / N:.4f}')


# Generate images
model_vae.eval()



with torch.no_grad():
    _, z_vae , _ , _ =model_vae(x_tensor.to(device))
    z_vae = z_vae.cpu().numpy()

    q_mu, _ = model.encode(x_tensor.to(device), u_tensor.to(device))
    z_encoded = q_mu.cpu().numpy()

    decoder_params_C, (g_c, v_c), z_c, prior_params_c = model_code(x_tensor.to(device), u_tensor.to(device))
    z_c = z_c.cpu().numpy()

    
    colors = ['orangered', 'gold', 'mediumseagreen', 'royalblue', 'orchid']
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # True Sources
    axs[0,0].scatter(z[:, 0], z[:, 1], c=[colors[i] for i in u], s=10, alpha=0.7, marker='+')
    axs[0,0].set_title('True Sources (z)')
    axs[0,0].set_xlabel('z1')
    axs[0,0].set_ylabel('z2')

    # Observations
    axs[1,0].scatter(x[:, 0], x[:, 1], c=[colors[i] for i in u], s=10, alpha=0.7, marker='+')
    axs[1,0].set_title('Observations (x)')
    axs[1, 0].set_xlabel('x1')
    axs[1, 0].set_ylabel('x2')

    # Encoded Latents from iVAE
    axs[1,1].scatter(z_encoded[:, 0], z_encoded[:, 1], c=[colors[i] for i in u], s=10, alpha=0.7, marker='+')
    axs[1,1].set_title('Encoded Latents from my iVAE')
    axs[1,1].set_xlabel('z1')
    axs[1,1].set_ylabel('z2')

    # Encoded Latents from VAE
    axs[0,1].scatter(z_vae[:, 0], z_vae[:, 1], c=[colors[i] for i in u], s=10, alpha=0.7, marker='+')
    axs[0,1].set_title('Encoded Latents from VAE')
    axs[0,1].set_xlabel('z1')
    axs[0,1].set_ylabel('z2')

    # code encoded
    # axs[4].scatter(z_c[:, 0], z_c[:, 1], c=[colors[i] for i in u], s=10, alpha=0.7, marker='+')
    # axs[4].set_title('paper code encoded (z)')
    # axs[4].set_xlabel('z1')
    # axs[4].set_ylabel('z2')


    plt.tight_layout()


    plt.show()