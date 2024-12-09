import torch
import numpy as np
from torch import nn


def init_weight(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)

def log_pdf_normal(x, mu, v, device, reduce=True):
    # the normal has independent dims so the v is diagonal
    c = 2 * np.pi * torch.ones(1).to(device)
    lpdf = -0.5 * (torch.log(c) + v.log() + (x - mu).pow(2).div(v))
    if reduce:
        return lpdf.sum(dim=-1)
    else:
        return lpdf


class MyVAE(nn.Module):
    def __init__(self, latent_dim, input_dim, decoder, encoder, binary_data):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.binary = binary_data
        

        if encoder is None:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 400),
                nn.ReLU(),
                nn.Linear(400, latent_dim*2)
            )
        else:
            self.encoder_net = encoder

        if decoder is None:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 400),
                nn.ReLU(),
                nn.Linear(400 , input_dim),
                nn.Sigmoid()
            )

        self.apply(init_weight)

    def encode(self, x):
        mu_logvar = self.encoder(x)
        return mu_logvar
        
    def decode(self, z):
        return self.decoder(z)
    
    def reparam_trick(self, mu, logvar):
        # print(f"mu shape is {mu.shape} and unique vals: {torch.unique(mu)}")
        # print(f"logvar shape is {logvar.shape} and unique vals: {torch.unique(logvar)}")
        return torch.exp(0.5*logvar) * torch.randn(mu.shape).to('mps')  +  mu
 
    def forward(self, x):
        mu_logvar = self.encode(x)
        mu = mu_logvar[:,:self.latent_dim]
        logvar = mu_logvar[:,self.latent_dim:]
        z = self.reparam_trick(mu, logvar)
        x_prime = self.decode(z)
        return x_prime, z, mu, logvar

    def elbo(self, x_prime, x, mu, logvar, prior=None):
        if self.binary:
            recon_loss = nn.functional.binary_cross_entropy(x_prime, x, reduction='sum')
        else:   
            #check for reduction if needed
            recon_loss = nn.functional.mse_loss(x_prime, x)
        if prior is None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl_loss = None
        return kl_loss+ recon_loss
    
    def generate_data(self, batch_size):
        sampeled_z = torch.randn((batch_size, self.latent_dim)).to('mps')
        return self.decode(sampeled_z)
        



class My_iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, hidden_dim, device):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim 
        self.aux_dim = aux_dim
        self.device = device

        #prior_params 
        # why in the codes of the paper one dim?
        self.prior_mean =  torch.zeros(1).to(device)
        self.logl = nn.Sequential(nn.Linear(aux_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, latent_dim))
        
        #decoder_params
        self.f = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, data_dim))
        self.decoder_var = .01 * torch.ones(1).to(device)

        #encoder_params
        self.g = nn.Sequential(nn.Linear(data_dim+aux_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, latent_dim))
        self.logv = nn.Sequential(nn.Linear(data_dim+aux_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, latent_dim))
        self.apply(init_weight)

    def encode(self, x, u):
        input = torch.cat([x,u], dim=1)
        mean = self.g(input)
        log_var = self.logv(input)
        return mean, log_var.exp()
    
    def decode(self, z):
        x_recon = self.f(z)
        # maybe change this 
        return x_recon, self.decoder_var

    def comp_prior(self,u):
        log_var = self.logl(u)
        return self.prior_mean, log_var.exp()
    
    def forward(self, data, aux):
        prior_mean, prior_var = self.comp_prior(aux)
        mean_z, var_z = self.encode(data, aux)
        sampled_z = var_z.sqrt() * torch.randn(mean_z.shape).to(self.device) + mean_z
        recon_x, decoder_var = self.decode(sampled_z)
        return mean_z, var_z, recon_x, sampled_z, prior_mean, prior_var


    def elbo(self, x, u):
        mean_z, var_z, recon_x,  sampled_z, prior_mean, prior_var = self.forward(x, u)
        log_px_z = log_pdf_normal(x, recon_x, self.decoder_var, self.device)
        log_qz_xu = log_pdf_normal(sampled_z, mean_z, var_z, self.device)
        log_pz_u = log_pdf_normal(sampled_z, prior_mean, prior_var, self.device)
        #changes the weight of the kl
        return (log_px_z +(log_pz_u - log_qz_xu)).mean(), sampled_z
    

class iVAE_anneal(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, hidden_dim, device, anneal=False):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim 
        self.aux_dim = aux_dim
        self.device = device
        self.anneal_params = anneal

        # prior_params
        self.prior_mean = torch.zeros(latent_dim).to(device)
        self.logl = nn.Sequential(nn.Linear(aux_dim, hidden_dim),
                                  nn.LeakyReLU(negative_slope=0.01),
                                  nn.Linear(hidden_dim, latent_dim))
        
        # decoder_params
        self.f = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                  nn.LeakyReLU(negative_slope=0.01),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.LeakyReLU(negative_slope=0.01),
                                  nn.Linear(hidden_dim, data_dim))
        self.decoder_var = .01 * torch.ones(1).to(device)

        # encoder_params
        self.g = nn.Sequential(nn.Linear(data_dim+aux_dim, hidden_dim),
                                  nn.LeakyReLU(negative_slope=0.01),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.LeakyReLU(negative_slope=0.01),
                                  nn.Linear(hidden_dim, latent_dim))
        self.logv = nn.Sequential(nn.Linear(data_dim+aux_dim, hidden_dim),
                                  nn.LeakyReLU(negative_slope=0.01),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.LeakyReLU(negative_slope=0.01),
                                  nn.Linear(hidden_dim, latent_dim))
        self.apply(init_weight)

        # Training hyperparameters for annealing
        self._training_hyperparams = [1., 1., 1., 1., 1]

    def encode(self, x, u):
        input = torch.cat([x, u], dim=1)
        mean = self.g(input)
        log_var = self.logv(input)
        return mean, log_var.exp()
    
    def decode(self, z):
        x_recon = self.f(z)
        return x_recon, self.decoder_var

    def comp_prior(self, u):
        log_var = self.logl(u)
        return self.prior_mean, log_var.exp()
    
    def forward(self, data, aux):
        prior_mean, prior_var = self.comp_prior(aux)
        mean_z, var_z = self.encode(data, aux)
        sampled_z = var_z.sqrt() * torch.randn(mean_z.shape).to(self.device) + mean_z
        recon_x, decoder_var = self.decode(sampled_z)
        return mean_z, var_z, recon_x, sampled_z, prior_mean, prior_var
    
    def elbo(self, x, u):
        mean_z, var_z, recon_x, sampled_z, prior_mean, prior_var = self.forward(x, u)
        log_px_z = log_pdf_normal(x, recon_x, self.decoder_var, self.device)
        log_qz_xu = log_pdf_normal(sampled_z, mean_z, var_z, self.device)
        log_pz_u = log_pdf_normal(sampled_z, prior_mean, prior_var, self.device)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = sampled_z.size(0)
            log_qz_tmp = log_pdf_normal(sampled_z.view(M, 1, self.latent_dim), mean_z.view(1, M, self.latent_dim),
                                        var_z.view(1, M, self.latent_dim), self.device, reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), sampled_z
        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), sampled_z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False
    
        


