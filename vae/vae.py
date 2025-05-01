import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, latent_dim: int, beta=0.1, loss_fcn=nn.MSELoss(),
                 h_encoder: nn.Sequential=None, decoder: nn.Sequential=None):
        
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.losses = []
        self.kld = 0.
        self.beta = beta
            
        self.loss_fcn = loss_fcn
        
        if h_encoder is None:
            self.h_encoder = nn.Sequential(
                nn.Linear(self.in_dim, 300), nn.ReLU(),
                nn.Linear(300, 200), nn.ReLU(),
                nn.Linear(200, self.h_dim), nn.ReLU()
            )
        else: 
            self.h_encoder = h_encoder
        
        self.mu_encoder = nn.Linear(self.h_dim, self.latent_dim)
        self.std_encoder = nn.Linear(self.h_dim, self.latent_dim)
            
        if decoder is None: 
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 100), nn.ReLU(),
                nn.Linear(100, self.in_dim),
            )
        else:
            self.decoder = decoder
        
        
    def encoder(self, x):
        h = self.h_encoder(x)
        mu = self.mu_encoder(h)
        std = torch.exp(self.std_encoder(h))
        return mu, std
        
    def forward(self, x):
        mu, std = self.encoder(x)
        return self.decoder(mu)
    
    def learn_forward(self, x):
        mu, sigma = self.encoder(x)
        # sample z
        z = mu + sigma*torch.randn_like(sigma)
        
        # kld loss
        self.kld = torch.mean(mu**2 + sigma**2 - 2.*torch.log(sigma) - 1.) / 2.
        y = self.decoder(z)
        return y
    
    def learn(self,  dl, epochs=10, lr=0.001):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in tqdm(range(epochs)):
            total_loss = 0.
            for x_batch, t_batch in dl:
                y_batch = self.learn_forward(x_batch)
                recon_loss = self.loss_fcn(y_batch, x_batch)
                loss = recon_loss + self.beta * self.kld
                # backprop
                optim.zero_grad()
                loss.backward()
                optim.step()

                # loss in the epoch
                total_loss += loss.item()*len(x_batch) # total sum of squared errors so far
                
            self.losses.append(total_loss/len(dl.dataset)) # total sum of squared errors / number of samples = MSE
        plt.plot(self.losses); plt.xlabel('Epochs');
        plt.yscale('log');