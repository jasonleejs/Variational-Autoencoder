import torch
import torch.nn as nn
from vae import VAE
from tqdm import tqdm
import matplotlib.pyplot as plt
    
class ConvVAE(nn.Module):
    def __init__(self,  latent_dim: int, beta=0.1, h_channels=32, loss_fcn = nn.MSELoss(),
                 h_encoder: nn.Sequential=None, z_decoder: nn.Sequential=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.losses = []
        self.kld = 0.
        self.beta = beta
        self.h_channels = h_channels
        
        self.loss_fcn = loss_fcn 
        
        if h_encoder is None:
            self.h_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),   # 28x28 -> 14x14
                nn.Conv2d(16, self.h_channels, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 14x14 -> 7x7
                nn.Flatten()
            )
        else: 
            self.h_encoder = h_encoder
        
        self.mu_encoder = nn.Linear(self.h_channels * 7 * 7, self.latent_dim)
        self.std_encoder = nn.Linear(self.h_channels * 7 * 7, self.latent_dim)
        
        if z_decoder is None:
            self.z_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.h_channels * 7 * 7), nn.ReLU(),
                nn.Unflatten(1, (self.h_channels, 7, 7)),
                nn.ConvTranspose2d(self.h_channels, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 7x7 -> 14x14
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()  # 14x14 -> 28x28
            )
        else:
            self.z_decoder = z_decoder
            
    def encoder(self, x):
        h = self.h_encoder(x)
        mu = self.mu_encoder(h)
        std = torch.exp(self.std_encoder(h))
        return mu, std
    
    def decoder(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.z_decoder(z)
    
    def forward(self, x):
        mu, std = self.encoder(x)
        return self.decoder(mu)
    

    def learn_forward(self, x):
        mu, sigma = self.encoder(x)
        # sample z
        z = mu + sigma * torch.randn_like(sigma)
        
        # kld loss
        self.kld = torch.mean(mu**2 + sigma**2 - 2.*torch.log(sigma) - 1.)/ 2
        y = self.decoder(z)
        return y

    def learn(self, dl, epochs=10, lr=0.001):
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
                total_loss += loss.item() * len(x_batch) 
                
            self.losses.append(total_loss / len(dl.dataset))
        plt.plot(self.losses); plt.xlabel('Epochs');
        plt.yscale('log');
        
        
        