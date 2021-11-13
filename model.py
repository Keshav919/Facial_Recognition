import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim = 96, num_classes = 10) -> None:
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(25088,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.classification = nn.Linear(512,num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.mu = nn.Linear(512,latent_dim)
        self.cov = nn.Linear(512,latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512,25088),
            nn.BatchNorm1d(25088),
            nn.ReLU(),
            nn.Unflatten(1,(128,14,14)),
            nn.ConvTranspose2d(128,128,3,stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,3,stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,3,stride=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self,x):

        feat = self.encoder(x)
        
        face = self.softmax(F.relu(self.classification(feat)))
        mu = self.mu(feat)
        cov = self.cov(feat)
        
        # # Reparameterise
        std = torch.exp(0.5*cov)
        eps = torch.randn_like(std)
        z = mu + std * eps

        recon = self.decoder(z)
        
        return (z, mu, cov, face), recon
