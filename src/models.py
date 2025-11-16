# Contenu de tp2_models.py
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        """
        Architecture de l'Auto-encodeur classique.
        Inspiré de la p.60 du Thème 5.
        """
        super(Autoencoder, self).__init__()
        
        # 140 -> 64 -> 32 -> 16 (espace latent)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 16 -> 32 -> 64 -> 140 (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
            # Pas de ReLU finale pour permettre des valeurs négatives (données normalisées)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, noise_factor=0.2):
        """
        Architecture de l'Auto-encodeur débruiteur (DAE).
        Identique à l'AE classique, mais on ajoute du bruit.
        """
        super(DenoisingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.noise_factor = noise_factor
        
        # L'encodeur et le décodeur peuvent être identiques à l'AE
        # 140 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 16 -> 32 -> 64 -> 140
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        # Le DAE ne prend pas en charge le bruit dans son forward
        # Le bruit est ajouté manuellement dans la boucle d'entraînement
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon