# Contenu de tp2_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_autoencoder(model, train_loader, n_epochs=50, lr=1e-3):
    """
    Fonction pour entraîner l'auto-encodeur classique.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Début de l'entraînement (AE) pour {n_epochs} époques...")
    start_time = time.time()
    
    model.train() # Mettre le modèle en mode entraînement
    for epoch in range(n_epochs):
        epoch_loss = 0
        for (inputs, targets) in train_loader:
            # inputs et targets sont X_train_tensor (données normales)
            
            # 1. Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 2. Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Époque [{epoch+1}/{n_epochs}], Perte (Loss): {epoch_loss/len(train_loader):.6f}")
    
    end_time = time.time()
    print(f"Entraînement terminé en {end_time - start_time:.2f} secondes.")
    
    return model # Retourne le modèle entraîné

def train_denoising_autoencoder(model, train_loader, n_epochs=50, lr=1e-3, noise_factor=0.2):
    """
    Fonction pour entraîner l'auto-encodeur débruiteur (DAE).
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Début de l'entraînement (DAE) pour {n_epochs} époques...")
    start_time = time.time()
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for (inputs, targets) in train_loader:
            # --- C'est la différence clé ---
            # 1. Ajouter du bruit aux entrées
            noise = torch.randn_like(inputs) * noise_factor
            noisy_inputs = inputs + noise
            
            # 2. Forward pass : reconstruire à partir du bruit
            outputs = model(noisy_inputs)
            
            # 3. Calculer la perte par rapport aux entrées ORIGINALES (targets)
            loss = criterion(outputs, targets)
            
            # 4. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Époque [{epoch+1}/{n_epochs}], Perte (Loss): {epoch_loss/len(train_loader):.6f}")
    
    end_time = time.time()
    print(f"Entraînement DAE terminé en {end_time - start_time:.2f} secondes.")
    
    return model