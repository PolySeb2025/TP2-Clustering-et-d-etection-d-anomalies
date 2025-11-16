from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd

################################################################################################################

def normalize_data(X):
    """
    Normalise les données en utilisant StandardScaler.

    Args:
        X (pd.DataFrame): DataFrame des features.

    Returns:
        pd.DataFrame: DataFrame normalisé.
    """
    print("Normalisation des données avec StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    return X_scaled_df


################################################################################################################


def reduce_dimension_for_viz(X, method='UMAP', n_components=2, random_state=42):
    """
    Réduit la dimension des données pour la visualisation.

    Args:
        X (pd.DataFrame): DataFrame des features.
        method (str): Méthode de réduction ('PCA', 'UMAP', 't-SNE').
        n_components (int): Nombre de composantes à conserver.

    Returns:
        pd.DataFrame: DataFrame des features réduites.
    """
    print(f"Réduction de dimension avec {method} à {n_components} composantes...")
    
    if method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, verbose=False)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, random_state=random_state, n_jobs=-1)
    else:
        print(method)
        raise ValueError("Méthode de réduction non supportée")

    X_reduced = reducer.fit_transform(X)
    
    # Création d'un DataFrame pour la visualisation
    cols = [f'{method}_C{i+1}' for i in range(n_components)]
    X_reduced_df = pd.DataFrame(X_reduced, index=X.index, columns=cols)
    return X_reduced_df


################################################################################################################


def reduce_dimension_for_clustering(data, method='PCA', n_components=100):
    """
    Applique une réduction de dimension (PCA ou UMAP) sur les données.
    
    Args:
        data (np.array): Les données d'entrée (X_scaled).
        method (str): 'PCA' ou 'UMAP'.
        n_components (int): Le nombre de dimensions cibles.

    Returns:
        np.array: Les données avec dimensions réduites.
    """
    
    print(f"Début de la réduction de dimension avec {method} à {n_components} composantes...")
    
    if method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
    
    elif method == 'UMAP':
        # Vous devez avoir 'umap-learn' d'installé (pip install umap-learn)
        reducer = umap.UMAP(n_components=n_components, 
                            random_state=42, 
                            n_neighbors=15, 
                            min_dist=0.1)
    else:
        raise ValueError("Méthode non supportée. Choisissez 'PCA' ou 'UMAP'.")

    X_reduced = reducer.fit_transform(data)
    
    print(f"Réduction {method} terminée. Nouvelle forme : {X_reduced.shape}")
    
    return X_reduced