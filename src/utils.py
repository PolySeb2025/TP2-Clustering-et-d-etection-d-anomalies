# --- Imports nécessaires pour les fonctions ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
import umap
import warnings
import torch
from sklearn.metrics import precision_recall_curve
import time
import tracemalloc


def load_dataset(data_path, labels_path):
    """
    Charge les données Hi-Seq et leurs labels à partir de fichiers CSV.
    
    Args:
        data_path (str): Chemin vers le fichier hiseq_data.csv.
        labels_path (str): Chemin vers le fichier hiseq_labels.csv.
    
    Returns:
        tuple: (X, y) où X est le DataFrame des features et y est la Series des labels,
               alignés sur les mêmes indices. Retourne (None, None) en cas d'erreur.
    """
    # Chargement des variables indépendantes (expression génique)
    print("Chargement des données Hi-Seq data")
    try:
        X = pd.read_csv(data_path, index_col=0)
        print(f"Données Hi-Seq chargées, forme de X: {X.shape}")
    except FileNotFoundError:
        print(f"Erreur: Fichier de données non trouvé à {data_path}")
        return None, None
    
    # Chargement de la variable dépendante (labels de type de tumeur)
    print("Chargement des données Hi-Seq labels")
    try:
        y = pd.read_csv(labels_path, index_col=0)
        print(f"Données Hi-Seq labels chargées, forme de y: {y.shape}")
    except FileNotFoundError:
        print(f"Erreur: Fichier de labels non trouvé à {labels_path}")
        return None, None
    
    # Alignement des données sur les mêmes indices
    X, y = X.align(y, join='inner', axis=0)
    
    # Conversion du DataFrame de labels en Series
    y = y.iloc[:, 0]
    
    return X, y


################################################################################################################



def evaluate_clustering(X, labels, y_true, allow_noise=False):
    """
    Calcule un ensemble de métriques internes et externes pour évaluer
    les résultats d'un algorithme de clustering.
    
    Args:
        X (np.array): Les données originales (attributs).
        labels (np.array): Les étiquettes de cluster prédites par l'algorithme.
        y_true (np.array): Les étiquettes de vérité terrain (labels réels).
        allow_noise (bool): Mettre à True si l'algorithme (ex: DBSCAN) peut
                            produire des points de bruit (label -1).
    
    Returns:
        dict: Un dictionnaire contenant les scores des métriques.
    """
    
    X_filtered = X
    labels_filtered = labels
    y_true_filtered = y_true
    noise_percentage = 0.0

    if allow_noise:
        # Trouver les indices des points qui ne sont PAS du bruit (non -1)
        non_noise_indices = np.where(labels != -1)[0]
        
        if len(non_noise_indices) == 0:
            print("Avertissement : Tous les points ont été classés comme bruit.")
            return {
                "Noise_Percentage": 100.0,
                "Silhouette": np.nan,
                "Homogeneity": np.nan,
                "Completeness": np.nan,
                "V-Measure": np.nan,
                "ARI": np.nan
            }

        # Filtrer les données pour exclure le bruit
        X_filtered = X[non_noise_indices]
        labels_filtered = labels[non_noise_indices]
        y_true_filtered = y_true[non_noise_indices]
        
        # Calculer le pourcentage de bruit
        noise_percentage = 100 * (len(labels) - len(labels_filtered)) / len(labels)

    try:
        # --- Métrique Interne (sans vérité terrain) ---
        # Nécessite au moins 2 clusters uniques (après filtrage du bruit)
        if len(np.unique(labels_filtered)) > 1:
            silhouette = metrics.silhouette_score(X_filtered, labels_filtered)
        else:
            silhouette = np.nan # Pas calculable avec 1 seul cluster

        # --- Métriques Externes (avec vérité terrain) ---
        homogeneity = metrics.homogeneity_score(y_true_filtered, labels_filtered)
        completeness = metrics.completeness_score(y_true_filtered, labels_filtered)
        v_measure = metrics.v_measure_score(y_true_filtered, labels_filtered)
        ari = metrics.adjusted_rand_score(y_true_filtered, labels_filtered)
        
        results = {
            "Silhouette": silhouette,
            "Homogeneity": homogeneity,
            "Completeness": completeness,
            "V-Measure": v_measure,
            "ARI": ari
        }
        
        if allow_noise:
            results["Noise_Percentage"] = noise_percentage

        return results

    except Exception as e:
        print(f"Erreur lors de l'évaluation : {e}")
        return {}
    
    
################################################################################################################


def plot_clusters(X, labels, title="Visualisation des Clusters", reducer_method='umap'):
    """
    Effectue une réduction de dimension (UMAP ou PCA) sur les données
    et affiche un scatter plot des clusters.
    
    Args:
        X (np.array): Les données (haute dimension).
        labels (np.array): Les étiquettes de cluster prédites.
        title (str): Titre du graphique.
        reducer_method (str): 'umap' (défaut) ou 'pca' pour la réduction.
    """
    
    print(f"Réduction de dimension pour la visualisation avec {reducer_method}...")
    
    # Gérer les warnings de UMAP (prédictions sur 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if reducer_method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        elif reducer_method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError("reducer_method doit être 'umap' ou 'pca'")
            
        X_2d = reducer.fit_transform(X)

    # Créer un DataFrame pour Seaborn
    df_plot = pd.DataFrame({
        'comp_1': X_2d[:, 0],
        'comp_2': X_2d[:, 1],
        'cluster': labels.astype(str) # Convertir en str pour que -1 soit une catégorie
    })

    # Déterminer les labels uniques (clusters + possible bruit)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Créer une palette de couleurs
    # 'tab10' est bien pour < 10 clusters, 'Paired' ou 'hls' pour plus
    palette = sns.color_palette('tab10', n_colors=n_clusters)
    
    # Gérer le bruit (-1) : le mettre en gris
    if -1 in unique_labels:
        # Trouver la position de '-1' dans la DataFrame
        label_map = {label: i for i, label in enumerate(df_plot['cluster'].unique())}
        noise_index = label_map['-1']
        palette[noise_index] = (0.5, 0.5, 0.5) # Gris
        
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_plot,
        x='comp_1',
        y='comp_2',
        hue='cluster',
        palette=palette,
        s=50,
        alpha=0.7
    )
    
    plt.title(title)
    plt.xlabel(f"{reducer_method.upper()} Composante 1")
    plt.ylabel(f"{reducer_method.upper()} Composante 2")
    plt.legend(loc='best', title='Cluster')
    plt.grid(True)
    plt.show()

###############################################################################################################


def evaluate_model_threshold(model, loader, y_true):
        print("Calcul des erreurs et du seuil optimal...")
        model.eval()
        errors = []
        with torch.no_grad():
            for inputs, _ in loader:
                recon = model(inputs)
                batch_errors = torch.mean((inputs - recon) ** 2, dim=1)
                errors.extend(batch_errors.numpy())
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, errors)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_scores = f1_scores[:-1] # le dernier f1 est nan
        threshold_best = thresholds[np.nan_to_num(f1_scores).argmax()]
        return threshold_best, np.max(np.nan_to_num(f1_scores))

###############################################################################################################

def run_stochastic_protocol(model_factory, data, true_labels, n_runs, experiment_name, evaluate_clustering_func):
    """
    Exécute un protocole expérimental (boucle de N exécutions) pour un algorithme STOCHASTIQUE.
    
    Args:
        model_factory (function): Une fonction (ex: lambda seed: KMeans(n_clusters=5, random_state=seed))
                                  qui prend une 'seed' et retourne un modèle non entraîné.
        data (np.array): Les données (ex: X_pca_100).
        true_labels (pd.Series): Les vrais labels (ex: y_hiseq).
        n_runs (int): Le nombre d'exécutions (ex: 10).
        experiment_name (str): Nom de l'expérience (ex: "KMeans (PCA)").
        evaluate_clustering_func (function): La fonction evaluate_clustering à utiliser.
        
    Returns:
        pd.DataFrame: Un DataFrame contenant les métriques de toutes les exécutions.
    """
    print(f"--- Protocole expérimental : {experiment_name} ---")
    results_list = []

    for i in range(n_runs):
        print(f"  Exécution {i+1}/{n_runs} ({experiment_name})...")
        current_seed = i
        
        # 1. Créer le modèle en utilisant la "fabrique"
        model = model_factory(seed=current_seed)
        
        # 2. Mesures
        tracemalloc.start()
        start_time = time.time()
        
        # 3. Entraînement
        labels = model.fit_predict(data)
        
        # 4. Arrêt des mesures
        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 5. Évaluation (par défaut: allow_noise=False)
        metrics_run = evaluate_clustering_func(data, labels, true_labels)
        
        # 6. Ajout des métriques computationnelles
        metrics_run['Time (s)'] = end_time - start_time
        metrics_run['Memory (MB)'] = peak_mem / (1024 ** 2)
        metrics_run['Seed'] = current_seed
        
        results_list.append(metrics_run)

    print(f"Protocole {experiment_name} terminé.")
    
    # Renvoyer le DataFrame
    return pd.DataFrame(results_list)


###############################################################################################################



def print_protocol_summary(results_df, experiment_name):
    """
    Calcule et affiche la moyenne (μ) et l'écart-type (σ) d'un DataFrame de résultats.
    
    Args:
        results_df (pd.DataFrame): Le DataFrame généré par run_stochastic_protocol.
        experiment_name (str): Nom de l'expérience (ex: "KMeans (PCA)").
        
    Returns:
        pd.Series: Les métriques moyennes (pour le tableau final).
    """
    mean_stats = results_df.mean()
    std_stats = results_df.std()
    
    print(f"\n--- Résultats {experiment_name} - (μ ± σ) sur {len(results_df)} exécutions ---")
    
    for metric in mean_stats.index:
        if metric == 'Seed':
            continue
        # L'alignement '<18' aide à formater la sortie
        print(f"{metric+':':<18} {mean_stats[metric]:.4f} ± {std_stats[metric]:.4f}")
    
    # Renvoyer les moyennes pour le tableau récapitulatif
    return mean_stats.drop('Seed')


###############################################################################################################



def run_deterministic_protocol(model, data, true_labels, experiment_name, evaluate_clustering_func):
    """
    Exécute une seule évaluation pour un algorithme DÉTERMINISTE (comme DBSCAN).
    
    Args:
        model (object): Un modèle scikit-learn initialisé (ex: DBSCAN(eps=...)).
        data (np.array): Les données (ex: X_pca_100).
        true_labels (pd.Series): Les vrais labels (ex: y_hiseq).
        experiment_name (str): Nom de l'expérience (ex: "DBSCAN (PCA)").
        evaluate_clustering_func (function): La fonction evaluate_clustering à utiliser.
        
    Returns:
        pd.Series: Une série contenant les métriques de l'unique exécution.
    """
    print(f"--- Exécution : {experiment_name} ---")

    tracemalloc.start()
    start_time = time.time()
    
    labels = model.fit_predict(data)
    
    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # IMPORTANT: On utilise allow_noise=True pour DBSCAN
    metrics_run = evaluate_clustering_func(data, labels, true_labels, allow_noise=True)
    
    metrics_run['Time (s)'] = end_time - start_time
    metrics_run['Memory (MB)'] = peak_mem / (1024 ** 2)
    
    print(f"Exécution {experiment_name} terminée.")
    
    # Afficher le résumé
    results_series = pd.Series(metrics_run)
    print(f"\n--- Résultats {experiment_name} - (1 exécution) ---")
    print(results_series.to_markdown(floatfmt=".4f"))
    
    return results_series