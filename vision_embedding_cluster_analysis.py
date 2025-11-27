#!/usr/bin/env python3
"""
Script to generate random clustered data or load embeddings from pickle, use vision model to recognize clusters, and perform clustering.

This script:
1. Generates random 2D vectors OR loads embeddings (2D vectors) from a pickle file
2. Creates a KDE (Kernel Density Estimation) plot of the data
3. Uses vision model (from master_config.json lines 139-146) to analyze the KDE plot and count clusters
4. Tests silhouette scores in a search range to find the best K value
5. Performs clustering (K-means, HDBSCAN, Spectral, or GMM) using the best K value
6. Outputs the final clustered visualization

Usage:
    # Generate random data
    python vision_embedding_cluster_analysis.py
    python vision_embedding_cluster_analysis.py --method kmeans --output-dir my_outputs
    
    # Load embeddings from pickle file
    python vision_embedding_cluster_analysis.py --embeddings-pkl path/to/embeddings.pkl
    python vision_embedding_cluster_analysis.py --embeddings-pkl embeddings.pkl --method spectral --search-range 2
"""

import os
import sys
import json
import base64
import io
import re
import pickle
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import requests
from dotenv import load_dotenv

# Add the parent directory to the path so we can import cryoagent
current_dir = Path.cwd()
sys.path.insert(0, str(current_dir))

from cryoagent.config.config_loader import ModelConfig

# Load environment variables
load_dotenv()

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def _merge_configs(master_config: dict, session_config: dict) -> dict:
    """
    Merge session configuration into master configuration.
    Session config takes precedence for overlapping keys.
    """
    merged = master_config.copy()
    
    for key, value in session_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = _merge_configs(merged[key], value)
        else:
            # Session config takes precedence
            merged[key] = value
    
    return merged


def load_vision_model_config(config_path: str = "configs/master_config.json") -> ModelConfig:
    """
    Load vision model configuration from master_config.json.
    
    Args:
        config_path: Path to the master config file
        
    Returns:
        ModelConfig object for the vision model
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Load and merge session.json if it exists (session.json takes precedence)
    session_config_path = config_file.parent / "session.json"
    if session_config_path.exists():
        with open(session_config_path, 'r') as f:
            session_config = json.load(f)
        # Merge session config into master config (session config takes precedence)
        config_data = _merge_configs(config_data, session_config)
    
    # Resolve environment variables
    def resolve_env_vars(data):
        if isinstance(data, dict):
            return {key: resolve_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [resolve_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            var_name = data[2:-1]
            return os.environ.get(var_name, "")
        else:
            return data
    
    config_data = resolve_env_vars(config_data)
    
    # Get vision model config (defaulting to panshi)
    vision_models = config_data.get("agent", {}).get("vision_models", {})
    vision_provider = "panshi"  # Default to panshi as shown in config
    
    if vision_provider not in vision_models:
        # Try to get the first available vision model
        vision_provider = list(vision_models.keys())[0] if vision_models else None
    
    if not vision_provider or vision_provider not in vision_models:
        raise ValueError("No vision model configuration found in config file")
    
    vision_config = vision_models[vision_provider]
    
    return ModelConfig(
        api_key=vision_config.get("api_key", ""),
        base_url=vision_config.get("base_url", ""),
        model_name=vision_config.get("model_name", ""),
        temperature=vision_config.get("temperature", 0.1),
        timeout=vision_config.get("timeout", 60)
    )


def load_embeddings_from_pickle(pickle_path: str) -> np.ndarray:
    """
    Load embeddings (2D vectors) from a pickle file.
    
    The pickle file can contain:
    - A numpy array directly (shape: [n_points, 2])
    - A dictionary with 'embeddings' or 'data' key
    - Any other structure that can be converted to a 2D numpy array
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        2D numpy array of shape (n_points, 2)
        
    Raises:
        FileNotFoundError: If pickle file doesn't exist
        ValueError: If data cannot be converted to 2D array
    """
    pickle_file = Path(pickle_path)
    if not pickle_file.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    print(f"Loading embeddings from {pickle_path}...")
    
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different possible formats
    if isinstance(data, np.ndarray):
        embeddings = data
    elif isinstance(data, dict):
        # Try common keys
        if 'embeddings' in data:
            embeddings = data['embeddings']
        elif 'data' in data:
            embeddings = data['data']
        elif 'vectors' in data:
            embeddings = data['vectors']
        elif 'points' in data:
            embeddings = data['points']
        else:
            # Try to use the first array-like value
            array_values = [v for v in data.values() if isinstance(v, np.ndarray)]
            if array_values:
                embeddings = array_values[0]
            else:
                raise ValueError(f"Could not find embeddings in dictionary. Available keys: {list(data.keys())}")
    elif isinstance(data, (list, tuple)):
        embeddings = np.array(data)
    else:
        raise ValueError(f"Unsupported data type in pickle file: {type(data)}")
    
    # Convert to numpy array if not already
    embeddings = np.asarray(embeddings)
    
    # Ensure it's 2D
    if embeddings.ndim == 1:
        # If 1D, assume it needs reshaping (unlikely for 2D vectors, but handle it)
        raise ValueError(f"Embeddings are 1D (shape: {embeddings.shape}). Expected 2D array [n_points, 2]")
    elif embeddings.ndim == 2:
        if embeddings.shape[1] != 2:
            # If more than 2 dimensions, try to reduce or raise error
            if embeddings.shape[1] > 2:
                print(f"Warning: Embeddings have {embeddings.shape[1]} dimensions. Using first 2 dimensions.")
                embeddings = embeddings[:, :2]
            else:
                raise ValueError(f"Embeddings must have 2 dimensions (shape: {embeddings.shape}). Expected [n_points, 2]")
    else:
        raise ValueError(f"Embeddings have {embeddings.ndim} dimensions. Expected 2D array [n_points, 2]")
    
    print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    return embeddings


def generate_random_clustered_data(
    n_points: int = 10000,
    n_clusters_range: Tuple[int, int] = (3, 6),
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Generate random 2D data points in random number of clusters.
    
    Args:
        n_points: Total number of data points
        n_clusters_range: Range (min, max) for number of clusters
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (data_points, true_n_clusters)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Randomly choose number of clusters
    min_clusters, max_clusters = n_clusters_range
    true_n_clusters = np.random.randint(min_clusters, max_clusters + 1)
    
    print(f"Generating {n_points} points in {true_n_clusters} clusters...")
    
    # Generate cluster centers
    cluster_centers = np.random.uniform(-10, 10, size=(true_n_clusters, 2))
    
    # Generate points for each cluster
    points_per_cluster = n_points // true_n_clusters
    remainder = n_points % true_n_clusters
    
    all_points = []
    for i in range(true_n_clusters):
        n_points_in_cluster = points_per_cluster + (1 if i < remainder else 0)
        
        # Generate points around cluster center with some spread
        center = cluster_centers[i]
        spread = np.random.uniform(1.0, 3.0)  # Random spread for each cluster
        
        cluster_points = np.random.normal(
            loc=center,
            scale=spread,
            size=(n_points_in_cluster, 2)
        )
        all_points.append(cluster_points)
    
    data = np.vstack(all_points)
    
    # Shuffle the points
    indices = np.random.permutation(len(data))
    data = data[indices]
    
    return data, true_n_clusters


def create_scatter_plot(data: np.ndarray, output_path: str) -> str:
    """
    Create a scatter plot of the data and save as image.
    
    Args:
        data: 2D array of data points
        output_path: Path to save the plot image
        
    Returns:
        Path to the saved image
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, c='blue', edgecolors='none')
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Scatter Plot of Data", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plot saved to {output_path}")
    return output_path


def create_kde_plot(data: np.ndarray, output_path: str) -> str:
    """
    Create a KDE (Kernel Density Estimation) plot of the data and save as image.
    
    Args:
        data: 2D array of data points
        output_path: Path to save the plot image
        
    Returns:
        Path to the saved image
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create KDE plot with seaborn
    sns.kdeplot(
        x=data[:, 0],
        y=data[:, 1],
        fill=True,
        cmap="viridis",
        alpha=0.7,
        levels=20,
        thresh=0.05,
        ax=ax
    )
    
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title("KDE Distribution Plot of Data", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar using the collection from the kdeplot
    # The kdeplot creates PolyCollection objects, use the last one for colorbar
    if ax.collections:
        plt.colorbar(ax.collections[-1], ax=ax, label="Density")
    
    plt.tight_layout()
    
    # Save to file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"KDE plot saved to {output_path}")
    return output_path


def create_combined_plot(data: np.ndarray, output_path: str) -> str:
    """
    Create a combined plot with KDE density contours and scatter points overlaid (alpha=0.1).
    
    Args:
        data: 2D array of data points
        output_path: Path to save the plot image
        
    Returns:
        Path to the saved image
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # First, create KDE plot with seaborn
    sns.kdeplot(
        x=data[:, 0],
        y=data[:, 1],
        fill=True,
        cmap="viridis",
        alpha=0.7,
        levels=20,
        thresh=0.05,
        ax=ax
    )
    
    # Overlay scatter points with low alpha
    ax.scatter(
        data[:, 0],
        data[:, 1],
        c='black',
        s=20,
        alpha=0.1,
        edgecolors='none'
    )
    
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title("Combined KDE and Scatter Plot", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar using the collection from the kdeplot
    if ax.collections:
        plt.colorbar(ax.collections[-1], ax=ax, label="Density")
    
    plt.tight_layout()
    
    # Save to file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved to {output_path}")
    return output_path


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ask_vision_model_count_clusters_from_image(
    image_path: str,
    model_config: ModelConfig,
    plot_type: str = "scatter"
) -> int:
    """
    Use vision model to analyze a plot and count number of clusters.
    
    Args:
        image_path: Path to the plot image
        model_config: Vision model configuration
        plot_type: Type of plot ("scatter" or "kde")
        
    Returns:
        Number of clusters detected by the vision model
    """
    plot_name = "scatter plot" if plot_type == "scatter" else "KDE plot"
    print(f"\nSending {plot_name} to vision model for cluster analysis...")
    
    # Encode image to base64
    base64_image = image_to_base64(image_path)
    
    # Prepare the prompt based on plot type
    if plot_type == "scatter":
        prompt = """Look at this scatter plot carefully. Count the number of distinct clusters or groups of data points you can see. 
Consider clusters as groups of points that are clearly separated from other groups.

Please respond with ONLY a single number representing the number of clusters you see. 
For example, if you see 4 clusters, respond with just: 4"""
    else:  # kde
        prompt = """Look at this KDE (Kernel Density Estimation) plot carefully. The plot shows density contours and colored regions representing different clusters of data points.

Count the number of distinct clusters or density peaks you can see. Each cluster appears as a separate colored region or density peak in the plot.

Please respond with ONLY a single number representing the number of clusters you see. 
For example, if you see 4 clusters, respond with just: 4"""

    try:
        chat_url = f"{model_config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": model_config.temperature,
            "max_tokens": 50
        }
        
        response = requests.post(
            chat_url,
            headers=headers,
            json=payload,
            timeout=model_config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()
            print(f"Vision model response from {plot_name}: {response_text}")
            
            # Extract number from response
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                detected_k = int(numbers[0])
                print(f"Detected number of clusters from {plot_name}: {detected_k}")
                return detected_k
            else:
                print(f"Warning: Could not extract number from response: {response_text}")
                print(f"Defaulting to K=4 for {plot_name}")
                return 4
        else:
            print(f"Error: API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            print(f"Defaulting to K=4 for {plot_name}")
            return 4
            
    except Exception as e:
        print(f"Error calling vision model for {plot_name}: {e}")
        print(f"Defaulting to K=4 for {plot_name}")
        return 4


def ask_vision_model_count_clusters_from_combined_plot(
    image_path: str,
    scatter_k: int,
    kde_k: int,
    model_config: ModelConfig
) -> int:
    """
    Send combined plot (scatter overlaid on KDE) to vision model to determine optimal number of clusters.
    
    Args:
        image_path: Path to the combined plot image
        scatter_k: Number of clusters detected from scatter plot
        kde_k: Number of clusters detected from KDE plot
        model_config: Vision model configuration
        
    Returns:
        Optimal number of clusters detected by the vision model from the combined plot
    """
    print(f"\nSending combined plot (scatter overlaid on KDE) to vision model...")
    print(f"Previous results: scatter={scatter_k} clusters, KDE={kde_k} clusters")
    
    # Encode image to base64
    base64_image = image_to_base64(image_path)
    
    prompt = f"""Look at this combined plot carefully. This plot shows both:
1. KDE (Kernel Density Estimation) density contours in the background
2. Scatter points overlaid on top (with low opacity)

I previously analyzed this data using two separate visualizations:
- Scatter plot detected: {scatter_k} clusters
- KDE (density) plot detected: {kde_k} clusters

These two methods gave different results. Now, looking at this combined visualization that shows both the density distribution and the actual data points together, please determine the optimal number of clusters.

Count the number of distinct clusters you can see in this combined plot. Consider clusters as groups of points that are clearly separated from other groups, taking into account both the density patterns and the point distribution.

Please respond with ONLY a single number representing the number of clusters you see. 
For example, if you see 4 clusters, respond with just: 4"""

    try:
        chat_url = f"{model_config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {model_config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": model_config.temperature,
            "max_tokens": 50
        }
        
        response = requests.post(
            chat_url,
            headers=headers,
            json=payload,
            timeout=model_config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"].strip()
            print(f"Vision model response from combined plot: {response_text}")
            
            # Extract number from response
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                detected_k = int(numbers[0])
                print(f"Detected number of clusters from combined plot: {detected_k}")
                return detected_k
            else:
                print(f"Warning: Could not extract number from response: {response_text}")
                print(f"Defaulting to scatter result: {scatter_k}")
                return scatter_k
        else:
            print(f"Error: API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            print(f"Defaulting to scatter result: {scatter_k}")
            return scatter_k
            
    except Exception as e:
        print(f"Error calling vision model: {e}")
        print(f"Defaulting to scatter result: {scatter_k}")
        return scatter_k


def test_silhouette_scores(data: np.ndarray, n: int, method: str = "kmeans", search_range: int = 1) -> Tuple[int, float]:
    """
    Test silhouette scores for clusters in the range [n-search_range, ..., n, ..., n+search_range] and return the best K.
    
    Args:
        data: 2D array of data points
        n: Base number of clusters (from vision model)
        method: Clustering method ("kmeans", "hdbscan", "spectral", or "gmm")
        search_range: Range around n to search (default: 0, meaning only n)
        
    Returns:
        Tuple of (best_k, best_score)
    """
    # Generate k_values in the range [n-search_range, ..., n, ..., n+search_range]
    k_values = list(range(max(2, n - search_range), n + search_range + 1))
    k_values = [k for k in k_values if k >= 2]  # Ensure at least 2 clusters
    
    if search_range == 0:
        k_str = f"K={n}"
    else:
        k_str = f"K={k_values[0]} to K={k_values[-1]}"
    
    print(f"\nTesting silhouette scores for {k_str} using {method.upper()}...")
    
    scores = {}
    
    for k in k_values:
        try:
            if method.lower() == "kmeans":
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
            elif method.lower() == "spectral":
                spectral = SpectralClustering(
                    n_clusters=k,
                    random_state=42,
                    affinity='rbf',
                    gamma=1.0,
                    assign_labels='kmeans'
                )
                labels = spectral.fit_predict(data)
            elif method.lower() == "hdbscan":
                # For HDBSCAN, we need to guide it to find approximately k clusters
                n_points = len(data)
                avg_cluster_size = n_points / k
                min_cluster_size = max(10, int(avg_cluster_size / 3))
                min_samples = max(5, int(min_cluster_size / 2))
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(data)
                
                # HDBSCAN may find different number of clusters, skip if too different
                unique_labels = np.unique(labels[labels != -1])
                if len(unique_labels) < 2:
                    print(f"  K={k}: Skipped (HDBSCAN found < 2 clusters)")
                    continue
            elif method.lower() == "gmm" or method.lower() == "gaussian_mixture":
                gmm = GaussianMixture(
                    n_components=k,
                    random_state=42,
                    covariance_type='full',
                    max_iter=100
                )
                labels = gmm.fit_predict(data)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Calculate silhouette score (need at least 2 clusters and no single cluster)
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
            
            if len(unique_labels) < 2:
                print(f"  K={k}: Skipped (found < 2 clusters)")
                continue
            
            # For silhouette score, we need to handle noise points in HDBSCAN
            # Only calculate score on non-noise points
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) < 2:
                print(f"  K={k}: Skipped (too few non-noise points)")
                continue
            
            score = silhouette_score(data[non_noise_mask], labels[non_noise_mask])
            scores[k] = score
            print(f"  K={k}: Silhouette score = {score:.4f}")
            
        except Exception as e:
            print(f"  K={k}: Error - {e}")
            continue
    
    if not scores:
        print("  Warning: No valid scores computed, using original K")
        return n, 0.0
    
    # Find the K with the highest silhouette score
    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    
    print(f"\nBest K: {best_k} (Silhouette score: {best_score:.4f})")
    return best_k, best_score


def perform_kmeans_clustering(data: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-means clustering on the data.
    
    Args:
        data: 2D array of data points
        n_clusters: Number of clusters (K)
        
    Returns:
        Tuple of (cluster_labels, cluster_centers)
    """
    print(f"\nPerforming K-means clustering with K={n_clusters}...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    cluster_centers = kmeans.cluster_centers_
    
    return cluster_labels, cluster_centers


def perform_spectral_clustering(data: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Spectral Clustering on the data.
    
    Spectral clustering uses the spectrum (eigenvalues) of the similarity matrix
    to perform dimensionality reduction before clustering.
    
    Args:
        data: 2D array of data points
        n_clusters: Number of clusters (K)
        
    Returns:
        Tuple of (cluster_labels, cluster_centers)
        Note: Cluster centers are computed as the mean of each cluster's points.
    """
    print(f"\nPerforming Spectral Clustering with K={n_clusters}...")
    
    # Perform Spectral Clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        random_state=42,
        affinity='rbf',  # Radial Basis Function kernel
        gamma=1.0,  # Kernel coefficient
        assign_labels='kmeans'  # How to assign labels in the embedding space
    )
    cluster_labels = spectral.fit_predict(data)
    
    # Calculate cluster centers as the mean of points in each cluster
    unique_labels = np.unique(cluster_labels)
    cluster_centers = []
    for label in unique_labels:
        cluster_points = data[cluster_labels == label]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)
    
    cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([]).reshape(0, 2)
    
    return cluster_labels, cluster_centers


def perform_gaussian_mixture_clustering(data: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Gaussian Mixture Model (GMM) clustering on the data.
    
    GMM assumes data points are generated from a mixture of Gaussian distributions.
    It uses the EM algorithm to fit the model and assign points to clusters.
    
    Args:
        data: 2D array of data points
        n_clusters: Number of clusters (components)
        
    Returns:
        Tuple of (cluster_labels, cluster_centers)
        Note: Cluster centers are the means of the Gaussian components.
    """
    print(f"\nPerforming Gaussian Mixture Model clustering with K={n_clusters}...")
    
    # Perform GMM clustering
    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=42,
        covariance_type='full',  # Full covariance matrix for each component
        max_iter=100,
        tol=1e-3
    )
    cluster_labels = gmm.fit_predict(data)
    cluster_centers = gmm.means_  # Use the means of the Gaussian components
    
    return cluster_labels, cluster_centers


def perform_hdbscan_clustering(data: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform HDBSCAN clustering on the data.
    
    HDBSCAN is a density-based clustering algorithm that doesn't require a fixed K.
    We use the detected K to guide the min_cluster_size parameter.
    
    Args:
        data: 2D array of data points
        n_clusters: Number of clusters detected (used to guide min_cluster_size)
        
    Returns:
        Tuple of (cluster_labels, cluster_centers)
        Note: HDBSCAN may assign -1 to noise points, and cluster_centers are computed
        as the mean of each cluster's points.
    """
    print(f"\nPerforming HDBSCAN clustering (guided by detected K={n_clusters})...")
    
    # Calculate min_cluster_size based on detected K
    # Use approximately 1/3 to 1/2 of the average cluster size
    n_points = len(data)
    avg_cluster_size = n_points / n_clusters
    min_cluster_size = max(10, int(avg_cluster_size / 3))
    min_samples = max(5, int(min_cluster_size / 2))
    
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples}")
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(data)
    
    # Calculate cluster centers as the mean of points in each cluster
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise points
    
    cluster_centers = []
    for label in unique_labels:
        cluster_points = data[cluster_labels == label]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)
    
    cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([]).reshape(0, 2)
    
    # Report results
    n_clusters_found = len(unique_labels)
    n_noise = np.sum(cluster_labels == -1)
    print(f"  HDBSCAN found {n_clusters_found} clusters")
    if n_noise > 0:
        print(f"  {n_noise} points classified as noise")
    
    return cluster_labels, cluster_centers


def create_final_clustered_plot(
    data: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    output_path: str,
    method: str = "hdbscan"
):
    """
    Create final visualization with clusters colored.
    
    Args:
        data: 2D array of data points
        cluster_labels: Cluster assignment for each point (may include -1 for noise)
        cluster_centers: Cluster center coordinates
        output_path: Path to save the final plot
        method: Clustering method used ("kmeans", "hdbscan", "spectral", or "gmm")
    """
    plt.figure(figsize=(12, 10))
    
    # Separate noise points from clustered points (only for HDBSCAN)
    noise_mask = cluster_labels == -1
    clustered_mask = ~noise_mask
    
    # Plot noise points in gray (only for HDBSCAN)
    if method.lower() == "hdbscan" and np.any(noise_mask):
        plt.scatter(
            data[noise_mask, 0],
            data[noise_mask, 1],
            c='gray',
            s=10,
            alpha=0.3,
            edgecolors='none',
            label='Noise',
            zorder=1
        )
    
    # Plot clustered points
    if method.lower() == "hdbscan" and np.any(clustered_mask):
        scatter = plt.scatter(
            data[clustered_mask, 0],
            data[clustered_mask, 1],
            c=cluster_labels[clustered_mask],
            cmap='tab10',
            s=20,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.3,
            zorder=2
        )
    else:  # K-means, Spectral, or GMM (no noise points)
        scatter = plt.scatter(
            data[:, 0],
            data[:, 1],
            c=cluster_labels,
            cmap='tab10',
            s=20,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.3,
            zorder=2
        )
    
    # Plot cluster centers if available
    if len(cluster_centers) > 0:
        plt.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            c='red',
            marker='x',
            s=300,
            linewidths=4,
            label='Cluster Centers',
            zorder=10
        )
    
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    method_name = method.upper()
    plt.title(f"Final Clustered Result ({method_name}: {n_clusters} clusters)", fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, label='Cluster')
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Final clustered plot saved to {output_path}")


def main():
    """Main function to run the complete workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate random clustered data, use vision model to detect clusters, and perform clustering"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10000,
        help="Number of data points to generate (default: 10000)"
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=3,
        help="Minimum number of clusters (default: 3)"
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=6,
        help="Maximum number of clusters (default: 6)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/master_config.json",
        help="Path to master config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["kmeans", "hdbscan", "spectral", "gmm"],
        default="kmeans",
        help="Clustering method to use for final clustering: 'kmeans', 'hdbscan', 'spectral', or 'gmm' (default: kmeans)"
    )
    parser.add_argument(
        "--silhouette-method",
        type=str,
        choices=["kmeans", "hdbscan", "spectral", "gmm"],
        default=None,
        help="Clustering method to use for silhouette score testing. If not specified, uses the same as --method"
    )
    parser.add_argument(
        "--search-range",
        type=int,
        default=0,
        metavar="N",
        help="Range around detected K to search for best cluster count. Tests K-N to K+N. Must be >= 0 (default: 0)"
    )
    parser.add_argument(
        "--embeddings-pkl",
        type=str,
        default=None,
        help="Path to pickle file containing embeddings (2D vectors). If provided, uses this instead of generating random data"
    )
    
    args = parser.parse_args()
    
    # Validate search_range
    if args.search_range < 0:
        print("Error: --search-range must be >= 0")
        return 1
    
    # Set silhouette_method to method if not specified
    if args.silhouette_method is None:
        args.silhouette_method = args.method
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Vision Model Cluster Recognition and {args.method.upper()} Clustering")
    if args.silhouette_method != args.method:
        print(f"Silhouette testing using: {args.silhouette_method.upper()}")
    print("=" * 60)
    
    # Step 1: Load embeddings from pickle or generate random clustered data
    if args.embeddings_pkl:
        print("\n[Step 1] Loading embeddings from pickle file...")
        try:
            data = load_embeddings_from_pickle(args.embeddings_pkl)
            true_n_clusters = None  # Unknown when loading from pickle
            print(f"Loaded {len(data)} embeddings from pickle file")
        except Exception as e:
            print(f"Error loading embeddings from pickle: {e}")
            return 1
    else:
        print("\n[Step 1] Generating random clustered data...")
        data, true_n_clusters = generate_random_clustered_data(
            n_points=args.n_points,
            n_clusters_range=(args.min_clusters, args.max_clusters),
            random_seed=args.random_seed
        )
        print(f"Generated {len(data)} points in {true_n_clusters} clusters (ground truth)")
    
    # Step 2: Create scatter plot (for reference)
    print("\n[Step 2] Creating scatter plot...")
    scatter_plot_path = os.path.join(args.output_dir, "initial_scatter_plot.png")
    create_scatter_plot(data, scatter_plot_path)
    
    # Step 3: Load vision model config
    print("\n[Step 3] Loading vision model configuration...")
    try:
        model_config = load_vision_model_config(args.config)
        print(f"Model: {model_config.model_name}")
        print(f"Base URL: {model_config.base_url}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Step 4: Send scatter plot to vision model in parallel with KDE plot generation
    print("\n[Step 4] Sending scatter plot to vision model and generating KDE plot in parallel...")
    kde_plot_path = os.path.join(args.output_dir, "kde_distribution_plot.png")
    
    # Use ThreadPoolExecutor to run scatter plot sending and KDE generation in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit scatter plot analysis (non-blocking)
        scatter_future = executor.submit(
            ask_vision_model_count_clusters_from_image,
            scatter_plot_path,
            model_config,
            "scatter"
        )
        
        # Submit KDE plot generation (non-blocking)
        kde_gen_future = executor.submit(create_kde_plot, data, kde_plot_path)
        
        # Wait for both to complete
        scatter_k = scatter_future.result()
        kde_gen_future.result()  # Wait for KDE plot to be generated
    
    # Step 5: Send KDE plot to vision model
    print("\n[Step 5] Sending KDE plot to vision model...")
    kde_k = ask_vision_model_count_clusters_from_image(
        kde_plot_path, model_config, "kde"
    )
    
    # Step 6: Compare responses and determine final K
    print("\n[Step 6] Comparing results from scatter and KDE plots...")
    print(f"Scatter plot detected: {scatter_k} clusters")
    print(f"KDE plot detected: {kde_k} clusters")
    
    if scatter_k == kde_k:
        print(f"✓ Both plots agree: {scatter_k} clusters")
        detected_k = scatter_k
    else:
        print(f"⚠ Disagreement detected between scatter ({scatter_k}) and KDE ({kde_k})")
        print("Creating combined plot (scatter overlaid on KDE) and sending to vision model...")
        
        # Create combined plot
        combined_plot_path = os.path.join(args.output_dir, "combined_plot.png")
        create_combined_plot(data, combined_plot_path)
        
        # Send combined plot to vision model for final decision
        detected_k = ask_vision_model_count_clusters_from_combined_plot(
            combined_plot_path, scatter_k, kde_k, model_config
        )
    
    print(f"\nFinal detected K: {detected_k} clusters")
    
    # Step 7: Test silhouette scores in the search range and select best K
    print("\n[Step 7] Testing silhouette scores to refine cluster count...")
    if args.silhouette_method != args.method:
        print(f"  Using {args.silhouette_method.upper()} for silhouette testing")
    if args.search_range > 0:
        print(f"  Search range: ±{args.search_range} around detected K={detected_k}")
    best_k, best_score = test_silhouette_scores(
        data, detected_k, method=args.silhouette_method, search_range=args.search_range
    )
    
    # Step 8: Perform clustering with best K
    print(f"\n[Step 8] Performing {args.method.upper()} clustering with K={best_k}...")
    if args.method.lower() == "kmeans":
        cluster_labels, cluster_centers = perform_kmeans_clustering(data, best_k)
    elif args.method.lower() == "spectral":
        cluster_labels, cluster_centers = perform_spectral_clustering(data, best_k)
    elif args.method.lower() == "gmm" or args.method.lower() == "gaussian_mixture":
        cluster_labels, cluster_centers = perform_gaussian_mixture_clustering(data, best_k)
    else:  # hdbscan
        cluster_labels, cluster_centers = perform_hdbscan_clustering(data, best_k)
    
    # Step 9: Create final visualization
    print("\n[Step 9] Creating final clustered visualization...")
    final_plot_path = os.path.join(args.output_dir, "final_clustered_result.png")
    create_final_clustered_plot(data, cluster_labels, cluster_centers, final_plot_path, method=args.method)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_clusters_found = len(np.unique(cluster_labels[cluster_labels != -1]))
    n_noise = np.sum(cluster_labels == -1) if args.method.lower() == "hdbscan" else 0
    
    if true_n_clusters is not None:
        print(f"True number of clusters: {true_n_clusters}")
    print(f"Vision model detected: {detected_k} clusters (from scatter + KDE analysis)")
    print(f"Best K (from silhouette test using {args.silhouette_method.upper()}): {best_k} (score: {best_score:.4f})")
    print(f"Final clustering ({args.method.upper()}) found: {n_clusters_found} clusters")
    if n_noise > 0:
        print(f"Noise points: {n_noise}")
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("  - initial_scatter_plot.png: Original scatter plot")
    print("  - kde_distribution_plot.png: KDE plot (sent to vision model)")
    if scatter_k != kde_k:
        print("  - combined_plot.png: Combined plot (scatter overlaid on KDE, sent to vision model)")
    print("  - final_clustered_result.png: Final clustered result")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
