#!/usr/bin/env python3
"""
Script de visualisation des résultats de benchmark
Génère les courbes de temps d'exécution et d'accélération
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration du style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 10

def load_and_process_data(filename):
    """Charge et traite les données de benchmark"""
    if not os.path.exists(filename):
        print(f"Fichier {filename} non trouvé")
        return None
    
    df = pd.read_csv(filename)
    return df

def calculate_speedup(df, time_col='mean_time', baseline_procs=1):
    """Calcule l'accélération par rapport à l'exécution séquentielle"""
    df = df.copy()
    
    # Trouver le temps de référence (1 processus ou le minimum disponible)
    if baseline_procs in df['num_procs'].values:
        baseline = df[df['num_procs'] == baseline_procs][time_col].values[0]
    else:
        baseline = df[time_col].max()  # Utiliser le plus grand temps comme référence
    
    df['speedup'] = baseline / df[time_col]
    df['efficiency'] = df['speedup'] / df['num_procs'] * 100
    return df

def plot_bucket_sort_results():
    """Génère les graphiques pour le Bucket Sort"""
    
    # Charger les données
    summary_file = 'results/bucket_sort_summary.csv'
    df = load_and_process_data(summary_file)
    
    if df is None:
        print("Impossible de générer les graphiques pour Bucket Sort")
        return
    
    # Grouper par taille de tableau
    sizes = df['array_size'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sizes)))
    
    for idx, size in enumerate(sorted(sizes)):
        subset = df[df['array_size'] == size].sort_values('num_procs')
        subset = calculate_speedup(subset)
        
        # Temps d'exécution
        axes[0].plot(subset['num_procs'], subset['mean_time'], 
                    marker='o', linewidth=2, markersize=8,
                    color=colors[idx], label=f'N = {size:,}')
        
        # Accélération
        axes[1].plot(subset['num_procs'], subset['speedup'], 
                    marker='s', linewidth=2, markersize=8,
                    color=colors[idx], label=f'N = {size:,}')
    
    # Ligne d'accélération idéale
    procs_range = df['num_procs'].unique()
    axes[1].plot(sorted(procs_range), sorted(procs_range), 
                'k--', linewidth=1, alpha=0.5, label='Idéal')
    
    # Configuration du premier graphique
    axes[0].set_xlabel('Nombre de processus', fontsize=12)
    axes[0].set_ylabel('Temps d\'exécution (s)', fontsize=12)
    axes[0].set_title('Temps d\'exécution - Bucket Sort', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Configuration du deuxième graphique
    axes[1].set_xlabel('Nombre de processus', fontsize=12)
    axes[1].set_ylabel('Accélération (Speedup)', fontsize=12)
    axes[1].set_title('Accélération - Bucket Sort', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/bucket_sort_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Graphique sauvegardé: results/bucket_sort_performance.png")

def plot_topk_results():
    """Génère les graphiques pour le Top-K"""
    
    summary_file = 'results/topk_summary.csv'
    df = load_and_process_data(summary_file)
    
    if df is None:
        print("Impossible de générer les graphiques pour Top-K")
        return
    
    # Grouper par valeur de K
    k_values = df['k'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(k_values)))
    
    for idx, k in enumerate(sorted(k_values)):
        subset = df[df['k'] == k].sort_values('num_procs')
        subset = calculate_speedup(subset)
        
        # Temps d'exécution
        axes[0].plot(subset['num_procs'], subset['mean_time'], 
                    marker='o', linewidth=2, markersize=8,
                    color=colors[idx], label=f'K = {k:,}')
        
        # Accélération
        axes[1].plot(subset['num_procs'], subset['speedup'], 
                    marker='s', linewidth=2, markersize=8,
                    color=colors[idx], label=f'K = {k:,}')
    
    # Ligne d'accélération idéale
    procs_range = df['num_procs'].unique()
    axes[1].plot(sorted(procs_range), sorted(procs_range), 
                'k--', linewidth=1, alpha=0.5, label='Idéal')
    
    # Configuration
    axes[0].set_xlabel('Nombre de processus', fontsize=12)
    axes[0].set_ylabel('Temps d\'exécution (s)', fontsize=12)
    axes[0].set_title('Temps d\'exécution - Top-K', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Nombre de processus', fontsize=12)
    axes[1].set_ylabel('Accélération (Speedup)', fontsize=12)
    axes[1].set_title('Accélération - Top-K', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/topk_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Graphique sauvegardé: results/topk_performance.png")

def plot_comparison():
    """Compare Bucket Sort et Top-K"""
    
    bucket_file = 'results/bucket_sort_summary.csv'
    topk_file = 'results/topk_summary.csv'
    
    df_bucket = load_and_process_data(bucket_file)
    df_topk = load_and_process_data(topk_file)
    
    if df_bucket is None or df_topk is None:
        print("Impossible de générer le graphique de comparaison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trouver une taille commune pour la comparaison
    # Utiliser la plus grande taille disponible pour bucket sort
    bucket_size = df_bucket['array_size'].max()
    
    df_bucket_subset = df_bucket[df_bucket['array_size'] == bucket_size].sort_values('num_procs')
    df_bucket_subset = calculate_speedup(df_bucket_subset)
    
    # Pour Top-K, utiliser K=1000 comme référence
    if 1000 in df_topk['k'].values:
        df_topk_subset = df_topk[df_topk['k'] == 1000].sort_values('num_procs')
    else:
        df_topk_subset = df_topk.groupby('num_procs').mean().reset_index()
    df_topk_subset = calculate_speedup(df_topk_subset)
    
    # Temps d'exécution
    axes[0].plot(df_bucket_subset['num_procs'], df_bucket_subset['mean_time'], 
                marker='o', linewidth=2, markersize=8,
                color='blue', label=f'Bucket Sort (N={bucket_size:,})')
    axes[0].plot(df_topk_subset['num_procs'], df_topk_subset['mean_time'], 
                marker='s', linewidth=2, markersize=8,
                color='red', label='Top-K (K=1000)')
    
    # Accélération
    axes[1].plot(df_bucket_subset['num_procs'], df_bucket_subset['speedup'], 
                marker='o', linewidth=2, markersize=8,
                color='blue', label='Bucket Sort')
    axes[1].plot(df_topk_subset['num_procs'], df_topk_subset['speedup'], 
                marker='s', linewidth=2, markersize=8,
                color='red', label='Top-K')
    
    # Ligne idéale
    procs_range = df_bucket_subset['num_procs'].unique()
    axes[1].plot(sorted(procs_range), sorted(procs_range), 
                'k--', linewidth=1, alpha=0.5, label='Idéal')
    
    # Configuration
    axes[0].set_xlabel('Nombre de processus', fontsize=12)
    axes[0].set_ylabel('Temps d\'exécution (s)', fontsize=12)
    axes[0].set_title('Comparaison des temps d\'exécution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Nombre de processus', fontsize=12)
    axes[1].set_ylabel('Accélération (Speedup)', fontsize=12)
    axes[1].set_title('Comparaison des accélérations', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Graphique sauvegardé: results/comparison.png")

def generate_efficiency_table():
    """Génère un tableau d'efficacité"""
    
    bucket_file = 'results/bucket_sort_summary.csv'
    df = load_and_process_data(bucket_file)
    
    if df is None:
        return
    
    print("\n=== Tableau d'Efficacité - Bucket Sort ===")
    print("-" * 60)
    
    for size in sorted(df['array_size'].unique()):
        subset = df[df['array_size'] == size].sort_values('num_procs')
        subset = calculate_speedup(subset)
        
        print(f"\nTaille du tableau: {size:,}")
        print(f"{'Processus':<12} {'Temps (s)':<12} {'Speedup':<12} {'Efficacité':<12}")
        print("-" * 48)
        
        for _, row in subset.iterrows():
            print(f"{int(row['num_procs']):<12} {row['mean_time']:<12.4f} {row['speedup']:<12.2f} {row['efficiency']:<12.1f}%")

def main():
    """Fonction principale"""
    
    print("=" * 50)
    print("Génération des graphiques de performance")
    print("=" * 50)
    print()
    
    # Créer le dossier de résultats s'il n'existe pas
    os.makedirs('results', exist_ok=True)
    
    # Générer les graphiques
    plot_bucket_sort_results()
    plot_topk_results()
    plot_comparison()
    
    # Afficher le tableau d'efficacité
    generate_efficiency_table()
    
    print()
    print("=" * 50)
    print("Génération terminée!")
    print("=" * 50)

if __name__ == "__main__":
    main()
