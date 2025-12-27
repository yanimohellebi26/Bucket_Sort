#!/usr/bin/env python3
"""
Script de visualisation des résultats - Version 2 (MPI + OpenMP)
Génère des graphiques pour analyser les performances hybrides
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration des chemins
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

# Style des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def load_data(filename):
    """Charge un fichier CSV depuis le répertoire des résultats"""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def plot_hybrid_comparison():
    """Compare les différentes configurations hybrides MPI/OpenMP"""
    df = load_data('hybrid_comparison.csv')
    if df is None:
        print("Fichier hybrid_comparison.csv non trouvé")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Temps d'exécution par configuration
    ax1 = axes[0]
    
    sizes = df['size'].unique()
    configs = df['config'].unique()
    
    x = np.arange(len(sizes))
    width = 0.15
    
    for i, config in enumerate(configs[:6]):  # Max 6 configurations
        config_data = df[df['config'] == config]
        times = [config_data[config_data['size'] == s]['avg_time'].values[0] 
                 if len(config_data[config_data['size'] == s]) > 0 else 0 
                 for s in sizes]
        ax1.bar(x + i * width, times, width, label=config, color=COLORS[i % len(COLORS)])
    
    ax1.set_xlabel('Taille du tableau')
    ax1.set_ylabel('Temps (secondes)')
    ax1.set_title('Temps d\'exécution par configuration hybride')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels([f'{s:,}'.replace(',', ' ') for s in sizes])
    ax1.legend(title='Configuration')
    ax1.set_yscale('log')
    
    # Graphique 2: Speedup par configuration
    ax2 = axes[1]
    
    for i, config in enumerate(configs[:6]):
        config_data = df[df['config'] == config]
        speedups = [config_data[config_data['size'] == s]['speedup'].values[0] 
                    if len(config_data[config_data['size'] == s]) > 0 else 1 
                    for s in sizes]
        ax2.plot(sizes, speedups, 'o-', label=config, color=COLORS[i % len(COLORS)], linewidth=2)
    
    ax2.set_xlabel('Taille du tableau')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup par rapport à la version séquentielle')
    ax2.set_xscale('log')
    ax2.legend(title='Configuration')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hybrid_comparison.png'), dpi=150, bbox_inches='tight')
    print("Graphique sauvegardé: hybrid_comparison.png")

def plot_mpi_vs_omp():
    """Compare l'approche MPI pure vs OpenMP pure vs hybride"""
    df = load_data('hybrid_comparison.csv')
    if df is None:
        print("Fichier hybrid_comparison.csv non trouvé")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filtrer les configurations intéressantes
    # MPI pur: *MPI_1OMP
    # OpenMP pur: 1MPI_*OMP
    # Hybride: reste
    
    largest_size = df['size'].max()
    df_large = df[df['size'] == largest_size]
    
    # Catégoriser les configurations
    categories = []
    for _, row in df_large.iterrows():
        config = row['config']
        mpi = row['mpi_procs']
        omp = row['omp_threads']
        
        if omp == 1:
            cat = 'MPI pur'
        elif mpi == 1:
            cat = 'OpenMP pur'
        else:
            cat = 'Hybride'
        categories.append(cat)
    
    df_large = df_large.copy()
    df_large['category'] = categories
    
    # Trier par nombre total de threads
    df_large = df_large.sort_values('total_threads')
    
    # Créer le graphique
    cat_colors = {'MPI pur': COLORS[0], 'OpenMP pur': COLORS[1], 'Hybride': COLORS[2]}
    
    for cat in ['MPI pur', 'OpenMP pur', 'Hybride']:
        cat_data = df_large[df_large['category'] == cat]
        ax.scatter(cat_data['total_threads'], cat_data['speedup'], 
                  label=cat, s=100, color=cat_colors[cat], alpha=0.7)
    
    ax.set_xlabel('Nombre total de threads (MPI × OpenMP)')
    ax.set_ylabel('Speedup')
    ax.set_title(f'Comparaison MPI vs OpenMP vs Hybride (taille={largest_size:,})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ligne de speedup linéaire idéal
    max_threads = df_large['total_threads'].max()
    ax.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.3, label='Speedup linéaire')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'mpi_vs_omp.png'), dpi=150, bbox_inches='tight')
    print("Graphique sauvegardé: mpi_vs_omp.png")

def plot_version_comparison():
    """Compare Version 1 et Version 2"""
    df = load_data('version_comparison.csv')
    if df is None:
        print("Fichier version_comparison.csv non trouvé (exécutez: make compare)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sizes = df['size'].unique()
    
    # Graphique 1: Temps par version et configuration
    ax1 = axes[0]
    
    configs = df['config'].unique()
    x = np.arange(len(sizes))
    width = 0.15
    
    for i, config in enumerate(configs):
        config_data = df[df['config'] == config]
        times = [config_data[config_data['size'] == s]['time'].values[0] 
                 if len(config_data[config_data['size'] == s]) > 0 else 0 
                 for s in sizes]
        
        version = 'V1' if 'v1' in df[df['config'] == config]['version'].values[0] else 'V2'
        label = f'{version}: {config}'
        
        ax1.bar(x + i * width, times, width, label=label, 
               color=COLORS[i % len(COLORS)])
    
    ax1.set_xlabel('Taille du tableau')
    ax1.set_ylabel('Temps (secondes)')
    ax1.set_title('Comparaison Version 1 vs Version 2')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels([f'{s:,}'.replace(',', ' ') for s in sizes])
    ax1.legend(fontsize=8)
    ax1.set_yscale('log')
    
    # Graphique 2: Speedup
    ax2 = axes[1]
    
    for i, config in enumerate(configs):
        config_data = df[df['config'] == config]
        speedups = [config_data[config_data['size'] == s]['speedup_vs_seq'].values[0] 
                    if len(config_data[config_data['size'] == s]) > 0 else 1 
                    for s in sizes]
        
        version = 'V1' if 'v1' in df[df['config'] == config]['version'].values[0] else 'V2'
        label = f'{version}: {config}'
        
        ax2.plot(sizes, speedups, 'o-', label=label, 
                color=COLORS[i % len(COLORS)], linewidth=2)
    
    ax2.set_xlabel('Taille du tableau')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup Version 1 vs Version 2')
    ax2.set_xscale('log')
    ax2.legend(fontsize=8)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'version_comparison.png'), dpi=150, bbox_inches='tight')
    print("Graphique sauvegardé: version_comparison.png")

def plot_topk_results():
    """Visualise les résultats du benchmark Top-K"""
    df = load_data('topk_hybrid_summary.csv')
    if df is None:
        print("Fichier topk_hybrid_summary.csv non trouvé")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Grouper par taille
    sizes = df['size'].unique()
    k_values = df['k'].unique()
    
    x = np.arange(len(sizes))
    width = 0.25
    
    for i, k in enumerate(k_values):
        k_data = df[df['k'] == k]
        speedups = [k_data[k_data['size'] == s]['speedup'].values[0] 
                    if len(k_data[k_data['size'] == s]) > 0 else 1 
                    for s in sizes]
        ax.bar(x + i * width, speedups, width, label=f'K={k}', color=COLORS[i])
    
    ax.set_xlabel('Taille du tableau')
    ax.set_ylabel('Speedup')
    ax.set_title('Performance Top-K Hybride par taille et valeur de K')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{s:,}'.replace(',', ' ') for s in sizes])
    ax.legend()
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'topk_hybrid_performance.png'), dpi=150, bbox_inches='tight')
    print("Graphique sauvegardé: topk_hybrid_performance.png")

def main():
    """Fonction principale"""
    print("=== Génération des graphiques - Version 2 ===")
    print(f"Répertoire des résultats: {RESULTS_DIR}")
    print()
    
    # Créer le répertoire si nécessaire
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Générer tous les graphiques
    plot_hybrid_comparison()
    plot_mpi_vs_omp()
    plot_version_comparison()
    plot_topk_results()
    
    print()
    print("=== Génération terminée ===")

if __name__ == '__main__':
    main()
