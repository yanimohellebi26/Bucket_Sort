# Bucket Sort Distribué avec MPI

## Description du Projet

Ce projet implémente l'algorithme **Bucket Sort** de manière distribuée en utilisant **MPI** (Message Passing Interface) et le langage **C**. Il comprend également une variante optimisée pour l'extraction des **Top-K** plus grandes valeurs.

## Structure du Projet

```
Bucket_Sort/
├── src/
│   ├── bucket_sort_mpi.c    # Implémentation du Bucket Sort distribué
│   └── topk_mpi.c           # Implémentation du Top-K distribué
├── scripts/
│   ├── benchmark_bucket_sort.sh  # Script de benchmark pour Bucket Sort
│   ├── benchmark_topk.sh         # Script de benchmark pour Top-K
│   ├── run_all_benchmarks.sh     # Script pour lancer tous les benchmarks
│   └── plot_results.py           # Script Python pour générer les graphiques
├── results/                  # Dossier des résultats (généré automatiquement)
├── Makefile                  # Fichier de compilation
└── README.md                 # Ce fichier
```

## Prérequis

- **Compilateur MPI** : `mpicc` (OpenMPI ou MPICH)
- **Python 3** avec les bibliothèques : `pandas`, `matplotlib`, `numpy`

### Installation des dépendances

```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev
pip3 install pandas matplotlib numpy

# Arch Linux
sudo pacman -S openmpi
pip install pandas matplotlib numpy
```

## Compilation

```bash
# Compilation standard (optimisée)
make

# Compilation en mode debug
make debug

# Nettoyage
make clean
```

## Utilisation

### Bucket Sort Distribué

```bash
# Syntaxe: mpirun -np <nb_processus> ./bucket_sort_mpi <taille_tableau>

# Exemple avec 4 processus et 1 million d'éléments
mpirun -np 4 ./bucket_sort_mpi 1000000

# Exemple avec 8 processus et 10 millions d'éléments
mpirun -np 8 ./bucket_sort_mpi 10000000
```

### Top-K Extraction

```bash
# Syntaxe: mpirun -np <nb_processus> ./topk_mpi <taille_tableau> <k>

# Exemple: extraire les 100 plus grandes valeurs avec 4 processus
mpirun -np 4 ./topk_mpi 1000000 100

# Exemple: extraire les 1000 plus grandes valeurs avec 8 processus
mpirun -np 8 ./topk_mpi 10000000 1000
```

## Benchmarks

### Lancer tous les benchmarks

```bash
make benchmark
```

### Benchmarks individuels

```bash
# Bucket Sort seulement
make benchmark-bucket

# Top-K seulement
make benchmark-topk

# Générer les graphiques
make plot
```

## Algorithmes

### 1. Bucket Sort Distribué

L'algorithme Bucket Sort distribué fonctionne en plusieurs étapes :

1. **Distribution** : Le processus 0 génère les données et les distribue équitablement entre tous les processus (via `MPI_Scatterv`)

2. **Création des buckets** : Chaque processus partitionne ses données locales en P buckets (P = nombre de processus), où le bucket i contient les valeurs dans l'intervalle `[i * range, (i+1) * range)`

3. **Échange All-to-All** : Les processus échangent les buckets entre eux via `MPI_Alltoallv`. Après cette étape, le processus i possède toutes les valeurs de l'intervalle i

4. **Tri local** : Chaque processus trie son bucket localement avec `qsort`

5. **Rassemblement** : Les buckets triés sont rassemblés sur le processus 0 via `MPI_Gatherv`

**Complexité** : O(n/p * log(n/p)) pour le tri local + O(n) pour les communications

### 2. Top-K Distribué

L'extraction des K plus grandes valeurs utilise une approche optimisée :

1. **Distribution** : Identique au Bucket Sort

2. **Tri local décroissant** : Chaque processus trie ses données en ordre décroissant

3. **Sélection locale** : Chaque processus garde ses K meilleurs éléments locaux

4. **Fusion** : Le processus 0 collecte les top-K locaux et effectue une fusion finale

**Avantages par rapport au tri complet** :
- Communication réduite : chaque processus envoie au maximum K éléments
- Tri partiel suffisant quand K << N

## Résultats Attendus

### Courbes de Performance

Les benchmarks génèrent automatiquement :
- `results/bucket_sort_performance.png` : Temps et accélération du Bucket Sort
- `results/topk_performance.png` : Temps et accélération du Top-K
- `results/comparison.png` : Comparaison des deux algorithmes

### Interprétation

**Facteurs limitant l'accélération** :
- **Loi d'Amdahl** : La fraction séquentielle (génération des données, vérification) limite l'accélération maximale
- **Communication** : L'échange All-to-All a une complexité O(p) en nombre de messages
- **Déséquilibre de charge** : La distribution des données peut créer des buckets de tailles inégales

**Comparaison Bucket Sort vs Top-K** :
- Le Top-K devrait être plus rapide car il communique moins de données
- L'accélération du Top-K peut être meilleure pour de petites valeurs de K
- Pour K proche de N, les performances convergent

## Fichiers de Sortie

| Fichier | Description |
|---------|-------------|
| `results/bucket_sort_results.csv` | Données brutes des benchmarks |
| `results/bucket_sort_summary.csv` | Statistiques agrégées |
| `results/topk_results.csv` | Données brutes Top-K |
| `results/topk_summary.csv` | Statistiques agrégées Top-K |

## Auteur

Projet Master 1 - Décembre 2025

## Licence

Ce projet est fourni à des fins éducatives.
