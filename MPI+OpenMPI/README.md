# Bucket Sort et Top-K Distribués - Version 2
## Implémentation Hybride MPI + OpenMP

Cette version améliore l'implémentation originale en combinant deux modèles de parallélisme:
- **MPI** (Message Passing Interface) pour le parallélisme distribué entre processus
- **OpenMP** pour le parallélisme partagé à l'intérieur de chaque processus

## Concepts Clés

### Parallélisme Hybride MPI + OpenMP

L'approche hybride permet de tirer parti des deux modèles:

```
┌─────────────────────────────────────────────────────────────┐
│                     Machine/Cluster                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Processus 0   │   Processus 1   │   Processus N-1         │
│   (MPI Rank 0)  │   (MPI Rank 1)  │   (MPI Rank N-1)        │
├─────────────────┼─────────────────┼─────────────────────────┤
│ Thread 0 │ T1   │ Thread 0 │ T1   │ Thread 0 │ T1           │
│ Thread 2 │ T3   │ Thread 2 │ T3   │ Thread 2 │ T3           │
│   (OpenMP)      │   (OpenMP)      │   (OpenMP)              │
└─────────────────┴─────────────────┴─────────────────────────┘
       │                  │                    │
       └──────────────────┼────────────────────┘
                   Communication MPI
```

### Avantages de l'Approche Hybride

1. **Réduction de l'overhead MPI**: Moins de processus MPI = moins de communication
2. **Meilleure localité mémoire**: Les threads OpenMP partagent la mémoire
3. **Flexibilité**: Adaptation aux architectures (clusters, multi-cœurs)
4. **Scalabilité**: Combine scalabilité MPI + efficacité OpenMP

## Structure du Projet

```
version2/
├── src/
│   ├── bucket_sort_hybrid.c    # Bucket Sort hybride MPI+OpenMP
│   └── topk_hybrid.c           # Top-K hybride MPI+OpenMP
├── scripts/
│   ├── benchmark_bucket_sort.sh  # Benchmark avec configurations hybrides
│   ├── benchmark_topk.sh         # Benchmark Top-K
│   ├── compare_versions.sh       # Comparaison V1 vs V2
│   ├── run_all_benchmarks.sh     # Script principal
│   └── plot_results.py           # Visualisation des résultats
├── bin/                          # Exécutables (générés)
├── results/                      # Résultats des benchmarks
├── Makefile                      # Système de build
└── README.md                     # Ce fichier
```

## Compilation

```bash
# Compiler tous les programmes
make

# Compiler avec OpenMP désactivé (pour comparaison)
make CFLAGS="-Wall -O3 -std=c99"
```

## Utilisation

### Bucket Sort Hybride

```bash
# Syntaxe: mpirun -np <NP> bucket_sort_hybrid <taille> [threads_omp]

# 4 processus MPI, 2 threads OpenMP chacun
OMP_NUM_THREADS=2 mpirun -np 4 bin/bucket_sort_hybrid 1000000 2

# 2 processus MPI, 4 threads OpenMP chacun
OMP_NUM_THREADS=4 mpirun -np 2 bin/bucket_sort_hybrid 1000000 4
```

### Top-K Hybride

```bash
# Syntaxe: mpirun -np <NP> topk_hybrid <taille> <K> [threads_omp]

# 4 processus MPI, 2 threads OpenMP, K=1000
OMP_NUM_THREADS=2 mpirun -np 4 bin/topk_hybrid 1000000 1000 2
```

### Tests Rapides

```bash
# Test Bucket Sort avec configuration par défaut
make test-bucket

# Test Top-K
make test-topk

# Comparer différentes configurations hybrides
make test-hybrid

# Paramètres personnalisés
make test-bucket NP=8 OMP_THREADS=2 SIZE=1000000
```

## Benchmarks

```bash
# Lancer tous les benchmarks
make benchmark

# Comparer avec la Version 1
make compare

# Générer les graphiques
make plot
```

## Configurations Hybrides Testées

| Configuration | Processus MPI | Threads OMP | Total |
|---------------|---------------|-------------|-------|
| 1MPI_4OMP     | 1             | 4           | 4     |
| 2MPI_2OMP     | 2             | 2           | 4     |
| 4MPI_1OMP     | 4             | 1           | 4     |
| 2MPI_4OMP     | 2             | 4           | 8     |
| 4MPI_2OMP     | 4             | 2           | 8     |
| 8MPI_1OMP     | 8             | 1           | 8     |

## Détails d'Implémentation

### Bucket Sort Hybride

1. **Génération des données** (OpenMP parallélisé)
   ```c
   #pragma omp parallel
   {
       unsigned int local_seed = seed + omp_get_thread_num();
       #pragma omp for
       for (int i = 0; i < size; i++)
           arr[i] = rand_r(&local_seed) % max_value;
   }
   ```

2. **Comptage des buckets** (OpenMP reduction)
   ```c
   #pragma omp parallel
   {
       int *local_counts = calloc(num_buckets, sizeof(int));
       #pragma omp for nowait
       for (int i = 0; i < local_size; i++)
           local_counts[bucket_id]++;
       #pragma omp critical
       for (int j = 0; j < num_buckets; j++)
           bucket_counts[j] += local_counts[j];
   }
   ```

3. **Communication MPI** (MPI_Alltoallv)
   - Échange des buckets entre processus

4. **Tri local** (OpenMP sections parallèles)
   - Division en chunks triés en parallèle
   - Fusion finale

### Top-K Hybride

1. **Extraction locale K max** (tri partiel parallélisé)
2. **Réduction arborescente** (MPI binaire)
3. **Fusion efficace** des Top-K partiels

### Initialisation MPI avec Support Threads

```c
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
```

Niveaux de support:
- `MPI_THREAD_SINGLE`: Un seul thread (pas d'OpenMP)
- `MPI_THREAD_FUNNELED`: Seul le thread maître appelle MPI
- `MPI_THREAD_SERIALIZED`: Threads sérialisés pour MPI
- `MPI_THREAD_MULTIPLE`: Threads multiples appelant MPI

## Analyse des Performances

### Temps de Calcul vs Communication

Le programme mesure séparément:
- **Temps de calcul** (`comp_time`): Opérations CPU (tri, comptage)
- **Temps de communication** (`comm_time`): Échanges MPI

```
=== Résultats ===
Temps total: 0.123456 secondes
Temps de calcul: 0.089000 secondes (72.1%)
Temps de communication: 0.034456 secondes (27.9%)
```

### Sorties CSV pour Analyse

Les programmes génèrent une ligne CSV pour faciliter le benchmarking:
```
CSV: <mpi_procs>,<omp_threads>,<size>,<total_time>,<comp_time>,<comm_time>
```

## Comparaison V1 vs V2

| Aspect | Version 1 | Version 2 |
|--------|-----------|-----------|
| Parallélisme | MPI seul | MPI + OpenMP |
| Overhead communication | Plus élevé | Réduit |
| Flexibilité | Limitée | Configurable |
| Complexité code | Simple | Modérée |
| Scalabilité | Bonne | Excellente |

## Dépendances

- **OpenMPI** ou MPICH
- **GCC** avec support OpenMP (`-fopenmp`)
- **Python 3** avec matplotlib, pandas, numpy (pour les graphiques)

```bash
# Installation des dépendances
sudo apt-get install openmpi-bin libopenmpi-dev
pip install matplotlib pandas numpy
```

## Références

- "Une introduction à la programmation parallèle avec Open MPI et OpenMP"
- Cours Ch6-MPI.pdf
- Documentation OpenMPI: https://www.open-mpi.org/doc/
- Documentation OpenMP: https://www.openmp.org/specifications/

## Auteur

Projet Master 1 - Décembre 2025
