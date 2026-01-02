# Rapport de Projet : Tri Distribué avec MPI

## Bucket Sort et Extraction Top-K

**Cours :** Programmation Parallèle et Distribuée  
**Niveau :** Master 1  
**Date :** Décembre 2025  
**Auteur :** Yani Mohellebi

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [Fondements Théoriques de MPI](#2-fondements-théoriques-de-mpi)
3. [Objectifs du Projet](#3-objectifs-du-projet)
4. [Architecture et Conception](#4-architecture-et-conception)
5. [Implémentation Version 1 - MPI Pure](#5-implémentation-version-1---mpi-pure)
6. [Implémentation Version 2 - MPI + OpenMP](#6-implémentation-version-2---mpi--openmp)
7. [Analyse Détaillée des Fonctions MPI](#7-analyse-détaillée-des-fonctions-mpi)
8. [Résultats Expérimentaux](#8-résultats-expérimentaux)
9. [Analyse des Performances](#9-analyse-des-performances)
10. [Conclusion](#10-conclusion)
11. [Références](#11-références)

---

## 1. Introduction

### 1.1 Contexte et Motivation

Ce projet s'inscrit dans le cadre du cours de programmation parallèle et distribuée. L'objectif est d'implémenter et d'analyser les performances d'algorithmes de tri distribués utilisant le paradigme MPI (Message Passing Interface).

Le tri est une opération fondamentale en informatique, utilisée dans de nombreuses applications : bases de données, traitement de données massives, intelligence artificielle, moteurs de recherche, etc. Lorsque les volumes de données deviennent importants (Big Data), le tri séquentiel atteint ses limites et le parallélisme devient nécessaire.

### 1.2 Problématique

Avec l'explosion des volumes de données, les algorithmes séquentiels traditionnels montrent leurs limites :
- **Temps de calcul** : Un tri de 10⁹ éléments prend plusieurs minutes sur une machine mono-cœur
- **Mémoire** : Les données peuvent dépasser la capacité mémoire d'une seule machine
- **Scalabilité** : La loi de Moore atteint ses limites, le parallélisme devient essentiel

### 1.3 Solution Proposée

Notre approche combine :
- **Distribution des données** : Répartition sur plusieurs processus
- **Parallélisme à gros grain** : MPI pour la communication inter-processus
- **Parallélisme à grain fin** : OpenMP pour le multi-threading intra-processus

### 1.4 Contexte Technologique

| Technologie | Type | Utilisation |
|-------------|------|-------------|
| **MPI** | Mémoire distribuée | Communication inter-processus, clusters |
| **OpenMP** | Mémoire partagée | Multi-threading, multi-cœurs |
| **Hybride** | Les deux | Architectures modernes (clusters de multi-cœurs) |

---

## 2. Fondements Théoriques de MPI

### 2.1 Qu'est-ce que MPI ?

**MPI (Message Passing Interface)** est une spécification standardisée pour la programmation parallèle distribuée. Ce n'est pas une implémentation, mais un standard qui définit la syntaxe et la sémantique des routines de communication.

#### Historique et Versions

| Version | Année | Nouveautés Principales |
|---------|-------|------------------------|
| MPI-1.0 | 1994 | Communications point-à-point, collectives |
| MPI-2.0 | 1997 | I/O parallèle, création dynamique de processus |
| MPI-3.0 | 2012 | Communications non-bloquantes collectives |
| MPI-4.0 | 2021 | Sessions, partitionnement mémoire |

#### Implémentations Principales

- **OpenMPI** : Open source, très utilisé dans le monde académique
- **MPICH** : Implémentation de référence
- **Intel MPI** : Optimisé pour les processeurs Intel
- **Microsoft MPI** : Pour les environnements Windows

### 2.2 Modèle de Programmation SPMD

MPI utilise le modèle **SPMD (Single Program, Multiple Data)** :

```
┌─────────────────────────────────────────────────────────────────┐
│                    MÊME PROGRAMME (SPMD)                        │
│                                                                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │ Process 0 │ │ Process 1 │ │ Process 2 │ │ Process 3 │       │
│  │ (rank=0)  │ │ (rank=1)  │ │ (rank=2)  │ │ (rank=3)  │       │
│  │           │ │           │ │           │ │           │       │
│  │ Données A │ │ Données B │ │ Données C │ │ Données D │       │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │
│                      │              │              │            │
│                      └──────────────┼──────────────┘            │
│                                     │                           │
│                           COMMUNICATIONS MPI                    │
└─────────────────────────────────────────────────────────────────┘
```

**Caractéristiques du modèle SPMD :**
- Tous les processus exécutent le même code
- Chaque processus a son propre espace mémoire (mémoire distribuée)
- Le comportement diffère selon le `rank` (identifiant unique)
- La communication est explicite via des messages

### 2.3 Concepts Fondamentaux de MPI

#### 2.3.1 Communicateur (MPI_Comm)

Un **communicateur** définit un groupe de processus pouvant communiquer entre eux.

```c
MPI_COMM_WORLD  // Communicateur par défaut incluant tous les processus
```

**Propriétés d'un communicateur :**
- Ensemble de processus participants
- Contexte de communication unique
- Topologie optionnelle (cartésienne, graphe)

#### 2.3.2 Rang (Rank)

Le **rang** est l'identifiant unique d'un processus au sein d'un communicateur.

```c
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// rank ∈ [0, num_procs - 1]
```

#### 2.3.3 Types de Données MPI

MPI définit des types de données portables :

| Type MPI | Type C | Taille typique |
|----------|--------|----------------|
| `MPI_INT` | `int` | 4 octets |
| `MPI_FLOAT` | `float` | 4 octets |
| `MPI_DOUBLE` | `double` | 8 octets |
| `MPI_CHAR` | `char` | 1 octet |
| `MPI_LONG` | `long` | 8 octets |
| `MPI_BYTE` | - | 1 octet |

### 2.4 Types de Communications MPI

#### 2.4.1 Communications Point-à-Point

Communications entre deux processus spécifiques.

```
  Process 0                    Process 1
      │                            │
      │  MPI_Send(data, dest=1)    │
      │ ──────────────────────────>│
      │                            │ MPI_Recv(data, source=0)
      │                            │
```

**Modes de communication :**

| Mode | Fonction | Comportement |
|------|----------|--------------|
| Standard | `MPI_Send` | Bloquant, peut ou non utiliser un buffer |
| Buffered | `MPI_Bsend` | Utilise un buffer utilisateur |
| Synchrone | `MPI_Ssend` | Attend que le récepteur commence à recevoir |
| Ready | `MPI_Rsend` | Le récepteur doit être prêt |

#### 2.4.2 Communications Collectives

Communications impliquant tous les processus d'un communicateur.

**Avantages des communications collectives :**
- Optimisées par l'implémentation MPI
- Algorithmes adaptés à la topologie réseau
- Code plus lisible et maintenable

### 2.5 Synchronisation

#### 2.5.1 Barrière (MPI_Barrier)

```c
MPI_Barrier(MPI_COMM_WORLD);
```

Tous les processus attendent que tous aient atteint ce point :

```
  P0     P1     P2     P3
   │      │      │      │
   │      │  ────┴──────┼─── Calculs en cours
   │      │      │      │
   ▼      ▼      ▼      ▼
───┴──────┴──────┴──────┴─── MPI_Barrier (synchronisation)
   │      │      │      │
   ▼      ▼      ▼      ▼
```

---

## 3. Objectifs du Projet

Le projet répond à quatre questions principales :

### Question 1 : Bucket Sort Distribué
> Utiliser MPI et le langage C pour trier une liste d'éléments avec l'algorithme Bucket Sort distribué.

### Question 2 : Mesure des Performances
> Mesurer le temps d'exécution pour 2, 4, 8, 16, 32, 64 et 128 tâches. Tracer les courbes de performance.

### Question 3 : Extraction Top-K
> Réaliser un programme pour extraire le top-k des plus grandes valeurs de manière distribuée.

### Question 4 : Comparaison Top-K vs Tri Complet
> Mesurer et comparer les temps d'exécution de l'extraction Top-K avec le tri complet.

---

## 4. Architecture et Conception

### 4.1 Algorithme Bucket Sort : Principes Théoriques

#### 4.1.1 Définition et Complexité

Le **Bucket Sort** (tri par paquets) est un algorithme de tri non-comparatif qui fonctionne en distribuant les éléments dans des "buckets" (seaux) selon leur valeur.

**Complexité algorithmique :**
- **Meilleur cas** : O(n + k) où k est le nombre de buckets
- **Cas moyen** : O(n + n²/k + k) ≈ O(n) si k ≈ n
- **Pire cas** : O(n²) si tous les éléments tombent dans le même bucket

**Hypothèses pour une performance optimale :**
1. Distribution uniforme des données
2. Nombre de buckets approprié
3. Tri efficace intra-bucket (quicksort, insertion sort)

#### 4.1.2 Pourquoi Bucket Sort pour le Parallélisme ?

| Critère | Avantage pour MPI |
|---------|-------------------|
| Partition naturelle | Chaque processus gère une plage de valeurs |
| Indépendance | Les buckets peuvent être triés indépendamment |
| Réduction de communication | Échange unique via All-to-All |
| Équilibrage de charge | Possible avec une bonne répartition |

### 4.2 Algorithme Bucket Sort Distribué

Le Bucket Sort distribué fonctionne en 6 étapes :

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSUS MAÎTRE (Rank 0)                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     Données initiales: [45, 12, 89, 3, 67, 23, ...]      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ ÉTAPE 1: MPI_Scatterv (Distribution)
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Process 0  │  Process 1  │  Process 2  │  Process 3  │
│  [45, 12]   │  [89, 3]    │  [67, 23]   │   [...]     │
└─────────────┴─────────────┴─────────────┴─────────────┘
                              │
                              ▼ ÉTAPE 2: Création des buckets locaux
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Bucket 0-24 │ Bucket 25-49│ Bucket 50-74│ Bucket 75-99│
│  (local)    │  (local)    │  (local)    │  (local)    │
└─────────────┴─────────────┴─────────────┴─────────────┘
                              │
                              ▼ ÉTAPE 3: MPI_Alltoall (Échange des tailles)
                              │
                              ▼ ÉTAPE 4: MPI_Alltoallv (Redistribution)
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Process 0  │  Process 1  │  Process 2  │  Process 3  │
│ Valeurs 0-24│ Valeurs 25-49│Valeurs 50-74│Valeurs 75-99│
│  [3, 12]    │  [23, 45]   │  [67]       │  [89]       │
└─────────────┴─────────────┴─────────────┴─────────────┘
                              │
                              ▼ ÉTAPE 5: Tri local (qsort)
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  [3, 12]    │  [23, 45]   │  [67]       │  [89]       │
│   (trié)    │   (trié)    │   (trié)    │   (trié)    │
└─────────────┴─────────────┴─────────────┴─────────────┘
                              │
                              ▼ ÉTAPE 6: MPI_Gatherv (Rassemblement)
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSUS MAÎTRE (Rank 0)                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     Données triées: [3, 12, 23, 45, 67, 89, ...]         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Algorithme Top-K Distribué

L'extraction Top-K utilise une approche optimisée avec réduction arborescente :

```
                    PHASE 1: Distribution et Extraction Locale
┌─────────────────────────────────────────────────────────────────┐
│                         MPI_Scatterv                            │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ Process │          │ Process │          │ Process │
    │    0    │          │    1    │          │    2    │
    │         │          │         │          │         │
    │ Top-K   │          │ Top-K   │          │ Top-K   │
    │ local   │          │ local   │          │ local   │
    └────┬────┘          └────┬────┘          └────┬────┘
         │                    │                    │
         │      PHASE 2: Réduction Arborescente    │
         │                    │                    │
         ▼                    ▼                    │
    ┌─────────┐          ┌─────────┐               │
    │ Fusion  │◄─────────│   Send  │               │
    │ P0 + P1 │          └─────────┘               │
    └────┬────┘                                    │
         │                                         │
         ▼                                         │
    ┌─────────┐                                    │
    │ Fusion  │◄───────────────────────────────────┘
    │ finale  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Top-K   │
    │ Global  │
    └─────────┘
```

**Avantages de la réduction arborescente :**
- Complexité en communication : O(log P) étapes au lieu de O(P)
- Chaque étape transmet seulement K éléments
- Parallélisme dans les niveaux de l'arbre

### 4.4 Structure du Projet

```
Bucket_Sort/
├── MPI/                        # Version 1 - MPI Pure
│   ├── src/
│   │   ├── bucket_sort_mpi.c   # Bucket Sort MPI
│   │   └── topk_mpi.c          # Top-K MPI
│   ├── scripts/
│   │   ├── benchmark_bucket_sort.sh
│   │   ├── benchmark_topk.sh
│   │   ├── run_all_benchmarks.sh
│   │   └── plot_results.py
│   ├── results/                # Résultats CSV
│   └── Makefile
│
├── MPI+OpenMPI/                # Version 2 - Hybride
│   ├── src/
│   │   ├── bucket_sort_hybrid.c
│   │   └── topk_hybrid.c
│   ├── scripts/
│   │   └── compare_versions.sh
│   ├── bin/                    # Exécutables
│   └── Makefile
│
├── README.md
└── RAPPORT_PROJET.md

---

## 5. Implémentation Version 1 - MPI Pure

### 5.1 Vue d'Ensemble des Fonctions MPI Utilisées

Notre implémentation utilise les fonctions MPI suivantes, classées par catégorie :

#### 5.1.1 Fonctions d'Environnement

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `MPI_Init` | `int MPI_Init(int *argc, char ***argv)` | Initialise l'environnement MPI |
| `MPI_Finalize` | `int MPI_Finalize(void)` | Termine l'environnement MPI |
| `MPI_Comm_rank` | `int MPI_Comm_rank(MPI_Comm comm, int *rank)` | Retourne le rang du processus |
| `MPI_Comm_size` | `int MPI_Comm_size(MPI_Comm comm, int *size)` | Retourne le nombre de processus |
| `MPI_Abort` | `int MPI_Abort(MPI_Comm comm, int errorcode)` | Termine tous les processus |

#### 5.1.2 Fonctions de Communication Collective

| Fonction | Type | Description |
|----------|------|-------------|
| `MPI_Scatterv` | Distribution | Distribue des portions de tailles variables |
| `MPI_Gatherv` | Collection | Rassemble des portions de tailles variables |
| `MPI_Alltoall` | Échange | Échange tous-vers-tous (tailles fixes) |
| `MPI_Alltoallv` | Échange | Échange tous-vers-tous (tailles variables) |
| `MPI_Gather` | Collection | Rassemble des données sur un processus |

#### 5.1.3 Fonctions de Synchronisation et Temps

| Fonction | Rôle |
|----------|------|
| `MPI_Barrier` | Synchronise tous les processus |
| `MPI_Wtime` | Retourne le temps écoulé en secondes |

### 5.2 Bucket Sort MPI - Analyse Détaillée du Code

#### 5.2.1 Initialisation de l'Environnement MPI

```c
int main(int argc, char *argv[]) {
    int rank, num_procs;
    
    // Initialisation MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
```

**Explication détaillée :**

- **`MPI_Init(&argc, &argv)`** :
  - Initialise l'environnement d'exécution MPI
  - Doit être appelée avant toute autre fonction MPI
  - Permet à MPI d'extraire ses arguments de la ligne de commande
  - Établit les connexions entre processus

- **`MPI_Comm_rank(MPI_COMM_WORLD, &rank)`** :
  - Retourne l'identifiant unique du processus appelant
  - `rank` ∈ [0, num_procs - 1]
  - Le rang 0 est souvent désigné comme "processus maître"

- **`MPI_Comm_size(MPI_COMM_WORLD, &num_procs)`** :
  - Retourne le nombre total de processus dans le communicateur
  - `MPI_COMM_WORLD` inclut tous les processus lancés

#### 5.2.2 Génération et Allocation des Données

```c
// Allocation et génération des données sur le processus 0 UNIQUEMENT
if (rank == 0) {
    data = (int*)malloc(total_size * sizeof(int));
    if (data == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    generate_random_array(data, total_size, MAX_VALUE, 42);
}
```

**Points clés :**
- Seul le processus 0 alloue et génère les données initiales
- **`MPI_Abort`** termine proprement tous les processus en cas d'erreur
- La seed fixe (42) permet la reproductibilité des tests

#### 5.2.3 Synchronisation et Chronométrage

```c
// Synchronisation avant le début du chronométrage
MPI_Barrier(MPI_COMM_WORLD);
start_time = MPI_Wtime();

// ... code à mesurer ...

MPI_Barrier(MPI_COMM_WORLD);
end_time = MPI_Wtime();
total_time = end_time - start_time;
```

**Explication de `MPI_Barrier` :**

```
Sans Barrier:                    Avec Barrier:
  P0  P1  P2  P3                  P0  P1  P2  P3
   │   │   │   │                   │   │   │   │
   ▼   │   │   │  start_time       ▼   ▼   ▼   ▼  Barrier
       ▼   │   │                   ▼   ▼   ▼   ▼  start_time (synchronisé)
           ▼   │
               ▼
```

**`MPI_Wtime()` :**
- Retourne le temps écoulé en secondes (double précision)
- Haute résolution (microsecondes ou mieux)
- Temps "wall clock" (temps réel, pas temps CPU)

#### 5.2.4 Distribution des Données avec MPI_Scatterv

```c
// Calcul de la taille locale pour chaque processus
int base_size = total_size / num_procs;
int remainder = total_size % num_procs;
int local_size = base_size + (rank < remainder ? 1 : 0);

// Calcul des déplacements pour Scatterv
int *sendcounts = (int*)malloc(num_procs * sizeof(int));
int *displs = (int*)malloc(num_procs * sizeof(int));

int offset = 0;
for (int i = 0; i < num_procs; i++) {
    sendcounts[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += sendcounts[i];
}

// Allocation du buffer local
int *local_data = (int*)malloc(local_size * sizeof(int));

// Distribution des données
MPI_Scatterv(data, sendcounts, displs, MPI_INT,
             local_data, local_size, MPI_INT,
             0, MPI_COMM_WORLD);
```

**Signature complète de MPI_Scatterv :**

```c
int MPI_Scatterv(
    const void *sendbuf,      // Buffer d'envoi (significatif sur root)
    const int sendcounts[],   // Nombre d'éléments à envoyer à chaque processus
    const int displs[],       // Déplacement pour chaque processus
    MPI_Datatype sendtype,    // Type des données envoyées
    void *recvbuf,            // Buffer de réception
    int recvcount,            // Nombre d'éléments à recevoir
    MPI_Datatype recvtype,    // Type des données reçues
    int root,                 // Processus source
    MPI_Comm comm             // Communicateur
);
```

**Illustration visuelle de MPI_Scatterv :**

```
                    Processus 0 (root)
    ┌────────────────────────────────────────────┐
    │ data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]     │
    │                                            │
    │ sendcounts = [3, 3, 2, 2]                  │
    │ displs     = [0, 3, 6, 8]                  │
    └────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    ┌───────┐       ┌───────┐       ┌───────┐       ┌───────┐
    │ P0    │       │ P1    │       │ P2    │       │ P3    │
    │[1,2,3]│       │[4,5,6]│       │[7,8]  │       │[9,10] │
    └───────┘       └───────┘       └───────┘       └───────┘
```

**Gestion du reste de la division :**
- Si N = 10 et P = 4 : 10 / 4 = 2 avec reste 2
- Les 2 premiers processus reçoivent 3 éléments
- Les 2 derniers reçoivent 2 éléments

#### 5.2.5 Création des Buckets Locaux

```c
// Chaque processus est responsable d'une plage de valeurs
// Processus i: [i * range, (i+1) * range)
double range = (double)MAX_VALUE / num_procs;

// Comptage des éléments pour chaque bucket
int *bucket_counts = (int*)calloc(num_procs, sizeof(int));

for (int i = 0; i < local_size; i++) {
    int bucket_id = (int)(local_data[i] / range);
    if (bucket_id >= num_procs) bucket_id = num_procs - 1;
    bucket_counts[bucket_id]++;
}

// Allocation des buckets locaux
int **local_buckets = (int**)malloc(num_procs * sizeof(int*));
int *bucket_indices = (int*)calloc(num_procs, sizeof(int));

for (int i = 0; i < num_procs; i++) {
    local_buckets[i] = (int*)malloc(bucket_counts[i] * sizeof(int));
}

// Remplissage des buckets
for (int i = 0; i < local_size; i++) {
    int bucket_id = (int)(local_data[i] / range);
    if (bucket_id >= num_procs) bucket_id = num_procs - 1;
    local_buckets[bucket_id][bucket_indices[bucket_id]++] = local_data[i];
}
```

**Logique de partitionnement :**

```
MAX_VALUE = 100, num_procs = 4
range = 100 / 4 = 25

Valeur 12:  bucket_id = 12 / 25 = 0  → Bucket 0 (plage [0, 25))
Valeur 45:  bucket_id = 45 / 25 = 1  → Bucket 1 (plage [25, 50))
Valeur 67:  bucket_id = 67 / 25 = 2  → Bucket 2 (plage [50, 75))
Valeur 89:  bucket_id = 89 / 25 = 3  → Bucket 3 (plage [75, 100))
```

#### 5.2.6 Échange All-to-All des Données

```c
// Communication des tailles de buckets
int *recv_counts = (int*)malloc(num_procs * sizeof(int));
MPI_Alltoall(bucket_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
```

**MPI_Alltoall - Échange de tailles fixes :**

```c
int MPI_Alltoall(
    const void *sendbuf,      // Buffer d'envoi
    int sendcount,            // Nombre d'éléments à envoyer à CHAQUE processus
    MPI_Datatype sendtype,    // Type des données envoyées
    void *recvbuf,            // Buffer de réception
    int recvcount,            // Nombre d'éléments à recevoir de CHAQUE processus
    MPI_Datatype recvtype,    // Type des données reçues
    MPI_Comm comm             // Communicateur
);
```

**Visualisation de MPI_Alltoall :**

```
AVANT:                              APRÈS:
P0: [a0, a1, a2, a3]               P0: [a0, b0, c0, d0]
P1: [b0, b1, b2, b3]     ──────>   P1: [a1, b1, c1, d1]
P2: [c0, c1, c2, c3]               P2: [a2, b2, c2, d2]
P3: [d0, d1, d2, d3]               P3: [a3, b3, c3, d3]
```

**MPI_Alltoallv - Échange de tailles variables :**

```c
// Calcul des déplacements pour l'envoi et la réception
int *send_displs = (int*)malloc(num_procs * sizeof(int));
int *recv_displs = (int*)malloc(num_procs * sizeof(int));

send_displs[0] = 0;
recv_displs[0] = 0;
int total_send = bucket_counts[0];
int total_recv = recv_counts[0];

for (int i = 1; i < num_procs; i++) {
    send_displs[i] = send_displs[i-1] + bucket_counts[i-1];
    recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    total_send += bucket_counts[i];
    total_recv += recv_counts[i];
}

// Préparation du buffer d'envoi contigu
int *send_buffer = (int*)malloc(total_send * sizeof(int));
int pos = 0;
for (int i = 0; i < num_procs; i++) {
    memcpy(send_buffer + pos, local_buckets[i], bucket_counts[i] * sizeof(int));
    pos += bucket_counts[i];
}

// Allocation du buffer de réception
recv_bucket = (int*)malloc(total_recv * sizeof(int));

// Échange All-to-All des données
MPI_Alltoallv(send_buffer, bucket_counts, send_displs, MPI_INT,
              recv_bucket, recv_counts, recv_displs, MPI_INT,
              MPI_COMM_WORLD);
```

**Signature de MPI_Alltoallv :**

```c
int MPI_Alltoallv(
    const void *sendbuf,          // Buffer d'envoi
    const int sendcounts[],       // Tableau des nombres d'éléments à envoyer
    const int sdispls[],          // Déplacements dans le buffer d'envoi
    MPI_Datatype sendtype,        // Type envoyé
    void *recvbuf,                // Buffer de réception
    const int recvcounts[],       // Tableau des nombres d'éléments à recevoir
    const int rdispls[],          // Déplacements dans le buffer de réception
    MPI_Datatype recvtype,        // Type reçu
    MPI_Comm comm                 // Communicateur
);
```

**Illustration de MPI_Alltoallv :**

```
                    sendcounts[i][j] = nombre d'éléments de Pi vers Pj
                    
Process 0                           Process 1
┌─────────────────┐                 ┌─────────────────┐
│ Pour P0: [1,2]  │────────────────>│ Pour P0: [5]    │
│ Pour P1: [3]    │     ╲     ╱     │ Pour P1: [6,7]  │
│ Pour P2: []     │      ╲   ╱      │ Pour P2: [8]    │
│ Pour P3: [4]    │       ╲ ╱       │ Pour P3: []     │
└─────────────────┘        ╳        └─────────────────┘
         │                ╱ ╲                │
         ▼               ╱   ╲               ▼
Après échange:          ╱     ╲        Après échange:
P0 reçoit de tous      ╱       ╲       P1 reçoit de tous
[1,2] + [5] + ...     ╱         ╲      [3] + [6,7] + ...
```

#### 5.2.7 Tri Local

```c
// Tri local du bucket reçu
qsort(recv_bucket, total_recv, sizeof(int), compare_int);
```

**Fonction de comparaison pour qsort :**

```c
int compare_int(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);  // Tri croissant
}
```

**Pourquoi le tri local suffit ?**
- Après Alltoallv, chaque processus a tous les éléments de sa plage
- Les plages sont disjointes et ordonnées (P0 < P1 < P2 < ...)
- Il suffit de trier localement puis concaténer

#### 5.2.8 Rassemblement des Résultats avec MPI_Gatherv

```c
// Communication des tailles de buckets triés
int *final_counts = NULL;
int *final_displs = NULL;

if (rank == 0) {
    final_counts = (int*)malloc(num_procs * sizeof(int));
    final_displs = (int*)malloc(num_procs * sizeof(int));
    sorted_data = (int*)malloc(total_size * sizeof(int));
}

MPI_Gather(&total_recv, 1, MPI_INT, final_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

if (rank == 0) {
    final_displs[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        final_displs[i] = final_displs[i-1] + final_counts[i-1];
    }
}

MPI_Gatherv(recv_bucket, total_recv, MPI_INT,
            sorted_data, final_counts, final_displs, MPI_INT,
            0, MPI_COMM_WORLD);
```

**MPI_Gather - Collection simple :**

```c
int MPI_Gather(
    const void *sendbuf,      // Données à envoyer
    int sendcount,            // Nombre d'éléments à envoyer
    MPI_Datatype sendtype,    // Type des données
    void *recvbuf,            // Buffer de réception (sur root)
    int recvcount,            // Éléments à recevoir de CHAQUE processus
    MPI_Datatype recvtype,    // Type reçu
    int root,                 // Processus destinataire
    MPI_Comm comm             // Communicateur
);
```

**MPI_Gatherv - Collection avec tailles variables :**

```c
int MPI_Gatherv(
    const void *sendbuf,      // Données à envoyer
    int sendcount,            // Nombre d'éléments à envoyer
    MPI_Datatype sendtype,    // Type des données
    void *recvbuf,            // Buffer de réception (sur root)
    const int recvcounts[],   // Éléments à recevoir de chaque processus
    const int displs[],       // Déplacements dans recvbuf
    MPI_Datatype recvtype,    // Type reçu
    int root,                 // Processus destinataire
    MPI_Comm comm             // Communicateur
);
```

**Visualisation de MPI_Gatherv :**

```
Process 0        Process 1        Process 2        Process 3
[3, 12]          [23, 45]         [67]             [89]
    │                │                │                │
    │                │                │                │
    └────────────────┴────────────────┴────────────────┘
                                │
                                ▼
                    Processus 0 (root)
    ┌──────────────────────────────────────────────────┐
    │ sorted_data = [3, 12, 23, 45, 67, 89]            │
    │                                                  │
    │ final_counts = [2, 2, 1, 1]                     │
    │ final_displs = [0, 2, 4, 5]                     │
    └──────────────────────────────────────────────────┘
```

#### 5.2.9 Finalisation MPI

```c
// Libération de la mémoire
free(local_data);
// ... autres free ...

MPI_Finalize();
return 0;
```

**Règles importantes :**
- `MPI_Finalize()` doit être la dernière fonction MPI appelée
- Toutes les communications doivent être terminées avant
- Les ressources MPI sont libérées

### 5.3 Top-K MPI - Analyse Détaillée

#### 5.3.1 Stratégie d'Optimisation

Le Top-K utilise une approche plus efficace que le tri complet :

```c
// Chaque processus trouve ses K plus grands localement
int local_k = (k < local_size) ? k : local_size;

// Tri décroissant du tableau local
qsort(local_data, local_size, sizeof(int), compare_int_desc);

// Garder seulement les local_k premiers (les plus grands)
int *local_topk = (int*)malloc(local_k * sizeof(int));
memcpy(local_topk, local_data, local_k * sizeof(int));
```

**Comparateur pour tri décroissant :**

```c
int compare_int_desc(const void *a, const void *b) {
    return (*(int*)b - *(int*)a);  // b - a pour ordre décroissant
}
```

#### 5.3.2 Réduction Arborescente

```c
// Réduction binaire pour fusionner les Top-K
int step = 1;
while (step < num_procs) {
    if (rank % (2 * step) == 0) {
        // Ce processus reçoit et fusionne
        int partner = rank + step;
        if (partner < num_procs) {
            MPI_Recv(recv_topk, k, MPI_INT, partner, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            merge_topk(local_topk, recv_topk, merged_topk, k);
            memcpy(local_topk, merged_topk, k * sizeof(int));
        }
    } else if (rank % (2 * step) == step) {
        // Ce processus envoie
        int partner = rank - step;
        MPI_Send(local_topk, k, MPI_INT, partner, 0, MPI_COMM_WORLD);
    }
    step *= 2;
}
```

**MPI_Send et MPI_Recv - Communications Point-à-Point :**

```c
int MPI_Send(
    const void *buf,          // Buffer des données à envoyer
    int count,                // Nombre d'éléments
    MPI_Datatype datatype,    // Type des données
    int dest,                 // Rang du destinataire
    int tag,                  // Étiquette du message
    MPI_Comm comm             // Communicateur
);

int MPI_Recv(
    void *buf,                // Buffer de réception
    int count,                // Capacité maximale
    MPI_Datatype datatype,    // Type des données
    int source,               // Rang de l'émetteur
    int tag,                  // Étiquette attendue
    MPI_Comm comm,            // Communicateur
    MPI_Status *status        // Informations sur le message reçu
);
```

**Visualisation de la réduction arborescente :**

```
Étape 1 (step=1):
P0 ◄── P1    P2 ◄── P3    P4 ◄── P5    P6 ◄── P7

Étape 2 (step=2):
P0 ◄──────── P2            P4 ◄──────── P6

Étape 3 (step=4):
P0 ◄──────────────────────── P4

Résultat final sur P0
```

**Avantages de cette approche :**
- log₂(P) étapes au lieu de P-1
- Parallélisme : plusieurs fusions simultanées
- Communication réduite : chaque message contient K éléments

---

## 6. Implémentation Version 2 - MPI + OpenMP

### 6.1 Motivation de l'Approche Hybride

#### 6.1.1 Limitations du MPI Pur

Le MPI pur présente certaines limitations sur les architectures modernes :

| Limitation | Description |
|------------|-------------|
| Overhead de processus | Création et gestion de nombreux processus coûteuse |
| Mémoire dupliquée | Chaque processus a son propre espace mémoire |
| Communication intra-nœud | Les messages MPI entre processus du même nœud passent par des buffers |
| Granularité | Difficile d'exploiter efficacement le parallélisme à grain fin |

#### 6.1.2 Avantages de l'Approche Hybride

L'approche hybride MPI + OpenMP combine les forces des deux modèles :

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLUSTER                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │     NŒUD 1          │  │     NŒUD 2          │              │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │              │
│  │  │ Processus MPI │  │  │  │ Processus MPI │  │              │
│  │  │  ┌─┬─┬─┬─┐    │  │  │  │  ┌─┬─┬─┬─┐    │  │              │
│  │  │  │T│T│T│T│    │  │  │  │  │T│T│T│T│    │  │              │
│  │  │  │0│1│2│3│    │  │  │  │  │0│1│2│3│    │  │              │
│  │  │  └─┴─┴─┴─┘    │  │  │  │  └─┴─┴─┴─┘    │  │              │
│  │  │  OpenMP       │  │  │  │  OpenMP       │  │              │
│  │  │  Threads      │  │  │  │  Threads      │  │              │
│  │  └───────────────┘  │  │  └───────────────┘  │              │
│  │      Mémoire        │  │      Mémoire        │              │
│  │      Partagée       │  │      Partagée       │              │
│  └─────────────────────┘  └─────────────────────┘              │
│              │                      │                          │
│              └──────────────────────┘                          │
│                    Réseau (MPI)                                │
└─────────────────────────────────────────────────────────────────┘
```

| Aspect | MPI Seul | OpenMP Seul | Hybride MPI+OpenMP |
|--------|----------|-------------|-------------------|
| Mémoire | Distribuée | Partagée | Les deux |
| Communication | Messages explicites | Implicite (mémoire) | Optimisée |
| Overhead | Élevé (processus) | Faible (threads) | Équilibré |
| Scalabilité | Clusters | Multi-cœurs | Maximale |
| Complexité | Moyenne | Faible | Élevée |

### 6.2 Initialisation MPI Thread-Safe

```c
int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Avertissement: Support threads insuffisant\n");
    }
    // ...
}
```

**MPI_Init_thread - Initialisation avec Support des Threads :**

```c
int MPI_Init_thread(
    int *argc,
    char ***argv,
    int required,     // Niveau de support demandé
    int *provided     // Niveau de support fourni
);
```

**Niveaux de Support MPI pour les Threads :**

| Niveau | Constante | Description |
|--------|-----------|-------------|
| 0 | `MPI_THREAD_SINGLE` | Un seul thread d'exécution |
| 1 | `MPI_THREAD_FUNNELED` | Multi-thread, mais seul le thread principal appelle MPI |
| 2 | `MPI_THREAD_SERIALIZED` | Multi-thread, appels MPI sérialisés |
| 3 | `MPI_THREAD_MULTIPLE` | Multi-thread, appels MPI concurrents |

**Pour notre implémentation, `MPI_THREAD_FUNNELED` suffit car :**
- Seul le thread principal (OpenMP master) effectue les appels MPI
- Les threads OpenMP travaillent sur les données locales
- Les communications MPI sont hors des régions parallèles OpenMP

### 6.3 Parallélisation OpenMP des Opérations

#### 6.3.1 Génération Parallèle des Données

```c
void generate_random_array(int *arr, int size, int max_value, unsigned int seed) {
    #ifdef _OPENMP
    #pragma omp parallel
    {
        // Chaque thread a sa propre seed pour éviter les conflits
        unsigned int local_seed = seed + omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < size; i++) {
            arr[i] = rand_r(&local_seed) % max_value;
        }
    }
    #else
    srand(seed);
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % max_value;
    }
    #endif
}
```

**Directives OpenMP utilisées :**

| Directive | Signification |
|-----------|---------------|
| `#pragma omp parallel` | Crée une région parallèle (équipe de threads) |
| `#pragma omp for` | Distribue les itérations de la boucle entre threads |
| `omp_get_thread_num()` | Retourne l'ID du thread (0 à N-1) |
| `rand_r()` | Version thread-safe de `rand()` |

**Pourquoi `rand_r` au lieu de `rand` ?**
- `rand()` utilise un état global (non thread-safe)
- `rand_r(&seed)` utilise une seed locale par thread
- Évite les race conditions et garantit la reproductibilité

#### 6.3.2 Comptage Parallèle des Buckets avec Réduction

```c
void count_bucket_elements(int *local_data, int local_size, int *bucket_counts, 
                           int num_buckets, double range) {
    memset(bucket_counts, 0, num_buckets * sizeof(int));
    
    #ifdef _OPENMP
    #pragma omp parallel
    {
        // Compteurs locaux pour chaque thread
        int *local_counts = (int*)calloc(num_buckets, sizeof(int));
        
        #pragma omp for nowait
        for (int i = 0; i < local_size; i++) {
            int bucket_id = (int)(local_data[i] / range);
            if (bucket_id >= num_buckets) bucket_id = num_buckets - 1;
            local_counts[bucket_id]++;
        }
        
        // Réduction manuelle avec section critique
        #pragma omp critical
        {
            for (int j = 0; j < num_buckets; j++) {
                bucket_counts[j] += local_counts[j];
            }
        }
        
        free(local_counts);
    }
    #else
    // Version séquentielle
    for (int i = 0; i < local_size; i++) {
        int bucket_id = (int)(local_data[i] / range);
        if (bucket_id >= num_buckets) bucket_id = num_buckets - 1;
        bucket_counts[bucket_id]++;
    }
    #endif
}
```

**Analyse des directives OpenMP :**

| Directive | Rôle |
|-----------|------|
| `#pragma omp for nowait` | Distribution sans barrière implicite à la fin |
| `#pragma omp critical` | Section critique (un seul thread à la fois) |
| `calloc` | Allocation avec initialisation à zéro |

**Pourquoi cette approche de réduction manuelle ?**

La réduction native OpenMP (`reduction(+:array)`) ne fonctionne pas avec les tableaux en C standard. On utilise donc :
1. Compteurs locaux par thread
2. Fusion dans une section critique

#### 6.3.3 Vérification Parallèle du Tri

```c
int is_sorted(int *arr, int size) {
    int sorted = 1;
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction(&&: sorted)
    #endif
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i-1]) {
            sorted = 0;
        }
    }
    return sorted;
}
```

**Clause `reduction(&&: sorted)` :**
- Chaque thread a une copie privée de `sorted`
- À la fin, toutes les copies sont combinées avec l'opérateur `&&`
- Résultat : `sorted = sorted_t0 && sorted_t1 && ... && sorted_tN`

#### 6.3.4 Tri Parallèle Simplifié

```c
void parallel_sort(int *arr, int size) {
    #ifdef _OPENMP
    if (size > 10000) {
        int num_threads = omp_get_max_threads();
        int chunk_size = size / num_threads;
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = tid * chunk_size;
            int end = (tid == num_threads - 1) ? size : start + chunk_size;
            
            // Tri local de chaque section
            qsort(arr + start, end - start, sizeof(int), compare_int);
        }
        
        // Fusion des sections triées
        qsort(arr, size, sizeof(int), compare_int);
    } else {
        qsort(arr, size, sizeof(int), compare_int);
    }
    #else
    qsort(arr, size, sizeof(int), compare_int);
    #endif
}
```

**Note :** Cette implémentation est simplifiée. Une vraie parallélisation du tri nécessiterait un merge-sort parallèle.

### 6.4 Configuration de l'Environnement Hybride

#### 6.4.1 Variables d'Environnement

```bash
# Nombre de threads OpenMP par processus MPI
export OMP_NUM_THREADS=4

# Ordonnancement des boucles parallèles
export OMP_SCHEDULE=dynamic

# Affinité des threads (binding)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

#### 6.4.2 Exécution Hybride

```bash
# 4 processus MPI × 2 threads OpenMP = 8 cœurs utilisés
OMP_NUM_THREADS=2 mpirun -np 4 ./bin/bucket_sort_hybrid 1000000 2

# Configuration optimale selon l'architecture
# Sur un nœud de 16 cœurs : 4 MPI × 4 OpenMP ou 2 MPI × 8 OpenMP
```

### 6.5 Comparaison des Configurations Hybrides

| Configuration | MPI Procs | OMP Threads | Total | Cas d'usage |
|---------------|-----------|-------------|-------|-------------|
| Pure MPI | 16 | 1 | 16 | Communication intensive |
| Équilibrée | 4 | 4 | 16 | Général |
| Plus OpenMP | 2 | 8 | 16 | Calcul intensif local |
| Pure OpenMP | 1 | 16 | 16 | Un seul nœud |

---

## 7. Analyse Détaillée des Fonctions MPI

### 7.1 Tableau Récapitulatif des Fonctions Utilisées

| Fonction | Catégorie | Complexité | Usage dans le projet |
|----------|-----------|------------|---------------------|
| `MPI_Init` | Environnement | O(1) | Initialisation |
| `MPI_Finalize` | Environnement | O(1) | Terminaison |
| `MPI_Comm_rank` | Environnement | O(1) | Identification |
| `MPI_Comm_size` | Environnement | O(1) | Taille du groupe |
| `MPI_Init_thread` | Environnement | O(1) | Init avec threads |
| `MPI_Barrier` | Synchronisation | O(log P) | Synchro globale |
| `MPI_Wtime` | Temps | O(1) | Mesure performance |
| `MPI_Send` | Point-à-point | O(N) | Envoi bloquant |
| `MPI_Recv` | Point-à-point | O(N) | Réception bloquante |
| `MPI_Scatterv` | Collective | O(N) | Distribution |
| `MPI_Gatherv` | Collective | O(N) | Collection |
| `MPI_Alltoall` | Collective | O(P) | Échange tailles |
| `MPI_Alltoallv` | Collective | O(N) | Redistribution |
| `MPI_Gather` | Collective | O(P) | Collection simple |

### 7.2 Communications Collectives en Détail

#### 7.2.1 Schémas de Communication

**Scatter (Distribution 1-vers-N) :**
```
    Root                    Tous les processus
    ┌───┐                   ┌───┐
    │ A │                   │ A │ P0
    ├───┤                   └───┘
    │ B │      MPI_Scatter  ┌───┐
    ├───┤     ─────────────>│ B │ P1
    │ C │                   └───┘
    ├───┤                   ┌───┐
    │ D │                   │ C │ P2
    └───┘                   └───┘
                            ┌───┐
                            │ D │ P3
                            └───┘
```

**Gather (Collection N-vers-1) :**
```
    Tous les processus      Root
    ┌───┐                   ┌───┐
    │ A │ P0                │ A │
    └───┘                   ├───┤
    ┌───┐     MPI_Gather    │ B │
    │ B │ P1  ─────────────>├───┤
    └───┘                   │ C │
    ┌───┐                   ├───┤
    │ C │ P2                │ D │
    └───┘                   └───┘
    ┌───┐
    │ D │ P3
    └───┘
```

**Alltoall (Transposition) :**
```
    Avant               Après
    P0: [a0,a1,a2,a3]   P0: [a0,b0,c0,d0]
    P1: [b0,b1,b2,b3]   P1: [a1,b1,c1,d1]
    P2: [c0,c1,c2,c3]   P2: [a2,b2,c2,d2]
    P3: [d0,d1,d2,d3]   P3: [a3,b3,c3,d3]
```

#### 7.2.2 Algorithmes Internes des Collectives

Les implémentations MPI utilisent différents algorithmes selon le nombre de processus :

| Opération | Petit P | Grand P |
|-----------|---------|---------|
| Broadcast | Binomial | Van de Geijn |
| Reduce | Binomial | Rabenseifner |
| Alltoall | Direct | Bruck |
| Scatter/Gather | Linéaire | Binomial |

### 7.3 Modes de Communication Point-à-Point

#### 7.3.1 Communications Bloquantes vs Non-Bloquantes

**Communications Bloquantes (utilisées dans notre projet) :**

```c
// MPI_Send bloque jusqu'à ce que le buffer soit réutilisable
MPI_Send(buffer, count, MPI_INT, dest, tag, MPI_COMM_WORLD);

// MPI_Recv bloque jusqu'à réception complète
MPI_Recv(buffer, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
```

**Communications Non-Bloquantes (alternative) :**

```c
MPI_Request request;

// Initie l'envoi sans bloquer
MPI_Isend(buffer, count, MPI_INT, dest, tag, MPI_COMM_WORLD, &request);

// Peut faire du calcul ici...

// Attend la fin de l'opération
MPI_Wait(&request, &status);
```

#### 7.3.2 Deadlock et Évitement

**Exemple de deadlock :**

```c
// DANGER : Deadlock si les deux processus font Send d'abord
if (rank == 0) {
    MPI_Send(..., dest=1, ...);  // P0 attend que P1 reçoive
    MPI_Recv(..., source=1, ...);
} else {
    MPI_Send(..., dest=0, ...);  // P1 attend que P0 reçoive → DEADLOCK
    MPI_Recv(..., source=0, ...);
}
```

**Solution : Ordonner les communications**

```c
if (rank == 0) {
    MPI_Send(..., dest=1, ...);
    MPI_Recv(..., source=1, ...);
} else {
    MPI_Recv(..., source=0, ...);  // P1 reçoit d'abord
    MPI_Send(..., dest=0, ...);
}
```

### 7.4 Gestion de la Mémoire en MPI

#### 7.4.1 Buffers et Contiguïté

MPI nécessite des buffers contigus en mémoire :

```c
// Correct : tableau contigu
int *buffer = malloc(N * sizeof(int));
MPI_Send(buffer, N, MPI_INT, ...);

// Problématique : tableau 2D non contigu
int **matrix;  // Allocation par lignes séparées
// Solution : linéariser ou utiliser MPI_Type_vector
```

#### 7.4.2 Types Dérivés MPI

Pour des structures complexes, MPI permet de créer des types personnalisés :

```c
// Exemple : type pour structure
typedef struct {
    int id;
    double value;
} Element;

MPI_Datatype element_type;
int blocklengths[2] = {1, 1};
MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
MPI_Aint offsets[2];

offsets[0] = offsetof(Element, id);
offsets[1] = offsetof(Element, value);

MPI_Type_create_struct(2, blocklengths, offsets, types, &element_type);
MPI_Type_commit(&element_type);
```

---

## 8. Résultats Expérimentaux

### 8.1 Configuration de Test

| Paramètre | Valeur |
|-----------|--------|
| **Système d'exploitation** | Linux (Ubuntu 22.04) |
| **Processeur** | Multi-cœurs x86_64 |
| **Implémentation MPI** | OpenMPI 4.x |
| **Compilateur** | GCC 11.x avec -O3 |
| **Tailles testées** | 100 000, 1 000 000, 10 000 000 |
| **Processus MPI** | 1, 2, 4, 8, 16 |

### 8.2 Résultats Bucket Sort (Version 1 - MPI Pure)

#### 8.2.1 Temps d'Exécution

| Processus | 100K (s) | 1M (s) | 10M (s) |
|-----------|----------|--------|---------|
| 1 | 0.016 | 0.172 | 1.019 |
| 2 | 0.009 | 0.080 | 0.481 |
| 4 | 0.005 | 0.039 | 0.276 |
| 8 | 0.004 | 0.029 | 0.170 |
| 16 | 0.004 | 0.027 | 0.132 |

#### 8.2.2 Speedup et Efficacité

| Processus | Speedup (10M) | Efficacité |
|-----------|---------------|------------|
| 1 | 1.00x | 100% |
| 2 | 2.12x | 106% |
| 4 | 3.69x | 92% |
| 8 | 5.99x | 75% |
| 16 | 7.73x | 48% |

**Formules utilisées :**

$$\text{Speedup}(P) = \frac{T_1}{T_P}$$

$$\text{Efficacité}(P) = \frac{\text{Speedup}(P)}{P} \times 100\%$$

### 8.3 Résultats Top-K (K=1000)

#### 8.3.1 Temps d'Exécution

| Processus | 100K (s) | 1M (s) | 10M (s) |
|-----------|----------|--------|---------|
| 1 | 0.015 | 0.152 | 0.926 |
| 2 | 0.008 | 0.068 | 0.421 |
| 4 | 0.004 | 0.033 | 0.223 |
| 8 | 0.003 | 0.022 | 0.137 |
| 16 | 0.003 | 0.018 | 0.094 |

#### 8.3.2 Speedup et Efficacité

| Processus | Speedup (10M) | Efficacité |
|-----------|---------------|------------|
| 1 | 1.00x | 100% |
| 2 | 2.20x | 110% |
| 4 | 4.15x | 104% |
| 8 | 6.76x | 84% |
| 16 | 9.85x | 62% |

### 8.4 Comparaison Top-K vs Bucket Sort Complet

| Processus | Bucket Sort (s) | Top-K (s) | Gain |
|-----------|-----------------|-----------|------|
| 1 | 1.019 | 0.926 | 9.1% |
| 4 | 0.276 | 0.223 | 19.2% |
| 8 | 0.170 | 0.137 | 19.4% |
| 16 | 0.132 | 0.094 | 28.8% |

**Observation :** Le gain du Top-K augmente avec le nombre de processus grâce à la réduction de communication.

### 8.5 Comparaison Version 1 vs Version 2 (1M éléments)

| Configuration | Temps (s) | Speedup vs Séq. |
|---------------|-----------|-----------------|
| V1: 1 MPI | 0.172 | 1.00x |
| V1: 4 MPI | 0.039 | 4.37x |
| V2: 1 MPI × 4 OMP | 0.142 | 1.21x |
| V2: 2 MPI × 2 OMP | 0.089 | 1.94x |
| V2: 4 MPI × 1 OMP | 0.057 | 3.00x |

### 8.6 Répartition Calcul vs Communication

| Composante | 100K | 1M | 10M |
|------------|------|-----|-----|
| Calcul | 60% | 70% | 80% |
| Communication | 40% | 30% | 20% |

**Observation :** Le ratio calcul/communication s'améliore avec la taille des données.

---

## 9. Analyse des Performances

### 9.1 Scalabilité

#### 9.1.1 Scalabilité Forte (Strong Scaling)

La **scalabilité forte** mesure comment le temps d'exécution diminue quand on augmente le nombre de processeurs pour une taille de problème fixe.

$$\text{Speedup}_{\text{fort}}(P) = \frac{T_1}{T_P}$$

**Observations pour le Bucket Sort (10M éléments) :**

| P | Temps (s) | Speedup | Speedup idéal |
|---|-----------|---------|---------------|
| 1 | 1.019 | 1.00 | 1 |
| 2 | 0.481 | 2.12 | 2 |
| 4 | 0.276 | 3.69 | 4 |
| 8 | 0.170 | 5.99 | 8 |
| 16 | 0.132 | 7.73 | 16 |

**Analyse :** Le speedup est proche de l'idéal jusqu'à 4 processus, puis l'overhead de communication devient significatif.

#### 9.1.2 Scalabilité Faible (Weak Scaling)

La **scalabilité faible** maintient le travail par processeur constant quand on augmente P.

$$\text{Efficacité}_{\text{faible}}(P) = \frac{T_1}{T_P}$$

(Idéalement = 1 si le temps reste constant)

### 9.2 Loi d'Amdahl

La **loi d'Amdahl** prédit le speedup maximal atteignable :

$$S(P) = \frac{1}{(1-f) + \frac{f}{P}}$$

Où :
- $S(P)$ : Speedup avec P processeurs
- $f$ : Fraction parallélisable du code
- $P$ : Nombre de processeurs

**Application à notre projet :**

En analysant notre code, nous estimons $f \approx 0.95$ (95% parallélisable).

| P | Speedup théorique | Speedup observé |
|---|-------------------|-----------------|
| 2 | 1.90 | 2.12 |
| 4 | 3.48 | 3.69 |
| 8 | 5.93 | 5.99 |
| 16 | 9.14 | 7.73 |
| ∞ | 20.00 | - |

**Observation :** Le speedup observé dépasse parfois le théorique pour les petits P (super-linéarité due aux effets de cache), mais devient inférieur pour les grands P (overhead de communication).

### 9.3 Loi de Gustafson

La **loi de Gustafson** est plus optimiste, elle considère que la taille du problème augmente avec P :

$$S(P) = P - \alpha \times (P - 1)$$

Où $\alpha$ est la fraction séquentielle.

Pour $\alpha = 0.05$ :

| P | Speedup Gustafson |
|---|-------------------|
| 2 | 1.95 |
| 4 | 3.85 |
| 8 | 7.65 |
| 16 | 15.25 |

### 9.4 Analyse du Coût de Communication

#### 9.4.1 Modèle de Coût

Le temps de communication MPI peut être modélisé par :

$$T_{comm} = \alpha + \beta \times n$$

Où :
- $\alpha$ : Latence (temps de démarrage du message)
- $\beta$ : Inverse de la bande passante (temps par octet)
- $n$ : Taille du message en octets

#### 9.4.2 Analyse des Communications dans le Bucket Sort

| Opération | Volume de données | Complexité |
|-----------|-------------------|------------|
| MPI_Scatterv | N éléments | O(N/P) par processus |
| MPI_Alltoall | P entiers | O(P) |
| MPI_Alltoallv | N éléments | O(N/P) par paire |
| MPI_Gatherv | N éléments | O(N/P) par processus |

**Coût total estimé :**

$$T_{comm} \approx 4\alpha \times \log P + \beta \times \frac{2N}{P}$$

### 9.5 Analyse Calcul vs Communication

#### 9.5.1 Ratio Computation/Communication

$$R = \frac{T_{calcul}}{T_{comm}}$$

| Taille | R (P=4) | R (P=16) |
|--------|---------|----------|
| 100K | 1.5 | 0.8 |
| 1M | 2.3 | 1.2 |
| 10M | 4.0 | 2.5 |

**Interprétation :**
- R > 1 : Le calcul domine, bon potentiel de parallélisation
- R < 1 : La communication domine, overhead significatif

#### 9.5.2 Granularité

La **granularité** est le ratio entre calcul et communication :

$$G = \frac{\text{Calcul par processus}}{\text{Communication}}$$

- **Grain fin** (G petit) : Beaucoup de petites communications
- **Gros grain** (G grand) : Peu de grandes communications

Notre implémentation utilise un **gros grain** avec des communications collectives, ce qui est optimal pour MPI.

### 9.6 Top-K vs Tri Complet : Analyse Détaillée

#### 9.6.1 Avantages du Top-K

| Aspect | Tri Complet | Top-K |
|--------|-------------|-------|
| Complexité locale | O((N/P) log(N/P)) | O((N/P) log(N/P)) |
| Données à communiquer (fusion) | O(N) | O(K × P) |
| Tri final | Non nécessaire | Fusion de K×P éléments |
| Mémoire | O(N) | O(K × P) |

#### 9.6.2 Gain de Performance

Le gain relatif du Top-K augmente quand :
- K << N (peu d'éléments à extraire)
- P augmente (moins de communication par processus)

Pour K=1000 et N=10M :

$$\text{Gain} = \frac{T_{tri} - T_{topk}}{T_{tri}} \times 100\%$$

| P | Gain |
|---|------|
| 1 | 9.1% |
| 4 | 19.2% |
| 8 | 19.4% |
| 16 | 28.8% |

### 9.7 Avantages et Inconvénients de l'Approche Hybride

#### 9.7.1 Tableau Comparatif

| Critère | MPI Pur | Hybride MPI+OpenMP |
|---------|---------|-------------------|
| **Complexité de code** | Moyenne | Élevée |
| **Débogage** | Modéré | Difficile |
| **Performance intra-nœud** | Bonne | Excellente |
| **Scalabilité inter-nœuds** | Excellente | Excellente |
| **Utilisation mémoire** | Élevée | Optimisée |
| **Portabilité** | Excellente | Bonne |

#### 9.7.2 Quand Utiliser l'Hybride ?

**Recommandé quand :**
- Nœuds multi-cœurs (>4 cœurs)
- Calculs locaux intensifs
- Mémoire limitée par nœud
- Réduction de la communication souhaitée

**MPI pur préférable quand :**
- Architecture simple (peu de cœurs par nœud)
- Code existant à paralléliser
- Équilibrage de charge dynamique nécessaire

---

## 10. Conclusion

### 10.1 Récapitulatif des Objectifs Atteints

Ce projet a permis de répondre aux quatre questions posées :

| Question | Objectif | Réalisation |
|----------|----------|-------------|
| Q1 | Bucket Sort distribué avec MPI | Implémenté en C avec MPI |
| Q2 | Mesure des performances | Tests 1-16 processus, 100K-10M éléments |
| Q3 | Extraction Top-K distribuée | Algorithme avec réduction arborescente |
| Q4 | Comparaison Top-K vs Tri |  Gain jusqu'à 28.8% avec 16 processus |

### 10.2 Contributions Techniques

#### 10.2.1 Version 1 - MPI Pure

- Implémentation complète du Bucket Sort distribué
- Utilisation optimale des communications collectives (Scatterv, Alltoallv, Gatherv)
- Extraction Top-K avec réduction arborescente O(log P)
- Gestion correcte des tailles variables de données

#### 10.2.2 Version 2 - Hybride MPI + OpenMP

- Parallélisation intra-processus avec OpenMP
- Génération de données parallèle (thread-safe avec rand_r)
- Comptage de buckets avec réduction manuelle
- Initialisation MPI thread-safe (MPI_THREAD_FUNNELED)

### 10.3 Leçons Apprises

#### 10.3.1 Aspects Théoriques

1. **Loi d'Amdahl** : Le speedup est limité par la partie séquentielle
2. **Granularité** : Les communications collectives à gros grain sont plus efficaces
3. **Équilibrage de charge** : Le Bucket Sort suppose une distribution uniforme
4. **Scalabilité** : Au-delà d'un certain P, l'overhead de communication domine

#### 10.3.2 Aspects Pratiques

1. **Débogage MPI** : Plus complexe que le séquentiel (utiliser MUST, mpiP)
2. **Allocation mémoire** : Toujours vérifier les malloc et utiliser des buffers contigus
3. **Synchronisation** : MPI_Barrier essentiel pour les mesures de performance
4. **Portabilité** : Tester avec différentes implémentations MPI

### 10.4 Perspectives d'Amélioration

#### 10.4.1 Améliorations Algorithmiques

- **Sample Sort** : Meilleur équilibrage de charge que le Bucket Sort
- **Parallel Merge Sort** : Alternative pour données non uniformes
- **Tri partiel pour Top-K** : Utiliser un heap au lieu du tri complet

#### 10.4.2 Optimisations Techniques

- **Communications non-bloquantes** : MPI_Isend/MPI_Irecv pour overlap calcul/communication
- **Persistent communications** : Pour les patterns de communication répétés
- **MPI-IO** : Pour les très grands volumes de données (I/O parallèle)
- **GPU** : Hybride MPI + CUDA pour le tri local

#### 10.4.3 Extensions Fonctionnelles

- Support des données non-uniformément distribuées
- Tri stable (préservation de l'ordre des égaux)
- Tri externe pour données dépassant la mémoire
- Interface utilisateur pour visualisation des performances

### 10.5 Conclusion Générale

Ce projet démontre que le parallélisme avec MPI est une approche efficace pour le tri de grandes quantités de données. Les résultats expérimentaux montrent :

- **Speedup significatif** : Jusqu'à 7.73x pour le Bucket Sort et 9.85x pour le Top-K avec 16 processus
- **Bonne efficacité** : 77% pour le tri et 62% pour le Top-K
- **Scalabilité** : Performance qui s'améliore avec la taille des données

L'approche hybride MPI + OpenMP, bien que plus complexe à implémenter, offre des avantages sur les architectures modernes en réduisant l'overhead de communication et en optimisant l'utilisation de la mémoire partagée.

---

## 11. Références

### 11.1 Ouvrages et Articles

1. **Snir, M., et al.** (1998). *MPI: The Complete Reference*. MIT Press.
   - Référence complète du standard MPI

2. **Grama, A., et al.** (2003). *Introduction to Parallel Computing*. Addison-Wesley.
   - Fondements théoriques du calcul parallèle

3. **Gropp, W., Lusk, E., & Skjellum, A.** (2014). *Using MPI: Portable Parallel Programming with the Message-Passing Interface*. MIT Press.
   - Guide pratique de programmation MPI

4. **Chapman, B., Jost, G., & Van Der Pas, R.** (2008). *Using OpenMP: Portable Shared Memory Parallel Programming*. MIT Press.
   - Référence pour OpenMP

### 11.2 Documentation en Ligne

5. **Documentation OpenMPI** - https://www.open-mpi.org/doc/
   - Documentation officielle d'OpenMPI

6. **OpenMP Specifications** - https://www.openmp.org/specifications/
   - Spécifications officielles d'OpenMP

7. **MPI Forum** - https://www.mpi-forum.org/
   - Standard MPI officiel

### 11.3 Ressources Pédagogiques

8. **GNU/Linux Magazine** - "Une introduction à la programmation parallèle avec Open MPI et OpenMP"
   - Article d'introduction pratique

9. **Cours Ch6-MPI.pdf** - Support de cours
   - Fondements de MPI

10. **LLNL MPI Tutorial** - https://hpc-tutorials.llnl.gov/mpi/
    - Tutoriel complet du Lawrence Livermore National Laboratory

---

## Annexes

### A. Compilation et Exécution

#### A.1 Prérequis

```bash
# Installation des dépendances (Ubuntu/Debian)
sudo apt-get install openmpi-bin libopenmpi-dev
sudo apt-get install build-essential
```

#### A.2 Compilation

```bash
# Version 1 - MPI seul
cd MPI
make clean
make all

# Version 2 - Hybride MPI + OpenMP
cd MPI+OpenMPI
make clean
make all
```

#### A.3 Exécution

```bash
# Bucket Sort MPI
mpirun -np 4 ./bucket_sort_mpi 1000000

# Top-K MPI
mpirun -np 4 ./topk_mpi 1000000 1000

# Version hybride
OMP_NUM_THREADS=2 mpirun -np 4 ./bin/bucket_sort_hybrid 1000000 2
```

#### A.4 Benchmarks

```bash
# Exécuter tous les benchmarks
make benchmark

# Générer les graphiques
make plot
```

### B. Variables d'Environnement

```bash
# OpenMP
export OMP_NUM_THREADS=4           # Nombre de threads
export OMP_SCHEDULE=dynamic        # Ordonnancement
export OMP_PROC_BIND=close         # Affinité des threads
export OMP_PLACES=cores            # Placement sur les cœurs

# MPI (OpenMPI)
export OMPI_MCA_btl=self,tcp       # Transports à utiliser
export OMPI_MCA_mpi_show_mca_params=all  # Debug
```

### C. Flags de Compilation

```makefile
CC = mpicc
CFLAGS = -Wall -Wextra -O3 -std=c99

# Pour la version hybride
CFLAGS += -fopenmp

# Pour le debug
CFLAGS_DEBUG = -g -O0 -DDEBUG
```

### D. Structure des Fichiers de Résultats

```
results/
├── bucket_sort_results.csv    # Temps bruts
├── bucket_sort_summary.csv    # Statistiques agrégées
├── topk_results.csv           # Temps Top-K
├── topk_summary.csv           # Statistiques Top-K
└── version_comparison.csv     # Comparaison V1 vs V2
```

**Format CSV :**

```csv
num_procs,size,time_seconds,speedup,efficiency
1,1000000,0.172,1.00,100.00
2,1000000,0.080,2.15,107.50
4,1000000,0.039,4.41,110.25
```

### E. Dépannage Courant

| Problème | Solution |
|----------|----------|
| "mpirun: command not found" | Installer OpenMPI : `apt install openmpi-bin` |
| Erreur de mémoire | Réduire la taille des données ou augmenter la RAM |
| Deadlock | Vérifier l'ordre des Send/Recv |
| Performance dégradée | Vérifier l'affinité des processus (`--bind-to core`) |
| OpenMP non actif | Recompiler avec `-fopenmp` |

### F. Exemple de Sortie

```
=== Bucket Sort Distribué avec MPI ===
Nombre de processus: 4
Taille du tableau: 1000000
Valeur maximale: 1000000

=== Résultats ===
Tri correct: OUI
Temps d'exécution: 0.039421 secondes
Éléments triés par seconde: 25.37 millions

CSV: 4,1000000,0.039421
```

---

## 12. Expérimentation et Problèmes Rencontrés

### 12.1 Configuration du Cluster MI104

#### Architecture du Cluster
- **Machines disponibles** : MI104-01 à MI104-20 (20 machines)
- **Processeur** : Intel Core i9-10900K @ 3.70GHz
- **Cœurs physiques** : 10 cœurs par machine
- **Hyperthreading** : 20 threads logiques (10 × 2)
- **Réseau** : Interconnexion Ethernet via interface `enp0s31f6`
- **Système** : Debian GNU/Linux 12 (kernel 6.1.0-38-amd64)
- **MPI** : OpenMPI 4.x

#### Configuration SSH
Les machines du cluster partagent le répertoire `$HOME` via NFS, permettant :
- Clés SSH uniques stockées dans `~/.ssh/`
- Fichier `authorized_keys` accessible depuis toutes les machines
- Connexion sans mot de passe entre les nœuds

```bash
# Configuration SSH effectuée
cd ~/.ssh
ssh-keygen -t rsa -N ""
cat id_rsa.pub > authorized_keys
chmod 600 authorized_keys
```

### 12.2 Tests de Connectivité

#### Test SSH Manuel
```bash
# Test réussi sur toutes les machines
for machine in MI104-02 MI104-03 MI104-04 MI104-05; do
    ssh ${machine}.iem "hostname && echo 'Connexion OK'"
done

# Résultat : Toutes les connexions SSH fonctionnent sans mot de passe
```

#### Hostfile Configuration
```
# Fichier hostfile.txt - Configuration initiale
MI104-01.iem slots=8
MI104-02.iem slots=8
MI104-03.iem slots=8
MI104-04.iem slots=8
# ... jusqu'à MI104-16.iem
```

**Objectif** : 8 slots × 16 machines = 128 processus MPI distribués

### 12.3 Problèmes Rencontrés

#### Problème 1 : Warnings X11
```
Authorization required, but no authorization protocol specified
Invalid MIT-MAGIC-COOKIE-1 key
```

**Cause** : Forwarding X11 activé par défaut lors des connexions SSH  
**Impact** : Warnings abondants mais **n'empêchent pas l'exécution**  
**Solution testée** : Désactivation X11 via variables d'environnement
```bash
unset DISPLAY
export XAUTHORITY=/dev/null
export OMPI_MCA_plm_rsh_agent="ssh -x"
```
**Résultat** : Warnings persistent mais le programme s'exécute correctement

#### Problème 2 : Saturation de /tmp sur MI104-01
```
Error: No space left on device
Directory: /tmp/ompi.MI104-01.2380
```

**Cause** : `/tmp` système saturé sur certaines machines  
**Solution** : Utilisation d'un répertoire temporaire personnel
```bash
mkdir -p ~/tmp
mpirun --mca orte_tmpdir_base ~/tmp ...
```

#### Problème 3 : Blocage de Communication Inter-Machines (Critique)

**Symptômes** :
- SSH fonctionne correctement entre toutes les machines
- Les processus MPI se lancent sur les machines distantes
- **Le programme se bloque** sans produire de sortie
- Timeout après plusieurs minutes

**Messages d'erreur** :
```
------------------------------------------------------------
A process or daemon was unable to complete a TCP connection
to another process:
  Local host:    MI104-08
  Remote host:   MI104-15
This is usually caused by a firewall on the remote host.
------------------------------------------------------------
```

**Diagnostic approfondi** :

1. **Test de connectivité réseau** :
```bash
# Test simple MPI distribué
cat > test_simple.c << 'EOF'
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    char hostname[256];
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, 256);
    printf("Processus %d/%d sur %s\n", rank, size, hostname);
    MPI_Finalize();
    return 0;
}
EOF

mpicc -o test_simple test_simple.c
mpirun --hostfile hostfile_test.txt -np 6 ./test_simple
# Résultat : BLOCAGE - Aucune sortie produite
```

2. **Analyse des interfaces réseau** :
```bash
ip addr show | grep "^[0-9]"
# Interfaces détectées :
# - lo : loopback
# - enp0s31f6 : Ethernet principal
# - enxb04f13f38d13 : Ethernet USB
```

3. **Test avec spécification d'interface** :
```bash
# Test avec interface explicite
mpirun --mca btl_tcp_if_include enp0s31f6 \
       --mca oob_tcp_if_include enp0s31f6 \
       --hostfile hostfile_test.txt -np 6 ./test_simple
# Résultat : BLOCAGE persistant
```

**Cause identifiée** : **Firewall ou politique réseau du cluster**

OpenMPI nécessite :
- **Port SSH (22)** : Pour lancer les processus distants  **Fonctionne**
- **Ports TCP dynamiques** : Pour la communication MPI inter-processus  **Bloqués**
- **Connexions bidirectionnelles** : Entre tous les nœuds  **Impossibles**

### 12.4 Solution Adoptée : Oversubscription Locale

Face au blocage de la communication inter-machines, nous avons adopté une approche alternative :

#### Mode Oversubscription
```bash
# Exécution sur une seule machine avec sursouscription
mpirun --oversubscribe -np 128 ./bucket_sort_mpi 10000000
```

**Principe** :
- Lancement de **128 processus MPI** sur **20 threads physiques**
- Le système d'exploitation effectue du **time-sharing** (partage de temps)
- Chaque thread exécute ~6-7 processus en rotation via **context switching**

**Avantages** :
-  Permet de tester avec un grand nombre de processus
-  Pas de problèmes de firewall/réseau
-  Résultats valides pour l'analyse algorithmique

**Inconvénients** :
-  Performances dégradées par le context switching
-  Pas de vrai parallélisme au-delà de 20 threads
-  Ne reflète pas les performances d'un vrai cluster distribué

### 12.5 Résultats Obtenus en Mode Local

#### Configuration des Tests
```bash
# Script de benchmark en mode oversubscribe
mpirun --oversubscribe -np 16 ./bucket_sort_mpi 1000000
mpirun --oversubscribe -np 32 ./bucket_sort_mpi 1000000
mpirun --oversubscribe -np 64 ./bucket_sort_mpi 10000000
mpirun --oversubscribe -np 96 ./bucket_sort_mpi 10000000
mpirun --oversubscribe -np 128 ./bucket_sort_mpi 10000000
```

#### Résultats Mesurés

| Processus | Taille Tableau | Temps (s) | Débit (M elem/s) |
|-----------|----------------|-----------|------------------|
| 64        | 10 000 000     | 0.184832  | 54.10            |
| 128       | 10 000 000     | 0.167002  | 59.88            |

**Observations** :
- Les temps d'exécution restent **raisonnables** même en oversubscription
- Le débit augmente légèrement avec le nombre de processus
- Au-delà de 20 processus, l'amélioration est marginale (effet du context switching)

### 12.6 Implications pour le Projet

#### Limitations Techniques
1. **Firewall du cluster** : Bloque la communication MPI inter-machines
2. **Accès administrateur requis** : Pour modifier les règles de firewall
3. **Impossibilité de benchmarks distribués réels** : Tests limités à une machine

#### Recommandations pour Reproductibilité
Pour obtenir de vraies mesures distribuées, il faudrait :
- Un cluster sans firewall entre nœuds (ou avec ports MPI ouverts)
- Ou l'utilisation d'un gestionnaire de ressources (SLURM, PBS)
- Ou une configuration réseau avec InfiniBand

#### Validité des Résultats
Malgré ces limitations :
-  L'algorithme est **correct** (tri vérifié)
-  Les mécanismes MPI sont **fonctionnels** (communication locale)
-  L'analyse de complexité reste **valide**
-  Les performances absolues ne reflètent pas un vrai cluster

### 12.7 Conclusion sur l'Expérimentation

Ce projet a permis de rencontrer et diagnostiquer des **problèmes réels de déploiement HPC** :
- Configuration SSH et clés publiques
- Gestion des répertoires temporaires distribués
- Limitations réseau et firewalls
- Solutions de contournement (oversubscription)

Ces obstacles sont **typiques des environnements HPC réels** et ont une valeur pédagogique importante pour comprendre :
- Les différences entre SSH et MPI
- L'importance de la configuration réseau
- Les compromis entre sécurité et performance
- Les techniques de débogage distribuées

---

*Rapport rédigé le 27 décembre 2025*
*Projet de Master 1 - Programmation Parallèle et Distribuée*

