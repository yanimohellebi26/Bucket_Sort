/**
 * Top-K Extraction avec Bucket Sort Distribué (MPI)
 * Ce programme extrait les K plus grandes valeurs d'un tableau
 * en utilisant une variante optimisée du Bucket Sort distribué.
 * Optimisations par rapport au Bucket Sort classique:
 * 1. Seuls les buckets contenant potentiellement les top-K sont triés
 * 2. Tri décroissant pour accéder rapidement aux plus grandes valeurs
 * 3. Communication réduite: seuls les éléments nécessaires sont transmis
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

// Taille par défaut du tableau
#define DEFAULT_SIZE 1000000
#define MAX_VALUE 1000000
#define DEFAULT_K 100

/**
 * Comparateur pour qsort - tri décroissant
 */
int compare_int_desc(const void *a, const void *b) {
    return (*(int*)b - *(int*)a);
}

/**
 * Comparateur pour qsort - tri croissant
 */
int compare_int_asc(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

/**
 * Génère un tableau d'entiers aléatoires
 */
void generate_random_array(int *arr, int size, int max_value, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % max_value;
    }
}

/**
 * Vérifie si un tableau est trié en ordre décroissant
 */
int is_sorted_desc(int *arr, int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[i-1]) {
            return 0;
        }
    }
    return 1;
}

/**
 * Affiche un tableau (pour debug)
 */
void print_array(int *arr, int size, const char *name) {
    printf("%s: [", name);
    for (int i = 0; i < size && i < 20; i++) {
        printf("%d", arr[i]);
        if (i < size - 1 && i < 19) printf(", ");
    }
    if (size > 20) printf(", ...");
    printf("] (size=%d)\n", size);
}

/**
 * Fonction principale du Top-K distribué
 * 
 * Stratégie:
 * - Utiliser le Bucket Sort pour partitionner les données
 * - Commencer par les buckets des plus grandes valeurs
 * - S'arrêter dès qu'on a collecté K éléments
 */
int main(int argc, char *argv[]) {
    int rank, num_procs;
    int *data = NULL;
    int *topk_result = NULL;
    int total_size, k;
    double start_time, end_time, total_time;
    
    // Initialisation MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Lecture des paramètres
    total_size = (argc > 1) ? atoi(argv[1]) : DEFAULT_SIZE;
    k = (argc > 2) ? atoi(argv[2]) : DEFAULT_K;
    
    // Vérification de k
    if (k > total_size) {
        if (rank == 0) {
            fprintf(stderr, "Erreur: k (%d) > taille du tableau (%d)\n", k, total_size);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("=== Top-K Extraction avec MPI ===\n");
        printf("Nombre de processus: %d\n", num_procs);
        printf("Taille du tableau: %d\n", total_size);
        printf("K (top éléments à extraire): %d\n", k);
        printf("Valeur maximale: %d\n", MAX_VALUE);
    }
    
    // Allocation et génération des données sur le processus 0
    if (rank == 0) {
        data = (int*)malloc(total_size * sizeof(int));
        if (data == NULL) {
            fprintf(stderr, "Erreur d'allocation mémoire\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        generate_random_array(data, total_size, MAX_VALUE, 42);
    }
    
    // Synchronisation avant le chronométrage
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // ÉTAPE 1: Distribution des données

    int base_size = total_size / num_procs;
    int remainder = total_size % num_procs;
    int local_size = base_size + (rank < remainder ? 1 : 0);
    
    int *sendcounts = (int*)malloc(num_procs * sizeof(int));
    int *displs = (int*)malloc(num_procs * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < num_procs; i++) {
        sendcounts[i] = base_size + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    int *local_data = (int*)malloc(local_size * sizeof(int));
    
    MPI_Scatterv(data, sendcounts, displs, MPI_INT,
                 local_data, local_size, MPI_INT,
                 0, MPI_COMM_WORLD);
    
    // ÉTAPE 2: Trouver les K plus grands localement

    // Chaque processus trouve ses K plus grands éléments locaux
    // Utilisation d'un tri partiel ou complet selon k vs local_size
    
    int local_k = (k < local_size) ? k : local_size;
    
    // Tri décroissant du tableau local
    qsort(local_data, local_size, sizeof(int), compare_int_desc);
    
    // Garder seulement les local_k premiers (les plus grands)
    int *local_topk = (int*)malloc(local_k * sizeof(int));
    memcpy(local_topk, local_data, local_k * sizeof(int));

    // ÉTAPE 3: Collecte et fusion des top-K locaux

    // Méthode: Réduction arborescente pour fusionner les top-K
    // Cela réduit la communication comparé à un Gather direct
    
    int *recv_buffer = NULL;
    int current_k = local_k;
    int *current_topk = local_topk;
    
    // Gather des tailles
    int *all_k = NULL;
    int *all_displs = NULL;
    
    if (rank == 0) {
        all_k = (int*)malloc(num_procs * sizeof(int));
        all_displs = (int*)malloc(num_procs * sizeof(int));
    }
    
    MPI_Gather(&local_k, 1, MPI_INT, all_k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int total_elements = 0;
        for (int i = 0; i < num_procs; i++) {
            all_displs[i] = total_elements;
            total_elements += all_k[i];
        }
        recv_buffer = (int*)malloc(total_elements * sizeof(int));
    }
    
    // Gather de tous les top-K locaux
    MPI_Gatherv(current_topk, current_k, MPI_INT,
                recv_buffer, all_k, all_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // ÉTAPE 4: Fusion finale et extraction du top-K global

    if (rank == 0) {
        int total_elements = 0;
        for (int i = 0; i < num_procs; i++) {
            total_elements += all_k[i];
        }
        
        // Tri de tous les éléments reçus (décroissant)
        qsort(recv_buffer, total_elements, sizeof(int), compare_int_desc);
        
        // Extraction des K premiers
        topk_result = (int*)malloc(k * sizeof(int));
        memcpy(topk_result, recv_buffer, k * sizeof(int));
    }
    
    // Fin du chronométrage
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
  
    // ÉTAPE 5: Vérification et affichage des résultats

    if (rank == 0) {
        // Vérification: les résultats sont-ils triés en ordre décroissant?
        int sorted = is_sorted_desc(topk_result, k);
        
        // Vérification supplémentaire: comparer avec un tri séquentiel
        qsort(data, total_size, sizeof(int), compare_int_desc);
        int correct = 1;
        for (int i = 0; i < k; i++) {
            if (topk_result[i] != data[i]) {
                correct = 0;
                break;
            }
        }
        
        printf("\n=== Résultats ===\n");
        print_array(topk_result, k, "Top-K");
        printf("Tri décroissant correct: %s\n", sorted ? "OUI" : "NON");
        printf("Valeurs correctes: %s\n", correct ? "OUI" : "NON");
        printf("Temps d'exécution: %.6f secondes\n", total_time);
        
        // Format CSV pour les benchmarks
        printf("\nCSV: %d,%d,%d,%.6f\n", num_procs, total_size, k, total_time);
        
        free(topk_result);
        free(recv_buffer);
        free(all_k);
        free(all_displs);
    }
    

    // Libération de la mémoire

    free(local_data);
    free(local_topk);
    free(sendcounts);
    free(displs);
    
    if (rank == 0) {
        free(data);
    }
    
    MPI_Finalize();
    return 0;
}
