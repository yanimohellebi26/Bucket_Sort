/**
 * Extraction Top-K Distribué - Version 2 (Hybride MPI + OpenMP)
 * 
 * Cette version utilise le parallélisme hybride:
 * - MPI pour le parallélisme lourd (distribution entre processus)
 * - OpenMP pour le parallélisme léger (threads au sein de chaque processus)
 * 
 * Optimisation: Utilise un tri partiel pour extraire seulement les K plus grandes valeurs
 * 
 * Auteur: Projet Master 1
 * Date: Décembre 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Configuration par défaut
#define DEFAULT_SIZE 1000000
#define MAX_VALUE 1000000
#define DEFAULT_K 100
#define DEFAULT_NUM_THREADS 4

/**
 * Comparateur pour tri décroissant
 */
int compare_int_desc(const void *a, const void *b) {
    return (*(int*)b - *(int*)a);
}

/**
 * Génère un tableau d'entiers aléatoires (parallélisé avec OpenMP)
 */
void generate_random_array(int *arr, int size, int max_value, unsigned int seed) {
    #ifdef _OPENMP
    #pragma omp parallel
    {
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

/**
 * Tri parallèle avec OpenMP
 */
void parallel_sort_desc(int *arr, int size) {
    #ifdef _OPENMP
    if (size > 10000) {
        int num_threads = omp_get_max_threads();
        int chunk_size = size / num_threads;
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int start = tid * chunk_size;
            int end = (tid == num_threads - 1) ? size : start + chunk_size;
            
            qsort(arr + start, end - start, sizeof(int), compare_int_desc);
        }
        
        // Fusion finale
        qsort(arr, size, sizeof(int), compare_int_desc);
    } else {
        qsort(arr, size, sizeof(int), compare_int_desc);
    }
    #else
    qsort(arr, size, sizeof(int), compare_int_desc);
    #endif
}

/**
 * Extraction parallèle des K plus grandes valeurs locales
 * Utilise un heap ou tri partiel pour optimiser
 */
void extract_local_topk(int *local_data, int local_size, int *local_topk, int k) {
    // Tri partiel: on ne trie que ce qui est nécessaire pour obtenir les K premiers
    // Pour simplifier, on utilise qsort complet puis copie des K premiers
    // Une optimisation serait d'utiliser un heap ou quickselect
    
    int *temp = (int*)malloc(local_size * sizeof(int));
    memcpy(temp, local_data, local_size * sizeof(int));
    
    parallel_sort_desc(temp, local_size);
    
    int copy_size = (k < local_size) ? k : local_size;
    memcpy(local_topk, temp, copy_size * sizeof(int));
    
    // Si local_size < k, remplir avec des valeurs minimales
    for (int i = copy_size; i < k; i++) {
        local_topk[i] = -1;
    }
    
    free(temp);
}

/**
 * Fusion parallèle de deux tableaux triés en décroissant
 * Garde seulement les K plus grandes valeurs
 */
void merge_topk(int *arr1, int size1, int *arr2, int size2, int *result, int k) {
    int i = 0, j = 0, r = 0;
    
    while (r < k && (i < size1 || j < size2)) {
        if (i >= size1) {
            result[r++] = arr2[j++];
        } else if (j >= size2) {
            result[r++] = arr1[i++];
        } else if (arr1[i] >= arr2[j]) {
            result[r++] = arr1[i++];
        } else {
            result[r++] = arr2[j++];
        }
    }
}

/**
 * Affiche les informations sur l'environnement d'exécution
 */
void print_execution_info(int rank, int num_procs, int k) {
    if (rank == 0) {
        printf("=== Top-K Hybride (MPI + OpenMP) - Version 2 ===\n");
        printf("Nombre de processus MPI: %d\n", num_procs);
        printf("Valeur de K: %d\n", k);
        
        #ifdef _OPENMP
        printf("OpenMP activé: OUI\n");
        printf("Nombre de threads OpenMP par processus: %d\n", omp_get_max_threads());
        printf("Nombre total de threads: %d\n", num_procs * omp_get_max_threads());
        #else
        printf("OpenMP activé: NON\n");
        #endif
    }
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    int *data = NULL;
    int total_size, k;
    int num_threads = DEFAULT_NUM_THREADS;
    double start_time, end_time, total_time;
    double comm_time = 0, comp_time = 0;
    
    // Initialisation MPI avec support des threads
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Lecture des arguments
    total_size = (argc > 1) ? atoi(argv[1]) : DEFAULT_SIZE;
    k = (argc > 2) ? atoi(argv[2]) : DEFAULT_K;
    if (argc > 3) {
        num_threads = atoi(argv[3]);
    }
    
    // Validation de K
    if (k > total_size) {
        k = total_size;
    }
    
    // Configuration OpenMP
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
    
    // Affichage des informations
    print_execution_info(rank, num_procs, k);
    
    if (rank == 0) {
        printf("Taille du tableau: %d\n", total_size);
        printf("\n");
    }
    
    // Génération des données sur le processus 0
    if (rank == 0) {
        data = (int*)malloc(total_size * sizeof(int));
        if (data == NULL) {
            fprintf(stderr, "Erreur d'allocation mémoire\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        double gen_start = MPI_Wtime();
        generate_random_array(data, total_size, MAX_VALUE, 42);
        double gen_end = MPI_Wtime();
        
        printf("Temps de génération: %.6f s\n", gen_end - gen_start);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // ============================================
    // ÉTAPE 1: Distribution des données (MPI_Scatterv)
    // ============================================
    double comm_start = MPI_Wtime();
    
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
    
    comm_time += MPI_Wtime() - comm_start;
    
    // ============================================
    // ÉTAPE 2: Extraction locale des K max (parallélisé avec OpenMP)
    // ============================================
    double comp_start = MPI_Wtime();
    
    int local_k = (k < local_size) ? k : local_size;
    int *local_topk = (int*)malloc(k * sizeof(int));
    
    extract_local_topk(local_data, local_size, local_topk, k);
    
    comp_time += MPI_Wtime() - comp_start;
    
    // ============================================
    // ÉTAPE 3: Réduction arborescente pour fusionner les Top-K
    // ============================================
    
    // Approche: réduction binaire pour minimiser la communication
    // Chaque étape, les processus pairs reçoivent et fusionnent
    
    int *recv_topk = (int*)malloc(k * sizeof(int));
    int *merged_topk = (int*)malloc(k * sizeof(int));
    
    int step = 1;
    while (step < num_procs) {
        if (rank % (2 * step) == 0) {
            int partner = rank + step;
            if (partner < num_procs) {
                comm_start = MPI_Wtime();
                MPI_Recv(recv_topk, k, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                comm_time += MPI_Wtime() - comm_start;
                
                comp_start = MPI_Wtime();
                merge_topk(local_topk, k, recv_topk, k, merged_topk, k);
                memcpy(local_topk, merged_topk, k * sizeof(int));
                comp_time += MPI_Wtime() - comp_start;
            }
        } else if (rank % (2 * step) == step) {
            int partner = rank - step;
            comm_start = MPI_Wtime();
            MPI_Send(local_topk, k, MPI_INT, partner, 0, MPI_COMM_WORLD);
            comm_time += MPI_Wtime() - comm_start;
        }
        step *= 2;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
    
    // ============================================
    // ÉTAPE 4: Affichage des résultats
    // ============================================
    
    if (rank == 0) {
        printf("\n=== Top-%d Résultats ===\n", k);
        
        // Afficher les 10 premiers (ou moins si k < 10)
        int display_count = (k < 10) ? k : 10;
        printf("Top %d valeurs: ", display_count);
        for (int i = 0; i < display_count; i++) {
            printf("%d ", local_topk[i]);
        }
        printf("...\n");
        
        // Vérification: les valeurs doivent être en ordre décroissant
        int sorted = 1;
        for (int i = 1; i < k; i++) {
            if (local_topk[i] > local_topk[i-1]) {
                sorted = 0;
                break;
            }
        }
        
        printf("Ordre correct (décroissant): %s\n", sorted ? "OUI" : "NON");
        printf("Valeur maximale: %d\n", local_topk[0]);
        printf("Valeur minimale du Top-K: %d\n", local_topk[k-1]);
        
        printf("\n=== Performances ===\n");
        printf("Temps total: %.6f secondes\n", total_time);
        printf("Temps de calcul: %.6f secondes (%.1f%%)\n", 
               comp_time, (comp_time/total_time)*100);
        printf("Temps de communication: %.6f secondes (%.1f%%)\n", 
               comm_time, (comm_time/total_time)*100);
        
        // Format CSV pour les benchmarks
        #ifdef _OPENMP
        printf("\nCSV: %d,%d,%d,%d,%.6f,%.6f,%.6f\n", 
               num_procs, omp_get_max_threads(), total_size, k,
               total_time, comp_time, comm_time);
        #else
        printf("\nCSV: %d,1,%d,%d,%.6f,%.6f,%.6f\n", 
               num_procs, total_size, k, total_time, comp_time, comm_time);
        #endif
    }
    
    // Libération mémoire
    free(local_data);
    free(sendcounts);
    free(displs);
    free(local_topk);
    free(recv_topk);
    free(merged_topk);
    
    if (rank == 0) {
        free(data);
    }
    
    MPI_Finalize();
    return 0;
}
