/**
 * Bucket Sort Distribué - Version 2 (Hybride MPI + OpenMP)
 * 
 * Cette version utilise le parallélisme hybride:
 * - MPI pour le parallélisme lourd (distribution entre processus)
 * - OpenMP pour le parallélisme léger (threads au sein de chaque processus)
 * 
 * Basé sur les ressources:
 * - "Une introduction à la programmation parallèle avec Open MPI et OpenMP"
 * - Cours Ch6-MPI.pdf
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
#define DEFAULT_NUM_THREADS 4

/**
 * Comparateur pour qsort - tri croissant
 */
int compare_int(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

/**
 * Génère un tableau d'entiers aléatoires (parallélisé avec OpenMP)
 */
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

/**
 * Vérifie si un tableau est trié (parallélisé avec OpenMP)
 */
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

/**
 * Calcule la somme locale pour le comptage des buckets (parallélisé)
 */
void count_bucket_elements(int *local_data, int local_size, int *bucket_counts, 
                           int num_buckets, double range) {
    // Initialisation à zéro
    memset(bucket_counts, 0, num_buckets * sizeof(int));
    
    #ifdef _OPENMP
    // Utilisation de reduction pour éviter les race conditions
    #pragma omp parallel
    {
        int *local_counts = (int*)calloc(num_buckets, sizeof(int));
        
        #pragma omp for nowait
        for (int i = 0; i < local_size; i++) {
            int bucket_id = (int)(local_data[i] / range);
            if (bucket_id >= num_buckets) bucket_id = num_buckets - 1;
            local_counts[bucket_id]++;
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < num_buckets; j++) {
                bucket_counts[j] += local_counts[j];
            }
        }
        
        free(local_counts);
    }
    #else
    for (int i = 0; i < local_size; i++) {
        int bucket_id = (int)(local_data[i] / range);
        if (bucket_id >= num_buckets) bucket_id = num_buckets - 1;
        bucket_counts[bucket_id]++;
    }
    #endif
}

/**
 * Distribue les éléments dans les buckets (parallélisé avec OpenMP)
 */
void distribute_to_buckets(int *local_data, int local_size, int **buckets, 
                          int *bucket_indices, int num_buckets, double range) {
    #ifdef _OPENMP
    // Version séquentielle pour éviter les conflits d'écriture
    // (la parallélisation nécessiterait des structures plus complexes)
    #endif
    
    for (int i = 0; i < local_size; i++) {
        int bucket_id = (int)(local_data[i] / range);
        if (bucket_id >= num_buckets) bucket_id = num_buckets - 1;
        buckets[bucket_id][bucket_indices[bucket_id]++] = local_data[i];
    }
}

/**
 * Tri parallèle avec OpenMP (fusion sort ou quicksort selon la taille)
 */
void parallel_sort(int *arr, int size) {
    #ifdef _OPENMP
    // Pour les grands tableaux, utiliser le tri parallèle
    if (size > 10000) {
        // Diviser le tableau en sections et trier en parallèle
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
        
        // Fusion des sections triées (version simplifiée)
        // Pour une vraie implémentation, utiliser un merge sort parallèle
        qsort(arr, size, sizeof(int), compare_int);
    } else {
        qsort(arr, size, sizeof(int), compare_int);
    }
    #else
    qsort(arr, size, sizeof(int), compare_int);
    #endif
}

/**
 * Affiche les informations sur l'environnement d'exécution
 */
void print_execution_info(int rank, int num_procs) {
    if (rank == 0) {
        printf("=== Bucket Sort Hybride (MPI + OpenMP) - Version 2 ===\n");
        printf("Nombre de processus MPI: %d\n", num_procs);
        
        #ifdef _OPENMP
        printf("OpenMP activé: OUI\n");
        printf("Nombre de threads OpenMP par processus: %d\n", omp_get_max_threads());
        printf("Nombre total de threads: %d\n", num_procs * omp_get_max_threads());
        #else
        printf("OpenMP activé: NON\n");
        #endif
        
        // Afficher la version MPI
        int version, subversion;
        MPI_Get_version(&version, &subversion);
        printf("Version MPI: %d.%d\n", version, subversion);
    }
}

/**
 * Fonction principale du Bucket Sort distribué hybride
 */
int main(int argc, char *argv[]) {
    int rank, num_procs;
    int *data = NULL;
    int *recv_bucket = NULL;
    int *sorted_data = NULL;
    int total_size;
    int num_threads = DEFAULT_NUM_THREADS;
    double start_time, end_time, total_time;
    double comm_time = 0, comp_time = 0;
    
    // Initialisation MPI avec support des threads
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Avertissement: Le niveau de thread MPI demandé n'est pas supporté\n");
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Lecture des arguments
    total_size = (argc > 1) ? atoi(argv[1]) : DEFAULT_SIZE;
    if (argc > 2) {
        num_threads = atoi(argv[2]);
    }
    
    // Configuration OpenMP
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
    
    // Affichage des informations d'exécution
    print_execution_info(rank, num_procs);
    
    if (rank == 0) {
        printf("Taille du tableau: %d\n", total_size);
        printf("Valeur maximale: %d\n", MAX_VALUE);
        printf("\n");
    }
    
    // Allocation et génération des données sur le processus 0
    if (rank == 0) {
        data = (int*)malloc(total_size * sizeof(int));
        if (data == NULL) {
            fprintf(stderr, "Erreur d'allocation mémoire\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        double gen_start = MPI_Wtime();
        generate_random_array(data, total_size, MAX_VALUE, 42);
        double gen_end = MPI_Wtime();
        
        printf("Temps de génération des données: %.6f s\n", gen_end - gen_start);
    }
    
    // Synchronisation avant le chronométrage
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
    // ÉTAPE 2: Création des buckets locaux (parallélisé avec OpenMP)
    // ============================================
    double comp_start = MPI_Wtime();
    
    double range = (double)MAX_VALUE / num_procs;
    
    // Comptage parallèle des éléments par bucket
    int *bucket_counts = (int*)calloc(num_procs, sizeof(int));
    count_bucket_elements(local_data, local_size, bucket_counts, num_procs, range);
    
    // Allocation et remplissage des buckets
    int **local_buckets = (int**)malloc(num_procs * sizeof(int*));
    int *bucket_indices = (int*)calloc(num_procs, sizeof(int));
    
    for (int i = 0; i < num_procs; i++) {
        local_buckets[i] = (int*)malloc((bucket_counts[i] + 1) * sizeof(int));
    }
    
    distribute_to_buckets(local_data, local_size, local_buckets, 
                         bucket_indices, num_procs, range);
    
    comp_time += MPI_Wtime() - comp_start;
    
    // ============================================
    // ÉTAPE 3: Échange All-to-All (MPI_Alltoallv)
    // ============================================
    comm_start = MPI_Wtime();
    
    int *recv_counts = (int*)malloc(num_procs * sizeof(int));
    MPI_Alltoall(bucket_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
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
    
    recv_bucket = (int*)malloc((total_recv + 1) * sizeof(int));
    
    MPI_Alltoallv(send_buffer, bucket_counts, send_displs, MPI_INT,
                  recv_bucket, recv_counts, recv_displs, MPI_INT,
                  MPI_COMM_WORLD);
    
    comm_time += MPI_Wtime() - comm_start;
    
    // ============================================
    // ÉTAPE 4: Tri local du bucket (parallélisé avec OpenMP)
    // ============================================
    comp_start = MPI_Wtime();
    
    parallel_sort(recv_bucket, total_recv);
    
    comp_time += MPI_Wtime() - comp_start;
    
    // ============================================
    // ÉTAPE 5: Rassemblement des résultats (MPI_Gatherv)
    // ============================================
    comm_start = MPI_Wtime();
    
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
    
    comm_time += MPI_Wtime() - comm_start;
    
    // Fin du chronométrage
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
    
    // ============================================
    // ÉTAPE 6: Vérification et affichage des résultats
    // ============================================
    
    if (rank == 0) {
        int sorted = is_sorted(sorted_data, total_size);
        
        printf("\n=== Résultats ===\n");
        printf("Tri correct: %s\n", sorted ? "OUI" : "NON");
        printf("Temps total: %.6f secondes\n", total_time);
        printf("Temps de calcul: %.6f secondes (%.1f%%)\n", 
               comp_time, (comp_time/total_time)*100);
        printf("Temps de communication: %.6f secondes (%.1f%%)\n", 
               comm_time, (comm_time/total_time)*100);
        printf("Éléments triés par seconde: %.2f millions\n", 
               (total_size / total_time) / 1000000.0);
        
        // Format CSV pour les benchmarks
        #ifdef _OPENMP
        printf("\nCSV: %d,%d,%d,%.6f,%.6f,%.6f\n", 
               num_procs, omp_get_max_threads(), total_size, 
               total_time, comp_time, comm_time);
        #else
        printf("\nCSV: %d,1,%d,%.6f,%.6f,%.6f\n", 
               num_procs, total_size, total_time, comp_time, comm_time);
        #endif
    }
    
    // ============================================
    // Libération de la mémoire
    // ============================================
    
    free(local_data);
    free(sendcounts);
    free(displs);
    free(bucket_counts);
    free(bucket_indices);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);
    free(send_buffer);
    free(recv_bucket);
    
    for (int i = 0; i < num_procs; i++) {
        free(local_buckets[i]);
    }
    free(local_buckets);
    
    if (rank == 0) {
        free(data);
        free(sorted_data);
        free(final_counts);
        free(final_displs);
    }
    
    MPI_Finalize();
    return 0;
}
