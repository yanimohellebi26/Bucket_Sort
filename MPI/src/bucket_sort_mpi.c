/**
 * Bucket Sort Distribué avec MPI
 * Ce programme implémente l'algorithme Bucket Sort de manière distribuée
 * en utilisant MPI pour paralléliser le tri.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

// Taille par défaut du tableau à trier
#define DEFAULT_SIZE 1000000
#define MAX_VALUE 1000000

/**
 * Comparateur pour qsort - tri croissant
 */
int compare_int(const void *a, const void *b) {
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
 * Vérifie si un tableau est trié
 */
int is_sorted(int *arr, int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i-1]) {
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
 * Fonction principale du Bucket Sort distribué
 */
int main(int argc, char *argv[]) {
    int rank, num_procs;
    int *data = NULL;
    int *recv_bucket = NULL;
    int *sorted_data = NULL;
    int total_size;
    double start_time, end_time, total_time;
    
    // Initialisation MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Lecture de la taille du tableau depuis les arguments
    total_size = (argc > 1) ? atoi(argv[1]) : DEFAULT_SIZE;
    
    if (rank == 0) {
        printf("=== Bucket Sort Distribué avec MPI ===\n");
        printf("Nombre de processus: %d\n", num_procs);
        printf("Taille du tableau: %d\n", total_size);
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
        // print_array(data, total_size, "Données initiales");
    }
    
    // Synchronisation avant le début du chronométrage
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // ÉTAPE 1: Distribution des données

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

    // ÉTAPE 2: Création des buckets locaux
 
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

    // ÉTAPE 3: Échange des buckets (All-to-All)

    // Communication des tailles de buckets
    int *recv_counts = (int*)malloc(num_procs * sizeof(int));
    MPI_Alltoall(bucket_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
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

    // ÉTAPE 4: Tri local du bucket

    qsort(recv_bucket, total_recv, sizeof(int), compare_int);

    // ÉTAPE 5: Rassemblement des résultats

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
    
    // Fin du chronométrage
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    total_time = end_time - start_time;

    // ÉTAPE 6: Vérification et affichage des résultats

    if (rank == 0) {
        // Vérification du tri
        int sorted = is_sorted(sorted_data, total_size);
        // print_array(sorted_data, total_size, "Données triées");
        
        printf("\n=== Résultats ===\n");
        printf("Tri correct: %s\n", sorted ? "OUI" : "NON");
        printf("Temps d'exécution: %.6f secondes\n", total_time);
        printf("Éléments triés par seconde: %.2f millions\n", 
               (total_size / total_time) / 1000000.0);
        
        // Format CSV pour les benchmarks
        printf("\nCSV: %d,%d,%.6f\n", num_procs, total_size, total_time);
    }

    // Libération de la mémoire

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
