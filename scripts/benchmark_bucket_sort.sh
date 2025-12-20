#!/bin/bash
#
# Script de benchmark pour le Bucket Sort distribué
# Mesure le temps d'exécution pour différents nombres de tâches
#

# Configuration
EXECUTABLE="./bucket_sort_mpi"
OUTPUT_FILE="results/bucket_sort_results.csv"
ARRAY_SIZES=(100000 1000000 10000000)
NUM_PROCS=(1 2 4 8 16 32 64 128)
NUM_RUNS=5  # Nombre d'exécutions pour moyenner

# Création du dossier de résultats
mkdir -p results

# En-tête du fichier CSV
echo "num_procs,array_size,run,time" > "$OUTPUT_FILE"
echo "Benchmark du Bucket Sort Distribué"


# Vérification de l'exécutable
if [ ! -f "$EXECUTABLE" ]; then
    echo "Erreur: L'exécutable $EXECUTABLE n'existe pas."
    echo "Veuillez d'abord compiler avec 'make'"
    exit 1
fi

# Vérification du nombre de cœurs disponibles
MAX_CORES=$(nproc)
echo "Nombre de cœurs disponibles: $MAX_CORES"
echo ""

for SIZE in "${ARRAY_SIZES[@]}"; do
    echo "=== Taille du tableau: $SIZE ==="
    
    for NP in "${NUM_PROCS[@]}"; do
        # Vérifier si on peut utiliser ce nombre de processus
        if [ $NP -gt $MAX_CORES ]; then
            echo "  Saut de $NP processus (> $MAX_CORES cœurs disponibles)"
            continue
        fi
        
        echo -n "  $NP processus: "
        
        for RUN in $(seq 1 $NUM_RUNS); do
            # Exécution et extraction du temps
            OUTPUT=$(mpirun --oversubscribe -np $NP $EXECUTABLE $SIZE 2>/dev/null)
            TIME=$(echo "$OUTPUT" | grep "CSV:" | cut -d',' -f3)
            
            if [ -n "$TIME" ]; then
                echo "$NP,$SIZE,$RUN,$TIME" >> "$OUTPUT_FILE"
                echo -n "."
            else
                echo -n "x"
            fi
        done
        echo " OK"
    done
    echo ""
done

echo "Résultats sauvegardés dans $OUTPUT_FILE"
echo ""
echo "Génération des statistiques..."

# Calcul des moyennes avec awk
echo ""
echo "=== Résumé des temps moyens (secondes) ==="
echo "num_procs,array_size,mean_time,std_dev" > results/bucket_sort_summary.csv

LC_NUMERIC=C awk -F',' 'NR>1 {
    key = $1","$2;
    sum[key] += $4;
    sumsq[key] += $4*$4;
    count[key]++;
}
END {
    for (key in sum) {
        mean = sum[key]/count[key];
        variance = (sumsq[key]/count[key]) - (mean*mean);
        if (variance < 0) variance = 0;
        std = sqrt(variance);
        printf "%s,%.6f,%.6f\n", key, mean, std;
    }
}' "$OUTPUT_FILE" | sort -t',' -k2,2n -k1,1n >> results/bucket_sort_summary.csv

cat results/bucket_sort_summary.csv

echo ""
echo "Benchmark terminé!"
