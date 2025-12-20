#!/bin/bash
#
# Script de benchmark pour le Top-K distribué
# Mesure le temps d'exécution pour différents nombres de tâches
#

# Configuration
EXECUTABLE="./topk_mpi"
OUTPUT_FILE="results/topk_results.csv"
ARRAY_SIZE=10000000
K_VALUES=(100 1000 10000)
NUM_PROCS=(1 2 4 8 16 32 64 128)
NUM_RUNS=5  # Nombre d'exécutions pour moyenner

# Création du dossier de résultats
mkdir -p results

# En-tête du fichier CSV
echo "num_procs,array_size,k,run,time" > "$OUTPUT_FILE"
echo "Benchmark du Top-K Distribué"

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

for K in "${K_VALUES[@]}"; do
    echo "=== K = $K (top éléments à extraire) ==="
    
    for NP in "${NUM_PROCS[@]}"; do
        # Vérifier si on peut utiliser ce nombre de processus
        if [ $NP -gt $MAX_CORES ]; then
            echo "  Saut de $NP processus (> $MAX_CORES cœurs disponibles)"
            continue
        fi
        
        echo -n "  $NP processus: "
        
        for RUN in $(seq 1 $NUM_RUNS); do
            # Exécution et extraction du temps
            OUTPUT=$(mpirun --oversubscribe -np $NP $EXECUTABLE $ARRAY_SIZE $K 2>/dev/null)
            TIME=$(echo "$OUTPUT" | grep "CSV:" | cut -d',' -f4)
            
            if [ -n "$TIME" ]; then
                echo "$NP,$ARRAY_SIZE,$K,$RUN,$TIME" >> "$OUTPUT_FILE"
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
echo "num_procs,array_size,k,mean_time,std_dev" > results/topk_summary.csv

awk -F',' 'NR>1 {
    key = $1","$2","$3;
    sum[key] += $5;
    sumsq[key] += $5*$5;
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
}' "$OUTPUT_FILE" | sort -t',' -k3,3n -k1,1n >> results/topk_summary.csv

cat results/topk_summary.csv

echo ""
echo "Benchmark terminé!"
