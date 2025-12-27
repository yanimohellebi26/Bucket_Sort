#!/bin/bash
# Script de comparaison entre Version 1 (MPI seul) et Version 2 (MPI + OpenMP)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_DIR="$SCRIPT_DIR/../../"
V2_DIR="$SCRIPT_DIR/../"
RESULTS_DIR="$SCRIPT_DIR/../results"

# Fichier de sortie
COMPARE_FILE="$RESULTS_DIR/version_comparison.csv"

# Configurations de test
SIZES=(100000 1000000 10000000)
RUNS=3

echo "============================================"
echo "  Comparaison Version 1 vs Version 2       "
echo "============================================"
echo ""

# Vérifier que les deux versions existent (V1 exécutables à la racine)
if [ ! -f "$V1_DIR/bucket_sort_mpi" ]; then
    echo "ERREUR: Version 1 non compilée. Compilez d'abord dans $V1_DIR"
    exit 1
fi

if [ ! -f "$V2_DIR/bin/bucket_sort_hybrid" ]; then
    echo "ERREUR: Version 2 non compilée. Compilez d'abord avec: make"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# En-tête du fichier de comparaison
echo "size,version,config,time,speedup_vs_seq" > "$COMPARE_FILE"

for SIZE in "${SIZES[@]}"; do
    echo "=== Taille: $SIZE ==="
    echo ""
    
    declare -A TIMES
    
    # Version 1 - Séquentiel (référence)
    echo "--- Version 1: 1 processus MPI (référence) ---"
    V1_SEQ_TIMES=()
    for RUN in $(seq 1 $RUNS); do
        OUTPUT=$(mpirun -np 1 "$V1_DIR/bucket_sort_mpi" $SIZE 2>&1)
        TIME=$(echo "$OUTPUT" | grep "Temps d'exécution:" | awk '{print $3}')
        if [ -n "$TIME" ]; then
            V1_SEQ_TIMES+=("$TIME")
            echo "  Run $RUN: $TIME s"
        fi
    done
    
    V1_SEQ_AVG=$(LC_NUMERIC=C awk -v times="${V1_SEQ_TIMES[*]}" 'BEGIN {
        n = split(times, arr, " ")
        sum = 0
        for (i = 1; i <= n; i++) sum += arr[i]
        printf "%.6f", sum/n
    }')
    echo "  Moyenne V1 (1 proc): $V1_SEQ_AVG s"
    echo "$SIZE,v1,1MPI,$V1_SEQ_AVG,1.00" >> "$COMPARE_FILE"
    echo ""
    
    # Version 1 - 4 processus MPI
    echo "--- Version 1: 4 processus MPI ---"
    V1_4P_TIMES=()
    for RUN in $(seq 1 $RUNS); do
        OUTPUT=$(mpirun -np 4 --oversubscribe "$V1_DIR/bucket_sort_mpi" $SIZE 2>&1)
        TIME=$(echo "$OUTPUT" | grep "Temps d'exécution:" | awk '{print $3}')
        if [ -n "$TIME" ]; then
            V1_4P_TIMES+=("$TIME")
            echo "  Run $RUN: $TIME s"
        fi
    done
    
    V1_4P_AVG=$(LC_NUMERIC=C awk -v times="${V1_4P_TIMES[*]}" 'BEGIN {
        n = split(times, arr, " ")
        sum = 0
        for (i = 1; i <= n; i++) sum += arr[i]
        printf "%.6f", sum/n
    }')
    V1_4P_SPEEDUP=$(LC_NUMERIC=C awk -v ref="$V1_SEQ_AVG" -v time="$V1_4P_AVG" 'BEGIN {printf "%.2f", ref/time}')
    echo "  Moyenne V1 (4 proc): $V1_4P_AVG s (Speedup: ${V1_4P_SPEEDUP}x)"
    echo "$SIZE,v1,4MPI,$V1_4P_AVG,$V1_4P_SPEEDUP" >> "$COMPARE_FILE"
    echo ""
    
    # Version 2 - 4 threads OpenMP (1 processus MPI)
    echo "--- Version 2: 1 processus MPI x 4 threads OpenMP ---"
    V2_OMP_TIMES=()
    for RUN in $(seq 1 $RUNS); do
        OUTPUT=$(OMP_NUM_THREADS=4 mpirun -np 1 "$V2_DIR/bin/bucket_sort_hybrid" $SIZE 4 2>&1)
        CSV=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/CSV: //')
        TIME=$(echo "$CSV" | cut -d',' -f4)
        if [ -n "$TIME" ]; then
            V2_OMP_TIMES+=("$TIME")
            echo "  Run $RUN: $TIME s"
        fi
    done
    
    V2_OMP_AVG=$(LC_NUMERIC=C awk -v times="${V2_OMP_TIMES[*]}" 'BEGIN {
        n = split(times, arr, " ")
        sum = 0
        for (i = 1; i <= n; i++) sum += arr[i]
        printf "%.6f", sum/n
    }')
    V2_OMP_SPEEDUP=$(LC_NUMERIC=C awk -v ref="$V1_SEQ_AVG" -v time="$V2_OMP_AVG" 'BEGIN {printf "%.2f", ref/time}')
    echo "  Moyenne V2 (1x4): $V2_OMP_AVG s (Speedup: ${V2_OMP_SPEEDUP}x)"
    echo "$SIZE,v2,1MPI_4OMP,$V2_OMP_AVG,$V2_OMP_SPEEDUP" >> "$COMPARE_FILE"
    echo ""
    
    # Version 2 - 2 processus MPI x 2 threads OpenMP
    echo "--- Version 2: 2 processus MPI x 2 threads OpenMP ---"
    V2_22_TIMES=()
    for RUN in $(seq 1 $RUNS); do
        OUTPUT=$(OMP_NUM_THREADS=2 mpirun -np 2 --oversubscribe "$V2_DIR/bin/bucket_sort_hybrid" $SIZE 2 2>&1)
        CSV=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/CSV: //')
        TIME=$(echo "$CSV" | cut -d',' -f4)
        if [ -n "$TIME" ]; then
            V2_22_TIMES+=("$TIME")
            echo "  Run $RUN: $TIME s"
        fi
    done
    
    V2_22_AVG=$(LC_NUMERIC=C awk -v times="${V2_22_TIMES[*]}" 'BEGIN {
        n = split(times, arr, " ")
        sum = 0
        for (i = 1; i <= n; i++) sum += arr[i]
        printf "%.6f", sum/n
    }')
    V2_22_SPEEDUP=$(LC_NUMERIC=C awk -v ref="$V1_SEQ_AVG" -v time="$V2_22_AVG" 'BEGIN {printf "%.2f", ref/time}')
    echo "  Moyenne V2 (2x2): $V2_22_AVG s (Speedup: ${V2_22_SPEEDUP}x)"
    echo "$SIZE,v2,2MPI_2OMP,$V2_22_AVG,$V2_22_SPEEDUP" >> "$COMPARE_FILE"
    echo ""
    
    # Version 2 - 4 processus MPI x 1 thread OpenMP (équivalent V1)
    echo "--- Version 2: 4 processus MPI x 1 thread OpenMP ---"
    V2_41_TIMES=()
    for RUN in $(seq 1 $RUNS); do
        OUTPUT=$(OMP_NUM_THREADS=1 mpirun -np 4 --oversubscribe "$V2_DIR/bin/bucket_sort_hybrid" $SIZE 1 2>&1)
        CSV=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/CSV: //')
        TIME=$(echo "$CSV" | cut -d',' -f4)
        if [ -n "$TIME" ]; then
            V2_41_TIMES+=("$TIME")
            echo "  Run $RUN: $TIME s"
        fi
    done
    
    V2_41_AVG=$(LC_NUMERIC=C awk -v times="${V2_41_TIMES[*]}" 'BEGIN {
        n = split(times, arr, " ")
        sum = 0
        for (i = 1; i <= n; i++) sum += arr[i]
        printf "%.6f", sum/n
    }')
    V2_41_SPEEDUP=$(LC_NUMERIC=C awk -v ref="$V1_SEQ_AVG" -v time="$V2_41_AVG" 'BEGIN {printf "%.2f", ref/time}')
    echo "  Moyenne V2 (4x1): $V2_41_AVG s (Speedup: ${V2_41_SPEEDUP}x)"
    echo "$SIZE,v2,4MPI_1OMP,$V2_41_AVG,$V2_41_SPEEDUP" >> "$COMPARE_FILE"
    echo ""
    
    # Résumé pour cette taille
    echo "=== Résumé pour taille $SIZE ==="
    echo "V1 (1 MPI):     $V1_SEQ_AVG s (référence)"
    echo "V1 (4 MPI):     $V1_4P_AVG s (speedup: ${V1_4P_SPEEDUP}x)"
    echo "V2 (1x4 OMP):   $V2_OMP_AVG s (speedup: ${V2_OMP_SPEEDUP}x)"
    echo "V2 (2x2):       $V2_22_AVG s (speedup: ${V2_22_SPEEDUP}x)"
    echo "V2 (4x1):       $V2_41_AVG s (speedup: ${V2_41_SPEEDUP}x)"
    echo ""
done

echo "============================================"
echo "  Comparaison terminée !                   "
echo "  Résultats: $COMPARE_FILE                 "
echo "============================================"
