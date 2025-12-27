#!/bin/bash
# Benchmark pour Top-K Hybride (MPI + OpenMP) - Version 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/../bin"
RESULTS_DIR="$SCRIPT_DIR/../results"
EXECUTABLE="$BIN_DIR/topk_hybrid"

# Créer le répertoire de résultats
mkdir -p "$RESULTS_DIR"

# Fichiers de sortie
RAW_FILE="$RESULTS_DIR/topk_hybrid_raw.csv"
SUMMARY_FILE="$RESULTS_DIR/topk_hybrid_summary.csv"

# Configurations à tester
SIZES=(100000 1000000 10000000)
K_VALUES=(100 1000 10000)
MPI_PROCS=(1 2 4 8 16)
OMP_THREADS=(1 2 4)
RUNS=3

echo "=== Benchmark Top-K Hybride (MPI + OpenMP) ==="
echo "Tailles: ${SIZES[*]}"
echo "Valeurs K: ${K_VALUES[*]}"
echo "Processus MPI: ${MPI_PROCS[*]}"
echo "Threads OpenMP: ${OMP_THREADS[*]}"
echo "Répétitions: $RUNS"
echo ""

# En-tête du fichier brut
echo "mpi_procs,omp_threads,total_threads,size,k,run,time,comp_time,comm_time" > "$RAW_FILE"

# En-tête du fichier de résumé
echo "size,k,best_config,best_mpi,best_omp,best_time,speedup" > "$SUMMARY_FILE"

# Stockage des temps de référence
declare -A REF_TIMES

for SIZE in "${SIZES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        # S'assurer que K <= SIZE
        if [ $K -gt $SIZE ]; then
            continue
        fi
        
        echo "=== Taille: $SIZE, K: $K ==="
        
        BEST_TIME=999999
        BEST_CONFIG=""
        BEST_MPI=1
        BEST_OMP=1
        
        for NP in "${MPI_PROCS[@]}"; do
            for THREADS in "${OMP_THREADS[@]}"; do
                TOTAL_THREADS=$((NP * THREADS))
                CONFIG="${NP}MPI_${THREADS}OMP"
                
                echo "Configuration: $NP processus MPI x $THREADS threads OpenMP"
                
                TIMES=()
                
                for RUN in $(seq 1 $RUNS); do
                    OUTPUT=$(OMP_NUM_THREADS=$THREADS mpirun -np $NP --oversubscribe "$EXECUTABLE" $SIZE $K $THREADS 2>&1)
                    
                    CSV_LINE=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/CSV: //')
                    
                    if [ -n "$CSV_LINE" ]; then
                        TIME=$(echo "$CSV_LINE" | cut -d',' -f5)
                        COMP_TIME=$(echo "$CSV_LINE" | cut -d',' -f6)
                        COMM_TIME=$(echo "$CSV_LINE" | cut -d',' -f7)
                        
                        echo "$NP,$THREADS,$TOTAL_THREADS,$SIZE,$K,$RUN,$TIME,$COMP_TIME,$COMM_TIME" >> "$RAW_FILE"
                        
                        TIMES+=("$TIME")
                        echo "  Run $RUN: $TIME s"
                    else
                        echo "  Run $RUN: ERREUR"
                    fi
                done
                
                # Calcul de la moyenne
                if [ ${#TIMES[@]} -gt 0 ]; then
                    AVG=$(LC_NUMERIC=C awk -v times="${TIMES[*]}" 'BEGIN {
                        n = split(times, arr, " ")
                        sum = 0
                        for (i = 1; i <= n; i++) sum += arr[i]
                        printf "%.6f", sum/n
                    }')
                    
                    echo "  Moyenne: $AVG s"
                    
                    # Stocker le temps de référence
                    if [ $NP -eq 1 ] && [ $THREADS -eq 1 ]; then
                        REF_TIMES["${SIZE}_${K}"]=$AVG
                    fi
                    
                    # Vérifier si c'est le meilleur temps
                    IS_BETTER=$(LC_NUMERIC=C awk -v avg="$AVG" -v best="$BEST_TIME" 'BEGIN {print (avg < best) ? 1 : 0}')
                    if [ "$IS_BETTER" -eq 1 ]; then
                        BEST_TIME=$AVG
                        BEST_CONFIG=$CONFIG
                        BEST_MPI=$NP
                        BEST_OMP=$THREADS
                    fi
                fi
                
                echo ""
            done
        done
        
        # Calculer le speedup
        REF=${REF_TIMES["${SIZE}_${K}"]:-$BEST_TIME}
        SPEEDUP=$(LC_NUMERIC=C awk -v ref="$REF" -v best="$BEST_TIME" 'BEGIN {printf "%.2f", ref/best}')
        
        echo "Meilleure configuration pour $SIZE/$K: $BEST_CONFIG (Temps: $BEST_TIME s, Speedup: ${SPEEDUP}x)"
        echo "$SIZE,$K,$BEST_CONFIG,$BEST_MPI,$BEST_OMP,$BEST_TIME,$SPEEDUP" >> "$SUMMARY_FILE"
        echo ""
    done
done

echo "=== Benchmark terminé ==="
echo "Résultats bruts: $RAW_FILE"
echo "Résumé: $SUMMARY_FILE"
