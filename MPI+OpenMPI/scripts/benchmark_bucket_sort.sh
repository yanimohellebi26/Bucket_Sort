#!/bin/bash
# Benchmark pour Bucket Sort Hybride (MPI + OpenMP) - Version 2
# Compare différentes configurations de processus MPI et threads OpenMP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/../bin"
RESULTS_DIR="$SCRIPT_DIR/../results"
EXECUTABLE="$BIN_DIR/bucket_sort_hybrid"

# Créer le répertoire de résultats
mkdir -p "$RESULTS_DIR"

# Fichiers de sortie
RAW_FILE="$RESULTS_DIR/bucket_sort_hybrid_raw.csv"
SUMMARY_FILE="$RESULTS_DIR/bucket_sort_hybrid_summary.csv"
HYBRID_FILE="$RESULTS_DIR/hybrid_comparison.csv"

# Configurations à tester
SIZES=(100000 1000000 10000000)
MPI_PROCS=(1 2 4 8 16)
OMP_THREADS=(1 2 4)
RUNS=3

echo "=== Benchmark Bucket Sort Hybride (MPI + OpenMP) ==="
echo "Tailles: ${SIZES[*]}"
echo "Processus MPI: ${MPI_PROCS[*]}"
echo "Threads OpenMP: ${OMP_THREADS[*]}"
echo "Répétitions: $RUNS"
echo ""

# En-tête du fichier brut
echo "mpi_procs,omp_threads,total_threads,size,run,time,comp_time,comm_time" > "$RAW_FILE"

# En-tête du fichier de comparaison hybride
echo "config,mpi_procs,omp_threads,total_threads,size,avg_time,speedup" > "$HYBRID_FILE"

# Stockage du temps de référence (1 proc, 1 thread)
declare -A REF_TIMES

for SIZE in "${SIZES[@]}"; do
    echo "=== Taille: $SIZE ==="
    
    for NP in "${MPI_PROCS[@]}"; do
        for THREADS in "${OMP_THREADS[@]}"; do
            TOTAL_THREADS=$((NP * THREADS))
            CONFIG="${NP}MPI_${THREADS}OMP"
            
            echo "Configuration: $NP processus MPI x $THREADS threads OpenMP (Total: $TOTAL_THREADS)"
            
            TIMES=()
            
            for RUN in $(seq 1 $RUNS); do
                # Exécution avec configuration hybride
                OUTPUT=$(OMP_NUM_THREADS=$THREADS mpirun -np $NP --oversubscribe "$EXECUTABLE" $SIZE $THREADS 2>&1)
                
                # Extraction du temps depuis la sortie CSV
                CSV_LINE=$(echo "$OUTPUT" | grep "^CSV:" | sed 's/CSV: //')
                
                if [ -n "$CSV_LINE" ]; then
                    TIME=$(echo "$CSV_LINE" | cut -d',' -f4)
                    COMP_TIME=$(echo "$CSV_LINE" | cut -d',' -f5)
                    COMM_TIME=$(echo "$CSV_LINE" | cut -d',' -f6)
                    
                    # Enregistrer dans le fichier brut
                    echo "$NP,$THREADS,$TOTAL_THREADS,$SIZE,$RUN,$TIME,$COMP_TIME,$COMM_TIME" >> "$RAW_FILE"
                    
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
                
                # Stocker le temps de référence (1 proc, 1 thread)
                if [ $NP -eq 1 ] && [ $THREADS -eq 1 ]; then
                    REF_TIMES[$SIZE]=$AVG
                fi
                
                # Calculer le speedup
                REF=${REF_TIMES[$SIZE]:-$AVG}
                SPEEDUP=$(LC_NUMERIC=C awk -v ref="$REF" -v avg="$AVG" 'BEGIN {printf "%.2f", ref/avg}')
                
                echo "$CONFIG,$NP,$THREADS,$TOTAL_THREADS,$SIZE,$AVG,$SPEEDUP" >> "$HYBRID_FILE"
            fi
            
            echo ""
        done
    done
done

# Générer le fichier de résumé avec les meilleures configurations
echo "=== Génération du résumé ==="
echo "size,best_config,best_mpi,best_omp,best_time,best_speedup" > "$SUMMARY_FILE"

for SIZE in "${SIZES[@]}"; do
    BEST=$(grep ",$SIZE," "$HYBRID_FILE" | sort -t',' -k6 -n | head -1)
    if [ -n "$BEST" ]; then
        CONFIG=$(echo "$BEST" | cut -d',' -f1)
        MPI=$(echo "$BEST" | cut -d',' -f2)
        OMP=$(echo "$BEST" | cut -d',' -f3)
        TIME=$(echo "$BEST" | cut -d',' -f6)
        SPEEDUP=$(echo "$BEST" | cut -d',' -f7)
        echo "$SIZE,$CONFIG,$MPI,$OMP,$TIME,$SPEEDUP" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "=== Benchmark terminé ==="
echo "Résultats bruts: $RAW_FILE"
echo "Comparaison hybride: $HYBRID_FILE"
echo "Résumé: $SUMMARY_FILE"
