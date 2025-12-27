#!/bin/bash
# Script principal pour lancer tous les benchmarks - Version 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  Benchmarks Bucket Sort & Top-K Hybrides  "
echo "  Version 2 (MPI + OpenMP)                 "
echo "============================================"
echo ""

# Vérifier que les exécutables existent
if [ ! -f "$SCRIPT_DIR/../bin/bucket_sort_hybrid" ] || [ ! -f "$SCRIPT_DIR/../bin/topk_hybrid" ]; then
    echo "ERREUR: Les exécutables n'existent pas."
    echo "Veuillez d'abord compiler avec: make"
    exit 1
fi

# Afficher les informations OpenMP
echo "=== Informations OpenMP ==="
if command -v lscpu &> /dev/null; then
    CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    echo "Nombre de cœurs CPU: $CORES"
fi
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-non défini}"
echo ""

# Lancement des benchmarks
echo "=== Benchmark Bucket Sort Hybride ==="
bash "$SCRIPT_DIR/benchmark_bucket_sort.sh"
echo ""

echo "=== Benchmark Top-K Hybride ==="
bash "$SCRIPT_DIR/benchmark_topk.sh"
echo ""

# Génération des graphiques
echo "=== Génération des graphiques ==="
if command -v python3 &> /dev/null; then
    python3 "$SCRIPT_DIR/plot_results.py"
else
    echo "Python3 non disponible, graphiques non générés"
fi

echo ""
echo "============================================"
echo "  Benchmarks terminés !                    "
echo "  Résultats dans: $SCRIPT_DIR/../results   "
echo "============================================"
