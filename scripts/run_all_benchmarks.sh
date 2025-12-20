#!/bin/bash
#
# Script de benchmark complet
# Lance tous les benchmarks et génère les graphiques
#

echo "=============================================="
echo "  Benchmark Complet - Bucket Sort & Top-K    "
echo "=============================================="
echo ""

# Création du dossier de résultats
mkdir -p results

# Benchmark Bucket Sort
echo "1. Lancement du benchmark Bucket Sort..."
./scripts/benchmark_bucket_sort.sh
echo ""

# Benchmark Top-K
echo "2. Lancement du benchmark Top-K..."
./scripts/benchmark_topk.sh
echo ""

# Génération des graphiques
echo "3. Génération des graphiques..."
if command -v python3 &> /dev/null; then
    python3 scripts/plot_results.py
    echo "   Graphiques générés dans le dossier 'results/'"
else
    echo "   Python3 non trouvé. Installez Python3 pour générer les graphiques."
fi

echo ""
echo "=============================================="
echo "  Benchmark terminé!                         "
echo "=============================================="
echo ""
echo "Fichiers générés:"
echo "  - results/bucket_sort_results.csv"
echo "  - results/bucket_sort_summary.csv"
echo "  - results/topk_results.csv"
echo "  - results/topk_summary.csv"
echo "  - results/bucket_sort_performance.png"
echo "  - results/topk_performance.png"
echo "  - results/comparison.png"
