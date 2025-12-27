# Makefile pour Bucket Sort et Top-K Hybride (MPI + OpenMP) - Version 2
# Auteur: Projet Master 1
# Date: Décembre 2025

# Compilateur MPI
CC = mpicc

# Flags de compilation
# -fopenmp: Active le support OpenMP
# -Wall: Active tous les avertissements
# -O3: Optimisation maximale
# -std=c99: Standard C99
CFLAGS = -Wall -O3 -std=c99 -fopenmp

# Flags MPI
MPIFLAGS = 

# Répertoires
SRC_DIR = src
BIN_DIR = bin
RESULTS_DIR = results
SCRIPTS_DIR = scripts

# Fichiers source
BUCKET_SORT_SRC = $(SRC_DIR)/bucket_sort_hybrid.c
TOPK_SRC = $(SRC_DIR)/topk_hybrid.c

# Exécutables
BUCKET_SORT_BIN = $(BIN_DIR)/bucket_sort_hybrid
TOPK_BIN = $(BIN_DIR)/topk_hybrid

# Nombre de processus MPI par défaut pour les tests
NP ?= 4

# Nombre de threads OpenMP par défaut
OMP_THREADS ?= 4

# Taille par défaut pour les tests
SIZE ?= 100000

# Valeur K par défaut pour Top-K
K ?= 100

# Cible par défaut
all: directories $(BUCKET_SORT_BIN) $(TOPK_BIN)

# Création des répertoires
directories:
	@mkdir -p $(BIN_DIR) $(RESULTS_DIR)

# Compilation du Bucket Sort hybride
$(BUCKET_SORT_BIN): $(BUCKET_SORT_SRC)
	$(CC) $(CFLAGS) $(MPIFLAGS) -o $@ $<

# Compilation du Top-K hybride
$(TOPK_BIN): $(TOPK_SRC)
	$(CC) $(CFLAGS) $(MPIFLAGS) -o $@ $<

# Test rapide du Bucket Sort
test-bucket: $(BUCKET_SORT_BIN)
	@echo "=== Test Bucket Sort Hybride ==="
	@echo "Processus MPI: $(NP), Threads OpenMP: $(OMP_THREADS), Taille: $(SIZE)"
	@OMP_NUM_THREADS=$(OMP_THREADS) mpirun -np $(NP) --oversubscribe $(BUCKET_SORT_BIN) $(SIZE) $(OMP_THREADS)

# Test rapide du Top-K
test-topk: $(TOPK_BIN)
	@echo "=== Test Top-K Hybride ==="
	@echo "Processus MPI: $(NP), Threads OpenMP: $(OMP_THREADS), Taille: $(SIZE), K: $(K)"
	@OMP_NUM_THREADS=$(OMP_THREADS) mpirun -np $(NP) --oversubscribe $(TOPK_BIN) $(SIZE) $(K) $(OMP_THREADS)

# Test avec différentes configurations hybrides
test-hybrid: $(BUCKET_SORT_BIN)
	@echo "=== Tests de configurations hybrides ==="
	@echo ""
	@echo "--- 4 processus MPI x 1 thread (total: 4) ---"
	@OMP_NUM_THREADS=1 mpirun -np 4 --oversubscribe $(BUCKET_SORT_BIN) $(SIZE) 1
	@echo ""
	@echo "--- 2 processus MPI x 2 threads (total: 4) ---"
	@OMP_NUM_THREADS=2 mpirun -np 2 --oversubscribe $(BUCKET_SORT_BIN) $(SIZE) 2
	@echo ""
	@echo "--- 1 processus MPI x 4 threads (total: 4) ---"
	@OMP_NUM_THREADS=4 mpirun -np 1 --oversubscribe $(BUCKET_SORT_BIN) $(SIZE) 4

# Test complet avec tous les programmes
test: test-bucket test-topk

# Benchmark complet
benchmark: $(BUCKET_SORT_BIN) $(TOPK_BIN)
	@echo "=== Lancement des benchmarks complets ==="
	@chmod +x $(SCRIPTS_DIR)/*.sh
	@$(SCRIPTS_DIR)/run_all_benchmarks.sh

# Benchmark Bucket Sort seulement
benchmark-bucket: $(BUCKET_SORT_BIN)
	@chmod +x $(SCRIPTS_DIR)/benchmark_bucket_sort.sh
	@$(SCRIPTS_DIR)/benchmark_bucket_sort.sh

# Benchmark Top-K seulement
benchmark-topk: $(TOPK_BIN)
	@chmod +x $(SCRIPTS_DIR)/benchmark_topk.sh
	@$(SCRIPTS_DIR)/benchmark_topk.sh

# Génération des graphiques
plot: 
	@echo "=== Génération des graphiques ==="
	@cd $(SCRIPTS_DIR) && python3 plot_results.py

# Comparaison avec la Version 1
compare: $(BUCKET_SORT_BIN)
	@chmod +x $(SCRIPTS_DIR)/compare_versions.sh
	@$(SCRIPTS_DIR)/compare_versions.sh

# Nettoyage des exécutables
clean:
	rm -rf $(BIN_DIR)

# Nettoyage complet (exécutables + résultats)
distclean: clean
	rm -rf $(RESULTS_DIR)

# Aide
help:
	@echo "=== Bucket Sort & Top-K Hybride (MPI + OpenMP) - Version 2 ==="
	@echo ""
	@echo "Compilation:"
	@echo "  make              - Compile tous les programmes"
	@echo "  make clean        - Supprime les exécutables"
	@echo "  make distclean    - Supprime exécutables et résultats"
	@echo ""
	@echo "Tests:"
	@echo "  make test         - Lance tous les tests"
	@echo "  make test-bucket  - Test Bucket Sort (NP=$(NP), OMP_THREADS=$(OMP_THREADS), SIZE=$(SIZE))"
	@echo "  make test-topk    - Test Top-K (NP=$(NP), OMP_THREADS=$(OMP_THREADS), SIZE=$(SIZE), K=$(K))"
	@echo "  make test-hybrid  - Compare différentes configurations MPI/OpenMP"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make benchmark       - Lance tous les benchmarks"
	@echo "  make benchmark-bucket - Benchmark Bucket Sort seulement"
	@echo "  make benchmark-topk   - Benchmark Top-K seulement"
	@echo "  make plot            - Génère les graphiques"
	@echo "  make compare         - Compare avec la Version 1"
	@echo ""
	@echo "Variables:"
	@echo "  NP=<n>            - Nombre de processus MPI (défaut: 4)"
	@echo "  OMP_THREADS=<n>   - Nombre de threads OpenMP (défaut: 4)"
	@echo "  SIZE=<n>          - Taille du tableau (défaut: 100000)"
	@echo "  K=<n>             - Valeur de K pour Top-K (défaut: 100)"
	@echo ""
	@echo "Exemples:"
	@echo "  make test-bucket NP=8 OMP_THREADS=2 SIZE=1000000"
	@echo "  make test-topk NP=4 OMP_THREADS=4 SIZE=500000 K=50"

.PHONY: all directories test test-bucket test-topk test-hybrid benchmark benchmark-bucket benchmark-topk plot compare clean distclean help
