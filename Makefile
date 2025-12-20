# Makefile pour le projet Bucket Sort Distribué
#
# Compilation des programmes MPI pour le Bucket Sort et Top-K
#

# Compilateur MPI
MPICC = mpicc

# Flags de compilation
CFLAGS = -Wall -O3 -std=c99
DEBUG_FLAGS = -Wall -g -O0 -std=c99 -DDEBUG

# Répertoires
SRC_DIR = src
BUILD_DIR = build
RESULTS_DIR = results
SCRIPTS_DIR = scripts

# Exécutables
BUCKET_SORT = bucket_sort_mpi
TOPK = topk_mpi

# Sources
BUCKET_SORT_SRC = $(SRC_DIR)/bucket_sort_mpi.c
TOPK_SRC = $(SRC_DIR)/topk_mpi.c

# Cibles par défaut
.PHONY: all clean debug run-bucket run-topk benchmark help

all: $(BUCKET_SORT) $(TOPK)
	@echo "Compilation terminée!"
	@echo "Exécutables créés: $(BUCKET_SORT), $(TOPK)"

# Compilation du Bucket Sort
$(BUCKET_SORT): $(BUCKET_SORT_SRC)
	$(MPICC) $(CFLAGS) -o $@ $<

# Compilation du Top-K
$(TOPK): $(TOPK_SRC)
	$(MPICC) $(CFLAGS) -o $@ $<

# Mode debug
debug: CFLAGS = $(DEBUG_FLAGS)
debug: clean all
	@echo "Compilation en mode debug terminée!"

# Nettoyage
clean:
	rm -f $(BUCKET_SORT) $(TOPK)
	rm -rf $(BUILD_DIR)
	@echo "Nettoyage terminé!"

# Nettoyage complet (incluant les résultats)
distclean: clean
	rm -rf $(RESULTS_DIR)
	@echo "Nettoyage complet terminé!"

# Création des répertoires
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

# Exécution du Bucket Sort
run-bucket: $(BUCKET_SORT)
	@echo "=== Exécution du Bucket Sort ==="
	mpirun -np 4 ./$(BUCKET_SORT) 1000000

# Exécution du Top-K
run-topk: $(TOPK)
	@echo "=== Exécution du Top-K ==="
	mpirun -np 4 ./$(TOPK) 1000000 100

# Test rapide avec différents nombres de processus
test: $(BUCKET_SORT) $(TOPK)
	@echo "=== Tests rapides ==="
	@echo ""
	@echo "--- Bucket Sort ---"
	@for np in 1 2 4; do \
		echo "Processus: $$np"; \
		mpirun -np $$np ./$(BUCKET_SORT) 100000; \
		echo ""; \
	done
	@echo "--- Top-K ---"
	@for np in 1 2 4; do \
		echo "Processus: $$np"; \
		mpirun -np $$np ./$(TOPK) 100000 50; \
		echo ""; \
	done

# Benchmark complet
benchmark: all $(RESULTS_DIR)
	@echo "=== Lancement des benchmarks ==="
	chmod +x $(SCRIPTS_DIR)/*.sh
	./$(SCRIPTS_DIR)/run_all_benchmarks.sh

# Benchmark Bucket Sort seulement
benchmark-bucket: $(BUCKET_SORT) $(RESULTS_DIR)
	chmod +x $(SCRIPTS_DIR)/benchmark_bucket_sort.sh
	./$(SCRIPTS_DIR)/benchmark_bucket_sort.sh

# Benchmark Top-K seulement
benchmark-topk: $(TOPK) $(RESULTS_DIR)
	chmod +x $(SCRIPTS_DIR)/benchmark_topk.sh
	./$(SCRIPTS_DIR)/benchmark_topk.sh

# Génération des graphiques
plot: $(RESULTS_DIR)
	python3 $(SCRIPTS_DIR)/plot_results.py

# Aide
help:
	@echo "=== Makefile - Bucket Sort Distribué ==="
	@echo ""
	@echo "Cibles disponibles:"
	@echo "  all              - Compile tous les programmes (défaut)"
	@echo "  debug            - Compile en mode debug"
	@echo "  clean            - Supprime les exécutables"
	@echo "  distclean        - Supprime tout (exécutables + résultats)"
	@echo ""
	@echo "  run-bucket       - Exécute le Bucket Sort (4 processus)"
	@echo "  run-topk         - Exécute le Top-K (4 processus)"
	@echo "  test             - Tests rapides avec 1, 2, 4 processus"
	@echo ""
	@echo "  benchmark        - Lance tous les benchmarks"
	@echo "  benchmark-bucket - Benchmark Bucket Sort seulement"
	@echo "  benchmark-topk   - Benchmark Top-K seulement"
	@echo "  plot             - Génère les graphiques"
	@echo ""
	@echo "  help             - Affiche cette aide"
	@echo ""
	@echo "Exemples d'utilisation:"
	@echo "  make                           # Compile tout"
	@echo "  mpirun -np 8 ./bucket_sort_mpi 1000000"
	@echo "  mpirun -np 8 ./topk_mpi 1000000 100"
