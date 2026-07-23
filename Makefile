CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter
LDFLAGS  :=

# OpenMP (disable with: make OPENMP=0)
OPENMP ?= 1
ifeq ($(OPENMP),1)
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
endif

# CUDA (enable with: make CUDA=1)
CUDA ?= 0
ifeq ($(CUDA),1)
    NVCC      := nvcc
    NVCCFLAGS := -std=c++17 -O3 -arch=native -Xcompiler -fopenmp
    LDFLAGS   += -lcudart -lm
endif

# Targets
.PHONY: all clean test

all: model 3body

model: model.cpp population.hpp
	$(CXX) $(CXXFLAGS) model.cpp -o model $(LDFLAGS)

3body: 3body.cpp population.hpp
	$(CXX) $(CXXFLAGS) 3body.cpp -o 3body $(LDFLAGS)

# CUDA-enabled model
model_cuda: model_cuda.cpp cuda_simulation.cu cuda_simulation.cuh population.hpp
	$(NVCC) $(NVCCFLAGS) cuda_simulation.cu -c -o cuda_simulation.o
	$(CXX) $(CXXFLAGS) -c model_cuda.cpp -o model_cuda.o
	$(CXX) model_cuda.o cuda_simulation.o -o model_cuda $(LDFLAGS) -lcudart

# Quick test: run the model for a few generations
test: model 3body output
	./model --popsize=256 --generations=20 --quiet
	@echo ""
	@echo "=== Best IC found ==="
	cat output/best.txt
	@echo ""
	@echo "=== Archive ==="
	cat output/archive.txt
	@echo ""
	@echo "=== Running 3-body simulation for best IC ==="
	./3body output/best.txt | head -5
	@echo "..."

# CUDA test
test-cuda: model_cuda 3body output
	./model_cuda --popsize=256 --generations=20 --quiet
	@echo ""
	@echo "=== Best IC found (CUDA) ==="
	cat output/best.txt
	@echo ""
	@echo "=== Running 3-body simulation for best IC ==="
	./3body output/best.txt | head -5
	@echo "..."

# Refinement test: refine an orbit from archive
test-refine: model 3body
	./model --refine=figure8.txt --generations=10 --mutation-sigma=0.02 --popsize=32 --seed=123 --quiet
	@echo ""
	@echo "=== Refined orbit ==="
	cat output/best.txt

# CUDA refinement test
test-refine-cuda: model_cuda 3body
	./model_cuda --refine=figure8.txt --generations=10 --mutation-sigma=0.02 --popsize=32 --seed=123 --quiet
	@echo ""
	@echo "=== Refined orbit (CUDA) ==="
	cat output/best.txt

# Run full pipeline with visualization (needs python3 + matplotlib + ffmpeg)
viz: model 3body
	./model --popsize=128 --generations=30 --seed=123
	./3body output/best.txt | python3 visualize.py --output=output/found_orbit.mp4 --fps=30

# Ensure output directory exists
output:
	mkdir -p output

# Clean build artifacts
clean:
	rm -f model 3body model_cuda *.o
	rm -rf output
