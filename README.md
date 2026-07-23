# 3-Body Orbit Search

A program that aims to find new solutions to the 3-body problem using evolutionary algorithms.

## Compiling

### CPU-only version (default)
```bash
make
```

This builds two executables:
- `model` - Evolutionary search for stable 3-body orbits
- `3body` - Simulate a specific orbit from initial conditions

### With OpenMP parallelization (default)
OpenMP is enabled by default for parallel fitness evaluation:
```bash
make OPENMP=1    # default
```

Disable with:
```bash
make OPENMP=0
```

### CUDA-accelerated version
If you have CUDA installed:
```bash
make CUDA=1
```

This builds `model_cuda` with GPU-accelerated fitness evaluation.

## Running

Quick test:
```bash
./run.sh quick
```

Full pipeline:
```bash
./run.sh all
```

See `./run.sh --help` or read the script header for all available modes.

## Manual compilation

If you prefer to compile manually:

```bash
# CPU version with OpenMP
g++ -std=c++17 -O3 -fopenmp model.cpp -o model
g++ -std=c++17 -O3 3body.cpp -o 3body

# CUDA version
make CUDA=1
```

## Files

- `model.cpp` - Main evolutionary search (CPU)
- `model_cuda.cpp` - CUDA-accelerated version
- `3body.cpp` - Orbit simulator
- `population.hpp` - Simulation engine, genetic operators, and fitness functions
- `constants.hpp` - All simulation constants
- `cuda_simulation.cu/cuh` - CUDA kernel implementations
- `Makefile` - Build system
- `run.sh` - Automated pipeline script
- `visualize.py` - Visualization tool

## Output

All output goes to the `output/` directory:
- `output/best.txt` - Best orbit found
- `output/archive.txt` - Archive of all good orbits found
- `output/history.csv` - Generation-by-generation statistics