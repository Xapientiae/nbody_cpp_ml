#ifndef CUDA_SIMULATION_CUH
#define CUDA_SIMULATION_CUH

#include <cuda_runtime.h>
#include "constants.hpp"

// CUDA simulation structures and kernels

// Simulation result
struct CudaSimulationResult {
    int steps;
    double closest_return;
    int reason;  // 0=none, 1=collision, 2=escape, 3=max_steps
    double pos_end[6];
    double vel_end[6];
    double checkpoint_states[NUM_ARCHIVE_CHECKPOINTS][STATE_SIZE];
    int checkpoint_count;
};

// Fitness result
struct CudaFitnessResult {
    double score;
    int steps;
    double closest_return;
    CudaSimulationResult sim_result;
};

// CUDA kernel: Evaluate fitness
__global__ void gpu_evaluate_fitness_kernel(
    const double* __restrict__ population,      // [pop_size * STATE_SIZE]
    CudaFitnessResult* __restrict__ results,    // [pop_size]
    int pop_size);

// CUDA kernel: Compute pairwise distance matrix
__global__ void gpu_compute_distance_matrix_kernel(
    const double* __restrict__ population,      // [pop_size * STATE_SIZE]
    double* __restrict__ distance_matrix,       // [pop_size * pop_size]
    int pop_size);

// CUDA kernel: Compute distances to archive
__global__ void gpu_archive_distance_kernel(
    const double* __restrict__ population,      // [pop_size * STATE_SIZE]
    const double* __restrict__ archive,         // [archive_size * STATE_SIZE]
    double* __restrict__ min_distances,         // [pop_size]
    int pop_size,
    int archive_size);

// Host wrapper: GPU fitness evaluation
cudaError_t cuda_evaluate_fitness(
    const double* population,
    CudaFitnessResult* results,
    int pop_size);

// Host wrapper: GPU distance matrix
cudaError_t cuda_compute_distance_matrix(
    const double* population,
    double* distance_matrix,
    int pop_size);

// Host wrapper: GPU archive distances
cudaError_t cuda_compute_archive_distances(
    const double* population,
    const double* archive,
    double* min_distances,
    int pop_size,
    int archive_size);

// Initialize CUDA
int cuda_init();

// Cleanup CUDA
void cuda_cleanup();

#endif // CUDA_SIMULATION_CUH