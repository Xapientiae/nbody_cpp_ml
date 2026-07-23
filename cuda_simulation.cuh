#ifndef CUDA_SIMULATION_CUH
#define CUDA_SIMULATION_CUH

#include <cuda_runtime.h>
#include "constants.hpp"

// ---------------------------------------------------------------------------
// CUDA-accelerated simulation structures and kernels
// ---------------------------------------------------------------------------

// Simulation result structure (GPU version)
struct CudaSimulationResult {
    int steps;
    double closest_return;
    int reason;  // 0=none, 1=collision, 2=escape, 3=max_steps
    double pos_end[6];
    double vel_end[6];
    double checkpoint_states[NUM_ARCHIVE_CHECKPOINTS][STATE_SIZE];
    int checkpoint_count;
};

// Fitness result structure (GPU version)
struct CudaFitnessResult {
    double score;
    int steps;
    double closest_return;
    CudaSimulationResult sim_result;
};

// ---------------------------------------------------------------------------
// CUDA kernel: Run multiple simulations in parallel
// Each thread handles one individual
// ---------------------------------------------------------------------------
__global__ void gpu_evaluate_fitness_kernel(
    const double* __restrict__ population,      // [pop_size * STATE_SIZE]
    CudaFitnessResult* __restrict__ results,    // [pop_size]
    int pop_size);

// ---------------------------------------------------------------------------
// CUDA kernel: Compute pairwise distance matrix for diversity checks
// Each thread computes one pair (i, j) where i < j
// ---------------------------------------------------------------------------
__global__ void gpu_compute_distance_matrix_kernel(
    const double* __restrict__ population,      // [pop_size * STATE_SIZE]
    double* __restrict__ distance_matrix,       // [pop_size * pop_size]
    int pop_size);

// ---------------------------------------------------------------------------
// CUDA kernel: Compute distances from population to archive entries
// Each thread computes distance from one individual to one archive entry
// ---------------------------------------------------------------------------
__global__ void gpu_archive_distance_kernel(
    const double* __restrict__ population,      // [pop_size * STATE_SIZE]
    const double* __restrict__ archive,         // [archive_size * STATE_SIZE]
    double* __restrict__ min_distances,         // [pop_size]
    int pop_size,
    int archive_size);

// ---------------------------------------------------------------------------
// Host wrapper: Launch GPU fitness evaluation
// ---------------------------------------------------------------------------
cudaError_t cuda_evaluate_fitness(
    const double* population,
    CudaFitnessResult* results,
    int pop_size);

// ---------------------------------------------------------------------------
// Host wrapper: Launch GPU distance matrix computation
// ---------------------------------------------------------------------------
cudaError_t cuda_compute_distance_matrix(
    const double* population,
    double* distance_matrix,
    int pop_size);

// ---------------------------------------------------------------------------
// Host wrapper: Launch GPU archive distance computation
// ---------------------------------------------------------------------------
cudaError_t cuda_compute_archive_distances(
    const double* population,
    const double* archive,
    double* min_distances,
    int pop_size,
    int archive_size);

// ---------------------------------------------------------------------------
// Initialize CUDA (check device, allocate memory)
// ---------------------------------------------------------------------------
int cuda_init();

// ---------------------------------------------------------------------------
// Cleanup CUDA resources
// ---------------------------------------------------------------------------
void cuda_cleanup();

#endif // CUDA_SIMULATION_CUH