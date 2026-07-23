#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "cuda_simulation.cuh"
#include "constants.hpp"

// CUDA device functions

__device__ double gpu_sqrt(double x) {
    return __fsqrt_rn(x);
}

__device__ double gpu_hypot(double dx, double dy) {
    return gpu_sqrt(dx * dx + dy * dy);
}

__device__ void gpu_compute_accelerations(
    const double x[3], const double y[3],
    const double m[3],
    double ax[3], double ay[3])
{
    ax[0] = ay[0] = 0.0;
    ax[1] = ay[1] = 0.0;
    ax[2] = ay[2] = 0.0;

    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double r2 = dx * dx + dy * dy;
            double inv_r3 = G / (r2 * gpu_sqrt(r2));
            double f_ij = inv_r3 * m[j];
            double f_ji = inv_r3 * m[i];

            ax[i] += f_ij * dx;
            ay[i] += f_ij * dy;
            ax[j] -= f_ji * dx;
            ay[j] -= f_ji * dy;
        }
    }
}

__device__ void gpu_yoshida_step(
    double x[3], double y[3],
    double vx[3], double vy[3],
    const double m[3])
{
    const double w1 = YOSHIDA_C1 * DT;
    const double w0 = YOSHIDA_C0 * DT;
    const double hw1 = 0.5 * w1;
    const double hw0 = 0.5 * w0;

    double ax[3], ay[3];

    for (int i = 0; i < 3; ++i) {
        x[i] += hw1 * vx[i];
        y[i] += hw1 * vy[i];
    }
    gpu_compute_accelerations(x, y, m, ax, ay);
    for (int i = 0; i < 3; ++i) {
        vx[i] += w1 * ax[i];
        vy[i] += w1 * ay[i];
    }
    for (int i = 0; i < 3; ++i) {
        x[i] += hw1 * vx[i];
        y[i] += hw1 * vy[i];
    }

    for (int i = 0; i < 3; ++i) {
        x[i] += hw0 * vx[i];
        y[i] += hw0 * vy[i];
    }
    gpu_compute_accelerations(x, y, m, ax, ay);
    for (int i = 0; i < 3; ++i) {
        vx[i] += w0 * ax[i];
        vy[i] += w0 * ay[i];
    }
    for (int i = 0; i < 3; ++i) {
        x[i] += hw0 * vx[i];
        y[i] += hw0 * vy[i];
    }

    for (int i = 0; i < 3; ++i) {
        x[i] += hw1 * vx[i];
        y[i] += hw1 * vy[i];
    }
    gpu_compute_accelerations(x, y, m, ax, ay);
    for (int i = 0; i < 3; ++i) {
        vx[i] += w1 * ax[i];
        vy[i] += w1 * ay[i];
    }
    for (int i = 0; i < 3; ++i) {
        x[i] += hw1 * vx[i];
        y[i] += hw1 * vy[i];
    }
}

__device__ bool gpu_is_escaping(int i, const double x[3], const double y[3],
                                const double vx[3], const double vy[3],
                                const double m[3])
{
    int j = (i + 1) % 3;
    int k = (i + 2) % 3;

    double M_jk = m[j] + m[k];
    double com_x = (m[j] * x[j] + m[k] * x[k]) / M_jk;
    double com_y = (m[j] * y[j] + m[k] * y[k]) / M_jk;

    double rx = x[i] - com_x;
    double ry = y[i] - com_y;
    double vx_rel = vx[i] - (m[j] * vx[j] + m[k] * vx[k]) / M_jk;
    double vy_rel = vy[i] - (m[j] * vy[j] + m[k] * vy[k]) / M_jk;

    double r2 = rx * rx + ry * ry;
    double r = gpu_sqrt(r2);
    double v2 = vx_rel * vx_rel + vy_rel * vy_rel;

    double E_spec = 0.5 * v2 - M_jk / r;
    double v_radial = (rx * vx_rel + ry * vy_rel) / r;

    return (E_spec > 0.0 && v_radial > 0.01);
}

__device__ double gpu_optimal_rotation_angle(
    const double x1[3], const double y1[3],
    const double x2[3], const double y2[3],
    const int perm[3])
{
    double num = 0.0, den = 0.0;
    for (int i = 0; i < 3; ++i) {
        int j = perm[i];
        num += x1[i] * y2[j] - y1[i] * x2[j];
        den += x1[i] * x2[j] + y1[i] * y2[j];
    }
    return atan2(num, den);
}

__device__ double gpu_permutation_rotation_distance(
    const double x1[3], const double y1[3],
    const double x2[3], const double y2[3])
{
    const int perms[6][3] = {
        {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
    };
    double best = INFINITY;
    for (int p = 0; p < 6; ++p) {
        double theta = gpu_optimal_rotation_angle(x1, y1, x2, y2, perms[p]);
        double c = cos(theta);
        double s = sin(theta);
        double d2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            int j = perms[p][i];
            double rx = x1[i] * c - y1[i] * s;
            double ry = x1[i] * s + y1[i] * c;
            double dx = rx - x2[j];
            double dy = ry - y2[j];
            d2 += dx * dx + dy * dy;
        }
        if (d2 < best) best = d2;
    }
    return sqrt(best);
}

__device__ void gpu_normalize_state(double state[STATE_SIZE]) {
    double *x = state;
    double *y = state + 3;
    double *vx = state + 6;
    double *vy = state + 9;

    double cm_x = (x[0] + x[1] + x[2]) / 3.0;
    double cm_y = (y[0] + y[1] + y[2]) / 3.0;
    for (int i = 0; i < 3; ++i) {
        x[i] -= cm_x;
        y[i] -= cm_y;
    }

    double cm_vx = (vx[0] + vx[1] + vx[2]) / 3.0;
    double cm_vy = (vy[0] + vy[1] + vy[2]) / 3.0;
    for (int i = 0; i < 3; ++i) {
        vx[i] -= cm_vx;
        vy[i] -= cm_vy;
    }
}

__device__ void gpu_normalize_scale(double state[STATE_SIZE]) {
    double *x = state;
    double *y = state + 3;

    double r2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        r2 += x[i] * x[i] + y[i] * y[i];
    }

    constexpr double TARGET_R2 = 12.0;
    double scale = sqrt(TARGET_R2 / r2);

    double *vx = state + 6;
    double *vy = state + 9;
    for (int i = 0; i < 3; ++i) {
        x[i] *= scale;
        y[i] *= scale;
        vx[i] *= scale;
        vy[i] *= scale;
    }
}

__device__ double gpu_permutation_rotation_state_distance(
    const double s1[STATE_SIZE],
    const double s2[STATE_SIZE])
{
    double n1[STATE_SIZE], n2[STATE_SIZE];
    for (int i = 0; i < STATE_SIZE; ++i) {
        n1[i] = s1[i];
        n2[i] = s2[i];
    }
    gpu_normalize_state(n1);
    gpu_normalize_state(n2);
    gpu_normalize_scale(n1);
    gpu_normalize_scale(n2);

    const double *x1 = n1;
    const double *y1 = n1 + 3;
    const double *vx1 = n1 + 6;
    const double *vy1 = n1 + 9;
    const double *x2 = n2;
    const double *y2 = n2 + 3;
    const double *vx2 = n2 + 6;
    const double *vy2 = n2 + 9;

    const int perms[6][3] = {
        {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
    };
    double best = INFINITY;

    for (int p = 0; p < 6; ++p) {
        double theta = gpu_optimal_rotation_angle(x1, y1, x2, y2, perms[p]);
        double c = cos(theta);
        double s = sin(theta);
        double d2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            int j = perms[p][i];
            double rx = x1[i] * c - y1[i] * s;
            double ry = x1[i] * s + y1[i] * c;
            double rvx = vx1[i] * c - vy1[i] * s;
            double rvy = vx1[i] * s + vy1[i] * c;
            double dx = rx - x2[j];
            double dy = ry - y2[j];
            double dvx = rvx - vx2[j];
            double dvy = rvy - vy2[j];
            d2 += dx*dx + dy*dy + dvx*dvx + dvy*dvy;
        }
        if (d2 < best) best = d2;
    }

    double vx1_rev[3], vy1_rev[3];
    for (int i = 0; i < 3; ++i) {
        vx1_rev[i] = -vx1[i];
        vy1_rev[i] = -vy1[i];
    }
    for (int p = 0; p < 6; ++p) {
        double theta = gpu_optimal_rotation_angle(x1, y1, x2, y2, perms[p]);
        double c = cos(theta);
        double s = sin(theta);
        double d2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            int j = perms[p][i];
            double rx = x1[i] * c - y1[i] * s;
            double ry = x1[i] * s + y1[i] * c;
            double rvx = vx1_rev[i] * c - vy1_rev[i] * s;
            double rvy = vx1_rev[i] * s + vy1_rev[i] * c;
            double dx = rx - x2[j];
            double dy = ry - y2[j];
            double dvx = rvx - vx2[j];
            double dvy = rvy - vy2[j];
            d2 += dx*dx + dy*dy + dvx*dvx + dvy*dvy;
        }
        if (d2 < best) best = d2;
    }

    return sqrt(best);
}

// CUDA Kernel: Evaluate fitness
__global__ void gpu_evaluate_fitness_kernel(
    const double* __restrict__ population,
    CudaFitnessResult* __restrict__ results,
    int pop_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    const double* state = &population[idx * STATE_SIZE];
    double x[3], y[3], vx[3], vy[3];
    double x0[3], y0[3];

    for (int i = 0; i < 3; ++i) {
        x[i] = x0[i] = state[i];
        y[i] = y0[i] = state[i + 3];
        vx[i] = state[i + 6];
        vy[i] = state[i + 9];
    }
    double m[3] = {1.0, 1.0, 1.0};

    CudaSimulationResult sim_result;
    sim_result.closest_return = INFINITY;
    sim_result.reason = 3;  // MAX_STEPS
    sim_result.checkpoint_count = 0;

    const int transient_steps = MAX_STEPS / TRANSIENT_RATIO;
    const double collision_dist_sq = COLLISION_DIST * COLLISION_DIST;

    int step;
    for (step = 0; step < MAX_STEPS; ++step) {
        gpu_yoshida_step(x, y, vx, vy, m);

            if (step > transient_steps) {
            double d = gpu_permutation_rotation_distance(x, y, x0, y0);
            if (d < sim_result.closest_return)
                sim_result.closest_return = d;
        }

            for (int c = 0; c < NUM_ARCHIVE_CHECKPOINTS; ++c) {
            if (step + 1 == ARCHIVE_CHECKPOINT_STEPS[c] && sim_result.checkpoint_count <= c) {
                for (int j = 0; j < 3; ++j) {
                    sim_result.checkpoint_states[c][j] = x[j];
                    sim_result.checkpoint_states[c][j + 3] = y[j];
                    sim_result.checkpoint_states[c][j + 6] = vx[j];
                    sim_result.checkpoint_states[c][j + 9] = vy[j];
                }
                sim_result.checkpoint_count = c + 1;
                break;
            }
        }

        // Collision check
        double dx01 = x[0] - x[1]; double dy01 = y[0] - y[1];
        double dx02 = x[0] - x[2]; double dy02 = y[0] - y[2];
        double dx12 = x[1] - x[2]; double dy12 = y[1] - y[2];

        if ((dx01*dx01 + dy01*dy01) < collision_dist_sq ||
            (dx02*dx02 + dy02*dy02) < collision_dist_sq ||
            (dx12*dx12 + dy12*dy12) < collision_dist_sq) {
            sim_result.reason = 1;  // COLLISION
            break;
        }

            if (gpu_is_escaping(0, x, y, vx, vy, m) ||
            gpu_is_escaping(1, x, y, vx, vy, m) ||
            gpu_is_escaping(2, x, y, vx, vy, m)) {
            sim_result.reason = 2;  // ESCAPE
            break;
        }
    }

    sim_result.steps = step;
    for (int i = 0; i < 3; ++i) {
        sim_result.pos_end[i] = x[i];
        sim_result.pos_end[i + 3] = y[i];
        sim_result.vel_end[i] = vx[i];
        sim_result.vel_end[i + 3] = vy[i];
    }

    double base = (double)sim_result.steps;
    double bonus = (sim_result.closest_return == INFINITY) ? 0.0 : exp(-sim_result.closest_return / RETURN_BONUS_SIGMA);
    results[idx].score = base * (1.0 + bonus);
    results[idx].steps = sim_result.steps;
    results[idx].closest_return = sim_result.closest_return;
    memcpy(&results[idx].sim_result, &sim_result, sizeof(CudaSimulationResult));
}

// CUDA Kernel: Compute pairwise distance matrix
__global__ void gpu_compute_distance_matrix_kernel(
    const double* __restrict__ population,
    double* __restrict__ distance_matrix,
    int pop_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= pop_size || j >= pop_size || i >= j) return;

    const double* s1 = &population[i * STATE_SIZE];
    const double* s2 = &population[j * STATE_SIZE];

    double d = gpu_permutation_rotation_state_distance(s1, s2);
    distance_matrix[i * pop_size + j] = d;
    distance_matrix[j * pop_size + i] = d;
}

// CUDA Kernel: Compute distances to archive
__global__ void gpu_archive_distance_kernel(
    const double* __restrict__ population,
    const double* __restrict__ archive,
    double* __restrict__ min_distances,
    int pop_size,
    int archive_size)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int arc = blockIdx.y * blockDim.y + threadIdx.y;

    if (ind >= pop_size || arc >= archive_size) return;

    const double* state = &population[ind * STATE_SIZE];
    const double* arch_entry = &archive[arc * STATE_SIZE];

    double d = gpu_permutation_rotation_state_distance(state, arch_entry);

    double old = atomicMin(&min_distances[ind], d);
    if (d < old) min_distances[ind] = d;
}

// Host wrapper functions

static int cuda_device_count = 0;
static bool cuda_initialized = false;

int cuda_init() {
    cudaError_t err = cudaGetDeviceCount(&cuda_device_count);
    if (err != cudaSuccess || cuda_device_count == 0) {
        fprintf(stderr, "WARNING: No CUDA device found\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# CUDA Device: %s (Compute Capability %d.%d)\n",
            prop.name, prop.major, prop.minor);

    cuda_initialized = true;
    return 1;
}

void cuda_cleanup() {
    if (cuda_initialized) {
        cudaDeviceReset();
        cuda_initialized = false;
    }
}

cudaError_t cuda_evaluate_fitness(
    const double* population,
    CudaFitnessResult* results,
    int pop_size)
{
    if (!cuda_initialized) return cudaErrorNotInitialized;

    int threads_per_block = 256;
    int blocks = (pop_size + threads_per_block - 1) / threads_per_block;

    gpu_evaluate_fitness_kernel<<<blocks, threads_per_block>>>(
        population, results, pop_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to launch fitness kernel: %s\n", cudaGetErrorString(err));
        return err;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: Kernel execution failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}

cudaError_t cuda_compute_distance_matrix(
    const double* population,
    double* distance_matrix,
    int pop_size)
{
    if (!cuda_initialized) return cudaErrorNotInitialized;

    dim3 threads_per_block(16, 16);
    dim3 blocks((pop_size + 15) / 16, (pop_size + 15) / 16);

    gpu_compute_distance_matrix_kernel<<<blocks, threads_per_block>>>(
        population, distance_matrix, pop_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to launch distance matrix kernel: %s\n", cudaGetErrorString(err));
        return err;
    }

    err = cudaDeviceSynchronize();
    return err;
}

cudaError_t cuda_compute_archive_distances(
    const double* population,
    const double* archive,
    double* min_distances,
    int pop_size,
    int archive_size)
{
    if (!cuda_initialized) return cudaErrorNotInitialized;

    cudaMemset(min_distances, 0xFF, pop_size * sizeof(double));  // NaN pattern for INFINITY

    dim3 threads_per_block(16, 16);
    dim3 blocks((pop_size + 15) / 16, (archive_size + 15) / 16);

    gpu_archive_distance_kernel<<<blocks, threads_per_block>>>(
        population, archive, min_distances, pop_size, archive_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to launch archive distance kernel: %s\n", cudaGetErrorString(err));
        return err;
    }

    err = cudaDeviceSynchronize();
    return err;
}