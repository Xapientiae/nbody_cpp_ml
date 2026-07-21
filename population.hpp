#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

#include "simulation.hpp"

// ---------------------------------------------------------------------------
// Random number generation (thread-safe with thread_local)
// ---------------------------------------------------------------------------
static thread_local std::mt19937_64 rng_local(std::random_device{}());

static inline double rand_uniform(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng_local);
}

static inline double rand_normal(double mean, double sigma) {
    std::normal_distribution<double> dist(mean, sigma);
    return dist(rng_local);
}

static inline size_t rand_index(size_t n) {
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    return dist(rng_local);
}

// ---------------------------------------------------------------------------
// Normalize: center of mass at origin, total momentum = 0
// Layout: [x0,x1,x2, y0,y1,y2, vx0,vx1,vx2, vy0,vy1,vy2]
// All masses = 1
// ---------------------------------------------------------------------------
static void normalize_state(double state[STATE_SIZE]) {
    double *x  = state;       // [0..2]
    double *y  = state + 3;   // [3..5]
    double *vx = state + 6;   // [6..8]
    double *vy = state + 9;   // [9..11]

    // Center of mass
    double cm_x = (x[0] + x[1] + x[2]) / 3.0;
    double cm_y = (y[0] + y[1] + y[2]) / 3.0;
    for (int i = 0; i < 3; ++i) {
        x[i] -= cm_x;
        y[i] -= cm_y;
    }

    // Center of mass velocity (total momentum / total mass)
    double cm_vx = (vx[0] + vx[1] + vx[2]) / 3.0;
    double cm_vy = (vy[0] + vy[1] + vy[2]) / 3.0;
    for (int i = 0; i < 3; ++i) {
        vx[i] -= cm_vx;
        vy[i] -= cm_vy;
    }
}

// ---------------------------------------------------------------------------
// Compute total energy (should be negative for bound systems)
// Units: G=1, m_i=1
// ---------------------------------------------------------------------------
static double total_energy(const double state[STATE_SIZE]) {
    const double *x  = state;
    const double *y  = state + 3;
    const double *vx = state + 6;
    const double *vy = state + 9;

    // Kinetic energy
    double KE = 0.0;
    for (int i = 0; i < 3; ++i) {
        KE += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
    }

    // Potential energy (-1/r_ij)
    double PE = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double r  = std::sqrt(dx * dx + dy * dy);
            PE -= 1.0 / r;
        }
    }

    return KE + PE;
}

// ---------------------------------------------------------------------------
// Ensure total energy is negative by rescaling velocities
// If E >= 0, we scale velocities by sqrt(0.9 * |PE| / KE) so that
// new KE = 0.9 * |PE|, giving E = -0.1 * |PE| < 0
// ---------------------------------------------------------------------------
static void ensure_bound(double state[STATE_SIZE]) {
    double E = total_energy(state);
    if (E < 0.0) return;   // already bound

    const double *x  = state;
    const double *y  = state + 3;
    double *vx = state + 6;
    double *vy = state + 9;

    // Recompute KE and PE
    double KE = 0.0;
    for (int i = 0; i < 3; ++i) {
        KE += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
    }

    double PE = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double r  = std::sqrt(dx * dx + dy * dy);
            PE -= 1.0 / r;
        }
    }

    // Desired KE = 0.9 * |PE| so E = KE + PE = -0.1 * |PE| < 0
    double scale = std::sqrt(0.9 * std::abs(PE) / KE);
    for (int i = 0; i < 3; ++i) {
        vx[i] *= scale;
        vy[i] *= scale;
    }
}

// ---------------------------------------------------------------------------
// Generate a random initial condition with center-of-mass normalization
// and E < 0.
// ---------------------------------------------------------------------------
static void generate_random_state(double state[STATE_SIZE]) {
    double *x  = state;
    double *y  = state + 3;
    double *vx = state + 6;
    double *vy = state + 9;

    // Generate positions in a box [-3, 3], but ensure no pair is too close
    for (int attempt = 0; attempt < 100; ++attempt) {
        for (int i = 0; i < 3; ++i) {
            x[i] = rand_uniform(-3.0, 3.0);
            y[i] = rand_uniform(-3.0, 3.0);
        }

        // Check minimum distance (avoid initial collision)
        double d01 = std::hypot(x[0] - x[1], y[0] - y[1]);
        double d02 = std::hypot(x[0] - x[2], y[0] - y[2]);
        double d12 = std::hypot(x[1] - x[2], y[1] - y[2]);

        if (d01 < 0.1 || d02 < 0.1 || d12 < 0.1) continue;

        // Also avoid a body being too far from the origin
        bool too_far = false;
        for (int i = 0; i < 3; ++i) {
            if (std::abs(x[i]) > 4.0 || std::abs(y[i]) > 4.0) {
                too_far = true;
                break;
            }
        }
        if (too_far) continue;

        break;  // acceptable positions
    }

    // Center positions
    double cm_x = (x[0] + x[1] + x[2]) / 3.0;
    double cm_y = (y[0] + y[1] + y[2]) / 3.0;
    for (int i = 0; i < 3; ++i) {
        x[i] -= cm_x;
        y[i] -= cm_y;
    }

    // Generate random velocities in [-1.2, 1.2], then zero total momentum
    for (int i = 0; i < 3; ++i) {
        vx[i] = rand_uniform(-1.2, 1.2);
        vy[i] = rand_uniform(-1.2, 1.2);
    }

    // Center velocities
    double cm_vx = (vx[0] + vx[1] + vx[2]) / 3.0;
    double cm_vy = (vy[0] + vy[1] + vy[2]) / 3.0;
    for (int i = 0; i < 3; ++i) {
        vx[i] -= cm_vx;
        vy[i] -= cm_vy;
    }

    // Ensure bound (E < 0)
    ensure_bound(state);
}

// ---------------------------------------------------------------------------
// Crossover: two parents produce one child.
// The child is a weighted average of parents + small random perturbation.
// ---------------------------------------------------------------------------
static void crossover(
    const double p1[STATE_SIZE],
    const double p2[STATE_SIZE],
    double child[STATE_SIZE],
    double mutation_sigma)
{
    // Blend (uniform crossover of each component)
    for (int i = 0; i < STATE_SIZE; ++i) {
        double alpha = rand_uniform(-0.2, 1.2);  // extrapolate a bit
        child[i] = alpha * p1[i] + (1.0 - alpha) * p2[i];
    }

    // Add Gaussian noise
    for (int i = 0; i < STATE_SIZE; ++i) {
        child[i] += rand_normal(0.0, mutation_sigma);
    }

    // Renormalize
    normalize_state(child);
    ensure_bound(child);
}

// ---------------------------------------------------------------------------
// Evaluate fitness of a single individual.
// score = steps (lifetime) * (1 + closest_return_bonus)
// Also stores the raw metrics for output.
// ---------------------------------------------------------------------------
struct FitnessResult {
    double score;
    int    steps;
    double closest_return;
};

static FitnessResult evaluate_fitness(const double state[STATE_SIZE]) {
    SimulationResult sim = run_simulation(state);

    FitnessResult fr;
    fr.steps = sim.steps;
    fr.closest_return = (sim.closest_return == INFINITY) ? 100.0 : sim.closest_return;

    // Base fitness: number of steps survived
    double base = (double)sim.steps;

    // Bonus for closest return: exp(-dist / sigma), sigma = 0.5
    double sigma = 0.5;
    double bonus = std::exp(-fr.closest_return / sigma);

    // Combined score
    fr.score = base * (1.0 + bonus);

    return fr;
}

// ---------------------------------------------------------------------------
// Tournament selection: pick k random individuals, return the best index.
// ---------------------------------------------------------------------------
static size_t tournament_select(
    const std::vector<double>& scores,
    size_t tournament_size)
{
    size_t n = scores.size();
    size_t best_idx = rand_index(n);
    double best_score = scores[best_idx];

    for (size_t i = 1; i < tournament_size; ++i) {
        size_t idx = rand_index(n);
        if (scores[idx] > best_score) {
            best_score = scores[idx];
            best_idx = idx;
        }
    }
    return best_idx;
}

// ---------------------------------------------------------------------------
// Diversity penalty: minimum Euclidean distance from state to all others
// in the given set (excluding exclude_idx).
// ---------------------------------------------------------------------------
static double crowding_distance(
    const double state[STATE_SIZE],
    const std::vector<std::vector<double>>& population,
    size_t exclude_idx = SIZE_MAX)
{
    double min_dist = INFINITY;
    for (size_t j = 0; j < population.size(); ++j) {
        if (j == exclude_idx) continue;
        const auto& other = population[j];
        double d2 = 0.0;
        for (int i = 0; i < STATE_SIZE; ++i) {
            double d = state[i] - other[i];
            d2 += d * d;
        }
        double d = std::sqrt(d2);
        if (d < min_dist) min_dist = d;
    }
    return min_dist;
}

// ---------------------------------------------------------------------------
// Permutation-aware distance between two full states (positions + velocities).
// Since bodies have equal mass, we try all 6 permutations of (0,1,2).
// Internal layout: [x1,x2,x3, y1,y2,y3, vx1,vx2,vx3, vy1,vy2,vy3]
// ---------------------------------------------------------------------------
static double permutation_state_distance(
    const double s1[STATE_SIZE],
    const double s2[STATE_SIZE])
{
    const int perms[6][3] = {
        {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
    };
    double best = INFINITY;
    for (int p = 0; p < 6; ++p) {
        double d2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            // x[i], y[i], vx[i], vy[i] vs x[perm[i]], y[perm[i]], vx[perm[i]], vy[perm[i]]
            double dx  = s1[i]     - s2[perms[p][i]];
            double dy  = s1[i + 3] - s2[perms[p][i] + 3];
            double dvx = s1[i + 6] - s2[perms[p][i] + 6];
            double dvy = s1[i + 9] - s2[perms[p][i] + 9];
            d2 += dx*dx + dy*dy + dvx*dvx + dvy*dvy;
        }
        if (d2 < best) best = d2;
    }
    return std::sqrt(best);
}

// ---------------------------------------------------------------------------
// Archive distance: minimum distance from state to all entries in archive.
// Uses permutation-aware distance (bodies are indistinguishable).
// Returns INFINITY if archive is empty.
// ---------------------------------------------------------------------------
static double archive_distance(
    const double state[STATE_SIZE],
    const std::vector<std::vector<double>>& archive)
{
    if (archive.empty()) return INFINITY;
    double min_dist = INFINITY;
    for (const auto& entry : archive) {
        double d = permutation_state_distance(state, entry.data());
        if (d < min_dist) min_dist = d;
    }
    return min_dist;
}

// ---------------------------------------------------------------------------
// Load a state from a file in 3body.cpp format.
// Skips comment lines (#), reads first non-comment line with 12 numbers.
// Returns 0 on success, 1 on failure.
// ---------------------------------------------------------------------------
static int load_state_from_file(const char *filename, double state[STATE_SIZE]) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", filename);
        return 1;
    }

    char line[1024];
    int found = 0;
    while (fgets(line, sizeof(line), fp)) {
        // Skip comment lines, blank lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        double vals[STATE_SIZE];
        int n = sscanf(line,
            "%lf %lf %lf %lf %lf %lf"
            "%lf %lf %lf %lf %lf %lf",
            &vals[0], &vals[1], &vals[2], &vals[3], &vals[4], &vals[5],
            &vals[6], &vals[7], &vals[8], &vals[9], &vals[10], &vals[11]);

        if (n == STATE_SIZE) {
            // File format: [x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3]
            // Internal format: [x1,x2,x3, y1,y2,y3, vx1,vx2,vx3, vy1,vy2,vy3]
            // Convert: internal[0..2] = {vals[0], vals[2], vals[4]} (x coords)
            //          internal[3..5] = {vals[1], vals[3], vals[5]} (y coords)
            //          internal[6..8] = {vals[6], vals[8], vals[10]} (vx)
            //          internal[9..11] = {vals[7], vals[9], vals[11]} (vy)
            state[0] = vals[0]; state[1] = vals[2]; state[2] = vals[4];
            state[3] = vals[1]; state[4] = vals[3]; state[5] = vals[5];
            state[6] = vals[6]; state[7] = vals[8]; state[8] = vals[10];
            state[9] = vals[7]; state[10] = vals[9]; state[11] = vals[11];
            found = 1;
            break;
        }
    }
    fclose(fp);

    if (!found) {
        fprintf(stderr, "ERROR: no valid state data found in '%s'\n", filename);
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Load all states from an archive file.
// Archive format: lines starting with # are metadata, lines with 12 nums are
// states. File format is same as 3body.cpp: x1 y1 x2 y2 x3 y3 vx1 vy1 vx2 vy2 vx3 vy3
// Returns number of states loaded.
// ---------------------------------------------------------------------------
static size_t load_states_from_archive(const char *filename,
                                       std::vector<std::vector<double>>& archive)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        // Archive doesn't exist yet — that's fine
        return 0;
    }

    char line[1024];
    size_t count = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        double vals[STATE_SIZE];
        int n = sscanf(line,
            "%lf %lf %lf %lf %lf %lf"
            "%lf %lf %lf %lf %lf %lf",
            &vals[0], &vals[1], &vals[2], &vals[3], &vals[4], &vals[5],
            &vals[6], &vals[7], &vals[8], &vals[9], &vals[10], &vals[11]);

        if (n == STATE_SIZE) {
            std::vector<double> state(STATE_SIZE);
            state[0] = vals[0]; state[1] = vals[2]; state[2] = vals[4];
            state[3] = vals[1]; state[4] = vals[3]; state[5] = vals[5];
            state[6] = vals[6]; state[7] = vals[8]; state[8] = vals[10];
            state[9] = vals[7]; state[10] = vals[9]; state[11] = vals[11];
            archive.push_back(state);
            ++count;
        }
    }

    fclose(fp);
    return count;
}

// ---------------------------------------------------------------------------
// Save a state to archive file (append mode).
// Format: # score=... steps=... closest_return=... seed=...
//         x1 y1 x2 y2 x3 y3  vx1 vy1 vx2 vy2 vx3 vy3
// ---------------------------------------------------------------------------
static void save_state_to_archive(
    const char *filename,
    const double state[STATE_SIZE],
    double score, int steps, double closest_return,
    int seed)
{
    FILE *fp = fopen(filename, "a");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot write to '%s'\n", filename);
        return;
    }

    // Convert internal format to file format
    const double *x  = state;       // [0..2]
    const double *y  = state + 3;   // [3..5]
    const double *vx = state + 6;   // [6..8]
    const double *vy = state + 9;   // [9..11]

    fprintf(fp, "# score=%.6g  steps=%d  closest_return=%.6g  seed=%d\n",
            score, steps, closest_return, seed);
    fprintf(fp, "%.12g %.12g  %.12g %.12g  %.12g %.12g  "
                "%.12g %.12g  %.12g %.12g  %.12g %.12g\n",
            x[0], y[0], x[1], y[1], x[2], y[2],
            vx[0], vy[0], vx[1], vy[1], vx[2], vy[2]);

    fclose(fp);
}

// ---------------------------------------------------------------------------
// Generate initial states for the refine mode:
// take the base state and create N mutated copies around it.
// ---------------------------------------------------------------------------
static void generate_refined_population(
    const double base_state[STATE_SIZE],
    std::vector<std::vector<double>>& population,
    double mutation_sigma)
{
    for (size_t i = 0; i < population.size(); ++i) {
        if (i == 0) {
            // Keep the original unchanged
            std::copy(base_state, base_state + STATE_SIZE, population[i].begin());
        } else {
            // Mutated copy
            std::copy(base_state, base_state + STATE_SIZE, population[i].begin());
            for (int j = 0; j < STATE_SIZE; ++j) {
                population[i][j] += rand_normal(0.0, mutation_sigma);
            }
            normalize_state(population[i].data());
            ensure_bound(population[i].data());
        }
    }
}

#endif // POPULATION_HPP