#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cstring>

// Physical Constants
constexpr double G = 1.0;              // gravitational constant
constexpr double DT = 0.01;            // timestep

// Simulation Parameters
constexpr int    N = 3;                 // number of bodies
constexpr int    STATE_SIZE = 12;       // x1..3, y1..3, vx1..3, vy1..3
constexpr int    MAX_STEPS = 100000;    // maximum simulation steps
constexpr double COLLISION_DIST = 0.05; // collision threshold (AU)
constexpr double ESCAPE_DIST = 100.0;   // escape threshold (AU)
constexpr double EJECTION_DIST = 15.0;  // ejection threshold (AU) - penalty if exceeded

// Yoshida 4th-order Coefficients
constexpr double YOSHIDA_C1 = 1.351207191959657;
constexpr double YOSHIDA_C0 = -1.702414383919314;

// Population / Evolutionary Algorithm Parameters
constexpr double POSITION_BOX_MIN = -3.0;  // min position for random initialization
constexpr double POSITION_BOX_MAX = 3.0;   // max position for random initialization
constexpr double POSITION_MAX_DIST = 4.0;  // max distance from origin
constexpr double POSITION_MIN_DIST = 0.1;  // min initial distance between bodies
constexpr double VELOCITY_RANGE_MIN = -1.2; // min velocity for random initialization
constexpr double VELOCITY_RANGE_MAX = 1.2;  // max velocity for random initialization
constexpr double MUTATION_ALPHA_MIN = -0.2; // extrapolation factor lower bound
constexpr double MUTATION_ALPHA_MAX = 1.2;  // extrapolation factor upper bound
constexpr double RETURN_BONUS_SIGMA = 0.5;  // sigma for return distance bonus
constexpr double TRANSIENT_RATIO = 20;      // transient period = MAX_STEPS / this

// Generator Parameters
constexpr double GENERATOR_POS_RANGE = 2.0;     // position range [-range, +range]
constexpr double GENERATOR_VEL_RANGE = 2.0;     // velocity range [-range, +range]
constexpr double GENERATOR_MIN_DIST = 0.2;      // minimum distance between bodies
constexpr double GENERATOR_TARGET_R2 = 12.0;    // target sum of squared radii

// Archive Checkpoint Times
constexpr int ARCHIVE_CHECKPOINT_STEPS[] = {3000, 7000, 13000, 17000, 29000, 51000, 67000, 100000};
constexpr int NUM_ARCHIVE_CHECKPOINTS = 8;

// Archive / Diversity Parameters
constexpr double DEFAULT_ARCHIVE_DIST_THRESHOLD = 0.3; // min dist from archive entries (stricter)
constexpr double DEFAULT_ARCHIVE_PENALTY = 0.7;         // penalty fraction if too close
constexpr double DEFAULT_DIVERSITY_THRESHOLD = 0.7;     // min crowding distance
constexpr double DEFAULT_DIVERSITY_PENALTY = 0.5;       // penalty fraction
constexpr double EJECTION_PENALTY = 0.4;                 // 40% score reduction per ejection

// Archive Penalty Parameters
constexpr double ARCHIVE_PENALTY_MAX = 0.95;  // maximum penalty at distance 0 (increased)
constexpr double ARCHIVE_PENALTY_EXPONENT = 3.0;  // exponential decay exponent (faster drop-off)

// Stop reasons
enum class StopReason : int {
    NONE,
    COLLISION,
    ESCAPE,
    MAX_STEPS
};

// Simulation result
struct SimulationResult {
    double t_end;
    int    steps;
    StopReason reason;
    double closest_return;        // best distance to initial state (permutation + rotation aware)
    double pos_end[6];            // final positions  [x1,y1, x2,y2, x3,y3]
    double vel_end[6];            // final velocities [vx1,vy1, vx2,vy2, vx3,vy3]
    double checkpoint_states[NUM_ARCHIVE_CHECKPOINTS][STATE_SIZE];  // states at checkpoint times
    int    checkpoint_count;      // number of checkpoints actually recorded
};

// Apply 2D rotation to points
static void rotate_points(double x[3], double y[3], double theta) {
    double c = std::cos(theta);
    double s = std::sin(theta);
    for (int i = 0; i < 3; ++i) {
        double xi = x[i], yi = y[i];
        x[i] = xi * c - yi * s;
        y[i] = xi * s + yi * c;
    }
}

// Apply 2D rotation to state
static void rotate_state(double state[STATE_SIZE], double theta) {
    double c = std::cos(theta);
    double s = std::sin(theta);
    double *x  = state;       // [0..2]
    double *y  = state + 3;   // [3..5]
    double *vx = state + 6;   // [6..8]
    double *vy = state + 9;   // [9..11]
    for (int i = 0; i < 3; ++i) {
        double xi = x[i], yi = y[i];
        x[i] = xi * c - yi * s;
        y[i] = xi * s + yi * c;
        double vxi = vx[i], vyi = vy[i];
        vx[i] = vxi * c - vyi * s;
        vy[i] = vxi * s + vyi * c;
    }
}

// Compute optimal rotation angle
static double optimal_rotation_angle(
    const double x1[3], const double y1[3],
    const double x2[3], const double y2[3],
    const int perm[3])
{
    double num = 0.0, den = 0.0;
    for (int i = 0; i < 3; ++i) {
        int j = perm[i];
        num  += x1[i] * y2[j] - y1[i] * x2[j];
        den  += x1[i] * x2[j] + y1[i] * y2[j];
    }
    return std::atan2(num, den);
}

// Permutation + rotation-aware distance
static double permutation_rotation_distance(
    const double x1[3], const double y1[3],
    const double x2[3], const double y2[3])
{
    const int perms[6][3] = {
        {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
    };
    double best = INFINITY;
    for (int p = 0; p < 6; ++p) {
        double theta = optimal_rotation_angle(x1, y1, x2, y2, perms[p]);

        double c = std::cos(theta);
        double s = std::sin(theta);

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
    return std::sqrt(best);
}

// Compute accelerations
static void compute_accelerations(
    const double x[3], const double y[3],
    const double m[3],
    double ax[3], double ay[3])
{
    ax[0] = ay[0] = 0.0;
    ax[1] = ay[1] = 0.0;
    ax[2] = ay[2] = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double r2 = dx * dx + dy * dy;
            double inv_r3 = G / (r2 * std::sqrt(r2));
            double f_ij = inv_r3 * m[j];
            double f_ji = inv_r3 * m[i];

            ax[i] += f_ij * dx;
            ay[i] += f_ij * dy;
            ax[j] -= f_ji * dx;
            ay[j] -= f_ji * dy;
        }
    }
}

// Yoshida step
static void yoshida_step(
    double x[3], double y[3],
    double vx[3], double vy[3],
    const double m[3])
{
    const double w1 = YOSHIDA_C1 * DT;
    const double w0 = YOSHIDA_C0 * DT;
    const double hw1 = 0.5 * w1;
    const double hw0 = 0.5 * w0;

    for (int i = 0; i < N; ++i) {
        x[i] += hw1 * vx[i];
        y[i] += hw1 * vy[i];
    }
    double ax[3], ay[3];
    compute_accelerations(x, y, m, ax, ay);
    for (int i = 0; i < N; ++i) {
        vx[i] += w1 * ax[i];
        vy[i] += w1 * ay[i];
    }
    for (int i = 0; i < N; ++i) {
        x[i] += hw1 * vx[i];
        y[i] += hw1 * vy[i];
    }

    for (int i = 0; i < N; ++i) {
        x[i] += hw0 * vx[i];
        y[i] += hw0 * vy[i];
    }
    compute_accelerations(x, y, m, ax, ay);
    for (int i = 0; i < N; ++i) {
        vx[i] += w0 * ax[i];
        vy[i] += w0 * ay[i];
    }
    for (int i = 0; i < N; ++i) {
        x[i] += hw0 * vx[i];
        y[i] += hw0 * vy[i];
    }

    for (int i = 0; i < N; ++i) {
        x[i] += hw1 * vx[i];
        y[i] += hw1 * vy[i];
    }
    compute_accelerations(x, y, m, ax, ay);
    for (int i = 0; i < N; ++i) {
        vx[i] += w1 * ax[i];
        vy[i] += w1 * ay[i];
    }
    for (int i = 0; i < N; ++i) {
        x[i] += hw1 * vx[i];
        y[i] += hw1 * vy[i];
    }
}

// Check if body is escaping
static bool is_escaping(int i, const double x[3], const double y[3],
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
    double r  = std::sqrt(r2);
    double v2 = vx_rel * vx_rel + vy_rel * vy_rel;

    double E_spec = 0.5 * v2 - M_jk / r;
    double v_radial = (rx * vx_rel + ry * vy_rel) / r;

    return (E_spec > 0.0 && v_radial > 0.01);
}

// Run simulation
static SimulationResult run_simulation(const double state[STATE_SIZE]) {
    double x[3], y[3], vx[3], vy[3];
    double x0[3], y0[3];   // copy of initial positions for periodicity check

    for (int i = 0; i < 3; ++i) {
        x[i]  = x0[i] = state[i];
        y[i]  = y0[i] = state[i + 3];
        vx[i] = state[i + 6];
        vy[i] = state[i + 9];
    }
    double m[3] = {1.0, 1.0, 1.0};

    SimulationResult result;
    result.closest_return = INFINITY;
    result.reason = StopReason::MAX_STEPS;
    result.checkpoint_count = 0;

    // Track closest return only after the initial transient
    const int transient_steps = MAX_STEPS / TRANSIENT_RATIO;
    const double collision_dist_sq = COLLISION_DIST * COLLISION_DIST;

    int step;
    for (step = 0; step < MAX_STEPS; ++step) {
        yoshida_step(x, y, vx, vy, m);

        // Closest return check
        if (step > transient_steps) {
            double d = permutation_rotation_distance(x, y, x0, y0);
            if (d < result.closest_return)
                result.closest_return = d;
        }

        // Checkpoint recording
        for (int c = 0; c < NUM_ARCHIVE_CHECKPOINTS; ++c) {
            if (step + 1 == ARCHIVE_CHECKPOINT_STEPS[c] && result.checkpoint_count <= c) {
                // Record the current state at this checkpoint
                for (int j = 0; j < 3; ++j) {
                    result.checkpoint_states[c][j] = x[j];
                    result.checkpoint_states[c][j + 3] = y[j];
                    result.checkpoint_states[c][j + 6] = vx[j];
                    result.checkpoint_states[c][j + 9] = vy[j];
                }
                result.checkpoint_count = c + 1;
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
            result.reason = StopReason::COLLISION;
            break;
        }

        // Escape check
        if (is_escaping(0, x, y, vx, vy, m) ||
            is_escaping(1, x, y, vx, vy, m) ||
            is_escaping(2, x, y, vx, vy, m)) {
            result.reason = StopReason::ESCAPE;
            break;
        }
    }

    result.steps = step;
    result.t_end = step * DT;
    for (int i = 0; i < 3; ++i) {
        result.pos_end[i]     = x[i];
        result.pos_end[i + 3] = y[i];
        result.vel_end[i]     = vx[i];
        result.vel_end[i + 3] = vy[i];
    }

    return result;
}

// Random number generation
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

// Normalize state
static void normalize_state(double state[STATE_SIZE]) {
    double *x  = state;       // [0..2]
    double *y  = state + 3;   // [3..5]
    double *vx = state + 6;   // [6..8]
    double *vy = state + 9;   // [9..11]

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

// Normalize scale
static void normalize_scale(double state[STATE_SIZE]) {
    double *x  = state;       // [0..2]
    double *y  = state + 3;   // [3..5]
    double *vx = state + 6;   // [6..8]
    double *vy = state + 9;   // [9..11]

    double r2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        r2 += x[i] * x[i] + y[i] * y[i];
    }

    constexpr double TARGET_R2 = 12.0;
    double scale = std::sqrt(TARGET_R2 / r2);

    for (int i = 0; i < 3; ++i) {
        x[i] *= scale;
        y[i] *= scale;
        vx[i] *= scale;
        vy[i] *= scale;
    }
}

// Compute energies
static void compute_energies(const double state[STATE_SIZE], double &KE, double &PE) {
    const double *x  = state;
    const double *y  = state + 3;
    const double *vx = state + 6;
    const double *vy = state + 9;

    KE = 0.0;
    for (int i = 0; i < 3; ++i) {
        KE += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);
    }

    PE = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double r  = std::sqrt(dx * dx + dy * dy);
            PE -= 1.0 / r;
        }
    }
}

// Ensure bound (E < 0)
static void ensure_bound(double state[STATE_SIZE]) {
    double KE, PE;
    compute_energies(state, KE, PE);
    if (KE + PE < 0.0) return;  // already bound

    double scale = std::sqrt(0.9 * std::abs(PE) / KE);
    double *vx = state + 6;
    double *vy = state + 9;
    for (int i = 0; i < 3; ++i) {
        vx[i] *= scale;
        vy[i] *= scale;
    }
}

// Generate random state
static void generate_random_state(double state[STATE_SIZE]) {
    double *x  = state;
    double *y  = state + 3;
    double *vx = state + 6;
    double *vy = state + 9;

    
    std::uniform_real_distribution<double> pos(-GENERATOR_POS_RANGE, GENERATOR_POS_RANGE);
    std::uniform_real_distribution<double> vel(-GENERATOR_VEL_RANGE, GENERATOR_VEL_RANGE);

    auto dist = [&](int a, int b) {
        double dx = x[a] - x[b];
        double dy = y[a] - y[b];
        return std::sqrt(dx * dx + dy * dy);
    };

    while (true) {
        x[0] = pos(rng_local);
        y[0] = pos(rng_local);
        x[1] = pos(rng_local);
        y[1] = pos(rng_local);
        x[2] = -(x[0] + x[1]);
        y[2] = -(y[0] + y[1]);

        // Normalize scale
        double sumr2 = 0;
        for (int i = 0; i < 3; i++)
            sumr2 += x[i] * x[i] + y[i] * y[i];
        double scale = std::sqrt(GENERATOR_TARGET_R2 / sumr2);
        for (int i = 0; i < 3; i++) {
            x[i] *= scale;
            y[i] *= scale;
        }

        if (dist(0, 1) < GENERATOR_MIN_DIST) continue;
        if (dist(0, 2) < GENERATOR_MIN_DIST) continue;
        if (dist(1, 2) < GENERATOR_MIN_DIST) continue;

        vx[0] = vel(rng_local);
        vy[0] = vel(rng_local);
        vx[1] = vel(rng_local);
        vy[1] = vel(rng_local);
        vx[2] = -(vx[0] + vx[1]);
        vy[2] = -(vy[0] + vy[1]);

        // Energy check
        double T = 0;
        for (int i = 0; i < 3; i++)
            T += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]);

        double U = -1.0 / dist(0, 1) - 1.0 / dist(0, 2) - 1.0 / dist(1, 2);

        if (T + U >= 0) continue;

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

        return;
    }
}

// Crossover
static void crossover(
    const double p1[STATE_SIZE],
    const double p2[STATE_SIZE],
    double child[STATE_SIZE],
    double mutation_sigma)
{
    for (int i = 0; i < STATE_SIZE; ++i) {
        double alpha = rand_uniform(MUTATION_ALPHA_MIN, MUTATION_ALPHA_MAX);  // extrapolate a bit
        child[i] = alpha * p1[i] + (1.0 - alpha) * p2[i];
    }

    for (int i = 0; i < STATE_SIZE; ++i) {
        child[i] += rand_normal(0.0, mutation_sigma);
    }

    normalize_state(child);
    ensure_bound(child);
}

// Evaluate fitness
struct FitnessResult {
    double score;
    int    steps;
    double closest_return;
    SimulationResult sim_result;  // Full simulation result including checkpoint states
};

static FitnessResult evaluate_fitness(const double state[STATE_SIZE]) {
    SimulationResult sim = run_simulation(state);

    FitnessResult fr;
    fr.sim_result = sim;
    fr.steps = sim.steps;
    fr.closest_return = sim.closest_return;

    double base = (double)sim.steps;

    double bonus = (sim.closest_return == INFINITY)
        ? 0.0
        : std::exp(-sim.closest_return / RETURN_BONUS_SIGMA);

    fr.score = base * (1.0 + bonus);

    return fr;
}

// Tournament selection
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

static double permutation_rotation_state_distance(
    const double s1[STATE_SIZE],
    const double s2[STATE_SIZE]);

// Diversity penalty
static double crowding_distance(
    const double state[STATE_SIZE],
    const std::vector<std::vector<double>>& population,
    size_t exclude_idx = std::numeric_limits<size_t>::max())
{
    double min_dist = INFINITY;
    for (size_t j = 0; j < population.size(); ++j) {
        if (j == exclude_idx) continue;
        double d = permutation_rotation_state_distance(state, population[j].data());
        if (d < min_dist) min_dist = d;
    }
    return min_dist;
}

// Permutation + rotation-aware distance (full state)
static double permutation_rotation_state_distance(
    const double s1[STATE_SIZE],
    const double s2[STATE_SIZE])
{
    double n1[STATE_SIZE], n2[STATE_SIZE];
    std::copy(s1, s1 + STATE_SIZE, n1);
    std::copy(s2, s2 + STATE_SIZE, n2);
    normalize_state(n1);
    normalize_state(n2);
    normalize_scale(n1);
    normalize_scale(n2);

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
    
    auto compute_dist = [&](const double* vx1_local, const double* vy1_local) {
        double best_local = INFINITY;
        for (int p = 0; p < 6; ++p) {
                double theta = optimal_rotation_angle(x1, y1, x2, y2, perms[p]);
            
                double c = std::cos(theta);
            double s = std::sin(theta);
            double d2 = 0.0;
            for (int i = 0; i < 3; ++i) {
                int j = perms[p][i];
                double rx = x1[i] * c - y1[i] * s;
                double ry = x1[i] * s + y1[i] * c;
                double rvx = vx1_local[i] * c - vy1_local[i] * s;
                double rvy = vx1_local[i] * s + vy1_local[i] * c;
                
                double dx  = rx - x2[j];
                double dy  = ry - y2[j];
                double dvx = rvx - vx2[j];
                double dvy = rvy - vy2[j];
                d2 += dx*dx + dy*dy + dvx*dvx + dvy*dvy;
            }
            if (d2 < best_local) best_local = d2;
        }
        return best_local;
    };
    
    best = compute_dist(vx1, vy1);
    
    double vx1_rev[3], vy1_rev[3];
    for (int i = 0; i < 3; ++i) {
        vx1_rev[i] = -vx1[i];
        vy1_rev[i] = -vy1[i];
    }
    double best_rev = compute_dist(vx1_rev, vy1_rev);
    if (best_rev < best) best = best_rev;
    
    return std::sqrt(best);
}

// Archive distance
static double archive_distance(
    const double state[STATE_SIZE],
    const std::vector<std::vector<double>>& archive)
{
    if (archive.empty()) return INFINITY;
    double min_dist = INFINITY;
    for (const auto& entry : archive) {
        double d = permutation_rotation_state_distance(state, entry.data());
        if (d < min_dist) min_dist = d;
    }
    return min_dist;
}

// Load state from file
static int load_state_from_file(const char *filename, double state[STATE_SIZE]) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", filename);
        return 1;
    }

    char line[1024];
    int found = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        double vals[STATE_SIZE];
        int n = sscanf(line,
            "%lf %lf %lf %lf %lf %lf"
            "%lf %lf %lf %lf %lf %lf",
            &vals[0], &vals[1], &vals[2], &vals[3], &vals[4], &vals[5],
            &vals[6], &vals[7], &vals[8], &vals[9], &vals[10], &vals[11]);

        if (n == STATE_SIZE) {
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

// Load states from archive
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
            normalize_state(state.data());
            normalize_scale(state.data());
            archive.push_back(state);
            ++count;
        }
    }

    fclose(fp);
    return count;
}

// Save state to archive
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

// Generate refined population
static void generate_refined_population(
    const double base_state[STATE_SIZE],
    std::vector<std::vector<double>>& population,
    double mutation_sigma)
{
    for (size_t i = 0; i < population.size(); ++i) {
        if (i == 0) {
            std::copy(base_state, base_state + STATE_SIZE, population[i].begin());
        } else {
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