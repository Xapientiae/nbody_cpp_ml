#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <cmath>
#include <cstring>
#include <algorithm>
#include "constants.hpp"

// ---------------------------------------------------------------------------
// Constants are now defined in constants.hpp
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Stop reasons
// ---------------------------------------------------------------------------
enum class StopReason : int {
    NONE,
    COLLISION,
    ESCAPE,
    MAX_STEPS
};

// ---------------------------------------------------------------------------
// Simulation result
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Apply 2D rotation to a set of 3 points (x[i], y[i]) by angle theta
// ---------------------------------------------------------------------------
static void rotate_points(double x[3], double y[3], double theta) {
    double c = std::cos(theta);
    double s = std::sin(theta);
    for (int i = 0; i < 3; ++i) {
        double xi = x[i], yi = y[i];
        x[i] = xi * c - yi * s;
        y[i] = xi * s + yi * c;
    }
}

// ---------------------------------------------------------------------------
// Apply 2D rotation to a full state (positions + velocities)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Compute optimal rotation angle that aligns set (x1,y1) to (x2,y2)
// (with a given permutation of indices).
// Minimizes Σ|R(θ)·p1_i - p2_perm[i]|²
// Returns the optimal angle.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Permutation + rotation-aware distance between two body configurations.
// Tries all 6 permutations, and for each finds the optimal 2D rotation
// that aligns the first set to the second.
// ---------------------------------------------------------------------------
static double permutation_rotation_distance(
    const double x1[3], const double y1[3],
    const double x2[3], const double y2[3])
{
    const int perms[6][3] = {
        {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
    };
    double best = INFINITY;
    for (int p = 0; p < 6; ++p) {
        // Find optimal rotation angle
        double theta = optimal_rotation_angle(x1, y1, x2, y2, perms[p]);

        // Compute cos and sin once, reuse for all 3 bodies
        double c = std::cos(theta);
        double s = std::sin(theta);

        double d2 = 0.0;
        for (int i = 0; i < 3; ++i) {
            int j = perms[p][i];
            // Rotate and compute distance in one step
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

// ---------------------------------------------------------------------------
// Compute accelerations for all bodies (Newtonian gravity)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// One Yoshida 4th-order symplectic step
// ---------------------------------------------------------------------------
static void yoshida_step(
    double x[3], double y[3],
    double vx[3], double vy[3],
    const double m[3])
{
    const double w1 = YOSHIDA_C1 * DT;
    const double w0 = YOSHIDA_C0 * DT;
    const double hw1 = 0.5 * w1;
    const double hw0 = 0.5 * w0;

    // Stage 1: w1
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

    // Stage 2: w0
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

    // Stage 3: w1
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

// ---------------------------------------------------------------------------
// Check if body i is escaping from the other two.
// Escape condition: specific orbital energy w.r.t. the other two > 0
// AND radial velocity points away from their center of mass.
// Masses are assumed equal (1.0) — the combined mass of the other two is 2.0.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Run a full simulation from initial conditions.
// Argument `state` layout: [x1,x2,x3, y1,y2,y3, vx1,vx2,vx3, vy1,vy2,vy3]
// Masses are assumed equal (1.0).
// ---------------------------------------------------------------------------
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

        // --- closest return check (permutation + rotation aware) ---
        if (step > transient_steps) {
            double d = permutation_rotation_distance(x, y, x0, y0);
            if (d < result.closest_return)
                result.closest_return = d;
        }

        // --- checkpoint recording for archive comparison ---
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

        // --- collision check (using squared distance to avoid sqrt) ---
        double dx01 = x[0] - x[1]; double dy01 = y[0] - y[1];
        double dx02 = x[0] - x[2]; double dy02 = y[0] - y[2];
        double dx12 = x[1] - x[2]; double dy12 = y[1] - y[2];

        if ((dx01*dx01 + dy01*dy01) < collision_dist_sq ||
            (dx02*dx02 + dy02*dy02) < collision_dist_sq ||
            (dx12*dx12 + dy12*dy12) < collision_dist_sq) {
            result.reason = StopReason::COLLISION;
            break;
        }

        // --- escape check (energy-based) ---
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

#endif // SIMULATION_HPP