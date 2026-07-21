/*
 * 3-Body Problem Simulation (2D)
 *
 * Units: G = 1, solar mass = 1, AU = 1, time = year / (2pi)
 * Integration: Yoshida 4th-order symplectic integrator
 *
 * Stopping conditions:
 *   1. Max 1000 steps reached
 *   2. Two bodies collide (distance < 0.01 AU)
 *   3. A body escapes (distance > 100 AU from both other bodies)
 *
 * Compile: g++ -std=c++17 -O3 3body.cpp -o 3body
 * Run:     ./3body [input_file]
 *          If no input_file given, reads from stdin.
 *
 * Input file format (space/tab separated, # for comments):
 *   x1 y1 x2 y2 x3 y3 vx1 vy1 vx2 vy2 vx3 vy3
 *
 * Example (Figure-8 orbit):
 *   -0.97000436  0.24308753   0.0  0.0   0.97000436  -0.24308753 \
 *     0.4662036850  0.4323657300  -0.9324073700  -0.8647314600 \
 *     0.4662036850  0.4323657300
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr double G  = 1.0;           // gravitational constant
constexpr double dt = 0.01;          // timestep

// Stopping thresholds
constexpr double COLLISION_DIST = 0.01;   // AU
constexpr double ESCAPE_DIST    = 100.0;  // AU

// Yoshida 4th-order coefficients (cbrt(2) ≈ 1.2599210498948732)
// c1 = 1/(2 - 2^(1/3)),  c0 = 1 - 2*c1
constexpr double c1 = 1.351207191959657;
constexpr double c0 = -1.702414383919314;

constexpr int N = 3;         // number of bodies

constexpr int MAX_STEPS = 20000;

// ---------------------------------------------------------------------------
// Acceleration calculator
// ---------------------------------------------------------------------------
static void compute_accelerations(
    const double x[3], const double y[3],
    const double m[3],
    double ax[3], double ay[3])
{
    ax[0] = 0.0; ay[0] = 0.0;
    ax[1] = 0.0; ay[1] = 0.0;
    ax[2] = 0.0; ay[2] = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double r2 = dx * dx + dy * dy;
            double r  = sqrt(r2);
            double r3 = r2 * r;

            double f = G / r3;               // force factor
            double f_i = f * m[j];           // acceleration on i from j
            double f_j = f * m[i];           // acceleration on j from i

            ax[i] += f_i * dx;
            ay[i] += f_i * dy;
            ax[j] -= f_j * dx;
            ay[j] -= f_j * dy;
        }
    }
}

// ---------------------------------------------------------------------------
// One full Yoshida 4th-order step (symplectic)
// ---------------------------------------------------------------------------
static void yoshida_step(
    double x[3], double y[3],
    double vx[3], double vy[3],
    const double m[3])
{
    // Coefficients for the three substages
    const double w[3] = {c1, c0, c1};

    for (int s = 0; s < 3; ++s) {
        double ws = w[s];
        double hh = 0.5 * ws * dt;   // half-step for position
        double h  = ws * dt;         // full-step for velocity

        // half position drift
        for (int i = 0; i < N; ++i) {
            x[i] += hh * vx[i];
            y[i] += hh * vy[i];
        }

        // full velocity kick (using updated positions)
        double ax[3], ay[3];
        compute_accelerations(x, y, m, ax, ay);
        for (int i = 0; i < N; ++i) {
            vx[i] += h * ax[i];
            vy[i] += h * ay[i];
        }

        // second half position drift
        for (int i = 0; i < N; ++i) {
            x[i] += hh * vx[i];
            y[i] += hh * vy[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Read initial conditions from a file/stream
// Returns 0 on success, 1 on failure.
// ---------------------------------------------------------------------------
static int read_initial_conditions(
    FILE *fp,
    double x[3], double y[3],
    double vx[3], double vy[3],
    double m[3])
{
    char line[512];

    while (fgets(line, sizeof(line), fp)) {
        // Skip blank lines and comments
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        // Parse the 12 numbers
        int n = sscanf(line,
            "%lf %lf %lf %lf %lf %lf"
            "%lf %lf %lf %lf %lf %lf",
            &x[0], &y[0], &x[1], &y[1], &x[2], &y[2],
            &vx[0], &vy[0], &vx[1], &vy[1], &vx[2], &vy[2]);

        if (n == 12) {
            // Default masses to 1.0
            m[0] = 1.0; m[1] = 1.0; m[2] = 1.0;
            return 0;
        }

        fprintf(stderr, "ERROR: expected 12 numbers, got %d\n", n);
        return 1;
    }

    fprintf(stderr, "ERROR: no valid data found in input\n");
    return 1;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // --- Open input file or stdin ------------------------------------------
    FILE *fp = stdin;
    if (argc > 1) {
        fp = fopen(argv[1], "r");
        if (!fp) {
            fprintf(stderr, "ERROR: cannot open '%s'\n", argv[1]);
            return 1;
        }
    }

    // --- Read initial conditions -------------------------------------------
    double x[3], y[3], vx[3], vy[3], m[3];

    if (read_initial_conditions(fp, x, y, vx, vy, m) != 0) {
        if (fp != stdin) fclose(fp);
        return 1;
    }

    if (fp != stdin) fclose(fp);

    // --- Print initial conditions summary to stderr ------------------------
    fprintf(stderr, "# Initial conditions:\n");
    for (int i = 0; i < 3; ++i) {
        fprintf(stderr, "#   body %d : (%.8g, %.8g)  v = (%.8g, %.8g)  m = %.8g\n",
                i + 1, x[i], y[i], vx[i], vy[i], m[i]);
    }

    // --- CSV header --------------------------------------------------------
    printf("t,x1,y1,x2,y2,x3,y3\n");

    // --- Print initial conditions ------------------------------------------
    printf("%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
           0.0, x[0], y[0], x[1], y[1], x[2], y[2]);

    // --- Time integration --------------------------------------------------
    double t = 0.0;
    int step = 0;
    const char *reason = nullptr;
    int body_a = -1, body_b = -1;

    for (step = 1; step <= MAX_STEPS; ++step) {
        yoshida_step(x, y, vx, vy, m);
        t += dt;

        // Print every step
        printf("%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
               t, x[0], y[0], x[1], y[1], x[2], y[2]);

        // --- Check collision and escape -----------------------------------
        // Compute all pairwise distances
        double d01 = hypot(x[0] - x[1], y[0] - y[1]);
        double d02 = hypot(x[0] - x[2], y[0] - y[2]);
        double d12 = hypot(x[1] - x[2], y[1] - y[2]);

        // Collision check: any pair closer than 0.01 AU
        if (d01 < COLLISION_DIST) { reason = "COLLISION"; body_a = 0; body_b = 1; break; }
        if (d02 < COLLISION_DIST) { reason = "COLLISION"; body_a = 0; body_b = 2; break; }
        if (d12 < COLLISION_DIST) { reason = "COLLISION"; body_a = 1; body_b = 2; break; }

        // Escape check: body farther than 100 AU from both other bodies
        if (d01 > ESCAPE_DIST && d02 > ESCAPE_DIST) { reason = "ESCAPE"; body_a = 0; break; }
        if (d01 > ESCAPE_DIST && d12 > ESCAPE_DIST) { reason = "ESCAPE"; body_a = 1; break; }
        if (d02 > ESCAPE_DIST && d12 > ESCAPE_DIST) { reason = "ESCAPE"; body_a = 2; break; }
    }

    // --- Termination message ----------------------------------------------
    if (reason) {
        if (body_b >= 0)
            printf("# %s: bodies %d and %d at t=%.12g\n", reason, body_a, body_b, t);
        else
            printf("# %s: body %d at t=%.12g\n", reason, body_a, t);
    } else {
        printf("# MAX STEPS REACHED at t=%.12g\n", t);
    }

    printf("# simulation time: %.12g, steps: %d\n", t, step - 1);

    return 0;
}