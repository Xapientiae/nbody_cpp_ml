#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// ---------------------------------------------------------------------------
// Physical Constants
// ---------------------------------------------------------------------------
constexpr double G = 1.0;              // gravitational constant
constexpr double DT = 0.01;            // timestep

// ---------------------------------------------------------------------------
// Simulation Parameters
// ---------------------------------------------------------------------------
constexpr int    N = 3;                 // number of bodies
constexpr int    STATE_SIZE = 12;       // x1..3, y1..3, vx1..3, vy1..3
constexpr int    MAX_STEPS = 100000;    // maximum simulation steps
constexpr double COLLISION_DIST = 0.05; // collision threshold (AU)
constexpr double ESCAPE_DIST = 100.0;   // escape threshold (AU)
constexpr double EJECTION_DIST = 15.0;  // ejection threshold (AU) - penalty if exceeded

// ---------------------------------------------------------------------------
// Yoshida 4th-order Coefficients
// c1 = 1/(2 - 2^(1/3)), c0 = 1 - 2*c1
// ---------------------------------------------------------------------------
constexpr double YOSHIDA_C1 = 1.351207191959657;
constexpr double YOSHIDA_C0 = -1.702414383919314;

// ---------------------------------------------------------------------------
// Population / Evolutionary Algorithm Parameters
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Generator Parameters (for generator.hpp integration)
// ---------------------------------------------------------------------------
constexpr double GENERATOR_POS_RANGE = 2.0;     // position range [-range, +range]
constexpr double GENERATOR_VEL_RANGE = 2.0;     // velocity range [-range, +range]
constexpr double GENERATOR_MIN_DIST = 0.2;      // minimum distance between bodies
constexpr double GENERATOR_TARGET_R2 = 12.0;    // target sum of squared radii

// ---------------------------------------------------------------------------
// Archive Checkpoint Times (for time-aware similarity detection)
// These are step numbers at which to check archive distance
// ---------------------------------------------------------------------------
constexpr int ARCHIVE_CHECKPOINT_STEPS[] = {3000, 7000, 13000, 17000, 29000, 51000, 67000, 100000};
constexpr int NUM_ARCHIVE_CHECKPOINTS = 8;

// ---------------------------------------------------------------------------
// Archive / Diversity Parameters
// ---------------------------------------------------------------------------
constexpr double DEFAULT_ARCHIVE_DIST_THRESHOLD = 0.3; // min dist from archive entries (stricter)
constexpr double DEFAULT_ARCHIVE_PENALTY = 0.7;         // penalty fraction if too close
constexpr double DEFAULT_DIVERSITY_THRESHOLD = 0.7;     // min crowding distance
constexpr double DEFAULT_DIVERSITY_PENALTY = 0.5;       // penalty fraction
constexpr double EJECTION_PENALTY = 0.4;                 // 40% score reduction per ejection

// ---------------------------------------------------------------------------
// Archive Penalty Function Parameters (for linear/exponential penalty)
// ---------------------------------------------------------------------------
constexpr double ARCHIVE_PENALTY_MAX = 0.95;  // maximum penalty at distance 0 (increased)
constexpr double ARCHIVE_PENALTY_EXPONENT = 3.0;  // exponential decay exponent (faster drop-off)

#endif // CONSTANTS_HPP