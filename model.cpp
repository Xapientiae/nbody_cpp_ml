/*
 * model.cpp — Evolutionary search for stable 3-body orbits
 *
 * Compile: g++ -std=c++17 -O3 -fopenmp model.cpp -o model
 * Run:     ./model --popsize=256 --generations=50 --seed=$(date +%s)
 *
 * Features:
 *   - Random search (default) or refine existing orbit (--refine)
 *   - Archive of found orbits (--archive)
 *   - Penalty for similarity to archive (encourages diversity)
 *   - All output goes to output/ directory
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <filesystem>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "population.hpp"

// Command-line configuration
struct Config {
    size_t pop_size           = 256;
    int    generations        = 50;
    int    seed               = 0;
    double mutation_sigma     = 0.05;
    size_t tournament_k       = 3;
    int    elite_count        = 2;
    double diversity_threshold = DEFAULT_DIVERSITY_THRESHOLD;
    double diversity_penalty  = DEFAULT_DIVERSITY_PENALTY;
    double archive_dist_threshold = DEFAULT_ARCHIVE_DIST_THRESHOLD;   // min dist from archive entries
    double archive_penalty    = DEFAULT_ARCHIVE_PENALTY;       // penalty fraction if too close to archive
    bool   verbose            = true;
    std::string output_dir    = "output";
    std::string archive_file  = "output/archive.txt";
    std::string refine_file   = "";        // if set, refine this orbit instead of random search
};

static Config parse_args(int argc, char *argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto get_val = [&]() -> const char* {
            auto eq = arg.find('=');
            if (eq != std::string::npos) return arg.c_str() + eq + 1;
            if (i + 1 < argc) return argv[++i];
            fprintf(stderr, "ERROR: %s requires a value\n", arg.c_str());
            exit(1);
            return nullptr;
        };

        std::string key = arg;
        auto eq = key.find('=');
        if (eq != std::string::npos) key = key.substr(0, eq);

        if (key == "--popsize" || key == "-p") {
            char *end = nullptr;
            long val = std::strtol(get_val(), &end, 10);
            if (end == nullptr || *end != '\0') {
                fprintf(stderr, "ERROR: invalid --popsize value\n");
                exit(1);
            }
            cfg.pop_size = (size_t)val;
        } else if (key == "--generations" || key == "-g") {
            char *end = nullptr;
            long val = std::strtol(get_val(), &end, 10);
            if (end == nullptr || *end != '\0') {
                fprintf(stderr, "ERROR: invalid --generations value\n");
                exit(1);
            }
            cfg.generations = (int)val;
        } else if (key == "--seed" || key == "-s") {
            char *end = nullptr;
            long val = std::strtol(get_val(), &end, 10);
            if (end == nullptr || *end != '\0') {
                fprintf(stderr, "ERROR: invalid --seed value\n");
                exit(1);
            }
            cfg.seed = (int)val;
        } else if (key == "--mutation-sigma" || key == "-m") {
            char *end = nullptr;
            double val = std::strtod(get_val(), &end);
            if (end == nullptr || *end != '\0') {
                fprintf(stderr, "ERROR: invalid --mutation-sigma value\n");
                exit(1);
            }
            cfg.mutation_sigma = val;
        } else if (key == "--output-dir") {
            cfg.output_dir = get_val();
            cfg.archive_file = cfg.output_dir + "/archive.txt";
        } else if (key == "--archive") {
            cfg.archive_file = get_val();
        } else if (key == "--refine") {
            cfg.refine_file = get_val();
        } else if (key == "--quiet" || key == "-q") {
            cfg.verbose = false;
        } else if (key == "--help" || key == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --popsize, -p N         Population size (default: 256)\n");
            printf("  --generations, -g N     Number of generations (default: 50)\n");
            printf("  --seed, -s N            RNG seed (0 = random, default: 0)\n");
            printf("  --mutation-sigma, -m    Mutation strength (default: 0.05)\n");
            printf("  --output-dir DIR        Output directory (default: output)\n");
            printf("  --archive FILE          Archive file (default: output/archive.txt)\n");
            printf("  --refine FILE           Refine existing orbit from FILE\n");
            printf("  --quiet, -q             Suppress progress output\n");
            printf("  --help, -h              Show this help\n");
            exit(0);
        } else {
            fprintf(stderr, "WARNING: unknown argument '%s', ignoring\n", arg.c_str());
        }
    }
    return cfg;
}

// Ensure directory exists
static void ensure_dir(const std::string& dir) {
    std::filesystem::create_directories(dir);
}

// Save best state to file
static void save_state(const std::string& filename, const double state[STATE_SIZE],
                       double score, int steps, double closest_return)
{
    FILE *fp = fopen(filename.c_str(), "a");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot write to '%s'\n", filename.c_str());
        return;
    }

    fprintf(fp, "# score=%.6g  steps=%d  closest_return=%.6g\n",
            score, steps, closest_return);
    fprintf(fp, "# format: x1 y1 x2 y2 x3 y3  vx1 vy1 vx2 vy2 vx3 vy3\n");

    const double *x  = state;
    const double *y  = state + 3;
    const double *vx = state + 6;
    const double *vy = state + 9;
    fprintf(fp, "%.12g %.12g  %.12g %.12g  %.12g %.12g  "
                "%.12g %.12g  %.12g %.12g  %.12g %.12g\n",
            x[0], y[0], x[1], y[1], x[2], y[2],
            vx[0], vy[0], vx[1], vy[1], vx[2], vy[2]);

    fclose(fp);
}

// Append history CSV
static void append_history(const std::string& filename, int gen,
                           double best_score, double avg_score,
                           int best_steps, double best_return)
{
    FILE *fp = fopen(filename.c_str(), "a");
    if (!fp) return;

    if (gen == 0) {
        fprintf(fp, "generation,best_score,avg_score,best_steps,best_closest_return\n");
    }
    fprintf(fp, "%d,%.6g,%.6g,%d,%.6g\n",
            gen, best_score, avg_score, best_steps, best_return);
    fclose(fp);
}

// Main
int main(int argc, char *argv[]) {
    Config cfg = parse_args(argc, argv);

    // Ensure output directory
    ensure_dir(cfg.output_dir);

    // Seed RNG
    if (cfg.seed == 0) {
        cfg.seed = (int)std::time(nullptr) ^
                   (int)std::chrono::steady_clock::now().time_since_epoch().count();
    }
    
    // Seed thread-local RNGs
    #ifdef _OPENMP
        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            unsigned int thread_seed = cfg.seed + thread_num * 1000;
            rng_local.seed(thread_seed);
        }
    #endif

    // Load archive
    std::vector<std::vector<double>> archive;
    size_t archive_count = load_states_from_archive(cfg.archive_file.c_str(), archive);

    // Determine base state (if refining)
    std::vector<double> refine_base(STATE_SIZE);
    bool is_refine = !cfg.refine_file.empty();

    if (is_refine) {
        if (load_state_from_file(cfg.refine_file.c_str(), refine_base.data()) != 0) {
            fprintf(stderr, "ERROR: failed to load state from '%s'\n",
                    cfg.refine_file.c_str());
            return 1;
        }
        // Normalize state
        normalize_state(refine_base.data());
        // Use smaller mutation sigma for refinement
        cfg.mutation_sigma = std::min(cfg.mutation_sigma, 0.02);

        if (cfg.verbose) {
            fprintf(stderr, "# Refining orbit from: %s\n", cfg.refine_file.c_str());
        }
    }

    // Echo config
    if (cfg.verbose) {
        fprintf(stderr, "# Model: Evolutionary 3-body orbit finder\n");
        fprintf(stderr, "# Population: %zu  Generations: %d  Seed: %d\n",
                cfg.pop_size, cfg.generations, cfg.seed);
        fprintf(stderr, "# Mutation sigma: %.4f  Tournament k: %zu  Elite: %d\n",
                cfg.mutation_sigma, cfg.tournament_k, cfg.elite_count);
        fprintf(stderr, "# Output dir: %s\n", cfg.output_dir.c_str());
        fprintf(stderr, "# Archive: %s (%zu entries)\n",
                cfg.archive_file.c_str(), archive_count);
        if (is_refine) {
            fprintf(stderr, "# Refining orbit from: %s\n", cfg.refine_file.c_str());
        }
        fprintf(stderr, "#\n");
    }

    // Allocate population
    std::vector<std::vector<double>> population(cfg.pop_size,
                                                 std::vector<double>(STATE_SIZE));
    std::vector<FitnessResult> fitness(cfg.pop_size);

    // Generate initial population
    if (cfg.verbose) fprintf(stderr, "# Gen 0: generating population...\n");

    if (is_refine) {
        generate_refined_population(refine_base.data(), population, cfg.mutation_sigma);
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < cfg.pop_size; ++i) {
            generate_random_state(population[i].data());
        }
    }

    // Evaluate generation 0
    if (cfg.verbose) fprintf(stderr, "# Gen 0: evaluating...\n");

    const int progress_interval = std::max(1, (int)(cfg.pop_size / 20)); // Report every 5%

    #pragma omp parallel for
    for (size_t i = 0; i < cfg.pop_size; ++i) {
        fitness[i] = evaluate_fitness(population[i].data());

        // Progress indicator
        #pragma omp critical
        if (cfg.verbose && (i + 1) % progress_interval == 0) {
            fprintf(stderr, "#   Progress: %zu/%zu (%.0f%%)\n",
                    i + 1, cfg.pop_size, 100.0 * (i + 1) / cfg.pop_size);
        }
    }

    // Track best
    double best_score = -1.0;
    std::vector<double> best_state(STATE_SIZE);
    int best_steps = 0;
    double best_return = 0.0;
    SimulationResult best_sim_result;  // Store full simulation result for checkpoint-based archive check

    auto update_best = [&](size_t idx) {
        if (fitness[idx].score > best_score) {
            best_score = fitness[idx].score;
            best_state = population[idx];
            best_steps = fitness[idx].steps;
            best_return = fitness[idx].closest_return;
            best_sim_result = fitness[idx].sim_result;
        }
    };

    for (size_t i = 0; i < cfg.pop_size; ++i) update_best(i);

    // History for gen 0
    double avg = 0.0;
    for (size_t i = 0; i < cfg.pop_size; ++i) avg += fitness[i].score;
    avg /= cfg.pop_size;

    std::string history_file = cfg.output_dir + "/history.csv";
    append_history(history_file.c_str(), 0, best_score, avg, best_steps, best_return);

    if (cfg.verbose) {
        fprintf(stderr, "# Gen 0: best=%.4g  avg=%.4g  steps=%d  return=%.4g\n",
                best_score, avg, best_steps, best_return);
    }

    // Evolutionary loop
    for (int gen = 1; gen <= cfg.generations; ++gen) {
        auto t_start = std::chrono::steady_clock::now();

        std::vector<std::vector<double>> next_pop(cfg.pop_size,
                                                   std::vector<double>(STATE_SIZE));
        std::vector<FitnessResult> next_fitness(cfg.pop_size);

        std::vector<size_t> indices(cfg.pop_size);
        for (size_t i = 0; i < cfg.pop_size; ++i) indices[i] = i;
        std::partial_sort(indices.begin(), indices.begin() + cfg.elite_count, indices.end(),
            [&](size_t a, size_t b) { return fitness[a].score > fitness[b].score; });

        for (int e = 0; e < cfg.elite_count; ++e) {
            size_t src = indices[e];
            std::copy(population[src].begin(), population[src].end(), next_pop[e].begin());
            next_fitness[e] = fitness[src];
        }

        #pragma omp parallel for
        for (size_t i = (size_t)cfg.elite_count; i < cfg.pop_size; ++i) {
            std::vector<double> scores(cfg.pop_size);
            for (size_t j = 0; j < cfg.pop_size; ++j) scores[j] = fitness[j].score;

            size_t p1 = tournament_select(scores, cfg.tournament_k);
            size_t p2 = tournament_select(scores, cfg.tournament_k);

            crossover(population[p1].data(), population[p2].data(),
                      next_pop[i].data(), cfg.mutation_sigma);

            // Evaluate
            next_fitness[i] = evaluate_fitness(next_pop[i].data());

            // Archive penalty
            double d_arch = archive_distance_checkpoints(next_fitness[i].sim_result, archive);
            double penalty = compute_archive_penalty(d_arch, cfg.archive_dist_threshold);
            next_fitness[i].score *= (1.0 - penalty);
        }

        // Diversity penalty
        for (size_t i = 0; i < cfg.pop_size; ++i) {
            double d = crowding_distance(next_pop[i].data(), next_pop, i);
            if (d < cfg.diversity_threshold) {
                next_fitness[i].score *= (1.0 - cfg.diversity_penalty);
            }
        }

        population.swap(next_pop);
        fitness.swap(next_fitness);

        for (size_t i = 0; i < cfg.pop_size; ++i) update_best(i);

        avg = 0.0;
        for (size_t i = 0; i < cfg.pop_size; ++i) avg += fitness[i].score;
        avg /= cfg.pop_size;

        auto t_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();

        append_history(history_file.c_str(), gen, best_score, avg, best_steps, best_return);

        if (cfg.verbose) {
            fprintf(stderr, "# Gen %d: best=%.4g  avg=%.4g  steps=%d  return=%.4g  [%.2fs]\n",
                    gen, best_score, avg, best_steps, best_return, elapsed);
        }

    }

    // Final save
    std::string best_file = cfg.output_dir + "/best.txt";
    save_state(best_file, best_state.data(), best_score, best_steps, best_return);

    // Add final best to archive if it has MAX_STEPS
    if (best_steps >= MAX_STEPS && is_novel_checkpoints(best_sim_result, archive, cfg.archive_dist_threshold)) {
        // Normalize before saving
        std::vector<double> best_state_normalized = best_state;
        normalize_state(best_state_normalized.data());
        normalize_scale(best_state_normalized.data());
        save_state_to_archive(cfg.archive_file.c_str(), best_state_normalized.data(),
                              best_score, best_steps, best_return, cfg.seed);
        archive.push_back(best_state_normalized);
        if (cfg.verbose) {
            fprintf(stderr, "# Final best added to archive (steps=%d)\n", best_steps);
        }
    }

    if (cfg.verbose) {
        fprintf(stderr, "#\n# Done! Best score: %.4g\n", best_score);
        fprintf(stderr, "# Best saved to: %s\n", best_file.c_str());
        fprintf(stderr, "# Archive: %s (%zu entries)\n",
                cfg.archive_file.c_str(), archive.size());
        fprintf(stderr, "# History: %s\n", history_file.c_str());
        fprintf(stderr, "#\n# To visualize:\n");
        fprintf(stderr, "#   ./3body %s | python3 visualize.py\n", best_file.c_str());
    }

    return 0;
}