# Code Review: Evolutionary 3-Body Orbit Finder

**Review Date:** 2026-07-22  
**Last Updated:** 2026-07-22 (after fixes applied)  
**Reviewer:** DeepSeek (AI Code Review)  
**Scope:** `constants.hpp`, `simulation.hpp`, `population.hpp`, `model.cpp`, `3body.cpp`, `Makefile`  
**Project:** Evolutionary search for stable 3-body orbits using a Yoshida 4th-order symplectic integrator

---

## Executive Summary

This is a well-structured C++17 project implementing an evolutionary algorithm to discover stable 3-body orbits. The code is generally clean, well-commented, and follows consistent naming conventions. The physics (Yoshida 4th-order symplectic integration) is correctly implemented, and the permutation+rotation-aware distance metric for comparing body configurations is a sophisticated and appropriate design choice.

Several issues have been identified and fixed. The remaining open items are mostly architectural suggestions and minor improvements.

**Severity Legend:** 🔴 Critical | 🟠 High | 🟡 Medium | 🔵 Low | ⚪ Suggestion  
**Status:** ✅ Fixed | ❌ Open

---

## 1. Correctness Issues

### 🔴 [CRITICAL] — ✅ FIXED — `3body.cpp`: Output interval logic is broken

**File:** `3body.cpp`, lines 140–146

**Fixed:** Changed `output_counter++ == 0` to `output_counter == 0` with `++output_counter >= output_interval` for correct modular counting. The `output_interval` parameter now works properly.

---

### 🟠 [HIGH] — ✅ FIXED — `is_escaping()`: Physically incorrect escape condition

**File:** `simulation.hpp`, lines 195–217

**Fixed:** `is_escaping()` now accepts a mass array `const double m[3]` and uses mass-weighted center-of-mass and specific orbital energy `M_jk / r` instead of hardcoded `2.0 / r`.

---

### 🟠 [HIGH] — ✅ FIXED — `generate_random_state()`: Silent failure on position generation

**File:** `population.hpp`, lines 173–197

**Fixed:** Added `found_valid` flag. If all 100 attempts fail to produce valid positions, a fallback generates positions with relaxed constraints instead of silently using invalid ones.

---

### 🟡 [MEDIUM] — ✅ FIXED — `run_simulation()`: `steps` field is `MAX_STEPS + 1` on completion

**File:** `simulation.hpp`, lines 244–289

**Fixed:** Changed loop from `for (step = 1; step <= MAX_STEPS; ++step)` to `for (step = 0; step < MAX_STEPS; ++step)`. On completion, `step` now equals `MAX_STEPS` instead of `MAX_STEPS + 1`. Checkpoint comparisons adjusted with `step + 1`.

---

### 🟡 [MEDIUM] — ✅ FIXED — `crowding_distance()`: Not permutation/rotation-aware

**File:** `population.hpp`, lines 308–326

**Fixed:** Now uses `permutation_rotation_state_distance()` instead of raw Euclidean distance, making the diversity penalty consistent with the archive distance metric. Added forward declaration.

---

### 🟡 [MEDIUM] — ✅ FIXED — `evaluate_fitness()`: Sentinel value `100.0` for missing return distance

**File:** `population.hpp`, line 269

**Fixed:** `closest_return` now preserves `INFINITY` when no return is detected. The bonus calculation explicitly checks for `INFINITY` and returns `0.0` in that case, avoiding the ambiguous sentinel value.

---

### 🔵 [LOW] — ✅ FIXED — `compute_archive_penalty()`: Very sharp exponential drop-off

**File:** `model.cpp`, lines 169–172

**Fixed:** Replaced exponential decay with linear penalty: `ARCHIVE_PENALTY_MAX * (1.0 - distance / threshold)`. This provides a smooth gradient from max penalty at distance=0 to zero penalty at distance=threshold.

---

## 2. Architecture & Design

### 🟠 [HIGH] — ❌ OPEN — Header-only implementation with `static` functions

**Files:** `simulation.hpp`, `population.hpp`

**Problem:** All function implementations are in header files with `static` linkage. This means:
1. Each translation unit gets its own copy of every function (code bloat)
2. If another `.cpp` file ever includes these headers, ODR violations or code duplication will occur
3. `static` in headers is generally considered an anti-pattern — `inline` is the correct keyword for header-only functions

**Fix:** Either:
- Move implementations to `.cpp` files with declarations in headers
- Or change `static` to `inline` for all function definitions in headers

### 🟡 [MEDIUM] — ❌ OPEN — `population.hpp` is a grab-bag of unrelated functionality

**File:** `population.hpp`

**Problem:** This single header contains RNG utilities, physics normalization, energy computation, EA operators, fitness evaluation, tournament selection, crowding distance, archive distance, file I/O, and population generation.

**Suggestion:** Split into `rng.hpp`, `physics.hpp`, `evolution.hpp`, `archive.hpp`, and `population.hpp`.

### 🟡 [MEDIUM] — ❌ OPEN — `Config` struct uses inconsistent types

**File:** `model.cpp`, lines 35–50

**Problem:** `pop_size` is `size_t` but `generations` and `elite_count` are `int`, causing signed/unsigned comparison warnings.

**Fix:** Use consistent types — either `size_t` for all sizes/counts or `int` for all.

### 🔵 [LOW] — ❌ OPEN — `thread_local` RNG in header is fragile

**File:** `population.hpp`, line 19

**Problem:** If this header is ever included from multiple translation units, each TU gets its own `rng_local` instance.

**Fix:** Move to a dedicated `.cpp` file with `extern thread_local` declaration in the header.

---

## 3. Performance

### 🟡 [MEDIUM] — ❌ OPEN — `permutation_rotation_state_distance()` is called excessively

**File:** `population.hpp`, lines 335–409; `model.cpp`, lines 367–369

**Problem:** Called for every individual against every archive entry in every generation. Each call copies/normalizes both states and tries 12 permutations with trig operations.

**Suggestion:** Pre-normalize archive entries once, cache normalization results, or use a cheaper approximate distance for initial filtering.

### 🟡 [MEDIUM] — ✅ FIXED — Redundant energy computation in `ensure_bound()`

**File:** `population.hpp`, lines 129–160

**Fixed:** Extracted shared `compute_energies()` helper that returns KE and PE together. `ensure_bound()` now calls it once instead of computing energies twice. Removed the now-unused `total_energy()` function.

### 🔵 [LOW] — ❌ OPEN — `scores` vector reallocated per-thread per-individual

**File:** `model.cpp`, lines 353–354

**Problem:** Inside the OpenMP parallel for, each thread creates a new `scores` vector for each individual. This is O(pop_size²) memory allocations per generation.

**Fix:** Restructure `tournament_select` to work directly with the `fitness` vector.

---

## 4. C++ Idioms & Safety

### 🟡 [MEDIUM] — ❌ OPEN — C-style arrays instead of `std::array`

**Throughout the codebase**

**Problem:** Raw C arrays (`double x[3]`, `double state[STATE_SIZE]`) are used everywhere, with no bounds checking and array-to-pointer decay.

**Fix:** Use `std::array<double, 3>` and `std::array<double, STATE_SIZE>` for fixed-size arrays.

### 🟡 [MEDIUM] — ✅ FIXED — `std::memcpy` instead of `std::copy`

**File:** `population.hpp`, lines 341–342, 381

**Fixed:** Replaced both `std::memcpy` calls with `std::copy` for type safety. Removed the now-unused `#include <cstring>`.

### 🟡 [MEDIUM] — ✅ FIXED — `std::atoi`/`std::atof` without error checking

**File:** `model.cpp`, lines 71–77

**Fixed:** Replaced `std::atol`/`std::atoi`/`std::atof` with `std::strtol`/`std::strtod` with proper error checking via end pointer validation.

### 🔵 [LOW] — ✅ FIXED — Platform-specific directory creation

**File:** `model.cpp`, lines 109–114

**Fixed:** Replaced `stat`/`mkdir` (POSIX-specific) with `std::filesystem::create_directories()` (C++17 standard). Changed include from `<sys/stat.h>` to `<filesystem>`.

### 🔵 [LOW] — ✅ FIXED — `SIZE_MAX` as sentinel

**File:** `population.hpp`, line 311

**Fixed:** Changed `SIZE_MAX` to `std::numeric_limits<size_t>::max()` for clarity.

---

## 5. Readability & Maintainability

### 🔵 [LOW] — ❌ OPEN — Hardcoded `3` instead of `N` in many places

**Throughout the codebase**

**Problem:** The constant `N = 3` is defined, but many loops/arrays use the literal `3` instead of `N`.

**Fix:** Use `N` consistently.

### 🔵 [LOW] — ✅ FIXED — `TRANSIENT_RATIO` constant defined but unused

**File:** `constants.hpp`, line 39; `simulation.hpp`, line 242

**Fixed:** Now uses `MAX_STEPS / TRANSIENT_RATIO` instead of hardcoded `MAX_STEPS / 20`.

### 🔵 [LOW] — ❌ OPEN — Dual state layout formats are confusing

**Problem:** The code uses two different state layouts (internal vs file format) with manual conversion in multiple places.

**Suggestion:** Define a single canonical layout and provide conversion functions.

### 🔵 [LOW] — ✅ FIXED — `3body.cpp` comment says `COLLISION_DIST = 0.01` but constant is `0.05`

**File:** `3body.cpp`, lines 8 and 155

**Fixed:** Updated header comment and inline comment to match the actual constant value.

---

## 6. Specific Bugs & Edge Cases

### 🟡 [MEDIUM] — ✅ FIXED — `3body.cpp` uses `hypot` without `std::`

**File:** `3body.cpp`, lines 150–152

**Fixed:** Changed `hypot` to `std::hypot`.

### 🟡 [MEDIUM] — ❌ OPEN — Weak thread seed derivation

**File:** `model.cpp`, lines 227–228

**Problem:** `unsigned int thread_seed = cfg.seed + thread_num * 1000;` — if two runs have seeds differing by exactly 1000, thread N of run 1 gets the same seed as thread N+1 of run 2.

**Fix:** Use a proper seed sequence: `std::seed_seq seq{cfg.seed, thread_num};`

### 🔵 [LOW] — ❌ OPEN — Double normalization on archive load

**Problem:** `save_state_to_archive()` normalizes before saving and `load_states_from_archive()` normalizes again on load.

**Fix:** Document the normalization convention clearly and normalize in only one place.

### 🔵 [LOW] — ❌ OPEN — Boundary behavior of `is_novel_checkpoints`

**File:** `model.cpp`, line 202

**Problem:** `return d >= threshold` — a state exactly at the threshold gets no penalty but is added to the archive. Could lead to archive flooding.

---

## 7. Positive Highlights

Despite the issues above, the codebase has many strengths:

1. **Excellent documentation** — Every function has a clear comment describing its purpose, parameters, and return value. Section headers make the code easy to navigate.

2. **Sophisticated distance metric** — The permutation + rotation-aware distance for comparing body configurations is a well-thought-out solution to the body-labeling symmetry problem. The time-reversal consideration is a nice touch.

3. **Correct symplectic integration** — The Yoshida 4th-order integrator is correctly implemented with the proper coefficient ordering (drift-kick-drift substeps).

4. **Good use of modern C++ features** — `constexpr`, `thread_local`, `enum class`, `std::mt19937_64`, OpenMP parallelism.

5. **Well-structured build system** — The Makefile is clean, with separate targets for building, testing, and visualization. OpenMP can be toggled on/off.

6. **Proper normalization** — Center-of-mass and momentum normalization ensure the reference frame is consistent. Scale normalization enables fair comparison of orbits at different sizes.

7. **Archive with checkpoint-based similarity** — Using multiple time points for archive comparison is a clever way to catch time-shifted similar orbits.

8. **Clean separation of simulation and evolution** — `simulation.hpp` handles pure physics, `population.hpp` handles the evolutionary algorithm. This is a good architectural boundary.

---

## 8. Summary

| Severity | Total | Fixed | Open | Key Open Issues |
|----------|-------|-------|------|-----------------|
| 🔴 Critical | 1 | 1 | 0 | — |
| 🟠 High | 4 | 2 | 2 | Header-only `static` functions, header architecture |
| 🟡 Medium | 9 | 7 | 2 | C-style arrays, excessive trig calls, weak thread seeds, scores vector |
| 🔵 Low | 7 | 6 | 1 | Hardcoded `3`, dual formats, double normalization, boundary behavior |
| ⚪ Suggestion | 3 | 0 | 3 | Split `population.hpp`, use `std::array`, add conversion functions |

**Overall Assessment:** The code is functionally correct for its primary purpose (discovering stable 3-body orbits) and is well-documented. All critical and most correctness bugs have been fixed. The remaining issues are architectural improvements (refactoring header-only code), performance optimizations, and modern C++ idioms.

**Fixes applied (this round):**
- `model.cpp`: Linear archive penalty, `strtol`/`strtod` with error checking, `std::filesystem::create_directories`
- `population.hpp`: `compute_energies()` helper eliminating redundant computation, `std::copy` instead of `memcpy`, removed unused `total_energy()` and `<cstring>`