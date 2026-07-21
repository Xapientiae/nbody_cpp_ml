#!/bin/bash
# run.sh — Automated pipeline for 3-body orbit search
#
# Usage:
#   ./run.sh quick     Fast test (small population, few generations)
#   ./run.sh explore   Search with multiple seeds → fill archive
#   ./run.sh refine    Refine all orbits in archive
#   ./run.sh deep      One big search (large population, many generations)
#   ./run.sh all       Full pipeline: explore → refine → summary
#   ./run.sh viz       Generate video from best.txt
#   ./run.sh summary   Show archive contents and stats
#
# All output goes to output/ directory
# Logs saved to output/run_<timestamp>.log

set -e

OUTPUT_DIR="output"
ARCHIVE="$OUTPUT_DIR/archive.txt"
BEST="$OUTPUT_DIR/best.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUTPUT_DIR/run_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() {
    echo -e "[${CYAN}$(date '+%H:%M:%S')${NC}] $1" | tee -a "$LOG"
}

ok() {
    echo -e "  ${GREEN}✓${NC} $1" | tee -a "$LOG"
}

warn() {
    echo -e "  ${YELLOW}⚠${NC} $1" | tee -a "$LOG"
}

fail() {
    echo -e "  ${RED}✗${NC} $1" | tee -a "$LOG"
    exit 1
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup() {
    # Create output dir BEFORE cleanup — log() needs it
    mkdir -p "$OUTPUT_DIR"
    log "Compiling..."
    # Only clean build artifacts, NOT output/ (we just created it)
    rm -f model 3body *.o 2>/dev/null || true
    if make -j$(nproc) 2>&1 | tee -a "$LOG"; then
        ok "Compilation successful"
    else
        fail "Compilation failed"
    fi
}

# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
run_quick() {
    log "Quick test: popsize=64, generations=10, seed=42"
    ./model --popsize=64 --generations=10 --seed=42 2>&1 | tee -a "$LOG" | grep -E '(Gen |Done|Best)'
    if [ -f "$BEST" ]; then
        ok "Best orbit saved to $BEST"
        echo "  Score: $(grep '^# score' "$BEST" | head -1)"
    fi
}

# ---------------------------------------------------------------------------
# Explore: multiple seeds
# ---------------------------------------------------------------------------
run_explore() {
    local SEEDS=(42 123 456 789 111 222 333 555 777 999)
    log "Explore mode: ${#SEEDS[@]} seeds, popsize=128, generations=20"

    for seed in "${SEEDS[@]}"; do
        log "Seed $seed..."
        ./model --popsize=128 --generations=20 --seed="$seed" --quiet 2>&1 | tee -a "$LOG" | tail -1
    done

    # Count archive entries
    if [ -f "$ARCHIVE" ]; then
        local count=$(grep -c '^[^#]' "$ARCHIVE" 2>/dev/null || echo 0)
        ok "Archive has $count entries (file: $ARCHIVE)"
    fi
}

# ---------------------------------------------------------------------------
# Refine all orbits in archive
# ---------------------------------------------------------------------------
run_refine() {
    if [ ! -f "$ARCHIVE" ]; then
        warn "Archive not found. Run 'explore' first or use --refine on a specific file."
        return
    fi

    # Count states in archive
    local lines=($(grep -n '^[^#]' "$ARCHIVE" 2>/dev/null | head -10 | cut -d: -f1))
    local count=${#lines[@]}

    if [ "$count" -eq 0 ]; then
        warn "Archive is empty."
        return
    fi

    log "Refine mode: refining $count orbit(s) from archive, 20 generations each"

    for ((i=0; i<count; i++)); do
        log "Refining orbit $((i+1))/$count..."
        local tmpfile="$OUTPUT_DIR/refine_input_$i.txt"

        # Extract the i-th non-comment line and its preceding metadata
        awk -v n="$((i+1))" '
            /^[^#]/ { found++; if (found == n) { print; exit } }
        ' "$ARCHIVE" > "$tmpfile"

        if [ -s "$tmpfile" ]; then
            ./model --refine="$tmpfile" --generations=20 --mutation-sigma=0.02 \
                    --popsize=64 --seed="$((i + 1000))" --quiet \
                    --output-dir="$OUTPUT_DIR/refine_$i" 2>&1 | tee -a "$LOG" | tail -1
            rm -f "$tmpfile"
        fi
    done

    ok "Refinement complete. Check output/refine_*/ directories"
}

# ---------------------------------------------------------------------------
# Deep run: one big search
# ---------------------------------------------------------------------------
run_deep() {
    log "Deep mode: popsize=512, generations=100"
    local DEEP_SEED=$(date +%s)
    ./model --popsize=512 --generations=100 --seed="$DEEP_SEED" 2>&1 | tee -a "$LOG" | grep -E '(Gen |Done)'
    ok "Deep search complete"
}

# ---------------------------------------------------------------------------
# Visualization: generate video from best.txt
# ---------------------------------------------------------------------------
run_viz() {
    if [ ! -f "$BEST" ]; then
        warn "No best.txt found. Run a search first."
        return
    fi

    log "Generating visualization from $BEST..."
    local VIDEO="$OUTPUT_DIR/orbit_best_$TIMESTAMP.mp4"

    if ./3body "$BEST" 2>/dev/null | python3 visualize.py --output="$VIDEO" --fps=30 2>&1 | tail -3 | tee -a "$LOG"; then
        ok "Video saved to $VIDEO"
    else
        warn "Visualization failed. Check python dependencies:"
        warn "  pip install numpy matplotlib"
        warn "  (ffmpeg needed for video encoding)"
    fi
}

# ---------------------------------------------------------------------------
# Visualize all orbits in archive
# ---------------------------------------------------------------------------
run_viz_all() {
    if [ ! -f "$ARCHIVE" ]; then
        warn "Archive not found. Run 'explore' first."
        return
    fi

    # Count states in archive
    local entries=$(grep -c '^[^#]' "$ARCHIVE" 2>/dev/null || echo 0)
    if [ "$entries" -eq 0 ]; then
        warn "Archive is empty."
        return
    fi

    log "Visualizing all $entries orbits from archive..."
    local VIZ_DIR="$OUTPUT_DIR/videos"
    mkdir -p "$VIZ_DIR"

    # Read archive lines, pairing metadata with state lines
    awk '
        /^# score/ {
            score = $2; sub(/.*=/, "", score)
            steps = $3; sub(/.*=/, "", steps)
            ret   = $4; sub(/.*=/, "", ret)
            seed  = $5; sub(/.*=/, "", seed)

            # Read next non-comment line (the state)
            getline next_line
            while (next_line ~ /^#/ || next_line ~ /^$/) getline next_line

            printf "ORBIT_SEED=%d\n", seed
            printf "ORBIT_SCORE=%.0f\n", score
            printf "ORBIT_RETURN=%.6f\n", ret
            print next_line
            print "---END---"
        }
    ' "$ARCHIVE" | while IFS= read -r line; do
        if [[ "$line" == ORBIT_SEED=* ]]; then
            SEED="${line#ORBIT_SEED=}"
        elif [[ "$line" == ORBIT_SCORE=* ]]; then
            SCORE="${line#ORBIT_SCORE=}"
        elif [[ "$line" == ORBIT_RETURN=* ]]; then
            RETURN="${line#ORBIT_RETURN=}"
        elif [[ "$line" == "---END---" ]]; then
            # We have all metadata + state, run simulation + visualization
            VIDNAME="orbit_seed${SEED}_score${SCORE}_return${RETURN}.mp4"
            VIDPATH="$VIZ_DIR/$VIDNAME"

            if [ -f "$VIDPATH" ]; then
                warn "  Skipping $VIDNAME (already exists)"
                continue
            fi

            log "  Rendering seed=$SEED score=$SCORE return=$RETURN ..."

            # Create temp file for this orbit
            TMPFILE=$(mktemp "$OUTPUT_DIR/tmp_orbit_XXXXXX.txt")
            echo "$STATE" > "$TMPFILE"

            if ./3body "$TMPFILE" 2>/dev/null | \
               python3 visualize.py --output="$VIDPATH" --fps=30 --max-frames=600 2>&1 | tail -1 >> "$LOG"; then
                ok "    -> $VIDNAME"
            else
                warn "    -> FAILED (might be short-lived orbit)"
            fi
            rm -f "$TMPFILE"
            STATE=""
        else
            # This is the state line (12 numbers)
            STATE="$line"
        fi
    done

    local count=$(ls "$VIZ_DIR"/*.mp4 2>/dev/null | wc -l)
    ok "Visualization complete! $count videos in $VIZ_DIR/"
    log "View them with:  ls $VIZ_DIR/"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
run_summary() {
    echo ""
    echo "============================================"
    echo "  3-Body Orbit Search — Summary"
    echo "============================================"
    echo ""

    if [ -f "$ARCHIVE" ]; then
        local entries=$(grep -c '^[^#]' "$ARCHIVE" 2>/dev/null || echo 0)
        echo "  Archive: $ARCHIVE ($entries entries)"
        echo ""
        echo "  Orbits found:"
        awk '
            /^# score/ {
                score = $2; sub(/.*=/, "", score)
                steps = $3; sub(/.*=/, "", steps)
                ret   = $4; sub(/.*=/, "", ret)
                seed  = $5; sub(/.*=/, "", seed)
                printf "    #%-2d  score=%-8.0f  steps=%-5d  return=%-8.4f  seed=%s\n", ++n, score, steps, ret, seed
            }
        ' "$ARCHIVE"
    else
        echo "  No archive found."
    fi

    if [ -f "$BEST" ]; then
        echo ""
        echo "  Best orbit: $BEST"
        head -2 "$BEST"
    fi

    echo ""
    echo "  Commands:"
    echo "    ./3body output/best.txt | python3 visualize.py"
    echo "    cat output/archive.txt"
    echo "    cat output/history.csv"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-quick}" in
    quick)
        setup
        run_quick
        run_summary
        ;;
    explore)
        setup
        run_explore
        run_summary
        ;;
    refine)
        setup
        run_refine
        run_summary
        ;;
    deep)
        setup
        run_deep
        run_summary
        ;;
    all)
        setup
        run_quick
        echo ""
        run_explore
        echo ""
        run_refine
        echo ""
        run_deep || true  # deep może być długie, nie przerywaj całego pipeline'u
        echo ""
        run_summary
        log "Full pipeline complete!"
        ;;
    viz)
        run_viz
        ;;
    viz-all)
        run_viz_all
        ;;
    summary)
        run_summary
        ;;
    *)
        echo "Usage: $0 {quick|explore|refine|deep|all|viz|viz-all|summary}"
        echo ""
        echo "  quick     — Fast test (popsize=64, generations=10, seed=42)"
        echo "  explore   — Search with 10 different seeds → fill archive"
        echo "  refine    — Refine all orbits in archive"
        echo "  deep      — One big search (popsize=512, generations=100)"
        echo "  all       — Full pipeline: explore → refine → deep"
        echo "  viz       — Generate video from best.txt"
        echo "  viz-all   — Generate videos for ALL orbits in archive"
        echo "  summary   — Show archive contents and stats"
        exit 1
        ;;
esac