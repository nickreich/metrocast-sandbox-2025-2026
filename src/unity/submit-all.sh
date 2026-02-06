#!/bin/bash
# Submit all GBQR models to Unity cluster
#
# Usage:
#   ./src/unity/submit-all.sh <today_date> [extra_args]
#
# Examples:
#   ./src/unity/submit-all.sh 2025-12-26
#   ./src/unity/submit-all.sh 2025-12-26 --short_run

set -e

TODAY_DATE="${1:-}"
EXTRA_ARGS="${@:2}"

if [ -z "$TODAY_DATE" ]; then
    echo "Usage: ./submit-all.sh <today_date> [extra_args]"
    echo "Example: ./submit-all.sh 2025-12-26"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Ensure logs directory exists
mkdir -p "$REPO_ROOT/logs"

# Model configurations: model_name:memory:time
# Adjust these based on observed runtimes (added 2h buffer for Unity)
declare -A MODEL_CONFIG
MODEL_CONFIG["gbqr"]="16G:03:30:00"
MODEL_CONFIG["gbqr_ili"]="16G:04:00:00"
MODEL_CONFIG["gbqr_flusurv"]="16G:04:00:00"
MODEL_CONFIG["gbqr_nhsn"]="20G:04:30:00"
MODEL_CONFIG["gbqr_nssp"]="24G:04:30:00"
MODEL_CONFIG["gbqr_5src"]="32G:05:30:00"
MODEL_CONFIG["gbqr_weather"]="16G:03:30:00"

echo "=============================================="
echo "Submitting GBQR models to Unity cluster"
echo "Today Date: $TODAY_DATE"
echo "Extra Args: $EXTRA_ARGS"
echo "=============================================="
echo ""

cd "$REPO_ROOT"

for model in gbqr gbqr_ili gbqr_flusurv gbqr_nhsn gbqr_nssp gbqr_5src gbqr_weather; do
    # Parse config
    IFS=':' read -r mem time <<< "${MODEL_CONFIG[$model]}"

    echo "Submitting $model (mem=$mem, time=$time)..."

    # Submit with model-specific resources
    job_id=$(sbatch \
        --job-name="$model" \
        --mem="$mem" \
        --time="$time" \
        --parsable \
        "$SCRIPT_DIR/run-model.sbatch" "$model" "$TODAY_DATE" $EXTRA_ARGS)

    echo "  Job ID: $job_id"
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo ""
echo "Monitor with: squeue --me"
echo "Job details:  sacct -j <job_id>"
echo "=============================================="
