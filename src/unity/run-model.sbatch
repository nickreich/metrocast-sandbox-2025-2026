#!/bin/bash
#SBATCH -c 4                          # CPUs per task
#SBATCH --mem=24G                     # Memory (adjust per model if needed)
#SBATCH -p cpu                        # Partition
#SBATCH -t 05:00:00                   # Time limit (5 hours default)
#SBATCH -o logs/slurm-%j-%x.out       # Output file (%j=job_id, %x=job_name)
#SBATCH --mail-type=END,FAIL          # Email on completion or failure

# GBQR Model Runner for Unity Cluster
#
# Usage:
#   sbatch src/unity/run-model.sbatch <model_name> <today_date> [extra_args]
#
# Examples:
#   sbatch src/unity/run-model.sbatch gbqr 2025-12-26
#   sbatch src/unity/run-model.sbatch gbqr_5src 2025-12-26
#   sbatch src/unity/run-model.sbatch gbqr 2025-12-26 --short_run
#
# Models: gbqr, gbqr_ili, gbqr_flusurv, gbqr_nhsn, gbqr_nssp, gbqr_5src, gbqr_weather

set -e

# Parse arguments
MODEL_NAME="${1:-}"
TODAY_DATE="${2:-}"
EXTRA_ARGS="${@:3}"

if [ -z "$MODEL_NAME" ] || [ -z "$TODAY_DATE" ]; then
    echo "Usage: sbatch run-model.sbatch <model_name> <today_date> [extra_args]"
    echo "Example: sbatch run-model.sbatch gbqr 2025-12-26"
    exit 1
fi

# Validate model name
VALID_MODELS="gbqr gbqr_5src gbqr_ili gbqr_flusurv gbqr_nhsn gbqr_nssp gbqr_weather"
if [[ ! " $VALID_MODELS " =~ " $MODEL_NAME " ]]; then
    echo "Error: Invalid model name '$MODEL_NAME'"
    echo "Valid models: $VALID_MODELS"
    exit 1
fi

# Set job name dynamically (for output file naming)
# Note: This is set via sbatch command line in submit-all.sh

# Determine repo root from submission directory
# SLURM_SUBMIT_DIR is the directory where sbatch was called
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "=============================================="
echo "GBQR Model: $MODEL_NAME"
echo "Today Date: $TODAY_DATE"
echo "Extra Args: $EXTRA_ARGS"
echo "Repo Root:  $REPO_ROOT"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start Time: $(date)"
echo "=============================================="

# Change to repo root
cd "$REPO_ROOT"

# Load Python module
module load python/3.11.0 2>/dev/null || module load python/3.10.0 2>/dev/null || true

# Activate model's virtual environment
VENV_PATH="$REPO_ROOT/src/$MODEL_NAME/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Run setup script first: ./src/unity/setup-unity.sh $MODEL_NAME"
    exit 1
fi

echo "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Verify Python
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Set PYTHONPATH for imports
export PYTHONPATH="$REPO_ROOT"

# Update weather data if running a weather model
if [[ "$MODEL_NAME" == *"weather"* ]]; then
    echo ""
    echo "Updating weather data..."
    python "$REPO_ROOT/src/gbqr_weather/update_weather.py"
fi

# Run the model
echo ""
echo "Running model..."
echo "Command: python src/$MODEL_NAME/main.py --today_date $TODAY_DATE $EXTRA_ARGS"
echo ""

python "src/$MODEL_NAME/main.py" --today_date "$TODAY_DATE" $EXTRA_ARGS

# Deactivate
deactivate

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "=============================================="
