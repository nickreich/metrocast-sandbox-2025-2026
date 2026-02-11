#!/bin/bash
# Setup script for GBQR models on Unity cluster
# Creates virtual environments and installs dependencies
#
# Usage:
#   ./src/unity/setup-unity.sh              # Set up all models
#   ./src/unity/setup-unity.sh gbqr         # Set up specific model(s)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SRC_DIR="$REPO_ROOT/src"

# Available models (excluding gbqr_3src)
ALL_MODELS="gbqr gbqr_5src gbqr_ili gbqr_flusurv gbqr_nhsn gbqr_nssp gbqr_weather"

# Determine which models to set up
if [ $# -eq 0 ]; then
    MODELS="$ALL_MODELS"
    echo "Setting up all models..."
else
    MODELS="$@"
    echo "Setting up specified models: $MODELS"
fi

# Load Python module on Unity
echo ""
echo "Loading Python module..."
module load python/3.11.0 2>/dev/null || module load python/3.10.0 2>/dev/null || {
    echo "Warning: Could not load Python module. Using system Python."
}

PYTHON_VERSION=$(python3 --version)
echo "Using: $PYTHON_VERSION"

# Create logs directory
mkdir -p "$REPO_ROOT/logs"

# Download auxiliary data if needed
if [ ! -f "$REPO_ROOT/auxiliary-data/locations.csv" ]; then
    echo ""
    echo "Downloading auxiliary data..."
    mkdir -p "$REPO_ROOT/auxiliary-data"
    curl -sL "https://raw.githubusercontent.com/reichlab/flu-metrocast/main/auxiliary-data/locations.csv" \
        -o "$REPO_ROOT/auxiliary-data/locations.csv"
    echo "  Downloaded: auxiliary-data/locations.csv"
fi

# Set up each model
for model in $MODELS; do
    MODEL_DIR="$SRC_DIR/$model"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "Warning: Model directory not found: $MODEL_DIR"
        continue
    fi

    echo ""
    echo "=== Setting up $model ==="

    VENV_DIR="$MODEL_DIR/.venv"

    # Remove existing venv if it exists (to ensure clean setup)
    if [ -d "$VENV_DIR" ]; then
        echo "  Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi

    # Create venv
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV_DIR"

    # Activate and install requirements
    echo "  Installing dependencies..."
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --quiet --upgrade pip

    # Install requirements (from model directory so relative paths work)
    cd "$MODEL_DIR"
    if [ -f "requirements.txt" ]; then
        pip install --quiet -r requirements.txt
    else
        pip install --quiet -r ../mchub_gbqr/requirements.txt
    fi
    cd "$REPO_ROOT"

    deactivate

    echo "  Done: $model"
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "To submit jobs:"
echo "  ./src/unity/submit-all.sh YYYY-MM-DD"
echo ""
echo "Or submit a single model:"
echo "  sbatch src/unity/run-model.sbatch <model> YYYY-MM-DD"
