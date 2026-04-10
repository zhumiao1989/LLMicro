#!/bin/bash
#
# LLMicro Complete Pipeline
#
# This script runs the complete LLMicro benchmarking pipeline:
# 1. Generate simulated data
# 2. Run LLM parameter recommendation
# 3. Run classification tools (Kraken2, Centrifuge, PathSeq)
# 4. Evaluate results
# 5. Generate figures
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/config"
DATA_DIR="$PROJECT_DIR/data"
RESULTS_DIR="$PROJECT_DIR/results"

# Default parameters
N_SAMPLES=20
SEED=42
PROVIDER="anthropic"
MODEL="claude-sonnet-4-5-20250929"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --skip-simulation)
            SKIP_SIMULATION=true
            shift
            ;;
        --skip-classification)
            SKIP_CLASSIFICATION=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-samples N        Number of samples per complexity type (default: 20)"
            echo "  --seed N             Random seed (default: 42)"
            echo "  --provider PROVIDER  LLM provider (default: anthropic)"
            echo "  --model MODEL        LLM model (default: claude-sonnet-4-5-20250929)"
            echo "  --skip-simulation    Skip data generation step"
            echo "  --skip-classification Skip classification step"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "LLMicro Pipeline Starting"
log_info "========================="
log_info "Project directory: $PROJECT_DIR"
log_info "Number of samples: $N_SAMPLES"
log_info "Random seed: $SEED"
log_info "LLM provider: $PROVIDER"
log_info "LLM model: $MODEL"

# Create directories
log_info "Creating directories..."
mkdir -p "$DATA_DIR"/{simulated,mock,reference}
mkdir -p "$RESULTS_DIR"/{figures,tables,metrics,classifications}

# Step 1: Generate simulated data
if [ "$SKIP_SIMULATION" = true ]; then
    log_info "Skipping data generation (user requested)"
else
    log_info "Step 1: Generating simulated data..."

    python "$PROJECT_DIR/src/simulate_data.py" \
        --config "$CONFIG_DIR/simulation_config.yaml" \
        --output "$DATA_DIR/simulated" \
        --type all \
        --n-samples "$N_SAMPLES" \
        --seed "$SEED"

    log_info "Data generation complete"
fi

# Step 2: Run LLM parameter recommendation
log_info "Step 2: Running LLM parameter recommendation..."

# For each tool, generate recommendations for each sample type
for tool in kraken2 centrifuge pathseq; do
    log_info "  Generating recommendations for $tool..."

    for complexity in low_complexity medium_complexity high_complexity mock; do
        sample_dir="$DATA_DIR/simulated/$complexity"

        if [ -d "$sample_dir" ]; then
            python "$PROJECT_DIR/src/llm_recommender.py" \
                --config "$CONFIG_DIR/llm_config.yaml" \
                --tool "$tool" \
                --input "$sample_dir" \
                --output "$RESULTS_DIR/recommendations/${tool}_${complexity}_params.json" \
                --provider "$PROVIDER" \
                --model "$MODEL"
        fi
    done
done

log_info "Parameter recommendation complete"

# Step 3: Run classification tools
if [ "$SKIP_CLASSIFICATION" = true ]; then
    log_info "Skipping classification (user requested)"
else
    log_info "Step 3: Running classification tools..."

    # This step requires the actual classification tools to be installed
    # and the reference database to be built

    for tool in kraken2 centrifuge pathseq; do
        log_info "  Running $tool..."

        for complexity in low_complexity medium_complexity high_complexity mock; do
            sample_dir="$DATA_DIR/simulated/$complexity"
            output_dir="$RESULTS_DIR/classifications/${tool}_${complexity}"

            if [ -d "$sample_dir" ]; then
                # Run with default parameters
                python "$PROJECT_DIR/src/run_${tool}.py" \
                    --input "$sample_dir" \
                    --database "$DATA_DIR/reference/${tool}" \
                    --output "$output_dir/default" \
                    --mode default

                # Run with LLM-recommended parameters
                params_file="$RESULTS_DIR/recommendations/${tool}_${complexity}_params.json"
                if [ -f "$params_file" ]; then
                    python "$PROJECT_DIR/src/run_${tool}.py" \
                        --input "$sample_dir" \
                        --database "$DATA_DIR/reference/${tool}" \
                        --output "$output_dir/llm" \
                        --params "$params_file" \
                        --mode llm
                fi
            fi
        done
    done

    log_info "Classification complete"
fi

# Step 4: Evaluate results
log_info "Step 4: Evaluating results..."

for tool in kraken2 centrifuge pathseq; do
    python "$PROJECT_DIR/src/evaluate.py" \
        --results "$RESULTS_DIR/classifications/${tool}" \
        --ground-truth "$DATA_DIR/simulated" \
        --tool "$tool" \
        --output "$RESULTS_DIR/metrics/${tool}_metrics.csv"
done

log_info "Evaluation complete"

# Step 5: Generate figures
log_info "Step 5: Generating figures..."

# Combine all metrics
python -c "
import pandas as pd
import glob

all_metrics = []
for f in glob.glob('$RESULTS_DIR/metrics/*_metrics.csv'):
    df = pd.read_csv(f)
    all_metrics.append(df)

combined = pd.concat(all_metrics, ignore_index=True)
combined.to_csv('$RESULTS_DIR/metrics/all_metrics.csv', index=False)
"

# Generate figures
python "$PROJECT_DIR/src/visualize.py" \
    --metrics "$RESULTS_DIR/metrics/all_metrics.csv" \
    --output "$RESULTS_DIR/figures"

log_info "Figures generated"

# Summary
log_info "========================="
log_info "Pipeline complete!"
log_info "Results directory: $RESULTS_DIR"
log_info ""
log_info "Output files:"
log_info "  Metrics: $RESULTS_DIR/metrics/"
log_info "  Figures: $RESULTS_DIR/figures/"
log_info "  Classifications: $RESULTS_DIR/classifications/"
