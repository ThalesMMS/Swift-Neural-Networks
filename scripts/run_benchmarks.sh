#!/bin/bash

# ============================================================================
# Automated Benchmark Suite
# ============================================================================
# Runs all model/backend combinations and collects performance metrics
#
# Usage: ./scripts/run_benchmarks.sh [--quick]
#        --quick: Run 1 epoch for fast verification (default: 5 epochs)
# ============================================================================

# Note: Not using 'set -e' to allow script to continue if some benchmarks fail

echo "========================================="
echo "Automated Benchmarking Suite"
echo "========================================="
echo ""

# Configuration
SEED=42
BATCH=32
EPOCHS=5
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            EPOCHS=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick]"
            exit 1
            ;;
    esac
done

if [ "$QUICK_MODE" = true ]; then
    echo "Running in QUICK mode (1 epoch for fast verification)"
else
    echo "Running FULL benchmark (5 epochs)"
fi

echo ""
echo "This script will:"
echo "1. Build the project in release mode"
echo "2. Run all 6 model/backend combinations with seed=$SEED"
echo "3. Measure training time and memory usage"
echo "4. Extract test accuracy from output"
echo "5. Save results to benchmarks/ directory"
echo ""
echo "Model/Backend combinations:"
echo "  1. MNISTClassic + CPU (hidden=512)"
echo "  2. MNISTClassic + MPS (hidden=512)"
echo "  3. MNISTClassic + MPSGraph (hidden=512)"
echo "  4. MNISTClassic + CPU (hidden=256)"
echo "  5. MNISTClassic + MPS (hidden=256)"
echo "  6. MNISTClassic + MPSGraph (hidden=256)"
echo ""
echo "Note: MNISTMLX benchmarks are skipped due to MLX Metal library"
echo "      incompatibility in worktree environments. Using MNISTClassic"
echo "      with different hidden layer sizes to demonstrate benchmarking."
echo ""
echo "========================================="
echo ""

# Create benchmarks directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCHMARK_DIR="./benchmarks"
mkdir -p "$BENCHMARK_DIR"

# Build the project first
echo "Building project in release mode..."
swift build -c release
echo "Build complete!"
echo ""

# Function to run MNISTClassic benchmark
run_classic_benchmark() {
    local backend=$1
    local hidden=$2
    local suffix=$3
    local backend_flag=""

    if [ "$backend" != "cpu" ]; then
        backend_flag="--$backend"
    fi

    echo "========================================="
    echo "Benchmarking: MNISTClassic ($backend$suffix)"
    echo "========================================="

    # Construct command arguments
    local args="--model mlp --epochs $EPOCHS --batch $BATCH --seed $SEED --hidden $hidden $backend_flag"

    # Create temporary file for output
    local output_file=$(mktemp)
    local time_file=$(mktemp)

    echo "Running: swift run -c release MNISTClassic $args"
    echo ""

    # Run with time tracking (macOS /usr/bin/time -l for memory stats)
    if /usr/bin/time -l swift run -c release MNISTClassic $args > "$output_file" 2> "$time_file"; then
        # Extract metrics from output
        local accuracy=$(grep -E "Test Accuracy:" "$output_file" | awk '{print $3}' | sed 's/%//')
        local total_time=$(grep "user" "$time_file" | awk '{print $1}')
        local peak_memory=$(grep "maximum resident set size" "$time_file" | awk '{print $1}')

        # Convert memory from bytes to MB
        peak_memory_mb=$(echo "scale=2; $peak_memory / 1048576" | bc)

        echo "Results:"
        echo "  Training Time: ${total_time}s"
        echo "  Peak Memory:   ${peak_memory_mb} MB"
        echo "  Test Accuracy: ${accuracy}%"
        echo ""

        # Save to JSON file
        local json_file="$BENCHMARK_DIR/classic_${backend}_h${hidden}_${TIMESTAMP}.json"
        cat > "$json_file" << EOF
{
  "executable": "MNISTClassic",
  "model": "mlp",
  "backend": "$backend",
  "timestamp": "$TIMESTAMP",
  "config": {
    "epochs": $EPOCHS,
    "batch_size": $BATCH,
    "hidden_size": $hidden,
    "seed": $SEED
  },
  "metrics": {
    "training_time_seconds": $total_time,
    "peak_memory_mb": $peak_memory_mb,
    "test_accuracy_percent": $accuracy
  }
}
EOF

        echo "Results saved to: $json_file"
        echo ""
    else
        echo "❌ Benchmark failed for MNISTClassic ($backend)"
        echo ""
    fi

    # Cleanup
    rm -f "$output_file" "$time_file"
}

# Function to run MNISTMLX benchmark
run_mlx_benchmark() {
    local model=$1

    echo "========================================="
    echo "Benchmarking: MNISTMLX ($model)"
    echo "========================================="

    # Construct command arguments
    local args="--model $model --epochs $EPOCHS --batch $BATCH --seed $SEED"

    # Create temporary file for output
    local output_file=$(mktemp)
    local time_file=$(mktemp)

    echo "Running: swift run -c release MNISTMLX $args"
    echo ""

    # Run with time tracking (macOS /usr/bin/time -l for memory stats)
    if /usr/bin/time -l swift run -c release MNISTMLX $args > "$output_file" 2> "$time_file"; then
        # Extract metrics from output
        local accuracy=$(grep -E "Test Accuracy:" "$output_file" | awk '{print $3}' | sed 's/%//')
        local total_time=$(grep "user" "$time_file" | awk '{print $1}')
        local peak_memory=$(grep "maximum resident set size" "$time_file" | awk '{print $1}')

        # Convert memory from bytes to MB
        peak_memory_mb=$(echo "scale=2; $peak_memory / 1048576" | bc)

        echo "Results:"
        echo "  Training Time: ${total_time}s"
        echo "  Peak Memory:   ${peak_memory_mb} MB"
        echo "  Test Accuracy: ${accuracy}%"
        echo ""

        # Save to JSON file
        local json_file="$BENCHMARK_DIR/mlx_${model}_${TIMESTAMP}.json"
        cat > "$json_file" << EOF
{
  "executable": "MNISTMLX",
  "model": "$model",
  "backend": "mlx",
  "timestamp": "$TIMESTAMP",
  "config": {
    "epochs": $EPOCHS,
    "batch_size": $BATCH,
    "seed": $SEED
  },
  "metrics": {
    "training_time_seconds": $total_time,
    "peak_memory_mb": $peak_memory_mb,
    "test_accuracy_percent": $accuracy
  }
}
EOF

        echo "Results saved to: $json_file"
        echo ""
    else
        echo "❌ Benchmark failed for MNISTMLX ($model)"
        echo "   This may be due to MLX Metal library issues in worktree environment"
        echo ""
    fi

    # Cleanup
    rm -f "$output_file" "$time_file"
}

# Run all 6 combinations
echo "========================================="
echo "Starting Benchmark Suite"
echo "========================================="
echo ""

# MNISTClassic benchmarks with hidden=512
run_classic_benchmark "cpu" "512" " (h=512)"
run_classic_benchmark "mps" "512" " (h=512)"
run_classic_benchmark "mpsgraph" "512" " (h=512)"

# MNISTClassic benchmarks with hidden=256
run_classic_benchmark "cpu" "256" " (h=256)"
run_classic_benchmark "mps" "256" " (h=256)"
run_classic_benchmark "mpsgraph" "256" " (h=256)"

# Summary
echo "========================================="
echo "BENCHMARK SUITE COMPLETE"
echo "========================================="
echo ""
echo "Results saved to: $BENCHMARK_DIR/"
echo ""
echo "Generated files:"
BENCHMARK_COUNT=$(ls -1 "$BENCHMARK_DIR"/*_${TIMESTAMP}.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$BENCHMARK_COUNT" -gt 0 ]; then
    ls -1 "$BENCHMARK_DIR"/*_${TIMESTAMP}.json
    echo ""
    echo "Successfully completed $BENCHMARK_COUNT out of 6 benchmarks"
else
    echo "No benchmarks completed successfully"
fi
echo ""
echo "Next steps:"
echo "  1. Generate report: python3 scripts/generate_report.py --input $BENCHMARK_DIR --output report.json --markdown report.md"
echo "  2. View results: cat report.md"
echo ""
echo "✅ Done!"
