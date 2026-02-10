#!/bin/bash

# ============================================================================
# Performance Benchmark Script
# ============================================================================
# Measures training time with and without progress bar to verify overhead <1%
#
# Usage: ./performance_benchmark.sh
# ============================================================================

set -e

echo "========================================="
echo "Performance Regression Test"
echo "========================================="
echo ""
echo "This script will:"
echo "1. Build the project"
echo "2. Run training WITH progress bar (5 runs)"
echo "3. Temporarily disable progress bar"
echo "4. Run training WITHOUT progress bar (5 runs)"
echo "5. Restore progress bar"
echo "6. Compare performance and verify overhead <1%"
echo ""
echo "========================================="
echo ""

# Configuration
MODEL="mlp"
EPOCHS=1
BATCH=128
RUNS=5

# Build the project first
echo "Building project..."
swift build -c release
echo "Build complete!"
echo ""

# Function to run training and measure time
run_benchmark() {
    local label=$1
    local total_time=0

    echo "Running benchmark: $label"
    echo "Performing $RUNS runs..."

    for i in $(seq 1 $RUNS); do
        echo -n "  Run $i/$RUNS... "

        # Run training and capture time
        start_time=$(date +%s.%N)
        ./.build/release/MNISTMLX --model $MODEL --epochs $EPOCHS --batch $BATCH > /dev/null 2>&1
        end_time=$(date +%s.%N)

        # Calculate elapsed time
        elapsed=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $elapsed" | bc)

        echo "${elapsed}s"
    done

    # Calculate average
    avg_time=$(echo "scale=3; $total_time / $RUNS" | bc)
    echo "  Average: ${avg_time}s"
    echo ""

    echo $avg_time
}

# Backup the training file
echo "Backing up CompiledTraining.swift..."
cp ./Sources/MNISTMLX/CompiledTraining.swift ./Sources/MNISTMLX/CompiledTraining.swift.backup

# Test 1: WITH progress bar (current implementation)
echo "========================================="
echo "Test 1: WITH Progress Bar"
echo "========================================="
time_with_progress=$(run_benchmark "WITH Progress Bar")

# Disable progress bar by commenting out the update calls
echo "========================================="
echo "Temporarily disabling progress bar..."
echo "========================================="

# Comment out progress bar in CompiledTraining.swift (trainMLPEpochCompiled)
sed -i.tmp 's/progressBar\.update/\/\/ progressBar.update/g' ./Sources/MNISTMLX/CompiledTraining.swift

# Rebuild
echo "Rebuilding without progress bar..."
swift build -c release > /dev/null 2>&1
echo "Rebuild complete!"
echo ""

# Test 2: WITHOUT progress bar
echo "========================================="
echo "Test 2: WITHOUT Progress Bar"
echo "========================================="
time_without_progress=$(run_benchmark "WITHOUT Progress Bar")

# Restore the original file
echo "========================================="
echo "Restoring original code..."
echo "========================================="
mv ./Sources/MNISTMLX/CompiledTraining.swift.backup ./Sources/MNISTMLX/CompiledTraining.swift
rm -f ./Sources/MNISTMLX/CompiledTraining.swift.tmp
echo "Code restored!"
echo ""

# Calculate overhead
echo "========================================="
echo "RESULTS"
echo "========================================="
echo ""
echo "Average time WITH progress bar:    ${time_with_progress}s"
echo "Average time WITHOUT progress bar: ${time_without_progress}s"
echo ""

# Calculate percentage overhead
overhead=$(echo "scale=2; ($time_with_progress - $time_without_progress) / $time_without_progress * 100" | bc)
overhead_absolute=$(echo "$overhead" | sed 's/-//')

echo "Performance overhead: ${overhead}%"
echo ""

# Check if overhead is acceptable
threshold=1.0
comparison=$(echo "$overhead_absolute < $threshold" | bc)

if [ "$comparison" -eq 1 ]; then
    echo "✓ PASS: Overhead is ${overhead}% (< ${threshold}%)"
    echo ""
    echo "The progress bar implementation has negligible performance impact."
    exit 0
else
    echo "✗ FAIL: Overhead is ${overhead}% (>= ${threshold}%)"
    echo ""
    echo "The progress bar implementation may have a performance impact."
    exit 1
fi
