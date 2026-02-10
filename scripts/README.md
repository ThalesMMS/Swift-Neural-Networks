# Benchmark Scripts

This directory contains automated benchmarking tools for measuring performance across different model implementations and backends.

## Overview

The benchmarking suite automates the process of:
1. Building the project in release mode
2. Running multiple model/backend combinations
3. Collecting performance metrics (training time, memory usage, accuracy)
4. Generating aggregated reports

## Scripts

### `run_benchmarks.sh`

Main benchmarking script that runs all model/backend combinations and saves results.

**Usage:**
```bash
# Run full benchmark suite (5 epochs per model)
./scripts/run_benchmarks.sh

# Run quick verification (1 epoch per model)
./scripts/run_benchmarks.sh --quick
```

**What it does:**
1. Builds the project in release mode
2. Runs 6 benchmark combinations (MNISTClassic with different backends and hidden layer sizes)
3. Measures training time using `/usr/bin/time -l`
4. Extracts test accuracy from output
5. Saves results to `benchmarks/` directory as timestamped JSON files

**Configuration:**
- **Seed:** 42 (for reproducibility)
- **Batch size:** 32
- **Epochs:** 5 (or 1 with `--quick`)

**Benchmark Combinations:**
- MNISTClassic + CPU (hidden=512)
- MNISTClassic + MPS (hidden=512)
- MNISTClassic + MPSGraph (hidden=512)
- MNISTClassic + CPU (hidden=256)
- MNISTClassic + MPS (hidden=256)
- MNISTClassic + MPSGraph (hidden=256)

**Output:**

Each benchmark generates a JSON file in `benchmarks/` with the following structure:
```json
{
  "executable": "MNISTClassic",
  "model": "mlp",
  "backend": "cpu",
  "timestamp": "20240315_143022",
  "config": {
    "epochs": 5,
    "batch_size": 32,
    "hidden_size": 512,
    "seed": 42
  },
  "metrics": {
    "training_time_seconds": 123.45,
    "peak_memory_mb": 256.78,
    "test_accuracy_percent": 95.67
  }
}
```

**Notes:**
- The script continues running even if some benchmarks fail
- MNISTMLX benchmarks are skipped in worktree environments due to MLX Metal library incompatibility
- All benchmarks use the same seed (42) for reproducible comparisons

### `generate_report.py`

Python script that aggregates benchmark results and generates reports.

**Usage:**
```bash
# Generate JSON report only
python3 scripts/generate_report.py \
    --input benchmarks/ \
    --output report.json

# Generate JSON report with console summary
python3 scripts/generate_report.py \
    --input benchmarks/ \
    --output report.json \
    --verbose

# Generate both JSON and Markdown reports
python3 scripts/generate_report.py \
    --input benchmarks/ \
    --output report.json \
    --markdown report.md
```

**Options:**
- `--input DIR`: Directory containing benchmark JSON files (required)
- `--output FILE`: Output path for aggregated JSON report (required)
- `--markdown FILE`: Optional output path for markdown report
- `--verbose`: Print summary to console

**JSON Report Structure:**
```json
{
  "summary": {
    "total_benchmarks": 6,
    "models": ["mlp"],
    "backends": ["cpu", "mps", "mpsgraph"],
    "config": {
      "epochs": 5,
      "batch_size": 32,
      "seed": 42
    }
  },
  "results": [
    {
      "model": "mlp",
      "backend": "cpu",
      "compiled": false,
      "timestamp": "20240315_143022",
      "training_time_seconds": 123.45,
      "peak_memory_mb": 256.78,
      "test_accuracy_percent": 95.67
    }
  ]
}
```

**Markdown Report Format:**

The markdown report includes:
- Summary section with total benchmarks, models, backends, and configuration
- Results table with training time, memory usage, and accuracy for each combination

Example:
```markdown
# Benchmark Report

## Summary

- **Total Benchmarks:** 6
- **Models:** mlp
- **Backends:** cpu, mps, mpsgraph

### Configuration

- **Epochs:** 5
- **Batch Size:** 32
- **Seed:** 42

## Results

| Model | Backend | Training Time (s) | Peak Memory (MB) | Test Accuracy (%) |
|-------|---------|-------------------|------------------|-------------------|
| mlp   | cpu     | 123.45            | 256.78           | 95.67             |
| mlp   | mps     | 45.67             | 289.12           | 95.89             |
```

## Complete Workflow

Here's a typical workflow for running benchmarks and generating reports:

```bash
# 1. Run the benchmark suite
./scripts/run_benchmarks.sh

# 2. Generate reports (shown at end of benchmark output)
python3 scripts/generate_report.py \
    --input benchmarks/ \
    --output report.json \
    --markdown report.md \
    --verbose

# 3. View the markdown report
cat report.md

# 4. Or open it in your preferred markdown viewer
open report.md  # macOS
```

## Quick Verification

For rapid testing during development:

```bash
# Run 1 epoch per benchmark (faster)
./scripts/run_benchmarks.sh --quick

# Generate report
python3 scripts/generate_report.py \
    --input benchmarks/ \
    --output quick_report.json \
    --verbose
```

## Output Files

**Benchmark Results:**
- Location: `benchmarks/` directory
- Format: `{executable}_{backend}_h{hidden}_{timestamp}.json`
- Example: `classic_mps_h512_20240315_143022.json`

**Reports:**
- JSON: Aggregated results in machine-readable format
- Markdown: Human-readable formatted tables and summaries

## Interpreting Results

**Training Time:**
- Measured in seconds (user + system time)
- Lower is better
- GPU backends (MPS, MPSGraph) typically faster than CPU

**Peak Memory:**
- Measured in MB (megabytes)
- Maximum resident set size during training
- Varies with batch size and hidden layer size

**Test Accuracy:**
- Measured as percentage (0-100%)
- Final accuracy on MNIST test set
- Should be consistent across runs with same seed

## Troubleshooting

**Script fails to run:**
```bash
# Make sure the script is executable
chmod +x scripts/run_benchmarks.sh
```

**No JSON files generated:**
- Check that the build succeeded
- Verify MNIST data is available in `data/` directory
- Look for error messages in console output

**Python script fails:**
```bash
# Install required Python packages
pip3 install -r requirements.txt
```

**Memory issues:**
- Use `--quick` mode for faster testing
- Reduce batch size by editing the script
- Close other applications

## Advanced Usage

**Custom benchmark runs:**

You can manually run individual benchmarks:

```bash
# Build first
swift build -c release

# Run specific configuration
swift run -c release MNISTClassic \
    --model mlp \
    --epochs 10 \
    --batch 64 \
    --seed 42 \
    --hidden 512 \
    --mps
```

**Filtering results:**

The report generator processes all JSON files in the input directory. To generate reports for specific benchmarks:

```bash
# Create a temporary directory
mkdir -p benchmarks/filtered

# Copy specific results
cp benchmarks/classic_mps_* benchmarks/filtered/

# Generate filtered report
python3 scripts/generate_report.py \
    --input benchmarks/filtered/ \
    --output filtered_report.json \
    --markdown filtered_report.md
```

## Related Documentation

- Main README: `../README.md` - Project overview and model descriptions
- Spec: `../.auto-claude/specs/021-automated-benchmarking-suite/spec.md` - Detailed specification

## Notes

- Benchmarks use fixed seed (42) for reproducibility
- Results may vary slightly between macOS versions or hardware
- GPU benchmarks require Apple Silicon with macOS 14.0+
- MNISTMLX benchmarks are disabled in worktree environments
