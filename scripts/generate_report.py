#!/usr/bin/env python3

"""
Benchmark Report Generator
Aggregates benchmark results from JSON files and generates reports
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


def load_benchmark_files(input_dir: str) -> List[Dict]:
    """
    Load all JSON benchmark files from the specified directory.

    Args:
        input_dir: Directory containing benchmark JSON files

    Returns:
        List of benchmark result dictionaries
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)

    # Find all JSON files
    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        print(f"Warning: No JSON files found in '{input_dir}'")
        return []

    results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse {json_file}: {e}")
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return results


def aggregate_results(results: List[Dict]) -> Dict:
    """
    Aggregate benchmark results into a standardized report format.

    Args:
        results: List of individual benchmark results

    Returns:
        Aggregated report dictionary
    """
    if not results:
        return {
            "summary": {
                "total_benchmarks": 0,
                "models": [],
                "backends": []
            },
            "results": []
        }

    # Extract unique models and backends
    models = sorted(set(r["model"] for r in results))
    backends = sorted(set(r["backend"] for r in results))

    # Get config from first result (should be same for all)
    config = results[0].get("config", {})

    report = {
        "summary": {
            "total_benchmarks": len(results),
            "models": models,
            "backends": backends,
            "config": config
        },
        "results": []
    }

    # Add each benchmark result
    for result in results:
        metrics = result.get("metrics", {})
        report["results"].append({
            "model": result.get("model"),
            "backend": result.get("backend"),
            "compiled": result.get("compiled", False),
            "timestamp": result.get("timestamp"),
            "training_time_seconds": metrics.get("training_time_seconds"),
            "peak_memory_mb": metrics.get("peak_memory_mb"),
            "test_accuracy_percent": metrics.get("test_accuracy_percent")
        })

    # Sort results by model then backend
    report["results"].sort(key=lambda x: (x["model"], x["backend"]))

    return report


def save_json_report(report: Dict, output_file: str) -> None:
    """
    Save aggregated report to JSON file.

    Args:
        report: Aggregated report dictionary
        output_file: Output file path
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved to: {output_file}")
    except Exception as e:
        print(f"Error: Failed to save JSON report: {e}")
        sys.exit(1)


def save_markdown_report(report: Dict, output_file: str) -> None:
    """
    Save aggregated report to Markdown file with formatted tables.

    Args:
        report: Aggregated report dictionary
        output_file: Output file path
    """
    try:
        with open(output_file, 'w') as f:
            # Write header
            f.write("# Benchmark Report\n\n")

            # Write summary section
            summary = report["summary"]
            f.write("## Summary\n\n")
            f.write(f"- **Total Benchmarks:** {summary['total_benchmarks']}\n")
            f.write(f"- **Models:** {', '.join(summary['models'])}\n")
            f.write(f"- **Backends:** {', '.join(summary['backends'])}\n")

            if summary.get("config"):
                config = summary["config"]
                f.write("\n### Configuration\n\n")
                f.write(f"- **Epochs:** {config.get('epochs', 'N/A')}\n")
                f.write(f"- **Batch Size:** {config.get('batch_size', 'N/A')}\n")
                f.write(f"- **Seed:** {config.get('seed', 'N/A')}\n")

            # Write results table
            f.write("\n## Results\n\n")
            f.write("| Model | Backend | Training Time (s) | Peak Memory (MB) | Test Accuracy (%) |\n")
            f.write("|-------|---------|-------------------|------------------|-------------------|\n")

            for result in report["results"]:
                model = result["model"]
                backend = result["backend"]
                time_s = result["training_time_seconds"]
                memory_mb = result["peak_memory_mb"]
                accuracy = result["test_accuracy_percent"]

                f.write(f"| {model} | {backend} | {time_s} | {memory_mb} | {accuracy} |\n")

            f.write("\n")

        print(f"Markdown report saved to: {output_file}")
    except Exception as e:
        print(f"Error: Failed to save markdown report: {e}")
        sys.exit(1)


def print_summary(report: Dict) -> None:
    """
    Print a summary of the benchmark results.

    Args:
        report: Aggregated report dictionary
    """
    print()
    print("=" * 60)
    print("BENCHMARK REPORT SUMMARY")
    print("=" * 60)
    print()

    summary = report["summary"]
    print(f"Total Benchmarks: {summary['total_benchmarks']}")
    print(f"Models: {', '.join(summary['models'])}")
    print(f"Backends: {', '.join(summary['backends'])}")

    if summary.get("config"):
        config = summary["config"]
        print()
        print("Configuration:")
        print(f"  Epochs: {config.get('epochs', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"  Seed: {config.get('seed', 'N/A')}")

    print()
    print("Results:")
    print()

    for result in report["results"]:
        model = result["model"]
        backend = result["backend"]
        time_s = result["training_time_seconds"]
        memory_mb = result["peak_memory_mb"]
        accuracy = result["test_accuracy_percent"]

        print(f"  {model} ({backend}):")
        print(f"    Training Time: {time_s}s")
        print(f"    Peak Memory:   {memory_mb} MB")
        print(f"    Test Accuracy: {accuracy}%")
        print()

    print("=" * 60)
    print()


def main():
    """Main entry point for the report generator."""
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate JSON report
  %(prog)s --input benchmarks/ --output report.json

  # Generate with summary output
  %(prog)s --input benchmarks/ --output report.json --verbose

  # Generate both JSON and markdown reports
  %(prog)s --input benchmarks/ --output report.json --markdown report.md
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing benchmark JSON files"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for aggregated JSON report"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print summary to console"
    )

    parser.add_argument(
        "--markdown",
        type=str,
        help="Optional output file path for markdown report"
    )

    args = parser.parse_args()

    # Load benchmark results
    print(f"Loading benchmark results from: {args.input}")
    results = load_benchmark_files(args.input)

    if not results:
        print("No benchmark results to process")
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark result(s)")

    # Aggregate results
    report = aggregate_results(results)

    # Save JSON report
    save_json_report(report, args.output)

    # Save markdown report if requested
    if args.markdown:
        save_markdown_report(report, args.markdown)

    # Print summary if requested
    if args.verbose:
        print_summary(report)

    print()
    print("✅ Report generation complete!")


if __name__ == "__main__":
    main()
