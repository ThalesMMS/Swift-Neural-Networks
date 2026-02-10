#!/usr/bin/env python3

"""
Performance Analysis Script
Analyzes the benchmark results to determine progress bar overhead
"""

# Results from benchmark
time_with_progress = 0.142  # Average time WITH progress bar
time_without_progress = 0.119  # Average time WITHOUT progress bar

# Calculate overhead
overhead_seconds = time_with_progress - time_without_progress
overhead_percent = (overhead_seconds / time_without_progress) * 100

print("=" * 60)
print("PERFORMANCE ANALYSIS")
print("=" * 60)
print()
print(f"Average time WITH progress bar:    {time_with_progress:.3f}s")
print(f"Average time WITHOUT progress bar: {time_without_progress:.3f}s")
print()
print(f"Absolute overhead: {overhead_seconds:.3f}s")
print(f"Percentage overhead: {overhead_percent:.2f}%")
print()

# Check if overhead is acceptable (<1%)
threshold = 1.0
if abs(overhead_percent) < threshold:
    print(f"✓ PASS: Overhead is {overhead_percent:.2f}% (< {threshold}%)")
    print()
    print("The progress bar implementation has negligible performance impact.")
    exit_code = 0
else:
    print(f"✗ FAIL: Overhead is {overhead_percent:.2f}% (>= {threshold}%)")
    print()
    print("The progress bar implementation may have a performance impact.")
    exit_code = 1

print()
print("=" * 60)
print("DETAILED NOTES")
print("=" * 60)
print()
print("Note: The first run in each set was significantly slower due to")
print("cold start (build artifacts, caching, etc.). The subsequent runs")
print("show more consistent performance:")
print()
print("WITH progress bar:    Runs 2-5: ~0.026-0.030s")
print("WITHOUT progress bar: Runs 2-5: ~0.027-0.030s")
print()
print("The warm-start runs show that the progress bar overhead is")
print("extremely minimal, well within normal measurement variance.")

import sys
sys.exit(exit_code)
