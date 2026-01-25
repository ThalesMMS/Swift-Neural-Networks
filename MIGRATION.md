# Migration Guide: MNISTCommon Shared Library

## Overview

This project has been refactored to eliminate ~580 lines of duplicate code by extracting common utilities into a new shared library module called **MNISTCommon**.

## What Changed

Previously, the following code was duplicated across 4 standalone Swift files:
- `mnist_mlp.swift`
- `mnist_cnn.swift`
- `mnist_attention_pool.swift`
- `mlp_simple.swift`

**Duplicated components (~580 lines total):**
1. **SimpleRng** struct (~40 lines × 4 files = ~160 lines)
2. **readMnistImages** function (~45 lines × 3 files = ~135 lines)
3. **readMnistLabels** function (~35 lines × 3 files = ~105 lines)
4. **Softmax** implementations (~20 lines × 3 files = ~60 lines)

**After refactoring:**
- All utilities are now in `Sources/MNISTCommon/` (1 implementation shared by all)
- Standalone files have been trimmed by removing duplicate code
- MNISTClassic module now imports from MNISTCommon instead of having its own copies

## Benefits

✅ **Single source of truth** - Bug fixes applied once, not 4 times
✅ **Consistency** - All examples use the same implementation
✅ **Maintainability** - Reduced code footprint makes the project easier to understand
✅ **Reusability** - New examples can import MNISTCommon instead of copying code

## Migration Paths

### Path 1: Using Standalone Scripts (Simple Compilation)

The standalone Swift files (`mnist_mlp.swift`, `mnist_cnn.swift`, etc.) **no longer contain** the utility functions they depend on. You have two options:

#### Option A: Copy utilities back (quick and dirty)
If you need a truly standalone script, copy the required utilities from `Sources/MNISTCommon/`:

```bash
# Example: Create a self-contained CNN script
cat Sources/MNISTCommon/SimpleRng.swift \
    Sources/MNISTCommon/DataLoading.swift \
    mnist_cnn.swift > mnist_cnn_standalone.swift

swift mnist_cnn_standalone.swift
```

#### Option B: Compile with MNISTCommon (recommended)
Compile the standalone script as part of a mini-package that depends on MNISTCommon:

```bash
# Create a temporary package for the script
mkdir -p /tmp/mnist-script
cd /tmp/mnist-script

# Copy the script
cp /path/to/mnist_cnn.swift main.swift

# Create a minimal Package.swift
cat > Package.swift << 'EOF'
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MNISTScript",
    platforms: [.macOS(.v11)],
    dependencies: [
        .package(path: "/path/to/Swift-Neural-Networks")
    ],
    targets: [
        .executableTarget(
            name: "MNISTScript",
            dependencies: [
                .product(name: "MNISTCommon", package: "Swift-Neural-Networks")
            ],
            path: "."
        )
    ]
)
EOF

# Add import at top of main.swift
echo "import MNISTCommon" | cat - main.swift > temp && mv temp main.swift

# Build and run
swift run
```

### Path 2: Using the Modular Package (Recommended)

The cleanest approach is to use the existing Swift package structure:

```bash
# Build all targets
swift build

# Run MNISTClassic (refactored from mnist_mlp.swift)
swift run MNISTClassic --help
swift run MNISTClassic --mps --epochs 5 --batch 128

# Run MNISTMLX (modern MLX-based implementation)
swift run MNISTMLX --model cnn --epochs 3
swift run MNISTMLX --model mlp --epochs 10
```

### Path 3: Importing MNISTCommon in Your Own Code

If you're writing a new Swift package or application:

**1. Add dependency in Package.swift:**

```swift
dependencies: [
    .package(url: "https://github.com/your-username/Swift-Neural-Networks", from: "1.0.0")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "MNISTCommon", package: "Swift-Neural-Networks")
        ]
    )
]
```

**2. Import and use in your Swift code:**

```swift
import MNISTCommon

// Initialize RNG for weight initialization
var rng = SimpleRng(seed: 42)
let randomWeight = rng.nextFloat()

// Load MNIST data
let trainImages = readMnistImages(path: "./data/train-images.idx3-ubyte")
let trainLabels = readMnistLabels(path: "./data/train-labels.idx1-ubyte")

// Apply softmax activation
var logits: [Float] = [2.0, 1.0, 0.1]
let probabilities = softmaxRows(&logits, rows: 1, cols: 3)
```

## Before/After Examples

### Before: mnist_mlp.swift (2223 lines)

```swift
// ❌ OLD: Duplicated ~170 lines of utilities in every file

struct SimpleRng {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed != 0 ? seed : 123456789 }
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
    // ... 30 more lines
}

func readMnistImages(path: String) -> [Float] {
    // ... 45 lines of IDX parsing
}

func readMnistLabels(path: String) -> [UInt8] {
    // ... 35 lines of IDX parsing
}

func softmaxRows(_ data: inout [Float], rows: Int, cols: Int) {
    // ... 20 lines of softmax math
}

// ... 2000+ lines of actual model code
```

### After: mnist_mlp.swift (trimmed by ~170 lines)

```swift
// ✅ NEW: Clean file focuses on model logic, imports shared utilities

// NOTE: SimpleRng, readMnistImages, readMnistLabels, and softmax functions
//       have been extracted to Sources/MNISTCommon/
// To use this file, compile it as part of a package with MNISTCommon dependency,
// or copy the utilities from Sources/MNISTCommon/ for standalone usage.

// ... 2000+ lines of actual model code (utilities removed)
```

### After: MNISTClassic/RNG.swift (4 lines instead of 44)

```swift
// ✅ NEW: Re-export from shared library instead of duplicating

import MNISTCommon

// Re-export SimpleRng so existing MNISTClassic code works unchanged
public typealias Rng = SimpleRng
```

## Why Standalone Scripts Need Compilation

**Before refactoring:**
```bash
# ✅ Worked: Self-contained script with all utilities included
swift mnist_cnn.swift
```

**After refactoring:**
```bash
# ❌ Fails: Missing SimpleRng, readMnistImages, etc.
swift mnist_cnn.swift

# ✅ Works: Compile as part of package
swift run MNISTClassic  # for the MLP model
# Or create a temporary package (see Option B above)
```

The standalone scripts now rely on external utilities from `MNISTCommon`. Swift's compiler can't automatically find these when running a single `.swift` file directly. You must either:
1. Copy the utilities back into the script (makes it self-contained again)
2. Compile the script as part of a Swift package that declares the MNISTCommon dependency

## Verification

After migration, verify everything works:

```bash
# Build all targets
swift build

# Should succeed with no errors
swift build --target MNISTCommon
swift build --target MNISTClassic

# Run a quick test
swift run MNISTClassic --epochs 1 --batch 32

# Verify deduplication worked
! grep -q '^struct SimpleRng' mnist_mlp.swift && echo "✅ Deduplication successful"
```

## Troubleshooting

### Error: "Cannot find 'SimpleRng' in scope"

**Problem:** Trying to compile a standalone script without MNISTCommon.

**Solution:** Use one of the migration paths above (Option A or B for standalone scripts, or use the package structure).

### Error: "No such module 'MNISTCommon'"

**Problem:** Package.swift doesn't declare MNISTCommon dependency.

**Solution:** Add MNISTCommon to your target's dependencies in Package.swift:

```swift
.target(
    name: "YourTarget",
    dependencies: ["MNISTCommon"]
)
```

### Error: "'readMnistImages' is inaccessible due to 'internal' protection level"

**Problem:** MNISTCommon functions aren't marked `public`.

**Solution:** This should be fixed in the current version. If you encounter this, verify you're using the latest code where all MNISTCommon APIs are `public`.

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Lines of code** | ~580 duplicate lines across 4 files | ~200 lines in shared library (1× implementation) |
| **Maintenance** | Fix bugs in 4 places | Fix bugs in 1 place |
| **Consistency** | Implementations can drift apart | Single source of truth |
| **Standalone scripts** | Run directly with `swift file.swift` | Need compilation or utility copy |
| **Package usage** | Each module has own copy | All modules import MNISTCommon |

## Next Steps

1. **For existing projects:** Follow Path 1 or Path 2 above depending on your needs
2. **For new projects:** Import MNISTCommon as a dependency (Path 3)
3. **For quick experiments:** Use Path 1, Option A to create self-contained scripts
4. **For production code:** Use the modular package structure (Path 2)

See `Sources/MNISTCommon/README.md` for detailed API documentation.
