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
- Root legacy files are thin wrappers that forward to SwiftPM targets
- MNISTClassic, MNISTManualCNN, and MNISTManualAttention import MNISTCommon instead of carrying their own copies

## Benefits

- **Single source of truth** - Bug fixes applied once, not 4 times
- **Consistency** - All examples use the same implementation
- **Maintainability** - Reduced code footprint makes the project easier to understand
- **Reusability** - New examples can import MNISTCommon instead of copying code

## Migration Paths

### Path 1: Using Legacy Wrapper Scripts

The root Swift files are now small compatibility wrappers. They preserve the familiar commands while delegating to modular SwiftPM targets:

```bash
swift mnist_mlp.swift --help              # forwards to MNISTClassic
swift mnist_cnn.swift --help              # forwards to MNISTManualCNN
swift mnist_attention_pool.swift --help   # forwards to MNISTManualAttention
```

Use the SwiftPM target names directly for normal development and testing.

### Path 2: Using the Modular Package (Recommended)

The cleanest approach is to use the existing Swift package structure:

```bash
# Build all targets
swift build

# Run MNISTClassic (refactored from mnist_mlp.swift)
swift run MNISTClassic --help
swift run MNISTClassic --mps --epochs 5 --batch 128

# Run modular manual educational models
swift run MNISTManualCNN --help
swift run MNISTManualAttention --help

# Run MNISTMLX (modern MLX-based implementation)
swift run MNISTMLX --model cnn --epochs 3
swift run MNISTMLX --model mlp --epochs 10
```

### Path 3: Importing MNISTCommon in Your Own Code

If you're writing a new Swift package or application:

**1. Add dependency in Package.swift:**

```swift
dependencies: [
    .package(url: "https://github.com/ThalesMMS/Swift-Neural-Networks.git", from: "1.0.0")
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
// OLD: Duplicated ~170 lines of utilities in every file

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

### After: root wrapper files

```swift
// NEW: root files are small compatibility wrappers

// mnist_mlp.swift              -> swift run MNISTClassic
// mnist_cnn.swift              -> swift run MNISTManualCNN
// mnist_attention_pool.swift   -> swift run MNISTManualAttention
```

### After: MNISTClassic/RNG.swift (4 lines instead of 44)

```swift
// NEW: Re-export from shared library instead of duplicating

import MNISTCommon

// Re-export SimpleRng so existing MNISTClassic code works unchanged
public typealias Rng = SimpleRng
```

## Why Legacy Scripts Forward to SwiftPM

**Before refactoring:**
```bash
# Worked: Self-contained script with all utilities included
swift mnist_cnn.swift
```

**After refactoring:**
```bash
# Still works, but forwards to the modular package target
swift mnist_cnn.swift

# Preferred direct command
swift run MNISTManualCNN
```

The manual implementations now live in SwiftPM targets so they can share `MNISTCommon`, stay testable, and keep each Swift file below 1000 lines.

## Verification

After migration, verify everything works:

```bash
# Build all targets
swift build

# Should succeed with no errors
swift build --target MNISTCommon
swift build --target MNISTClassic
swift build --target MNISTManualCNN
swift build --target MNISTManualAttention

# Run a quick test
swift run MNISTClassic --epochs 1 --batch 32
swift run MNISTManualCNN --help
swift run MNISTManualAttention --help

# Verify deduplication worked
! grep -q '^struct SimpleRng' mnist_mlp.swift && echo "Deduplication successful"
```

## Troubleshooting

### Error: "Cannot find 'SimpleRng' in scope"

**Problem:** Compiling extracted implementation files outside SwiftPM without declaring `MNISTCommon`.

**Solution:** Use the package targets or add `MNISTCommon` to your own package target dependencies.

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
| **Standalone scripts** | Large self-contained files | Small wrappers that forward to SwiftPM targets |
| **Package usage** | Each module has own copy | All modules import MNISTCommon |

## Next Steps

1. **For existing projects:** Follow Path 1 or Path 2 above depending on your needs
2. **For new projects:** Import MNISTCommon as a dependency (Path 3)
3. **For quick experiments:** Use the root wrapper scripts or run the SwiftPM targets directly
4. **For production code:** Use the modular package structure (Path 2)

See `Sources/MNISTCommon/README.md` for detailed API documentation.
