// swift-tools-version: 5.9
// ============================================================================
// Package.swift - Swift Package Manager manifest for MLX Swift Neural Networks
// ============================================================================
//
// This package configures the project to use Apple's MLX Swift framework for
// GPU-accelerated machine learning on Apple Silicon. MLX provides:
//
//   - Automatic differentiation (no manual backward passes!)
//   - GPU acceleration via Metal (unified memory architecture)
//   - Pre-built neural network layers (Conv2d, Linear, etc.)
//   - Optimizers (SGD, Adam, etc.)
//
// REQUIREMENTS:
//   - macOS 14.0+ (Sonoma or later)
//   - Apple Silicon (M1/M2/M3)
//   - Xcode 15+ or Swift 5.9+
//
// BUILD:
//   swift build
//
// RUN:
//   swift run MNISTMLX --model cnn --epochs 3
//
// ============================================================================

import PackageDescription

let package = Package(
    // ---------------------------------------------------------------------------
    // MARK: - Package Identity
    // ---------------------------------------------------------------------------
    name: "Swift-Neural-Networks",
    
    // ---------------------------------------------------------------------------
    // MARK: - Platform Requirements
    // ---------------------------------------------------------------------------
    // MLX requires macOS 14+ for Metal 3 features and unified memory access.
    platforms: [
        .macOS(.v14)
    ],
    
    // ---------------------------------------------------------------------------
    // MARK: - Products
    // ---------------------------------------------------------------------------
    // Products define what this package exports to other packages or executables.
    products: [
        // Library: Shared utilities for MNIST projects
        // Contains SimpleRng, data loaders, activation functions, and ANSIColors
        // for optional colored terminal output (ANSI_COLORS=1).
        .library(
            name: "MNISTCommon",
            targets: ["MNISTCommon"]
        ),

        // Library: Reusable MNIST data loading utilities
        // Other packages can import this to load MNIST data into MLXArrays.
        .library(
            name: "MNISTData",
            targets: ["MNISTData"]
        ),

        // Executable: PRODUCTION RECOMMENDED - Main training program with all models
        // Uses MLX framework for optimal performance on Apple Silicon.
        // Supports CNN, MLP, and Attention models with automatic differentiation.
        // Run with: swift run MNISTMLX --model cnn --epochs 3
        .executable(
            name: "MNISTMLX",
            targets: ["MNISTMLX"]
        ),

        // Executable: EDUCATIONAL REFERENCE - Classic MLP implementation
        // Refactored from mnist_mlp.swift (2222 lines) to demonstrate CPU/GPU backends
        // using system frameworks (Accelerate, Metal, MPS). For production use,
        // prefer MNISTMLX which uses the modern MLX framework.
        // Run with: swift run MNISTClassic --help
        .executable(
            name: "MNISTClassic",
            targets: ["MNISTClassic"]
        ),

        // Executable: Proof-of-concept for MLX compilation
        // Run with: swift run POCCompile
        // TEMPORARILY DISABLED: Missing file in this worktree
        // .executable(
        //     name: "POCCompile",
        //     targets: ["POCCompile"]
        // ),
    ],
    
    // ---------------------------------------------------------------------------
    // MARK: - Dependencies
    // ---------------------------------------------------------------------------
    // External packages this project depends on.
    dependencies: [
        // MLX Swift: Apple's ML framework for Apple Silicon
        // Repository: https://github.com/ml-explore/mlx-swift
        //
        // This single dependency provides multiple products:
        //   - MLX: Core array operations (like NumPy for Swift)
        //   - MLXNN: Neural network layers (Conv2d, Linear, etc.)
        //   - MLXOptimizers: SGD, Adam, etc.
        //   - MLXRandom: Random number generation for initialization
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            from: "0.21.0"
        ),
    ],
    
    // ---------------------------------------------------------------------------
    // MARK: - Targets
    // ---------------------------------------------------------------------------
    // Targets are the basic building blocks of a package.
    targets: [
        // -----------------------------------------------------------------------
        // MNISTCommon: Shared utilities library
        // -----------------------------------------------------------------------
        // This target provides common utilities extracted from duplicated code:
        //   - SimpleRng: Reproducible random number generator
        //   - Data loaders: readMnistImages, readMnistLabels
        //   - Activations: softmaxRows, softmaxRowsPointer
        //   - ANSIColors: Optional colored terminal output (ANSI_COLORS=1)
        // Pure Swift with no external dependencies.
        .target(
            name: "MNISTCommon",
            dependencies: [],
            path: "Sources/MNISTCommon"
        ),

        // -----------------------------------------------------------------------
        // MNISTData: Shared data loading library
        // -----------------------------------------------------------------------
        // This target provides reusable MNIST loading utilities.
        // It's a separate target so it can be imported by tests or other targets.
        .target(
            name: "MNISTData",
            dependencies: [
                // We need MLX to create MLXArray tensors from the raw data
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MNISTData"
        ),
        
        // -----------------------------------------------------------------------
        // MNISTMLX: Main executable with all models
        // -----------------------------------------------------------------------
        // This is the main training program. It includes:
        //   - CNN model (Conv2d + MaxPool + Linear)
        //   - MLP model (Linear + ReLU + Linear)
        //   - Attention model (Transformer-style self-attention)
        //   - CLI interface for model selection and hyperparameters
        .executableTarget(
            name: "MNISTMLX",
            dependencies: [
                // Shared utilities (ANSIColors for colored terminal output)
                "MNISTCommon",

                // Our data loading library
                "MNISTData",

                // MLX core for array operations
                .product(name: "MLX", package: "mlx-swift"),

                // Neural network layers (Conv2d, Linear, Embedding, etc.)
                .product(name: "MLXNN", package: "mlx-swift"),

                // Optimizers (SGD, Adam, AdamW, etc.)
                .product(name: "MLXOptimizers", package: "mlx-swift"),

                // Random number generation for weight initialization
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/MNISTMLX"
        ),

        // -----------------------------------------------------------------------
        // MNISTClassic: Classic MLP implementation with CPU/GPU backends
        // -----------------------------------------------------------------------
        // This is the refactored version of mnist_mlp.swift (2222 lines).
        // It uses system frameworks for CPU (Accelerate) and GPU (Metal/MPS).
        // Run with: swift run MNISTClassic --help
        //   - CPU backend: Pure Swift + Accelerate (vDSP for GEMM)
        //   - MPS backend: Metal Performance Shaders for GPU acceleration
        //   - MPSGraph backend: Higher-level graph API for training
        .executableTarget(
            name: "MNISTClassic",
            dependencies: [
                // Shared utilities (SimpleRng, data loaders, activations)
                "MNISTCommon",
            ],
            path: "Sources/MNISTClassic",
            resources: [
                // Metal shader files for GPU acceleration
                .process("Shaders")
            ]
        ),

        // -----------------------------------------------------------------------
        // POCCompile: Proof-of-concept for MLX function compilation
        // -----------------------------------------------------------------------
        // Demonstrates how to use compile() to optimize training loops.
        // Run with: swift run POCCompile
        // TEMPORARILY DISABLED: Missing file in this worktree
        // .executableTarget(
        //     name: "POCCompile",
        //     dependencies: [
        //         .product(name: "MLX", package: "mlx-swift"),
        //         .product(name: "MLXNN", package: "mlx-swift"),
        //         .product(name: "MLXOptimizers", package: "mlx-swift"),
        //         .product(name: "MLXRandom", package: "mlx-swift"),
        //     ],
        //     path: ".auto-claude/specs/006-use-mlx-compiled-functions-for-training-loops",
        //     sources: ["poc_compile.swift"]
        // ),

        // -----------------------------------------------------------------------
        // MARK: - Test Targets
        // -----------------------------------------------------------------------

        // -----------------------------------------------------------------------
        // MNISTCommonTests: Tests for shared utilities
        // -----------------------------------------------------------------------
        // Tests for:
        //   - SimpleRng: Reproducibility and random number generation
        //   - Activations: softmaxRows, softmaxRowsPointer correctness
        //   - Data loaders: readMnistImages, readMnistLabels
        .testTarget(
            name: "MNISTCommonTests",
            dependencies: [
                "MNISTCommon",
            ],
            path: "Tests/MNISTCommonTests"
        ),

        // -----------------------------------------------------------------------
        // MNISTDataTests: Tests for MNIST data loading
        // -----------------------------------------------------------------------
        // Tests for:
        //   - IDX file format parsing (headers, magic numbers)
        //   - Image normalization ([0,255] → [0.0,1.0])
        //   - Batch creation and shuffling
        //   - Error handling (missing files, invalid format)
        .testTarget(
            name: "MNISTDataTests",
            dependencies: [
                "MNISTData",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/MNISTDataTests"
        ),

        // -----------------------------------------------------------------------
        // MNISTMLXTests: Tests for MLX-based models
        // -----------------------------------------------------------------------
        // Tests for:
        //   - MLP model: Forward pass shapes, gradient flow
        //   - CNN model: Convolution, pooling, forward pass shapes
        //   - Loss functions: Cross-entropy correctness
        //   - Accuracy computation
        .testTarget(
            name: "MNISTMLXTests",
            dependencies: [
                "MNISTMLX",
                "MNISTData",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Tests/MNISTMLXTests"
        ),

        // -----------------------------------------------------------------------
        // MNISTClassicTests: Tests for classic CPU/GPU backends
        // -----------------------------------------------------------------------
        // Tests for:
        //   - CPU GEMM operations (matrix multiplication)
        //   - GEMM transpose modes (transposeA, transposeB)
        //   - GEMM scaling (alpha, beta parameters)
        //   - Backend correctness
        .testTarget(
            name: "MNISTClassicTests",
            dependencies: [
                "MNISTClassic",
                "MNISTCommon",
            ],
            path: "Tests/MNISTClassicTests"
        ),
    ]
)
