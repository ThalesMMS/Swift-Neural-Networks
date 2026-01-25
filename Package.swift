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
        // Library: Reusable MNIST data loading utilities
        // Other packages can import this to load MNIST data into MLXArrays.
        .library(
            name: "MNISTData",
            targets: ["MNISTData"]
        ),

        // Executable: Main training program with all models
        // Run with: swift run MNISTMLX --model cnn
        .executable(
            name: "MNISTMLX",
            targets: ["MNISTMLX"]
        ),

        // Executable: Classic MLP implementation (refactored from mnist_mlp.swift)
        // Run with: swift run MNISTClassic --help
        .executable(
            name: "MNISTClassic",
            targets: ["MNISTClassic"]
        ),
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
            dependencies: [],
            path: "Sources/MNISTClassic"
        ),
    ]
)
