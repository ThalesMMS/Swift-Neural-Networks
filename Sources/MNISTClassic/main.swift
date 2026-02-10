// ============================================================================
// main.swift - Entry Point for MNISTClassic Training
// ============================================================================
//
// This is the main executable for the classic MNIST neural network trainer.
// It provides a simple command-line interface for training a 2-layer MLP
// on the MNIST dataset using CPU (Accelerate), MPS, or MPSGraph backends.
//
// USAGE:
//   swift run MNISTClassic [OPTIONS]
//
// AVAILABLE BACKENDS:
//   - CPU (default): Uses Accelerate framework (vDSP)
//   - MPS:           Use --mps flag for Metal Performance Shaders
//   - MPSGraph:      Use --mpsgraph flag for Metal Performance Shaders Graph
//
// COMMAND-LINE OPTIONS:
//   --batch <n>      Batch size (default: 64)
//   --hidden <n>     Hidden layer size (default: 512)
//   --epochs <n>     Number of training epochs (default: 10)
//   --lr <f>         Learning rate (default: 0.01)
//   --seed <n>       Random seed (default: 1)
//   --mps            Use Metal Performance Shaders backend
//   --mpsgraph       Use Metal Performance Shaders Graph backend
//   --help           Show usage information
//
// ============================================================================

import Foundation
import MNISTCommon

// =============================================================================
// MARK: - Main Entry Point
// =============================================================================

func main() {
    // Apply CLI overrides to global configuration
    applyCliOverrides()

    // Check which backend to use
    let useMpsGraph = CommandLine.arguments.contains("--mpsgraph")
    let useMPS = CommandLine.arguments.contains("--mps") || useMpsGraph

    let programStart = Date()

    // -------------------------------------------------------------------------
    // Data Loading
    // -------------------------------------------------------------------------
    ColoredPrint.progress("Loading training data...")
    let loadStart = Date()
    let trainImages = readMnistImages(path: "./data/train-images.idx3-ubyte", count: trainSamples)
    let trainLabels = readMnistLabels(path: "./data/train-labels.idx1-ubyte", count: trainSamples)

    ColoredPrint.progress("Loading test data...")
    let testImages = readMnistImages(path: "./data/t10k-images.idx3-ubyte", count: testSamples)
    let testLabels = readMnistLabels(path: "./data/t10k-labels.idx1-ubyte", count: testSamples)
    let loadTime = Date().timeIntervalSince(loadStart)
    ColoredPrint.info(String(format: "Data loading time: %.2f seconds", loadTime))

    // -------------------------------------------------------------------------
    // Model Initialization
    // -------------------------------------------------------------------------
    ColoredPrint.progress("Initializing neural network...")
    ColoredPrint.info("Config: hidden=\(numHidden) batch=\(batchSize) epochs=\(epochs) lr=\(learningRate) seed=\(rngSeed)")
    var rng = SimpleRng(seed: rngSeed)
    var nn = initializeNetwork(rng: &rng)

    // -------------------------------------------------------------------------
    // Training
    // -------------------------------------------------------------------------
    ColoredPrint.progress("Training neural network...")
    let trainStart = Date()
    var usedGraph = false

    if useMpsGraph {
        #if canImport(MetalPerformanceShadersGraph)
        ColoredPrint.info("Using MPSGraph backend.")
        trainMpsGraph(
            nn: &nn,
            images: trainImages,
            labels: trainLabels,
            numSamples: trainImages.count / numInputs,
            rng: &rng
        )
        usedGraph = true
        #else
        ColoredPrint.warning("MPSGraph not available, falling back to MPS kernels.")
        #endif
    }

    if !usedGraph {
        let backend = selectGemmBackend(useMPS: useMPS)
        switch backend {
        case .cpu(let cpu):
            train(
                nn: &nn,
                images: trainImages,
                labels: trainLabels,
                numSamples: trainImages.count / numInputs,
                engine: cpu,
                rng: &rng
            )
        #if canImport(MetalPerformanceShaders)
        case .mps(let mps):
            trainMps(
                nn: &nn,
                images: trainImages,
                labels: trainLabels,
                numSamples: trainImages.count / numInputs,
                engine: mps,
                rng: &rng
            )
        #endif
        }
    }

    let trainTime = Date().timeIntervalSince(trainStart)
    ColoredPrint.info(String(format: "Total training time: %.2f seconds", trainTime))

    // -------------------------------------------------------------------------
    // Testing
    // -------------------------------------------------------------------------
    ColoredPrint.progress("Testing neural network...")
    let testStart = Date()
    var testedOnGPU = false

    #if canImport(MetalPerformanceShadersGraph)
    if useMpsGraph {
        ColoredPrint.info("Testing with MPSGraph backend.")
        testMpsGraph(
            nn: nn,
            images: testImages,
            labels: testLabels,
            numSamples: testImages.count / numInputs
        )
        testedOnGPU = true
    }
    #endif

    if !testedOnGPU && useMPS {
        #if canImport(MetalPerformanceShaders)
        if let engine = MpsGemmEngine() {
            ColoredPrint.info("Testing with MPS GEMM backend.")
            testMps(
                nn: nn,
                images: testImages,
                labels: testLabels,
                numSamples: testImages.count / numInputs,
                engine: engine
            )
            testedOnGPU = true
        }
        #endif
    }

    if !testedOnGPU {
        test(
            nn: nn,
            images: testImages,
            labels: testLabels,
            numSamples: testImages.count / numInputs
        )
    }

    let testTime = Date().timeIntervalSince(testStart)
    ColoredPrint.info(String(format: "Testing time: %.2f seconds", testTime))

    // -------------------------------------------------------------------------
    // Model Persistence
    // -------------------------------------------------------------------------
    ColoredPrint.progress("Saving model...")
    saveModel(nn: nn, filename: "mnist_model.bin")

    // -------------------------------------------------------------------------
    // Performance Summary
    // -------------------------------------------------------------------------
    let totalTime = Date().timeIntervalSince(programStart)
    ColoredPrint.success("\n=== Performance Summary ===")
    ColoredPrint.info(String(format: "Data loading time: %.2f seconds", loadTime))
    ColoredPrint.info(String(format: "Total training time: %.2f seconds", trainTime))
    ColoredPrint.info(String(format: "Testing time: %.2f seconds", testTime))
    ColoredPrint.info(String(format: "Total program time: %.2f seconds", totalTime))
    ColoredPrint.success("========================")
}

// =============================================================================
// Execute main
// =============================================================================

// Force unbuffered output for immediate console feedback
import Darwin
setbuf(__stdoutp, nil)

main()
