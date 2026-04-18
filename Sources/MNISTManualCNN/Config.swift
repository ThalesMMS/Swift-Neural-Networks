import Foundation
import Accelerate
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

#if canImport(Metal)
import Metal
#endif

#if canImport(MetalPerformanceShaders)
import MetalPerformanceShaders
#endif

// =============================================================================
// MARK: - MNIST Constants and Configuration
// =============================================================================

// MNIST constants (images are flat 28x28 in row-major order).
let imgH = 28
let imgW = 28
let numInputs = imgH * imgW // 784
let numClasses = 10
let trainSamples = 60_000
let testSamples  = 10_000

// CNN topology: 1x28x28 -> conv -> ReLU -> 2x2 maxpool -> FC(10).
let convOut = 8
let kernel = 3
let pad = 1
let pool = 2

let poolH = imgH / pool
let poolW = imgW / pool
let fcIn = convOut * poolH * poolW // 1568

// =============================================================================
// MARK: - Command-Line Argument Parsing
// =============================================================================

/// Configuration parsed from command-line arguments
struct Config {
    var epochs: Int = 3
    var batchSize: Int = 32
    var learningRate: Float = 0.01
    var dataPath: String = "./data"
    var seed: UInt64 = 1
    var useGpu: Bool = false

    /// Parses command-line arguments into configuration
    ///
    /// This is a simple hand-rolled parser. For production code,
    /// Parses command-line arguments and returns a populated `Config`.
    /// 
    /// Recognizes the following options and updates the corresponding `Config` fields:
    /// - `--epochs` / `-e` <Int>
    /// - `--batch` / `-b` <Int>
    /// - `--lr` / `-l` <Float>
    /// - `--data` / `-d` <String>
    /// - `--seed` / `-s` <UInt64>
    /// - `--gpu`
    /// - `--help` / `-h`
    /// 
    /// Invalid or missing option values are reported to stderr and exit with code 1. Invoking `--help`/`-h` prints usage text and exits with code 0. An unrecognized argument prints an error, prints usage, and exits with code 1.
    /// - Returns: A `Config` populated from `CommandLine.arguments` with recognized options applied; fields not set remain at their defaults.
    static func parse() -> Config {
        var config = Config()
        let args = CommandLine.arguments
        var i = 1

        func requireValue(for option: String) -> String {
            i += 1
            guard i < args.count else {
                fputs("Missing value for \(option)\n", stderr)
                exit(1)
            }
            return args[i]
        }

        func failInvalidValue(_ option: String, _ value: String, expected: String) -> Never {
            fputs("Invalid value for \(option) \(value); expected \(expected)\n", stderr)
            exit(1)
        }

        while i < args.count {
            let arg = args[i]

            switch arg {
            case "--epochs", "-e":
                let value = requireValue(for: arg)
                guard let val = Int(value), val > 0 else {
                    failInvalidValue(arg, value, expected: "a positive Int")
                }
                config.epochs = val

            case "--batch", "-b":
                let value = requireValue(for: arg)
                guard let val = Int(value), val > 0 else {
                    failInvalidValue(arg, value, expected: "a positive Int")
                }
                config.batchSize = val

            case "--lr", "-l":
                let value = requireValue(for: arg)
                guard let val = Float(value), val.isFinite, val > 0 else {
                    failInvalidValue(arg, value, expected: "a positive finite Float")
                }
                config.learningRate = val

            case "--data", "-d":
                let value = requireValue(for: arg)
                guard !value.isEmpty else {
                    failInvalidValue(arg, value, expected: "a non-empty path")
                }
                config.dataPath = value

            case "--seed", "-s":
                let value = requireValue(for: arg)
                guard let val = UInt64(value) else {
                    failInvalidValue(arg, value, expected: "a UInt64")
                }
                config.seed = val

            case "--gpu":
                config.useGpu = true

            case "--help", "-h":
                printUsage()
                exit(0)

            default:
                print("Unknown argument: \(arg)")
                printUsage()
                exit(1)
            }

            i += 1
        }

        return config
    }
}

/// Prints the program usage text describing command-line options, example invocations, the model architecture, expected MNIST data filenames, and output log location.///
/// The help text includes supported flags (`--epochs/-e`, `--batch/-b`, `--lr/-l`, `--data/-d`, `--seed/-s`, `--gpu`, `--help/-h`), default values, example commands, a brief model summary, required dataset filenames under the data path, and the default training log file.
func printUsage() {
    print("""
    MNIST CNN - Convolutional Neural Network for MNIST
    ===================================================

    USAGE:
      swift run MNISTManualCNN [OPTIONS]
      swift mnist_cnn.swift [OPTIONS]  # legacy wrapper

    OPTIONS:
      --epochs, -e <n>      Number of training epochs (default: 3)
      --batch, -b <n>       Batch size (default: 32)
      --lr, -l <f>          Learning rate (default: 0.01)
      --data, -d <path>     Path to MNIST data directory (default: ./data)
      --seed, -s <n>        Random seed for reproducibility (default: 1)
      --gpu                 Enable GPU acceleration (Metal/MPS, default: off)
                            Note: GPU and CPU may produce different convergence paths
                            due to floating-point precision and operation ordering.
                            This is expected behavior for GPU acceleration.
      --help, -h            Show this help message

    EXAMPLES:
      swift run MNISTManualCNN --epochs 5
      swift run MNISTManualCNN -e 10 -b 64 -l 0.005
      swift mnist_cnn.swift --seed 42

    MODEL ARCHITECTURE:
      Input:  28×28 grayscale images (784 pixels)
      Conv:   3×3 kernel, 8 filters, ReLU activation
      Pool:   2×2 max pooling
      FC:     Fully connected layer to 10 classes
      Output: 10-class softmax (digits 0-9)

    EXPECTED DATA FILES:
      <data-path>/train-images.idx3-ubyte
      <data-path>/train-labels.idx1-ubyte
      <data-path>/t10k-images.idx3-ubyte
      <data-path>/t10k-labels.idx1-ubyte

    OUTPUT:
      logs/training_loss_cnn.txt - Training loss per epoch
    """)
}
