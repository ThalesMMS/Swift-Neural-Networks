import Foundation

let imgH = 28
let imgW = 28
let numInputs = imgH * imgW
let numClasses = 10

// Patch grid and tokenization.
let patch = 4
let grid = imgH / patch          // 7
let seqLen = grid * grid         // 49
let patchDim = patch * patch     // 16
let dModel = 32                  // model dimension (increased for capacity test)
let ffDim = 64                   // feed-forward hidden size (2x dModel)

// Dataset sizes.
let trainSamples = 60_000
let testSamples = 10_000

// =============================================================================
// MARK: - Configuration
// =============================================================================

/// Configuration for training hyperparameters
struct Config {
    var learningRate: Float = 0.005
    var epochs: Int = 5
    var batchSize: Int = 32
    var dataPath: String = "./data"
    var seed: UInt64 = 1

    /// Parses command-line arguments and returns a populated `Config`.
    ///
    /// Supported options: `--batch`/`-b` <int>, `--epochs`/`-e` <int>, `--lr`/`-l` <float>,
    /// `--data`/`-d` <path>, `--seed`/`-s` <uint64>, `--help`/`-h`.
    /// Exits with code 1 on invalid arguments; exits with code 0 on `--help`.
    /// - Returns: A `Config` populated from the command-line arguments; unset fields keep their defaults.
    static func parse() -> Config {
        var config = Config()
        let args = CommandLine.arguments
        var i = 1

        while i < args.count {
            let arg = args[i]

            switch arg {
            case "--batch", "-b":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                let token = args[valueIndex]
                guard let val = Int(token), val > 0 else {
                    print("Invalid value for \(arg): \(token)")
                    printUsage()
                    exit(1)
                }
                config.batchSize = val
                i = valueIndex

            case "--epochs", "-e":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                let token = args[valueIndex]
                guard let val = Int(token), val > 0 else {
                    print("Invalid value for \(arg): \(token)")
                    printUsage()
                    exit(1)
                }
                config.epochs = val
                i = valueIndex

            case "--lr", "-l":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                let token = args[valueIndex]
                guard let val = Float(token), val > 0 else {
                    print("Invalid value for \(arg): \(token)")
                    printUsage()
                    exit(1)
                }
                config.learningRate = val
                i = valueIndex

            case "--data", "-d":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                config.dataPath = args[valueIndex]
                i = valueIndex

            case "--seed", "-s":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                let token = args[valueIndex]
                guard let val = UInt64(token) else {
                    print("Invalid value for \(arg): \(token)")
                    printUsage()
                    exit(1)
                }
                config.seed = val
                i = valueIndex

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

/// Prints the command-line usage and help message for the MNISTAttentionPool example.
/// 
/// The help text includes supported CLI options with defaults, example invocations, and a brief model architecture summary.
func printUsage() {
    print("""
    MNIST Attention Pool - Self-Attention Model for MNIST
    ======================================================

    USAGE:
      swift run MNISTManualAttention [OPTIONS]
      swift mnist_attention_pool.swift [OPTIONS]  # legacy wrapper

    OPTIONS:
      --batch, -b <n>    Batch size (default: 32)
      --epochs, -e <n>   Number of training epochs (default: 5)
      --lr, -l <f>       Learning rate (default: 0.005)
      --data, -d <path>  Path to MNIST data directory (default: ./data)
      --seed, -s <n>     RNG seed for reproducibility (default: 1)
      --help, -h         Show this help message

    EXAMPLES:
      swift run MNISTManualAttention --epochs 10
      swift run MNISTManualAttention -b 64 -e 5 -l 0.005
      swift mnist_attention_pool.swift --data ./data --seed 42

    MODEL ARCHITECTURE:
      - 4×4 patches → 49 tokens
      - Self-attention with Q/K/V projections
      - Feed-forward MLP per token
      - Mean-pool → logits → softmax
    """)
}
