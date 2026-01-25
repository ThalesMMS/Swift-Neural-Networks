import Foundation

// MARK: - Global Configuration Parameters

/// Number of training samples in MNIST dataset
let trainSamples = 60_000

/// Number of test samples in MNIST dataset
let testSamples = 10_000

/// Random number generator seed for reproducibility
var rngSeed: UInt64 = 1

// MARK: - CLI Argument Parsing

/// Parses command-line arguments and overrides global configuration parameters.
///
/// Supported arguments:
/// - `--batch N`: Set batch size (default: 64)
/// - `--hidden N`: Set number of hidden units (default: 512)
/// - `--epochs N`: Set number of training epochs (default: 10)
/// - `--lr F`: Set learning rate (default: 0.01)
/// - `--seed N`: Set RNG seed for reproducibility (default: 1)
/// - `--help`: Display usage information and exit
///
/// Invalid arguments are silently ignored. Invalid values for recognized arguments
/// will print an error message and exit with status code 1.
func applyCliOverrides() {
    let args = CommandLine.arguments
    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "--batch":
            guard i + 1 < args.count, let value = Int(args[i + 1]), value > 0 else {
                print("Invalid value for --batch")
                exit(1)
            }
            batchSize = value
            i += 1
        case "--hidden":
            guard i + 1 < args.count, let value = Int(args[i + 1]), value > 0 else {
                print("Invalid value for --hidden")
                exit(1)
            }
            numHidden = value
            i += 1
        case "--epochs":
            guard i + 1 < args.count, let value = Int(args[i + 1]), value > 0 else {
                print("Invalid value for --epochs")
                exit(1)
            }
            epochs = value
            i += 1
        case "--lr":
            guard i + 1 < args.count, let value = Float(args[i + 1]), value > 0 else {
                print("Invalid value for --lr")
                exit(1)
            }
            learningRate = value
            i += 1
        case "--seed":
            guard i + 1 < args.count, let value = UInt64(args[i + 1]) else {
                print("Invalid value for --seed")
                exit(1)
            }
            rngSeed = value
            i += 1
        case "--help":
            print("""
Usage: mnist_mlp_swift [--mps] [--mpsgraph] [--batch N] [--hidden N] [--epochs N] [--lr F] [--seed N]
""")
            exit(0)
        default:
            break
        }
        i += 1
    }
}
