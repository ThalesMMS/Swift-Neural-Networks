// ============================================================================
// Config.swift - Command-Line Configuration for MNIST Neural Network Training
// ============================================================================

import Foundation
import MNISTCommon

// =============================================================================
// MARK: - Command-Line Argument Parsing
// =============================================================================

/// Configuration parsed from command-line arguments
struct Config {
    var modelType: String = "mlp"
    var epochs: Int = 5
    var batchSize: Int = 32
    var learningRate: Float = 0.01
    var learningRateProvided: Bool = false
    var dataPath: String = "./data"
    var seed: UInt64 = 1
    var useCompile: Bool = false
    var exportJson: Bool = false
    var checkpointInterval: Int? = nil
    var resumeFrom: String? = nil

    /// Parses command-line arguments into configuration
    ///
    /// This is a simple hand-rolled parser. For production code,
    /// consider using Swift Argument Parser package.
    static func parse() -> Config {
        var config = Config()
        let args = CommandLine.arguments
        var i = 1

        func fail(_ message: String) -> Never {
            FileHandle.standardError.write(Data("\(message)\n".utf8))
            exit(1)
        }

        func requireValue(for option: String) -> String {
            i += 1
            guard i < args.count else {
                fail("Missing value for \(option)")
            }
            return args[i]
        }

        func requireNonFlagValue(for option: String) -> String {
            let value = requireValue(for: option)
            guard !value.hasPrefix("-") else {
                fail("Missing value for \(option)")
            }
            return value
        }
        
        while i < args.count {
            let arg = args[i]
            
            switch arg {
            case "--model", "-m":
                let value = requireNonFlagValue(for: "--model/-m").lowercased()
                let validModels = ["mlp", "cnn", "resnet", "attention", "transformer"]
                guard validModels.contains(value) else {
                    fail("Invalid value for --model/-m: \(value). Expected one of: \(validModels.joined(separator: ", ")).")
                }
                config.modelType = value
                
            case "--epochs", "-e":
                let value = requireValue(for: "--epochs/-e")
                guard let val = Int(value), val > 0 else {
                    fail("Invalid value for --epochs/-e: \(value). Expected a positive integer.")
                }
                config.epochs = val
                
            case "--batch", "-b":
                let value = requireValue(for: "--batch/-b")
                guard let val = Int(value), val > 0 else {
                    fail("Invalid value for --batch/-b: \(value). Expected a positive integer.")
                }
                config.batchSize = val
                
            case "--lr", "-l":
                let value = requireValue(for: "--lr/-l")
                guard let val = Float(value), val.isFinite, val > 0 else {
                    fail("Invalid value for --lr/-l: \(value). Expected a positive finite number.")
                }
                config.learningRate = val
                config.learningRateProvided = true
                
            case "--data", "-d":
                config.dataPath = requireNonFlagValue(for: "--data/-d")

            case "--seed", "-s":
                let value = requireValue(for: "--seed/-s")
                guard let val = UInt64(value) else {
                    fail("Invalid value for --seed/-s: \(value). Expected a non-negative integer.")
                }
                config.seed = val

            case "--compile", "-c":
                config.useCompile = true

            case "--export-json":
                config.exportJson = true

            case "--checkpoint-interval":
                let value = requireValue(for: "--checkpoint-interval")
                guard let val = Int(value), val > 0 else {
                    fail("Invalid value for --checkpoint-interval: \(value). Expected a positive integer.")
                }
                config.checkpointInterval = val

            case "--resume":
                config.resumeFrom = requireNonFlagValue(for: "--resume")

            case "--help", "-h":
                printUsage()
                exit(0)
                
            default:
                ColoredPrint.error("Unknown argument: \(arg)")
                printUsage()
                exit(1)
            }
            
            i += 1
        }
        
        return config
    }
}

/// Prints usage information
func printUsage() {
    print("""
    MNIST Neural Networks with MLX Swift
    =====================================

    USAGE:
      swift run MNISTMLX [OPTIONS]

    OPTIONS:
      --model, -m <name>    Model to train: mlp, cnn, resnet, attention, or transformer (default: mlp)
      --epochs, -e <n>      Number of training epochs (default: 5)
      --batch, -b <n>       Batch size (default: 32)
      --lr, -l <f>          Learning rate (default: 0.01)
      --data, -d <path>     Path to MNIST data directory (default: ./data)
      --seed, -s <n>        Random seed for reproducibility (default: 1)
      --compile, -c         Enable compiled training for faster execution
      --export-json         Export training results to JSON file
      --checkpoint-interval <n>  Save checkpoint every N epochs (default: disabled)
      --resume <path>       Resume training from checkpoint file
      --help, -h            Show this help message

    ENVIRONMENT:
      ANSI_COLORS=1         Enable colored terminal output
                            (errors=red, warnings=yellow, success=green, progress=cyan)

    EXAMPLES:
      swift run MNISTMLX --model cnn --epochs 3
      swift run MNISTMLX -m mlp -e 10 -b 64 -l 0.005
      ANSI_COLORS=1 swift run MNISTMLX --model attention --epochs 5

    MODELS:
      mlp        Multi-Layer Perceptron (784→512→10)
                 - Fastest training
                 - Good baseline (~97% accuracy)

      cnn        Convolutional Neural Network
                 - Conv(3×3, 8 filters) → MaxPool → Linear
                 - Best accuracy (~98%)

      resnet     Residual Network with skip connections
                 - ResidualBlock × 3 with skip connections
                 - Demonstrates how ResNet enables deeper networks (~98% accuracy)

      attention  Transformer-style self-attention
                 - 4×4 patches → 49 tokens → attention → pooling
                 - Educational (demonstrates attention mechanism)

      transformer Full Transformer block with multi-head attention
                 - Patches → Multi-head Self-Attention → LayerNorm → FFN → LayerNorm
                 - Demonstrates complete transformer architecture
    """)
}
