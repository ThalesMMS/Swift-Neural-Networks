// ============================================================================
// Config.swift - Command-Line Configuration for MNIST Neural Network Training
// ============================================================================

import Foundation
import MNISTCommon

// =============================================================================
// MARK: - Command-Line Argument Parsing
// =============================================================================

/// Configuration parsed from command-line arguments
struct ConfigError: Error, CustomStringConvertible {
    let message: String

    var description: String {
        message
    }
}

let supportedModelTypes = ["mlp", "cnn", "resnet", "attention", "transformer"]

struct Config {
    var modelType: String = "mlp"
    var modelTypeProvided: Bool = false
    var epochs: Int = 5
    var epochsProvided: Bool = false
    var batchSize: Int = 32
    var learningRate: Float = 0.01
    var learningRateProvided: Bool = false
    var dataPath: String = "./data"
    var seed: UInt64 = 1
    var useCompile: Bool = false
    var exportJson: Bool = false
    var checkpointInterval: Int? = nil
    var resumeFrom: String? = nil
    var evaluatePath: String? = nil
    var earlyStoppingPatience: Int? = nil
    var earlyStoppingMinDelta: Float? = nil

    var isEvaluationMode: Bool {
        evaluatePath != nil
    }

    /// Parses command-line arguments into configuration
    ///
    /// This is a simple hand-rolled parser. For production code,
    /// consider using Swift Argument Parser package.
    static func parse() -> Config {
        do {
            return try parseOrThrow(arguments: CommandLine.arguments)
        } catch let error as ConfigError {
            FileHandle.standardError.write(Data("\(error.description)\n".utf8))
            exit(1)
        } catch {
            FileHandle.standardError.write(Data("\(error)\n".utf8))
            exit(1)
        }
    }

    static func parse(arguments args: [String]) -> Config {
        do {
            return try parseOrThrow(arguments: args)
        } catch let error as ConfigError {
            FileHandle.standardError.write(Data("\(error.description)\n".utf8))
            exit(1)
        } catch {
            FileHandle.standardError.write(Data("\(error)\n".utf8))
            exit(1)
        }
    }

    static func parseOrThrow(arguments args: [String]) throws -> Config {
        var config = Config()
        var i = 1

        func fail(_ message: String) throws -> Never {
            throw ConfigError(message: message)
        }

        func requireValue(for option: String) throws -> String {
            i += 1
            guard i < args.count else {
                try fail("Missing value for \(option)")
            }
            return args[i]
        }

        func requireNonFlagValue(for option: String) throws -> String {
            let value = try requireValue(for: option)
            guard !value.hasPrefix("-") else {
                try fail("Missing value for \(option)")
            }
            return value
        }
        
        while i < args.count {
            let arg = args[i]
            
            switch arg {
            case "--model", "-m":
                let value = try requireNonFlagValue(for: "--model/-m").lowercased()
                guard supportedModelTypes.contains(value) else {
                    try fail("Invalid value for --model/-m: \(value). Expected one of: \(supportedModelTypes.joined(separator: ", ")).")
                }
                config.modelType = value
                config.modelTypeProvided = true
                
            case "--epochs", "-e":
                let value = try requireValue(for: "--epochs/-e")
                guard let val = Int(value), val > 0 else {
                    try fail("Invalid value for --epochs/-e: \(value). Expected a positive integer.")
                }
                config.epochs = val
                config.epochsProvided = true
                
            case "--batch", "-b":
                let value = try requireValue(for: "--batch/-b")
                guard let val = Int(value), val > 0 else {
                    try fail("Invalid value for --batch/-b: \(value). Expected a positive integer.")
                }
                config.batchSize = val
                
            case "--lr", "-l":
                let value = try requireValue(for: "--lr/-l")
                guard let val = Float(value), val.isFinite, val > 0 else {
                    try fail("Invalid value for --lr/-l: \(value). Expected a positive finite number.")
                }
                config.learningRate = val
                config.learningRateProvided = true
                
            case "--data", "-d":
                config.dataPath = try requireNonFlagValue(for: "--data/-d")

            case "--seed", "-s":
                let value = try requireValue(for: "--seed/-s")
                guard let val = UInt64(value) else {
                    try fail("Invalid value for --seed/-s: \(value). Expected a non-negative integer.")
                }
                config.seed = val

            case "--compile", "-c":
                config.useCompile = true

            case "--export-json":
                config.exportJson = true

            case "--checkpoint-interval":
                let value = try requireValue(for: "--checkpoint-interval")
                guard let val = Int(value), val > 0 else {
                    try fail("Invalid value for --checkpoint-interval: \(value). Expected a positive integer.")
                }
                config.checkpointInterval = val

            case "--early-stopping-patience":
                let value = try requireValue(for: "--early-stopping-patience")
                guard let val = Int(value), val > 0 else {
                    try fail("Invalid value for --early-stopping-patience: \(value). Expected a positive integer.")
                }
                config.earlyStoppingPatience = val

            case "--early-stopping-min-delta":
                let value = try requireValue(for: "--early-stopping-min-delta")
                guard let val = Float(value), val.isFinite, val >= 0 else {
                    try fail("Invalid value for --early-stopping-min-delta: \(value). Expected a non-negative finite number.")
                }
                config.earlyStoppingMinDelta = val

            case "--resume":
                config.resumeFrom = try requireNonFlagValue(for: "--resume")

            case "--evaluate", "-E":
                config.evaluatePath = try requireNonFlagValue(for: "--evaluate/-E")

            case "--help", "-h":
                printUsage()
                exit(0)
                
            default:
                try fail("Unknown argument: \(arg)")
            }
            
            i += 1
        }

        if config.isEvaluationMode {
            if config.epochsProvided {
                try fail("--evaluate cannot be combined with --epochs/-e because evaluation does not train.")
            }
            if config.resumeFrom != nil {
                try fail("--evaluate cannot be combined with --resume. Pass the checkpoint path to --evaluate instead.")
            }
            if config.useCompile {
                try fail("--evaluate cannot be combined with --compile")
            }
            if config.checkpointInterval != nil {
                try fail("--evaluate cannot be combined with --checkpoint-interval")
            }
            if config.earlyStoppingPatience != nil {
                try fail("--evaluate cannot be combined with --early-stopping-patience")
            }
            if config.earlyStoppingMinDelta != nil {
                try fail("--evaluate cannot be combined with --early-stopping-min-delta")
            }
            if config.exportJson {
                try fail("--evaluate cannot be combined with --export-json")
            }
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
      --model, -m <name>    Model to train: \(supportedModelTypes.joined(separator: ", ")) (default: mlp)
      --epochs, -e <n>      Number of training epochs (default: 5)
      --batch, -b <n>       Batch size (default: 32)
      --lr, -l <f>          Learning rate (default: 0.01)
      --data, -d <path>     Path to MNIST data directory (default: ./data)
      --seed, -s <n>        Random seed for reproducibility (default: 1)
      --compile, -c         Enable compiled training for faster execution
      --export-json         Export training results to JSON file
      --checkpoint-interval <n>  Save checkpoint every N epochs (default: disabled)
      --early-stopping-patience <n>  Stop after N epochs without meaningful validation accuracy improvement (default: disabled)
      --early-stopping-min-delta <f> Minimum validation accuracy improvement required to reset patience (default: 0.0)
      --resume <path>       Resume training from checkpoint file
      --evaluate, -E <path> Evaluate a saved MNISTMLX checkpoint on the MNIST test set
      --help, -h            Show this help message

    ENVIRONMENT:
      ANSI_COLORS=1         Enable colored terminal output
                            (errors=red, warnings=yellow, success=green, progress=cyan)

    EXAMPLES:
      swift run MNISTMLX --model cnn --epochs 3
      swift run MNISTMLX --evaluate best_model_cnn.json
      swift run MNISTMLX --evaluate best_model_cnn.json --model cnn
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
