// ============================================================================
// ConfigTests.swift - Tests for Config Struct (CLI Argument Parsing)
// ============================================================================
//
// Tests for the Config struct defined in Config.swift:
// - Default values for all configuration fields
// - Struct mutability and value assignment
// - Optional fields default to nil
// - Boolean flags default to false
//
// Tests use Config.parse(arguments:) for direct parser coverage without
// mutating process-global CommandLine.arguments.
//
// ============================================================================

import XCTest
@testable import MNISTMLX

final class ConfigTests: XCTestCase {

    private func assertEvaluateCombinationRejected(
        arguments: [String],
        expectedMessage: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try Config.parseOrThrow(arguments: arguments), file: file, line: line) { error in
            guard let configError = error as? ConfigError else {
                XCTFail("Should throw ConfigError", file: file, line: line)
                return
            }

            XCTAssertEqual(configError.description, expectedMessage, file: file, line: line)
        }
    }

    // =============================================================================
    // MARK: - Default Value Tests
    // =============================================================================

    func testDefaultModelType() {
        let config = Config()
        XCTAssertEqual(config.modelType, "mlp",
                       "Default model type should be 'mlp'")
    }

    func testDefaultModelTypeProvided() {
        let config = Config()
        XCTAssertFalse(config.modelTypeProvided,
                       "modelTypeProvided should default to false")
    }

    func testDefaultEpochs() {
        let config = Config()
        XCTAssertEqual(config.epochs, 5,
                       "Default epochs should be 5")
    }

    func testDefaultEpochsProvided() {
        let config = Config()
        XCTAssertFalse(config.epochsProvided,
                       "epochsProvided should default to false")
    }

    func testDefaultBatchSize() {
        let config = Config()
        XCTAssertEqual(config.batchSize, 32,
                       "Default batch size should be 32")
    }

    func testDefaultLearningRate() {
        let config = Config()
        XCTAssertEqual(config.learningRate, 0.01, accuracy: 1e-6,
                       "Default learning rate should be 0.01")
    }

    func testDefaultLearningRateProvided() {
        let config = Config()
        XCTAssertFalse(config.learningRateProvided,
                       "learningRateProvided should default to false")
    }

    func testDefaultDataPath() {
        let config = Config()
        XCTAssertEqual(config.dataPath, "./data",
                       "Default data path should be './data'")
    }

    func testDefaultSeed() {
        let config = Config()
        XCTAssertEqual(config.seed, 1,
                       "Default seed should be 1")
    }

    func testDefaultUseCompile() {
        let config = Config()
        XCTAssertFalse(config.useCompile,
                       "useCompile should default to false")
    }

    func testDefaultExportJson() {
        let config = Config()
        XCTAssertFalse(config.exportJson,
                       "exportJson should default to false")
    }

    func testDefaultCheckpointInterval() {
        let config = Config()
        XCTAssertNil(config.checkpointInterval,
                     "checkpointInterval should default to nil (disabled)")
    }

    func testDefaultResumeFrom() {
        let config = Config()
        XCTAssertNil(config.resumeFrom,
                     "resumeFrom should default to nil")
    }

    func testDefaultEvaluatePath() {
        let config = Config()
        XCTAssertNil(config.evaluatePath,
                     "evaluatePath should default to nil")
        XCTAssertFalse(config.isEvaluationMode,
                       "Default config should not be in evaluation mode")
    }

    func testDefaultEarlyStoppingPatience() {
        let config = Config()
        XCTAssertNil(config.earlyStoppingPatience,
                     "earlyStoppingPatience should default to nil (disabled)")
    }

    func testDefaultEarlyStoppingMinDelta() {
        let config = Config()
        XCTAssertNil(config.earlyStoppingMinDelta,
                     "earlyStoppingMinDelta should default to nil")
    }

    // =============================================================================
    // MARK: - Struct Mutation Tests
    // =============================================================================

    func testMutableModelType() {
        var config = Config()
        config.modelType = "cnn"
        config.modelTypeProvided = true
        XCTAssertEqual(config.modelType, "cnn",
                       "modelType should be mutable")
        XCTAssertTrue(config.modelTypeProvided,
                      "modelTypeProvided should be mutable")
    }

    func testMutableEpochs() {
        var config = Config()
        config.epochs = 10
        config.epochsProvided = true
        XCTAssertEqual(config.epochs, 10,
                       "epochs should be mutable")
        XCTAssertTrue(config.epochsProvided,
                      "epochsProvided should be mutable")
    }

    func testMutableBatchSize() {
        var config = Config()
        config.batchSize = 64
        XCTAssertEqual(config.batchSize, 64,
                       "batchSize should be mutable")
    }

    func testMutableLearningRate() {
        var config = Config()
        config.learningRate = 0.001
        XCTAssertEqual(config.learningRate, 0.001, accuracy: 1e-7,
                       "learningRate should be mutable")
    }

    func testMutableLearningRateProvided() {
        var config = Config()
        config.learningRateProvided = true
        XCTAssertTrue(config.learningRateProvided,
                      "learningRateProvided should be mutable")
    }

    func testMutableDataPath() {
        var config = Config()
        config.dataPath = "/custom/path/to/data"
        XCTAssertEqual(config.dataPath, "/custom/path/to/data",
                       "dataPath should be mutable")
    }

    func testMutableSeed() {
        var config = Config()
        config.seed = 42
        XCTAssertEqual(config.seed, 42,
                       "seed should be mutable")
    }

    func testMutableUseCompile() {
        var config = Config()
        config.useCompile = true
        XCTAssertTrue(config.useCompile,
                      "useCompile should be mutable")
    }

    func testMutableExportJson() {
        var config = Config()
        config.exportJson = true
        XCTAssertTrue(config.exportJson,
                      "exportJson should be mutable")
    }

    func testMutableCheckpointInterval() {
        var config = Config()
        config.checkpointInterval = 5
        XCTAssertEqual(config.checkpointInterval, 5,
                       "checkpointInterval should be mutable")
    }

    func testMutableResumeFrom() {
        var config = Config()
        config.resumeFrom = "./checkpoints/epoch_5.json"
        XCTAssertEqual(config.resumeFrom, "./checkpoints/epoch_5.json",
                       "resumeFrom should be mutable")
    }

    func testMutableEvaluatePath() {
        var config = Config()
        config.evaluatePath = "./best_model_mlp.json"
        XCTAssertEqual(config.evaluatePath, "./best_model_mlp.json",
                       "evaluatePath should be mutable")
        XCTAssertTrue(config.isEvaluationMode,
                      "evaluatePath should enable evaluation mode")
    }

    func testMutableEarlyStoppingPatience() {
        var config = Config()
        config.earlyStoppingPatience = 3
        XCTAssertEqual(config.earlyStoppingPatience, 3,
                       "earlyStoppingPatience should be mutable")
    }

    func testMutableEarlyStoppingMinDelta() {
        var config = Config()
        config.earlyStoppingMinDelta = 0.001
        XCTAssertEqual(config.earlyStoppingMinDelta ?? -1, 0.001, accuracy: 1e-7,
                       "earlyStoppingMinDelta should be mutable")
    }

    // =============================================================================
    // MARK: - Parser Tests
    // =============================================================================

    func testParseEarlyStoppingPatience() {
        let config = Config.parse(arguments: [
            "MNISTMLX",
            "--early-stopping-patience", "3"
        ])

        XCTAssertEqual(config.earlyStoppingPatience, 3)
        XCTAssertNil(config.earlyStoppingMinDelta)
    }

    func testParseEarlyStoppingMinDelta() {
        let config = Config.parse(arguments: [
            "MNISTMLX",
            "--early-stopping-min-delta", "0.001"
        ])

        XCTAssertNil(config.earlyStoppingPatience)
        XCTAssertEqual(config.earlyStoppingMinDelta ?? -1, 0.001, accuracy: 1e-7)
    }

    func testParseEarlyStoppingOptionsTogether() {
        let config = Config.parse(arguments: [
            "MNISTMLX",
            "--epochs", "20",
            "--early-stopping-patience", "4",
            "--early-stopping-min-delta", "0.0005"
        ])

        XCTAssertEqual(config.epochs, 20)
        XCTAssertEqual(config.earlyStoppingPatience, 4)
        XCTAssertEqual(config.earlyStoppingMinDelta ?? -1, 0.0005, accuracy: 1e-7)
    }

    func testParseEvaluatePath() {
        let config = Config.parse(arguments: [
            "MNISTMLX",
            "--evaluate", "./best_model_mlp.json"
        ])

        XCTAssertEqual(config.evaluatePath, "./best_model_mlp.json")
        XCTAssertTrue(config.isEvaluationMode)
        XCTAssertFalse(config.modelTypeProvided)
    }

    func testParseEvaluateShortFlagWithModelOverride() {
        let config = Config.parse(arguments: [
            "MNISTMLX",
            "-E", "./best_model_cnn.json",
            "--model", "cnn"
        ])

        XCTAssertEqual(config.evaluatePath, "./best_model_cnn.json")
        XCTAssertEqual(config.modelType, "cnn")
        XCTAssertTrue(config.modelTypeProvided)
        XCTAssertTrue(config.isEvaluationMode)
    }

    func testParseEvaluateRejectsEpochs() {
        assertEvaluateCombinationRejected(
            arguments: [
                "MNISTMLX",
                "--evaluate", "./best_model_mlp.json",
                "--epochs", "5"
            ],
            expectedMessage: "--evaluate cannot be combined with --epochs/-e because evaluation does not train."
        )
    }

    func testParseEvaluateRejectsResume() {
        assertEvaluateCombinationRejected(
            arguments: [
                "MNISTMLX",
                "--evaluate", "./best_model_mlp.json",
                "--resume", "./checkpoint.json"
            ],
            expectedMessage: "--evaluate cannot be combined with --resume. Pass the checkpoint path to --evaluate instead."
        )
    }

    func testParseEvaluateRejectsCompile() {
        assertEvaluateCombinationRejected(
            arguments: [
                "MNISTMLX",
                "--evaluate", "./best_model_mlp.json",
                "--compile"
            ],
            expectedMessage: "--evaluate cannot be combined with --compile"
        )
    }

    func testParseEvaluateRejectsCheckpointInterval() {
        assertEvaluateCombinationRejected(
            arguments: [
                "MNISTMLX",
                "--evaluate", "./best_model_mlp.json",
                "--checkpoint-interval", "1"
            ],
            expectedMessage: "--evaluate cannot be combined with --checkpoint-interval"
        )
    }

    func testParseEvaluateRejectsEarlyStoppingPatience() {
        assertEvaluateCombinationRejected(
            arguments: [
                "MNISTMLX",
                "--evaluate", "./best_model_mlp.json",
                "--early-stopping-patience", "3"
            ],
            expectedMessage: "--evaluate cannot be combined with --early-stopping-patience"
        )
    }

    func testParseEvaluateRejectsEarlyStoppingMinDelta() {
        assertEvaluateCombinationRejected(
            arguments: [
                "MNISTMLX",
                "--evaluate", "./best_model_mlp.json",
                "--early-stopping-min-delta", "0.001"
            ],
            expectedMessage: "--evaluate cannot be combined with --early-stopping-min-delta"
        )
    }

    func testParseEvaluateRejectsExportJson() {
        assertEvaluateCombinationRejected(
            arguments: [
                "MNISTMLX",
                "--evaluate", "./best_model_mlp.json",
                "--export-json"
            ],
            expectedMessage: "--evaluate cannot be combined with --export-json"
        )
    }

    // =============================================================================
    // MARK: - Valid Model Type Names
    // =============================================================================

    func testAllValidModelTypes() {
        // Verify all documented model types can be stored
        let validModelTypes = ["mlp", "cnn", "resnet", "attention", "transformer"]
        for modelType in validModelTypes {
            var config = Config()
            config.modelType = modelType
            XCTAssertEqual(config.modelType, modelType,
                           "Config should accept model type '\(modelType)'")
        }
    }

    func testModelTypeIsLowercaseString() {
        // The default model type is lowercase
        let config = Config()
        XCTAssertEqual(config.modelType, config.modelType.lowercased(),
                       "Default model type should be lowercase")
    }

    // =============================================================================
    // MARK: - Edge Case Values
    // =============================================================================

    func testZeroEpochsAllowed() {
        // Config struct doesn't enforce positive epochs at struct level
        var config = Config()
        config.epochs = 0
        XCTAssertEqual(config.epochs, 0,
                       "Config should allow setting epochs to 0 at struct level")
    }

    func testLargeSeedValue() {
        // UInt64 can hold large seed values
        var config = Config()
        config.seed = UInt64.max
        XCTAssertEqual(config.seed, UInt64.max,
                       "Config should store maximum UInt64 seed value")
    }

    func testSmallLearningRate() {
        var config = Config()
        config.learningRate = 1e-6
        XCTAssertEqual(config.learningRate, 1e-6, accuracy: 1e-10,
                       "Config should store very small learning rate")
    }

    func testLargeLearningRate() {
        var config = Config()
        config.learningRate = 1.0
        XCTAssertEqual(config.learningRate, 1.0, accuracy: 1e-6,
                       "Config should store learning rate of 1.0")
    }

    func testCheckpointIntervalZero() {
        // Zero interval stored (even if semantically unusual)
        var config = Config()
        config.checkpointInterval = 0
        XCTAssertNotNil(config.checkpointInterval,
                        "Setting checkpointInterval to 0 should make it non-nil")
        XCTAssertEqual(config.checkpointInterval!, 0)
    }

    func testCheckpointIntervalCanBeCleared() {
        // Can set and then clear checkpointInterval
        var config = Config()
        config.checkpointInterval = 3
        config.checkpointInterval = nil
        XCTAssertNil(config.checkpointInterval,
                     "checkpointInterval should be clearable back to nil")
    }

    func testResumeFromCanBeCleared() {
        // Can set and then clear resumeFrom
        var config = Config()
        config.resumeFrom = "./checkpoint.json"
        config.resumeFrom = nil
        XCTAssertNil(config.resumeFrom,
                     "resumeFrom should be clearable back to nil")
    }

    func testEarlyStoppingOptionsCanBeCleared() {
        var config = Config()
        config.earlyStoppingPatience = 3
        config.earlyStoppingMinDelta = 0.001
        config.earlyStoppingPatience = nil
        config.earlyStoppingMinDelta = nil
        XCTAssertNil(config.earlyStoppingPatience,
                     "earlyStoppingPatience should be clearable back to nil")
        XCTAssertNil(config.earlyStoppingMinDelta,
                     "earlyStoppingMinDelta should be clearable back to nil")
    }

    // =============================================================================
    // MARK: - Config Isolation Tests
    // =============================================================================

    func testConfigInstancesAreIndependent() {
        // Verify Config is a value type - mutations don't affect other instances
        var config1 = Config()
        var config2 = Config()

        config1.epochs = 10
        config2.epochs = 20

        XCTAssertEqual(config1.epochs, 10,
                       "config1 epochs should be 10")
        XCTAssertEqual(config2.epochs, 20,
                       "config2 epochs should be 20")
        XCTAssertNotEqual(config1.epochs, config2.epochs,
                          "Config instances should be independent")
    }

    func testConfigCopyPreservesValues() {
        // Value semantics: copy should have same values
        var config1 = Config()
        config1.modelType = "transformer"
        config1.epochs = 3
        config1.batchSize = 16
        config1.learningRate = 0.005
        config1.useCompile = true
        config1.exportJson = true
        config1.seed = 42
        config1.checkpointInterval = 1
        config1.resumeFrom = "./checkpoint.json"
        config1.evaluatePath = "./best_model_transformer.json"
        config1.earlyStoppingPatience = 4
        config1.earlyStoppingMinDelta = 0.002

        let config2 = config1

        XCTAssertEqual(config2.modelType, "transformer")
        XCTAssertEqual(config2.epochs, 3)
        XCTAssertEqual(config2.batchSize, 16)
        XCTAssertEqual(config2.learningRate, 0.005, accuracy: 1e-7)
        XCTAssertTrue(config2.useCompile)
        XCTAssertTrue(config2.exportJson)
        XCTAssertEqual(config2.seed, 42)
        XCTAssertEqual(config2.checkpointInterval, 1)
        XCTAssertEqual(config2.resumeFrom, "./checkpoint.json")
        XCTAssertEqual(config2.evaluatePath, "./best_model_transformer.json")
        XCTAssertTrue(config2.isEvaluationMode)
        XCTAssertEqual(config2.earlyStoppingPatience, 4)
        XCTAssertEqual(config2.earlyStoppingMinDelta ?? -1, 0.002, accuracy: 1e-7)
    }
}
