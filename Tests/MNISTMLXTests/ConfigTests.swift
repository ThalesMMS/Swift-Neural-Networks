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
// Note: Config.parse() reads CommandLine.arguments at runtime, which cannot
// be modified in unit tests without process-level manipulation. These tests
// focus on the struct's default values and direct construction.
//
// ============================================================================

import XCTest
@testable import MNISTMLX

final class ConfigTests: XCTestCase {

    // =============================================================================
    // MARK: - Default Value Tests
    // =============================================================================

    func testDefaultModelType() {
        let config = Config()
        XCTAssertEqual(config.modelType, "mlp",
                       "Default model type should be 'mlp'")
    }

    func testDefaultEpochs() {
        let config = Config()
        XCTAssertEqual(config.epochs, 5,
                       "Default epochs should be 5")
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

    // =============================================================================
    // MARK: - Struct Mutation Tests
    // =============================================================================

    func testMutableModelType() {
        var config = Config()
        config.modelType = "cnn"
        XCTAssertEqual(config.modelType, "cnn",
                       "modelType should be mutable")
    }

    func testMutableEpochs() {
        var config = Config()
        config.epochs = 10
        XCTAssertEqual(config.epochs, 10,
                       "epochs should be mutable")
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
    }
}