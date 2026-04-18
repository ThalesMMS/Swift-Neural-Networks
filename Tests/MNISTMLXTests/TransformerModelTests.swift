// ============================================================================
// TransformerModelTests.swift - Tests for TransformerModel and Training Utils
// ============================================================================
//
// This test suite validates the TransformerModel class and associated functions
// now defined in TransformerModel.swift (extracted to its own file in this PR).
//
// Tests cover (distinct from existing TransformerBlockTests.swift):
// - Model stored properties (numLayers, dModel, numHeads, ffDim)
// - Block array length matches numLayers
// - Positional embeddings have expected shape [49, dModel]
// - Default init uses TRANSFORMER_* constants from TransformerConfig.swift
// - trainTransformerEpoch: returns finite Float loss
// - trainTransformerEpoch: model parameters update after training
// - trainTransformerEpoch: handles minimal dataset sizes
// - verifyLayerNorm: runs without crashing
// - transformerLoss and transformerAccuracy property contracts
//
// ============================================================================

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import MNISTMLX

final class TransformerModelTests: MLXTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    private func makeFlatImages(batch: Int) -> MLXArray {
        // Returns deterministic [batch, 784] input in [0, 1].
        let values = (0..<(batch * 784)).map { Float($0 % 251) / 250.0 }
        return MLXArray(values, [batch, 784])
    }

    private func makeLabels(batch: Int) -> MLXArray {
        MLXArray((0..<batch).map { Int32($0 % 10) })
    }

    private func assertAllFinite(
        _ array: MLXArray,
        _ message: String = "Should be finite",
        file: StaticString = #file, line: UInt = #line
    ) {
        eval(array)
        let values = array.asArray(Float.self)
        for v in values {
            XCTAssertTrue(v.isFinite, "\(message): found \(v)", file: file, line: line)
        }
    }

    // =============================================================================
    // MARK: - Stored Property Tests
    // =============================================================================

    func testDefaultNumLayers() {
        let model = TransformerModel()
        XCTAssertEqual(model.numLayers, 2,
                       "Default numLayers should be 2")
    }

    func testDefaultDModel() {
        let model = TransformerModel()
        XCTAssertEqual(model.dModel, TRANSFORMER_D_MODEL,
                       "Default dModel should match TRANSFORMER_D_MODEL constant")
    }

    func testDefaultNumHeads() {
        let model = TransformerModel()
        XCTAssertEqual(model.numHeads, TRANSFORMER_NUM_HEADS,
                       "Default numHeads should match TRANSFORMER_NUM_HEADS constant")
    }

    func testDefaultFFDim() {
        let model = TransformerModel()
        XCTAssertEqual(model.ffDim, TRANSFORMER_FF_DIM,
                       "Default ffDim should match TRANSFORMER_FF_DIM constant")
    }

    func testCustomNumLayers() {
        let model = TransformerModel(numLayers: 4)
        XCTAssertEqual(model.numLayers, 4)
    }

    func testCustomDModel() {
        let model = TransformerModel(numLayers: 1, dModel: 64, numHeads: 8, ffDim: 128)
        XCTAssertEqual(model.dModel, 64)
    }

    func testCustomNumHeads() {
        let model = TransformerModel(numLayers: 1, dModel: 64, numHeads: 8, ffDim: 128)
        XCTAssertEqual(model.numHeads, 8)
    }

    func testCustomFFDim() {
        let model = TransformerModel(numLayers: 1, dModel: 64, numHeads: 8, ffDim: 128)
        XCTAssertEqual(model.ffDim, 128)
    }

    // =============================================================================
    // MARK: - Block Array Tests
    // =============================================================================

    func testBlockCountMatchesNumLayersDefault() {
        let model = TransformerModel()
        XCTAssertEqual(model.blocks.count, model.numLayers,
                       "Number of transformer blocks should equal numLayers")
    }

    func testBlockCountMatchesNumLayersCustom() {
        for layers in [1, 2, 3, 4] {
            let model = TransformerModel(numLayers: layers)
            XCTAssertEqual(model.blocks.count, layers,
                           "blocks.count should equal numLayers=\(layers)")
        }
    }

    func testSingleLayerModel() {
        let model = TransformerModel(numLayers: 1)
        XCTAssertEqual(model.numLayers, 1)
        XCTAssertEqual(model.blocks.count, 1)
    }

    // =============================================================================
    // MARK: - Positional Embeddings Tests
    // =============================================================================

    func testPositionalEmbeddingsShape() {
        // posEmbeddings should have shape [49, d_model] for 7x7 grid of 4x4 patches
        let model = TransformerModel()
        eval(model)
        XCTAssertEqual(model.posEmbeddings.shape, [49, TRANSFORMER_D_MODEL],
                       "posEmbeddings should have shape [49, dModel]")
    }

    func testPositionalEmbeddingsShapeCustomDModel() {
        let dModel = 64
        let model = TransformerModel(numLayers: 1, dModel: dModel, numHeads: 8, ffDim: 128)
        eval(model)
        XCTAssertEqual(model.posEmbeddings.shape, [49, dModel],
                       "posEmbeddings should have shape [49, \(dModel)] for custom dModel")
    }

    func testPositionalEmbeddingsAreFinite() {
        let model = TransformerModel()
        eval(model)
        assertAllFinite(model.posEmbeddings, "Positional embeddings should be finite at initialization")
    }

    func testPositionalEmbeddingsAreSmall() {
        // Initialized with * 0.02, so values should be small
        let model = TransformerModel()
        eval(model)
        let values = model.posEmbeddings.asArray(Float.self)
        let maxAbs = values.map { abs($0) }.max() ?? 0.0
        XCTAssertLessThan(maxAbs, 1.0,
                          "Positional embeddings should be small at init (scaled by 0.02)")
    }

    // =============================================================================
    // MARK: - trainTransformerEpoch Tests
    // =============================================================================

    func testTrainTransformerEpochReturnsFiniteFloat() {
        let model = TransformerModel()
        eval(model)
        let optimizer = SGD(learningRate: 0.01)

        let images = makeFlatImages(batch: 16)
        let labels = makeLabels(batch: 16)

        let loss = trainTransformerEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: images,
            trainLabels: labels,
            batchSize: 8
        )

        XCTAssertTrue(loss.isFinite, "Training epoch should return finite loss, got \(loss)")
    }

    func testTrainTransformerEpochLossIsPositive() {
        let model = TransformerModel()
        eval(model)
        let optimizer = SGD(learningRate: 0.01)

        let images = makeFlatImages(batch: 16)
        let labels = makeLabels(batch: 16)

        let loss = trainTransformerEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: images,
            trainLabels: labels,
            batchSize: 8
        )

        XCTAssertGreaterThan(loss, 0.0, "Cross-entropy loss should be positive")
    }

    func testTrainTransformerEpochLossIsReasonable() {
        // For random predictions on 10 classes, initial loss ≈ ln(10) ≈ 2.303
        let model = TransformerModel()
        eval(model)
        let optimizer = SGD(learningRate: 0.01)

        let images = makeFlatImages(batch: 32)
        let labels = makeLabels(batch: 32)

        let loss = trainTransformerEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: images,
            trainLabels: labels,
            batchSize: 32
        )

        // Initial loss should be near ln(10) ≈ 2.30 for a 10-class random model
        // Allow wide range [0.01, 100.0] to accommodate various random initializations
        XCTAssertGreaterThan(loss, 0.01, "Loss should be greater than 0.01")
        XCTAssertLessThan(loss, 100.0, "Loss should be less than 100.0")
    }

    func testTrainTransformerEpochUpdatesModelParameters() {
        // Training should change model parameters
        let model = TransformerModel()
        eval(model)

        let initialParams = model.parameters().flattened()
        XCTAssertFalse(initialParams.isEmpty, "Model should have parameters")

        var initialValuesByName: [String: (shape: [Int], values: [Float])] = [:]
        for (name, param) in initialParams {
            eval(param)
            initialValuesByName[name] = (shape: param.shape, values: param.asArray(Float.self))
        }

        let optimizer = SGD(learningRate: 0.01)
        let images = makeFlatImages(batch: 16)
        let labels = makeLabels(batch: 16)

        let _ = trainTransformerEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: images,
            trainLabels: labels,
            batchSize: 16
        )

        // Check that the parameters changed
        let updatedParams = model.parameters().flattened()
        XCTAssertEqual(updatedParams.count, initialParams.count,
                       "Parameter tensor count should not change after training")

        var hasChanged = false
        for (name, updatedParam) in updatedParams {
            guard let initial = initialValuesByName[name] else {
                XCTFail("Parameter \(name) should still exist after training")
                continue
            }

            eval(updatedParam)
            let updatedValues = updatedParam.asArray(Float.self)
            XCTAssertEqual(updatedParam.shape, initial.shape,
                           "Parameter \(name) shape should not change after training")
            XCTAssertEqual(updatedValues.count, initial.values.count,
                           "Parameter \(name) element count should not change after training")

            if zip(initial.values, updatedValues).contains(where: { abs($0.0 - $0.1) > 1e-8 }) {
                hasChanged = true
            }
        }

        // At least one parameter tensor should have changed
        XCTAssertTrue(hasChanged,
                      "Model parameters should change after one training step")
    }

    func testTrainTransformerEpochWithSingleBatch() {
        // Batch size equals dataset size (single batch per epoch)
        let model = TransformerModel()
        eval(model)
        let optimizer = SGD(learningRate: 0.01)

        let images = makeFlatImages(batch: 8)
        let labels = makeLabels(batch: 8)

        let loss = trainTransformerEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: images,
            trainLabels: labels,
            batchSize: 8  // same as dataset size
        )

        XCTAssertTrue(loss.isFinite)
        XCTAssertGreaterThan(loss, 0.0)
    }

    func testTrainTransformerEpochWithBatchSizeLargerThanDataset() {
        // If batchSize > dataset, should still process all samples in one "batch"
        let model = TransformerModel()
        eval(model)
        let optimizer = SGD(learningRate: 0.01)

        let images = makeFlatImages(batch: 4)
        let labels = makeLabels(batch: 4)

        let loss = trainTransformerEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: images,
            trainLabels: labels,
            batchSize: 100  // larger than dataset
        )

        XCTAssertTrue(loss.isFinite,
                      "Training with batchSize > dataset should still work")
    }

    // =============================================================================
    // MARK: - verifyLayerNorm Tests
    // =============================================================================

    func testVerifyLayerNormDoesNotCrash() {
        // verifyLayerNorm() should run without throwing or crashing
        XCTAssertNoThrow(verifyLayerNorm(),
                         "verifyLayerNorm() should complete without error")
    }

    // =============================================================================
    // MARK: - transformerLoss Tests
    // =============================================================================

    func testTransformerLossIsScalar() {
        let model = TransformerModel()
        let images = makeFlatImages(batch: 8)
        let labels = makeLabels(batch: 8)
        let loss = transformerLoss(model: model, images: images, labels: labels)
        XCTAssertEqual(loss.ndim, 0, "Loss should be a scalar (0-dimensional tensor)")
    }

    func testTransformerLossIsPositive() {
        let model = TransformerModel()
        let images = makeFlatImages(batch: 8)
        let labels = makeLabels(batch: 8)
        let loss = transformerLoss(model: model, images: images, labels: labels)
        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0,
                             "Cross-entropy loss should be positive")
    }

    func testTransformerLossIsFinite() {
        let model = TransformerModel()
        let images = makeFlatImages(batch: 8)
        let labels = makeLabels(batch: 8)
        let loss = transformerLoss(model: model, images: images, labels: labels)
        let lossValue = loss.item(Float.self)
        XCTAssertTrue(lossValue.isFinite, "Loss should be finite")
    }

    func testTransformerLossWithPerfectPredictions() {
        // Loss should be near 0 for perfect predictions (very high confidence correct class)
        let model = TransformerModel()
        // We can't easily engineer perfect predictions with random init,
        // but we can verify it's in a reasonable range
        let images = makeFlatImages(batch: 4)
        let labels = makeLabels(batch: 4)
        let loss = transformerLoss(model: model, images: images, labels: labels)
        let lossValue = loss.item(Float.self)
        XCTAssertLessThan(lossValue, 50.0, "Initial loss should be well below 50")
    }

    // =============================================================================
    // MARK: - transformerAccuracy Tests
    // =============================================================================

    func testTransformerAccuracyInRange() {
        let model = TransformerModel()
        let images = makeFlatImages(batch: 32)
        let labels = makeLabels(batch: 32)
        let acc = transformerAccuracy(model: model, images: images, labels: labels)
        XCTAssertGreaterThanOrEqual(acc, 0.0, "Accuracy should be >= 0")
        XCTAssertLessThanOrEqual(acc, 1.0, "Accuracy should be <= 1")
    }

    func testTransformerAccuracyIsFinite() {
        let model = TransformerModel()
        let images = makeFlatImages(batch: 16)
        let labels = makeLabels(batch: 16)
        let acc = transformerAccuracy(model: model, images: images, labels: labels)
        XCTAssertTrue(acc.isFinite, "Accuracy should be finite")
    }

    func testTransformerAccuracyWithSingleSample() {
        let model = TransformerModel()
        let image = makeFlatImages(batch: 1)
        let label = makeLabels(batch: 1)
        let acc = transformerAccuracy(model: model, images: image, labels: label)
        // With a single sample, accuracy is either 0.0 or 1.0
        XCTAssertTrue(acc == 0.0 || acc == 1.0,
                      "Accuracy for a single sample should be 0.0 or 1.0, got \(acc)")
    }

    func testTransformerAccuracyReturnType() {
        let model = TransformerModel()
        let images = makeFlatImages(batch: 8)
        let labels = makeLabels(batch: 8)
        let acc: Float = transformerAccuracy(model: model, images: images, labels: labels)
        // The fact that it compiles as Float is the test
        XCTAssertTrue(acc >= 0.0)
    }

    // =============================================================================
    // MARK: - Default Constants Integration Tests
    // =============================================================================

    func testDefaultModelUsesTransformerDModel() {
        let model = TransformerModel()
        XCTAssertEqual(model.dModel, TRANSFORMER_D_MODEL,
                       "Default TransformerModel should use TRANSFORMER_D_MODEL")
    }

    func testDefaultModelUsesTransformerNumHeads() {
        let model = TransformerModel()
        XCTAssertEqual(model.numHeads, TRANSFORMER_NUM_HEADS,
                       "Default TransformerModel should use TRANSFORMER_NUM_HEADS")
    }

    func testDefaultModelUsesTransformerFFDim() {
        let model = TransformerModel()
        XCTAssertEqual(model.ffDim, TRANSFORMER_FF_DIM,
                       "Default TransformerModel should use TRANSFORMER_FF_DIM")
    }

    // =============================================================================
    // MARK: - Regression: trainTransformerEpoch Does Not Diverge
    // =============================================================================

    func testTrainTransformerEpochDoesNotDiverge() {
        // After several training steps on the same data, loss should stay bounded
        let model = TransformerModel(numLayers: 1, dModel: 16, numHeads: 2, ffDim: 32)
        eval(model)
        let optimizer = SGD(learningRate: 0.01)

        // Create a small toy dataset
        let numSamples = 32
        let images = makeFlatImages(batch: numSamples)
        let labels = makeLabels(batch: numSamples)
        eval(images, labels)

        var losses: [Float] = []
        for _ in 0..<3 {
            let loss = trainTransformerEpoch(
                model: model,
                optimizer: optimizer,
                trainImages: images,
                trainLabels: labels,
                batchSize: 16
            )
            losses.append(loss)
        }

        // All losses should be finite
        for (i, loss) in losses.enumerated() {
            XCTAssertTrue(loss.isFinite, "Loss at epoch \(i+1) should be finite, got \(loss)")
        }

        // With this small toy dataset, require bounded loss rather than strict
        // improvement so the test catches divergence without becoming flaky.
        let firstLoss = losses[0]
        let lastLoss = losses[2]
        XCTAssertLessThan(lastLoss, firstLoss * 2.0,
                          "Loss should not diverge over training (first: \(firstLoss), last: \(lastLoss))")
    }
}
