import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class ModelLossFunctionTests: LossFunctionTestSupport {
    func testMLPLossFunction() {
        // Test the mlpLoss wrapper function
        let model = MLPModel()
        let batchSize = 16

        // Create random input images [batch_size, 784]
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = mlpLoss(model: model, images: images, labels: labels)
        eval(loss)

        // Verify loss properties
        XCTAssertEqual(loss.size, 1, "MLP loss should be scalar")
        assertAllFinite(loss, "MLP loss should be finite")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "MLP loss should be positive")
        XCTAssertLessThan(lossValue, 20.0, "MLP loss should be reasonable for untrained model")
    }

    func testMLPLossConsistency() {
        // Test that mlpLoss produces consistent results for same inputs
        let model = MLPModel()
        let batchSize = 8
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let loss1 = mlpLoss(model: model, images: images, labels: labels)
        let loss2 = mlpLoss(model: model, images: images, labels: labels)

        eval(loss1, loss2)

        let lossValue1 = loss1.item(Float.self)
        let lossValue2 = loss2.item(Float.self)

        XCTAssertEqual(lossValue1, lossValue2, accuracy: 1e-6,
                      "Same inputs should produce same loss")
    }

    func testMLPLossIsFiniteForUntrainedModel() {
        // Test MLP loss for an untrained model.
        let model = MLPModel()
        let batchSize = 4
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        // Just verify loss is computed (can't force perfect predictions without training)
        let loss = mlpLoss(model: model, images: images, labels: labels)
        eval(loss)

        assertAllFinite(loss, "MLP loss should be finite")
        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "Loss should be positive for untrained model")
    }

    // =============================================================================
    // MARK: - CNN Model Loss Function Tests
    // =============================================================================

    func testCNNLossFunction() {
        // Test the cnnLoss wrapper function
        let model = CNNModel()
        let batchSize = 8

        // Create random input images [batch_size, 28, 28, 1]
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = cnnLoss(model: model, images: images, labels: labels)
        eval(loss)

        // Verify loss properties
        XCTAssertEqual(loss.size, 1, "CNN loss should be scalar")
        assertAllFinite(loss, "CNN loss should be finite")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "CNN loss should be positive")
        XCTAssertLessThan(lossValue, 20.0, "CNN loss should be reasonable for untrained model")
    }

    func testCNNLossConsistency() {
        // Test that cnnLoss produces consistent results for same inputs
        let model = CNNModel()
        let batchSize = 4
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let loss1 = cnnLoss(model: model, images: images, labels: labels)
        let loss2 = cnnLoss(model: model, images: images, labels: labels)

        eval(loss1, loss2)

        let lossValue1 = loss1.item(Float.self)
        let lossValue2 = loss2.item(Float.self)

        XCTAssertEqual(lossValue1, lossValue2, accuracy: 1e-6,
                      "Same inputs should produce same loss")
    }

    // =============================================================================
    // MARK: - Edge Cases and Numerical Stability Tests
    // =============================================================================

}
