import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class CrossEntropyGradientTests: LossFunctionTestSupport {
    // =============================================================================
    // MARK: - Gradient Flow Tests
    // =============================================================================

    func testCrossEntropyGradientsExist() {
        // Test that cross-entropy loss produces gradients for backpropagation
        let model = MLPModel()
        let batchSize = 8
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let lossAndGrad = valueAndGrad(model: model, mlpLoss)
        let (loss, grads) = lossAndGrad(model, images, labels)

        eval(loss)

        // Verify gradients exist
        let flatGrads = grads.flattened()
        XCTAssertGreaterThan(flatGrads.count, 0,
                            "Gradients should exist after loss computation")

        // Verify gradients are finite and non-zero
        for (_, gradArray) in flatGrads {
            eval(gradArray)
            assertAllFinite(gradArray, "Gradients should be finite")
        }
    }

    func testCrossEntropyGradientsNonZero() {
        // Test that cross-entropy gradients are non-zero (indicating proper flow)
        let model = MLPModel()
        let batchSize = 8
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let lossAndGrad = valueAndGrad(model: model, mlpLoss)
        let (_, grads) = lossAndGrad(model, images, labels)

        // Check that gradients have non-zero norm
        let flatGrads = grads.flattened()
        var totalGradNorm: Float = 0.0

        for (_, gradArray) in flatGrads {
            eval(gradArray)
            let gradNorm = sum(gradArray * gradArray).item(Float.self)
            totalGradNorm += gradNorm
        }

        XCTAssertGreaterThan(totalGradNorm, 0.0,
                            "Total gradient norm should be non-zero")
    }

}
