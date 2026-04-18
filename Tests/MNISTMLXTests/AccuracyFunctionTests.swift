import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class AccuracyFunctionTests: LossFunctionTestSupport {
    // =============================================================================
    // MARK: - Accuracy Computation Tests
    // =============================================================================

    func testAccuracyPerfectPredictions() {
        // Test accuracy computation with perfect predictions (100%)
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        // Create perfect logits that match the labels
        let perfectLogits = createPerfectLogits(labels: labels)

        // Test the accuracy logic directly
        let predictions = argMax(perfectLogits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Perfect predictions should give 100% accuracy
        XCTAssertEqual(accuracy, 1.0, accuracy: 1e-6,
                      "Perfect predictions should yield 100% accuracy")
    }

    func testAccuracyWorstPredictions() {
        // Test accuracy computation with worst-case predictions (0%)
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        // Create worst logits (always predict wrong class)
        let worstLogits = createWorstLogits(labels: labels)

        let predictions = argMax(worstLogits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Worst predictions should give 0% accuracy
        XCTAssertEqual(accuracy, 0.0, accuracy: 1e-6,
                      "Worst-case predictions should yield 0% accuracy")
    }

    func testAccuracyRandomPredictions() {
        // Test accuracy computation with random predictions
        // Random guessing on 10 classes should give ~10% accuracy
        let batchSize = 1000  // Large batch for statistical stability
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Random predictions should give roughly 10% accuracy for 10 classes
        // Allow wide range due to randomness: [0.0, 0.3]
        assertInRange(accuracy, 0.0, 0.3,
                     "Random predictions should yield roughly 10% accuracy")
    }

    func testAccuracyPartialCorrect() {
        // Test accuracy computation with partially correct predictions
        let batchSize = 10
        let numClasses = 10

        // Create labels [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let labels = MLXArray((0..<batchSize).map { Int32($0) })

        // Create logits where first 5 are correct, last 5 are wrong
        var allLogits: [Float] = []
        for i in 0..<batchSize {
            for j in 0..<numClasses {
                if i < 5 {
                    // First 5: correct predictions
                    if j == i {
                        allLogits.append(10.0)
                    } else {
                        allLogits.append(-10.0)
                    }
                } else {
                    // Last 5: wrong predictions (predict class 0)
                    if j == 0 {
                        allLogits.append(10.0)
                    } else {
                        allLogits.append(-10.0)
                    }
                }
            }
        }
        let logits = MLXArray(allLogits, [batchSize, numClasses])

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // 5 out of 10 correct = 50% accuracy
        XCTAssertEqual(accuracy, 0.5, accuracy: 1e-6,
                      "50% correct predictions should yield 50% accuracy")
    }

    func testAccuracyScalarOutput() {
        // Test that accuracy computation produces a scalar value
        let batchSize = 16
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracyArray = mean(correct)

        eval(accuracyArray)

        XCTAssertEqual(accuracyArray.size, 1, "Accuracy should be a scalar")
        XCTAssertEqual(accuracyArray.ndim, 0, "Accuracy should be 0-dimensional")
    }

    func testAccuracyInValidRange() {
        // Test that accuracy is always in valid range [0.0, 1.0]
        let batchSizes = [1, 10, 32, 100]

        for batchSize in batchSizes {
            let logits = createRandomLogits(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            let predictions = argMax(logits, axis: 1)
            let correct = predictions .== labels
            let accuracy = mean(correct).item(Float.self)

            assertInRange(accuracy, 0.0, 1.0,
                         "Accuracy should be in [0.0, 1.0] for batch size \(batchSize)")
        }
    }

    func testAccuracySingleSample() {
        // Test accuracy computation with single sample
        let logits = MLXArray([10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], [1, 10])
        let labels = MLXArray([Int32(0)])  // Correct class is 0

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Single correct prediction should give 100% accuracy
        XCTAssertEqual(accuracy, 1.0, accuracy: 1e-6,
                      "Single correct prediction should yield 100% accuracy")
    }

    func testAccuracySingleSampleWrong() {
        // Test accuracy computation with single incorrect sample
        let logits = MLXArray([-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], [1, 10])
        let labels = MLXArray([Int32(0)])  // Correct class is 0, but we predict 1

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Single wrong prediction should give 0% accuracy
        XCTAssertEqual(accuracy, 0.0, accuracy: 1e-6,
                      "Single wrong prediction should yield 0% accuracy")
    }

    // =============================================================================
    // MARK: - MLP Model Accuracy Tests
    // =============================================================================

    func testMLPAccuracyFunction() {
        // Test the mlpAccuracy wrapper function
        let model = MLPModel()
        let batchSize = 32
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = mlpAccuracy(model: model, images: images, labels: labels)

        // Verify accuracy is in valid range
        assertInRange(accuracy, 0.0, 1.0,
                     "MLP accuracy should be in [0.0, 1.0]")
    }

    func testMLPAccuracyConsistency() {
        // Test that mlpAccuracy produces consistent results for same inputs
        let model = MLPModel()
        let batchSize = 16
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy1 = mlpAccuracy(model: model, images: images, labels: labels)
        let accuracy2 = mlpAccuracy(model: model, images: images, labels: labels)

        XCTAssertEqual(accuracy1, accuracy2, accuracy: 1e-6,
                      "Same inputs should produce same accuracy")
    }

    func testMLPAccuracyUntrainedModel() {
        // Test accuracy of untrained model (should be around random guessing)
        let model = MLPModel()
        let batchSize = 100
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = mlpAccuracy(model: model, images: images, labels: labels)

        // Untrained model should have low accuracy (not much better than random)
        // Random guessing on 10 classes = 10%, allow range [0.0, 0.5]
        assertInRange(accuracy, 0.0, 0.5,
                     "Untrained MLP accuracy should be low (roughly random guessing)")
    }

    func testMLPAccuracyVariousBatchSizes() {
        // Test MLP accuracy with various batch sizes
        let model = MLPModel()
        let batchSizes = [1, 4, 16, 64]

        for batchSize in batchSizes {
            let images = abs(MLXRandom.normal([batchSize, 784]))
            let labels = createRandomLabels(batchSize: batchSize)

            let accuracy = mlpAccuracy(model: model, images: images, labels: labels)

            assertInRange(accuracy, 0.0, 1.0,
                         "MLP accuracy should be in [0.0, 1.0] for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - CNN Model Accuracy Tests
    // =============================================================================

    func testCNNAccuracyFunction() {
        // Test the cnnAccuracy wrapper function
        let model = CNNModel()
        let batchSize = 16
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = cnnAccuracy(model: model, images: images, labels: labels)

        // Verify accuracy is in valid range
        assertInRange(accuracy, 0.0, 1.0,
                     "CNN accuracy should be in [0.0, 1.0]")
    }

    func testCNNAccuracyConsistency() {
        // Test that cnnAccuracy produces consistent results for same inputs
        let model = CNNModel()
        let batchSize = 8
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy1 = cnnAccuracy(model: model, images: images, labels: labels)
        let accuracy2 = cnnAccuracy(model: model, images: images, labels: labels)

        XCTAssertEqual(accuracy1, accuracy2, accuracy: 1e-6,
                      "Same inputs should produce same accuracy")
    }

    func testCNNAccuracyUntrainedModel() {
        // Test accuracy of untrained CNN model (should be around random guessing)
        let model = CNNModel()
        let batchSize = 100
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = cnnAccuracy(model: model, images: images, labels: labels)

        // Untrained model should have low accuracy (not much better than random)
        // Random guessing on 10 classes = 10%, allow range [0.0, 0.5]
        assertInRange(accuracy, 0.0, 0.5,
                     "Untrained CNN accuracy should be low (roughly random guessing)")
    }

    func testCNNAccuracyVariousBatchSizes() {
        // Test CNN accuracy with various batch sizes
        let model = CNNModel()
        let batchSizes = [1, 4, 16, 32]

        for batchSize in batchSizes {
            let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
            let labels = createRandomLabels(batchSize: batchSize)

            let accuracy = cnnAccuracy(model: model, images: images, labels: labels)

            assertInRange(accuracy, 0.0, 1.0,
                         "CNN accuracy should be in [0.0, 1.0] for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Accuracy vs Loss Correlation Tests
    // =============================================================================

    func testAccuracyLossCorrelation() {
        // Test that higher accuracy correlates with lower loss
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        // Perfect predictions
        let perfectLogits = createPerfectLogits(labels: labels)
        let perfectPredictions = argMax(perfectLogits, axis: 1)
        let perfectCorrect = perfectPredictions .== labels
        let perfectAccuracy = mean(perfectCorrect).item(Float.self)
        let perfectLoss = crossEntropy(logits: perfectLogits, targets: labels, reduction: .mean)
        eval(perfectLoss)
        let perfectLossValue = perfectLoss.item(Float.self)

        // Random predictions
        let randomLogits = createRandomLogits(batchSize: batchSize)
        let randomPredictions = argMax(randomLogits, axis: 1)
        let randomCorrect = randomPredictions .== labels
        let randomAccuracy = mean(randomCorrect).item(Float.self)
        let randomLoss = crossEntropy(logits: randomLogits, targets: labels, reduction: .mean)
        eval(randomLoss)
        let randomLossValue = randomLoss.item(Float.self)

        // Perfect accuracy should be higher and loss should be lower
        XCTAssertGreaterThan(perfectAccuracy, randomAccuracy,
                            "Perfect predictions should have higher accuracy")
        XCTAssertLessThan(perfectLossValue, randomLossValue,
                         "Perfect predictions should have lower loss")
    }

    func testAccuracyLossInverseRelationship() {
        // Test that as accuracy increases, loss generally decreases
        let batchSize = 50
        let labels = createRandomLabels(batchSize: batchSize)

        // Create three scenarios: worst, random, perfect
        let worstLogits = createWorstLogits(labels: labels)
        let randomLogits = createRandomLogits(batchSize: batchSize)
        let perfectLogits = createPerfectLogits(labels: labels)

        // Compute accuracies
        let randomAcc = mean(argMax(randomLogits, axis: 1) .== labels).item(Float.self)
        let perfectAcc = mean(argMax(perfectLogits, axis: 1) .== labels).item(Float.self)

        // Compute losses
        let worstLoss = crossEntropy(logits: worstLogits, targets: labels, reduction: .mean).item(Float.self)
        let randomLoss = crossEntropy(logits: randomLogits, targets: labels, reduction: .mean).item(Float.self)
        let perfectLoss = crossEntropy(logits: perfectLogits, targets: labels, reduction: .mean).item(Float.self)

        // Verify ordering: perfect > random for accuracy
        XCTAssertGreaterThan(perfectAcc, randomAcc, "Perfect accuracy > Random accuracy")

        // Verify ordering: worst > random > perfect for loss
        XCTAssertGreaterThan(worstLoss, randomLoss, "Worst loss > Random loss")
        XCTAssertGreaterThan(randomLoss, perfectLoss, "Random loss > Perfect loss")
    }

}
