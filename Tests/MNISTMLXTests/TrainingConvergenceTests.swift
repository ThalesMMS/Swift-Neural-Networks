// ============================================================================
// TrainingConvergenceTests.swift - Tests for Training Convergence
// ============================================================================
//
// This test suite validates that models can learn by verifying training
// convergence on small toy datasets:
// - MLP converges on 100-sample toy dataset
// - CNN converges on 50-sample toy dataset
// - Attention converges on 30-sample toy dataset
// - Loss decreases over multiple epochs
// - Final loss is lower than initial loss
// - Models can overfit small datasets (sanity check)
//
// WHY TEST CONVERGENCE?
//   Testing convergence ensures the training pipeline works end-to-end:
//   - Forward pass computes correct outputs
//   - Loss function measures error properly
//   - Backward pass computes valid gradients
//   - Optimizer updates parameters in the right direction
//   - Model capacity is sufficient to learn patterns
//
//   If a model cannot overfit a tiny dataset, something is fundamentally broken!
//
// TOY DATASETS:
//   We use small synthetic datasets for fast, deterministic testing:
//   - Small size (30-100 samples) for quick test execution
//   - Random but repeatable (can be made deterministic with seeds)
//   - Simple enough to overfit quickly (sanity check)
//   - Large enough to show meaningful convergence trends
//
// ============================================================================

import XCTest
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
@testable import MNISTMLX

final class TrainingConvergenceTests: MLXTestCase {

    // =============================================================================
    // MARK: - Toy Dataset Creation Utilities
    // =============================================================================

    /// Creates a small toy dataset for convergence testing
    ///
    /// This generates random images and labels for testing if models can learn.
    /// The dataset is intentionally small so models can quickly overfit, which
    /// serves as a sanity check that the training pipeline works.
    ///
    /// - Parameters:
    ///   - numSamples: Number of samples in the toy dataset
    ///   - imageSize: Size of each flattened image (default: 784 for MNIST)
    /// - Returns: Tuple of (images: [numSamples, imageSize], labels: [numSamples])
    ///
    /// ## Example Usage
    /// ```swift
    /// let (images, labels) = createToyDataset(numSamples: 100)
    /// // Train model on this tiny dataset - it should overfit!
    /// ```
    private func createToyDataset(
        numSamples: Int,
        imageSize: Int = 784
    ) -> (images: MLXArray, labels: MLXArray) {
        // Create random images normalized to [0, 1] range
        let images = abs(MLXRandom.normal([numSamples, imageSize])) * 0.1

        // Create random labels (0-9 for MNIST)
        let labels = (0..<numSamples).map { _ in Int32.random(in: 0..<10) }

        return (images: images, labels: MLXArray(labels))
    }

    /// Creates a small 2D toy dataset for CNN/Attention models
    ///
    /// - Parameters:
    ///   - numSamples: Number of samples in the toy dataset
    ///   - height: Image height (default: 28 for MNIST)
    ///   - width: Image width (default: 28 for MNIST)
    ///   - channels: Number of channels (default: 1 for grayscale)
    /// - Returns: Tuple of (images: [numSamples, height, width, channels], labels: [numSamples])
    private func createToy2DDataset(
        numSamples: Int,
        height: Int = 28,
        width: Int = 28,
        channels: Int = 1
    ) -> (images: MLXArray, labels: MLXArray) {
        // Create random 2D images
        let images = abs(MLXRandom.normal([numSamples, height, width, channels])) * 0.1

        // Create random labels
        let labels = (0..<numSamples).map { _ in Int32.random(in: 0..<10) }

        return (images: images, labels: MLXArray(labels))
    }

    /// Creates a simple toy dataset with a known pattern for testing basic learning
    ///
    /// This creates a dataset where different classes have slightly different
    /// distributions, making them learnable but not trivially separable.
    ///
    /// - Parameters:
    ///   - numSamples: Number of samples to create
    ///   - imageSize: Size of flattened feature vector
    /// - Returns: Tuple of (images, labels)
    ///
    /// ## Implementation Note
    /// Creates samples with some correlation between features and labels, making
    /// the dataset learnable. This serves as a sanity check - if a model can't
    /// show any improvement on this dataset, something is wrong!
    private func createPatternedToyDataset(
        numSamples: Int,
        imageSize: Int = 784
    ) -> (images: MLXArray, labels: MLXArray) {
        // Create random images
        var imageData: [Float] = []
        var labelData: [Int32] = []

        for i in 0..<numSamples {
            let label = Int32(i % 10)  // Distribute evenly across classes
            labelData.append(label)

            // Add slight bias based on label (helps with learning)
            let bias = Float(label) * 0.05
            for _ in 0..<imageSize {
                imageData.append(Float.random(in: 0..<1) * 0.1 + bias)
            }
        }

        let images = MLXArray(imageData, [numSamples, imageSize])
        let labels = MLXArray(labelData)

        return (images: images, labels: labels)
    }

    // =============================================================================
    // MARK: - Convergence Verification Utilities
    // =============================================================================

    /// Verifies that a loss value is valid (finite and non-negative)
    ///
    /// - Parameters:
    ///   - loss: The loss value to check
    ///   - message: Custom assertion message
    ///   - file: Source file (auto-filled by Swift)
    ///   - line: Source line (auto-filled by Swift)
    private func assertValidLoss(
        _ loss: Float,
        _ message: String = "Loss should be finite and non-negative",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertFalse(loss.isNaN, "Loss should not be NaN", file: file, line: line)
        XCTAssertFalse(loss.isInfinite, "Loss should not be infinite", file: file, line: line)
        XCTAssertGreaterThanOrEqual(loss, 0, "Loss should be non-negative", file: file, line: line)
    }

    /// Verifies that loss decreased over training (convergence check)
    ///
    /// This is the core convergence test: if training works, loss should decrease!
    ///
    /// - Parameters:
    ///   - initialLoss: Loss before training (or at start of epoch)
    ///   - finalLoss: Loss after training
    ///   - minimumDecrease: Minimum expected decrease (default: 0.0 for any decrease)
    ///   - message: Custom assertion message
    ///   - file: Source file (auto-filled by Swift)
    ///   - line: Source line (auto-filled by Swift)
    ///
    /// ## Why This Matters
    /// If loss doesn't decrease on a tiny toy dataset, one of these is broken:
    /// - Forward pass (wrong outputs)
    /// - Loss function (not measuring error correctly)
    /// - Backward pass (wrong gradients)
    /// - Optimizer (not updating parameters correctly)
    /// - Learning rate (too small or too large)
    private func assertLossDecreased(
        initialLoss: Float,
        finalLoss: Float,
        minimumDecrease: Float = 0.0,
        _ message: String = "Loss should decrease during training",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        let decrease = initialLoss - finalLoss
        XCTAssertGreaterThanOrEqual(
            decrease,
            minimumDecrease,
            "\(message) (initial: \(initialLoss), final: \(finalLoss), decrease: \(decrease))",
            file: file,
            line: line
        )
    }

    /// Verifies that loss decreased monotonically (or mostly so) across epochs
    ///
    /// - Parameters:
    ///   - losses: Array of loss values, one per epoch
    ///   - tolerance: Number of allowed non-decreasing steps (default: 1)
    ///   - message: Custom assertion message
    ///   - file: Source file (auto-filled by Swift)
    ///   - line: Source line (auto-filled by Swift)
    ///
    /// ## Why Tolerance?
    /// Loss might not decrease EVERY epoch due to:
    /// - Stochastic gradient descent randomness
    /// - Small datasets with high variance
    /// - Convergence plateaus near minimum
    ///
    /// We allow a few non-decreasing steps while still checking overall trend.
    private func assertLossDecreasedMonotonically(
        losses: [Float],
        tolerance: Int = 1,
        _ message: String = "Loss should generally decrease across epochs",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        var nonDecreasingCount = 0

        for i in 1..<losses.count {
            if losses[i] >= losses[i - 1] {
                nonDecreasingCount += 1
            }
        }

        XCTAssertLessThanOrEqual(
            nonDecreasingCount,
            tolerance,
            "\(message) (found \(nonDecreasingCount) non-decreasing steps, tolerance: \(tolerance))",
            file: file,
            line: line
        )
    }

    /// Trains a model for multiple epochs and returns loss history
    ///
    /// This is a simplified training loop for convergence testing that records
    /// loss values at each epoch for analysis.
    ///
    /// - Parameters:
    ///   - model: The model to train
    ///   - optimizer: The optimizer to use
    ///   - images: Training images
    ///   - labels: Training labels
    ///   - batchSize: Batch size for training
    ///   - epochs: Number of epochs to train
    ///   - lossFunction: Function that computes loss given (model, images, labels)
    /// - Returns: Array of average loss values, one per epoch
    ///
    /// ## Example Usage
    /// ```swift
    /// let losses = trainAndRecordLosses(
    ///     model: mlp,
    ///     optimizer: sgd,
    ///     images: toyImages,
    ///     labels: toyLabels,
    ///     batchSize: 32,
    ///     epochs: 5,
    ///     lossFunction: mlpLoss
    /// )
    /// // Verify losses[4] < losses[0]
    /// ```
    private func trainAndRecordLosses<Model: Module>(
        model: Model,
        optimizer: SGD,
        images: MLXArray,
        labels: MLXArray,
        batchSize: Int,
        epochs: Int,
        lossFunction: @escaping (Model, MLXArray, MLXArray) -> MLXArray
    ) -> [Float] {
        var lossHistory: [Float] = []

        for _ in 0..<epochs {
            let epochLoss = trainEpoch(
                model: model,
                optimizer: optimizer,
                images: images,
                labels: labels,
                batchSize: batchSize,
                lossFunction: lossFunction
            )
            lossHistory.append(epochLoss)
        }

        return lossHistory
    }

    /// Trains a model for one epoch (simplified, no progress bar)
    ///
    /// - Parameters:
    ///   - model: The model to train
    ///   - optimizer: The optimizer
    ///   - images: Training images
    ///   - labels: Training labels
    ///   - batchSize: Batch size
    ///   - lossFunction: Loss computation function
    /// - Returns: Average loss for the epoch
    private func trainEpoch<Model: Module>(
        model: Model,
        optimizer: SGD,
        images: MLXArray,
        labels: MLXArray,
        batchSize: Int,
        lossFunction: @escaping (Model, MLXArray, MLXArray) -> MLXArray
    ) -> Float {
        let n = images.shape[0]
        var totalLoss: Float = 0
        var batchCount = 0

        // Setup automatic differentiation
        let lossAndGrad = valueAndGrad(model: model, lossFunction)

        // Shuffle indices for SGD
        var indices = Array(0..<n)
        indices.shuffle()

        // Training loop
        var start = 0
        while start < n {
            let end = min(start + batchSize, n)
            let batchIndices = Array(indices[start..<end]).map { Int32($0) }
            let idxArray = MLXArray(batchIndices)

            let batchImages = images[idxArray]
            let batchLabels = labels[idxArray]

            // Forward + backward pass
            let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)

            // Update parameters
            optimizer.update(model: model, gradients: grads)
            eval(model, optimizer)

            totalLoss += loss.item(Float.self)
            batchCount += 1

            start = end
        }

        return totalLoss / Float(batchCount)
    }

    // =============================================================================
    // MARK: - MLP Convergence Tests
    // =============================================================================

    func testMLPConvergenceOnToyDataset() {
        // Test that MLP can learn (converge) on a tiny 100-sample dataset
        //
        // WHAT WE'RE TESTING:
        // - Forward pass works correctly
        // - Loss computation is valid
        // - Backward pass computes gradients
        // - Optimizer updates parameters
        // - Model capacity is sufficient to learn
        //
        // SUCCESS CRITERIA:
        // - Loss decreases over epochs (any decrease is good!)
        // - Final loss is lower than initial loss
        // - No NaN or Inf values appear

        // Create MLP model (default: 784 → 128 → 10)
        let model = MLPModel()

        // Create 100-sample toy dataset
        let (images, labels) = createToyDataset(numSamples: 100)

        // Verify dataset creation
        XCTAssertEqual(images.shape, [100, 784], "Toy dataset should have 100 samples")
        XCTAssertEqual(labels.shape, [100], "Labels should match sample count")

        // Create optimizer with learning rate appropriate for toy dataset
        let optimizer = SGD(learningRate: 0.01)

        // Train for 5 epochs (should be enough to see convergence on tiny dataset)
        let losses = trainAndRecordLosses(
            model: model,
            optimizer: optimizer,
            images: images,
            labels: labels,
            batchSize: 32,
            epochs: 5,
            lossFunction: mlpLoss
        )

        // Verify we got loss values for all epochs
        XCTAssertEqual(losses.count, 5, "Should have 5 loss values (one per epoch)")

        // Verify all losses are valid (finite, non-negative)
        for (epoch, loss) in losses.enumerated() {
            assertValidLoss(loss, "Epoch \(epoch + 1) loss should be valid")
        }

        // Core convergence check: loss should decrease from start to finish
        let initialLoss = losses[0]
        let finalLoss = losses[4]

        assertLossDecreased(
            initialLoss: initialLoss,
            finalLoss: finalLoss,
            "MLP should converge on 100-sample toy dataset"
        )

        // Additional check: final loss should be reasonably low
        // (not too strict, just checking it learned something)
        XCTAssertLessThan(
            finalLoss,
            initialLoss * 0.9,
            "Final loss should be at least 10% lower than initial loss (initial: \(initialLoss), final: \(finalLoss))"
        )
    }

    // =============================================================================
    // MARK: - CNN Convergence Tests
    // =============================================================================

    func testCNNConvergenceOnToyDataset() {
        // Test that CNN can learn (converge) on a tiny 50-sample dataset
        //
        // WHAT WE'RE TESTING:
        // - CNN forward pass works correctly with 4D input [N, C, H, W]
        // - Convolutional layers compute features properly
        // - Loss computation is valid
        // - Backward pass computes gradients through conv layers
        // - Optimizer updates parameters
        // - Model capacity is sufficient to learn spatial patterns
        //
        // SUCCESS CRITERIA:
        // - Loss decreases over epochs (any decrease is good!)
        // - Final loss is lower than initial loss
        // - No NaN or Inf values appear

        // Create CNN model
        let model = CNNModel()

        // Create 50-sample toy dataset in [N, C, H, W] format (channels-first)
        // CNN expects [batch, channels, height, width]
        let numSamples = 50
        let images = abs(MLXRandom.normal([numSamples, 1, 28, 28])) * 0.1
        let labelData = (0..<numSamples).map { _ in Int32.random(in: 0..<10) }
        let labels = MLXArray(labelData)

        // Verify dataset creation
        XCTAssertEqual(images.shape, [50, 1, 28, 28], "Toy dataset should have 50 samples in [N, C, H, W] format")
        XCTAssertEqual(labels.shape, [50], "Labels should match sample count")

        // Create optimizer with learning rate appropriate for toy dataset
        let optimizer = SGD(learningRate: 0.01)

        // Train for 5 epochs (should be enough to see convergence on tiny dataset)
        let losses = trainAndRecordLosses(
            model: model,
            optimizer: optimizer,
            images: images,
            labels: labels,
            batchSize: 16,
            epochs: 5,
            lossFunction: cnnLoss
        )

        // Verify we got loss values for all epochs
        XCTAssertEqual(losses.count, 5, "Should have 5 loss values (one per epoch)")

        // Verify all losses are valid (finite, non-negative)
        for (epoch, loss) in losses.enumerated() {
            assertValidLoss(loss, "Epoch \(epoch + 1) loss should be valid")
        }

        // Core convergence check: loss should decrease from start to finish
        let initialLoss = losses[0]
        let finalLoss = losses[4]

        assertLossDecreased(
            initialLoss: initialLoss,
            finalLoss: finalLoss,
            "CNN should converge on 50-sample toy dataset"
        )

        // Additional check: final loss should be reasonably low
        // (not too strict, just checking it learned something)
        XCTAssertLessThan(
            finalLoss,
            initialLoss * 0.9,
            "Final loss should be at least 10% lower than initial loss (initial: \(initialLoss), final: \(finalLoss))"
        )
    }

    // =============================================================================
    // MARK: - Attention Convergence Tests
    // =============================================================================

    func testAttentionConvergenceOnToyDataset() {
        // Test that Attention model can learn (converge) on a tiny 30-sample dataset
        //
        // WHAT WE'RE TESTING:
        // - Attention forward pass works correctly with flat input [N, 784]
        // - Patch embedding converts flat images to patch sequences
        // - Self-attention mechanism processes patch interactions
        // - Loss computation is valid
        // - Backward pass computes gradients through attention layers
        // - Optimizer updates parameters
        // - Model capacity is sufficient to learn
        //
        // SUCCESS CRITERIA:
        // - Loss decreases over epochs (any decrease is good!)
        // - Final loss is lower than initial loss
        // - No NaN or Inf values appear

        // Create Attention model
        let model = AttentionModel()

        // Create 30-sample toy dataset (smallest of the three models)
        let (images, labels) = createToyDataset(numSamples: 30)

        // Verify dataset creation
        XCTAssertEqual(images.shape, [30, 784], "Toy dataset should have 30 samples")
        XCTAssertEqual(labels.shape, [30], "Labels should match sample count")

        // Create optimizer with learning rate appropriate for toy dataset
        let optimizer = SGD(learningRate: 0.01)

        // Train for 5 epochs (should be enough to see convergence on tiny dataset)
        let losses = trainAndRecordLosses(
            model: model,
            optimizer: optimizer,
            images: images,
            labels: labels,
            batchSize: 16,
            epochs: 5,
            lossFunction: attentionLoss
        )

        // Verify we got loss values for all epochs
        XCTAssertEqual(losses.count, 5, "Should have 5 loss values (one per epoch)")

        // Verify all losses are valid (finite, non-negative)
        for (epoch, loss) in losses.enumerated() {
            assertValidLoss(loss, "Epoch \(epoch + 1) loss should be valid")
        }

        // Core convergence check: loss should decrease from start to finish
        let initialLoss = losses[0]
        let finalLoss = losses[4]

        assertLossDecreased(
            initialLoss: initialLoss,
            finalLoss: finalLoss,
            "Attention should converge on 30-sample toy dataset"
        )

        // Additional check: final loss should be reasonably low
        // (not too strict, just checking it learned something)
        XCTAssertLessThan(
            finalLoss,
            initialLoss * 0.9,
            "Final loss should be at least 10% lower than initial loss (initial: \(initialLoss), final: \(finalLoss))"
        )
    }

    // =============================================================================
    // MARK: - Test Utilities Validation
    // =============================================================================

    func testToyDatasetCreation() {
        // Verify toy dataset utilities work correctly
        let (images, labels) = createToyDataset(numSamples: 100)

        XCTAssertEqual(images.shape, [100, 784], "Images should have correct shape")
        XCTAssertEqual(labels.shape, [100], "Labels should have correct shape")

        // Basic validation - just check that we can evaluate the arrays
        eval(images, labels)
    }

    func testToy2DDatasetCreation() {
        // Verify 2D toy dataset utilities work correctly
        let (images, labels) = createToy2DDataset(numSamples: 50)

        XCTAssertEqual(images.shape, [50, 28, 28, 1], "2D images should have correct shape")
        XCTAssertEqual(labels.shape, [50], "Labels should have correct shape")
    }

    func testPatternedDatasetCreation() {
        // Verify patterned dataset creation
        let (images, labels) = createPatternedToyDataset(numSamples: 100)

        XCTAssertEqual(images.shape, [100, 784], "Images should have correct shape")
        XCTAssertEqual(labels.shape, [100], "Labels should have correct shape")

        // Basic validation - check that we can evaluate the arrays
        eval(images, labels)
    }

    func testValidLossAssertion() {
        // Test valid loss checking utility
        assertValidLoss(0.5, "Valid loss should pass")
        assertValidLoss(2.3, "Valid loss should pass")
        assertValidLoss(0.0, "Zero loss should pass")
    }

    func testLossDecreasedAssertion() {
        // Test loss decrease verification utility
        assertLossDecreased(
            initialLoss: 2.0,
            finalLoss: 1.5,
            "Loss decreased from 2.0 to 1.5"
        )

        assertLossDecreased(
            initialLoss: 2.0,
            finalLoss: 1.0,
            minimumDecrease: 0.5,
            "Loss decreased by at least 0.5"
        )
    }
}
