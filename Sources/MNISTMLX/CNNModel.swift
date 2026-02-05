// ============================================================================
// CNNModel.swift - Convolutional Neural Network for MNIST using MLX Swift
// ============================================================================
//
// This file implements a simple CNN (Convolutional Neural Network) for
// classifying MNIST digits using Apple's MLX Swift framework.
//
// ARCHITECTURE:
//   Input: [N, 1, 28, 28] - Batch of grayscale images
//     ↓
//   Conv2d: 1→8 channels, 3×3 kernel, padding=1 → [N, 8, 28, 28]
//     ↓
//   ReLU activation
//     ↓
//   MaxPool2d: 2×2, stride=2 → [N, 8, 14, 14]
//     ↓
//   Flatten → [N, 1568]
//     ↓
//   Linear: 1568→10 → [N, 10]
//     ↓
//   Output: logits for 10 classes
//
// WHY CNN FOR IMAGES?
//   - Convolutions preserve spatial structure (nearby pixels are related)
//   - Weight sharing: same filter applied across the image (fewer parameters)
//   - Translation invariance: can recognize digits regardless of position
//
// ============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

// =============================================================================
// MARK: - CNN Architecture Constants
// =============================================================================

/// Number of output channels from the convolutional layer
/// More channels = more features the network can learn
/// We use 8 for a lightweight model; production models might use 32-64
private let CNN_CONV_CHANNELS = 8

/// Convolution kernel size (3×3 is standard for small images)
/// Larger kernels capture more context but have more parameters
private let CNN_KERNEL_SIZE = 3

/// Pooling size (2×2 with stride 2 halves spatial dimensions)
private let CNN_POOL_SIZE = 2

/// Size of the flattened feature vector after conv + pool
/// = channels × (height / pool) × (width / pool)
/// = 8 × 14 × 14 = 1568
private let CNN_FLATTEN_SIZE = CNN_CONV_CHANNELS * 14 * 14

// =============================================================================
// MARK: - CNN Model Definition
// =============================================================================

/// Convolutional Neural Network for MNIST classification
///
/// This model uses MLX's Module protocol which provides:
/// - Automatic parameter tracking (weights and biases)
/// - Easy gradient computation with valueAndGrad()
/// - Built-in serialization for saving/loading
///
/// ## Architecture Details
///
/// The architecture follows a classic pattern:
/// 1. **Convolution**: Extract local features (edges, curves, etc.)
/// 2. **ReLU**: Non-linearity to learn complex patterns
/// 3. **MaxPool**: Downsample to reduce computation and add invariance
/// 4. **Linear**: Map features to class predictions
///
/// ## Example Usage
/// ```swift
/// let model = CNNModel()
/// let images = MLXArray(...)  // [32, 1, 28, 28]
/// let logits = model(images)  // [32, 10]
/// ```
public class CNNModel: Module {
    // -------------------------------------------------------------------------
    // MARK: - Layers
    // -------------------------------------------------------------------------
    // The @ModuleInfo property wrapper tells MLX to track this layer's parameters.
    // The `key` parameter specifies the name used when saving/loading weights.
    
    /// Convolutional layer: 1 input channel (grayscale) → 8 output channels
    ///
    /// Conv2d parameters:
    /// - inputChannels: 1 (grayscale images)
    /// - outputChannels: 8 (number of feature maps)
    /// - kernelSize: 3×3 (receptive field size)
    /// - stride: 1 (move 1 pixel at a time)
    /// - padding: 1 (preserve spatial dimensions: 28×28 → 28×28)
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    
    /// Max pooling layer: reduces spatial dimensions by 2×
    ///
    /// MaxPool2d takes the maximum value in each 2×2 window.
    /// This provides:
    /// - Dimensionality reduction (28×28 → 14×14)
    /// - Translation invariance (small shifts don't change output)
    /// - Feature selection (keeps strongest activations)
    @ModuleInfo(key: "pool") var pool: MaxPool2d
    
    /// Fully connected layer: maps flattened features to class logits
    ///
    /// Linear(in, out) computes: output = input @ weights + bias
    /// - Input: 1568 features (8 channels × 14 × 14 spatial)
    /// - Output: 10 logits (one per digit class)
    @ModuleInfo(key: "fc") var fc: Linear
    
    // -------------------------------------------------------------------------
    // MARK: - Initialization
    // -------------------------------------------------------------------------
    
    /// Creates a new CNN model with randomly initialized weights
    ///
    /// MLX uses Xavier/Glorot initialization by default, which helps
    /// maintain stable gradients during training.
    public override init() {
        // Convolutional layer
        // - 1 input channel (grayscale)
        // - 8 output channels (feature maps)
        // - 3×3 kernel
        // - padding=1 keeps same spatial size
        _conv1 = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: 1,
                outputChannels: CNN_CONV_CHANNELS,
                kernelSize: IntOrPair(CNN_KERNEL_SIZE),
                stride: IntOrPair(1),
                padding: IntOrPair(1)
            ),
            key: "conv1"
        )
        
        // Max pooling layer
        // - 2×2 window
        // - stride=2 (non-overlapping)
        _pool = ModuleInfo(
            wrappedValue: MaxPool2d(
                kernelSize: IntOrPair(CNN_POOL_SIZE),
                stride: IntOrPair(CNN_POOL_SIZE)
            ),
            key: "pool"
        )
        
        // Fully connected (dense) layer
        // - 1568 inputs (flattened feature maps)
        // - 10 outputs (digit classes)
        _fc = ModuleInfo(
            wrappedValue: Linear(CNN_FLATTEN_SIZE, 10),
            key: "fc"
        )
    }
    
    // -------------------------------------------------------------------------
    // MARK: - Forward Pass
    // -------------------------------------------------------------------------
    
    /// Forward pass: computes class logits from input images
    ///
    /// This method defines how data flows through the network.
    /// MLX will automatically track operations for gradient computation.
    ///
    /// - Parameter x: Input images of shape [N, 1, 28, 28]
    /// - Returns: Logits of shape [N, 10] (unnormalized class scores)
    ///
    /// ## Note on Logits vs Probabilities
    /// We return raw logits (not softmax) because:
    /// 1. Cross-entropy loss includes softmax internally (numerically stable)
    /// 2. At test time, argmax on logits gives same result as on probabilities
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // =====================================================================
        // Step 1: Convolutional Layer
        // =====================================================================
        // Apply 3×3 convolution with 8 filters.
        // This extracts local features like edges and corners.
        // Input:  [N, 1, 28, 28]
        // Output: [N, 8, 28, 28] (same spatial size due to padding=1)
        var h = conv1(x)
        
        // =====================================================================
        // Step 2: ReLU Activation
        // =====================================================================
        // ReLU(x) = max(0, x)
        // Non-linearity allows the network to learn complex patterns.
        // Without non-linearities, stacking layers would just be linear math.
        h = relu(h)
        
        // =====================================================================
        // Step 3: Max Pooling
        // =====================================================================
        // Take the maximum in each 2×2 window.
        // This reduces spatial dimensions and provides translation invariance.
        // Input:  [N, 8, 28, 28]
        // Output: [N, 8, 14, 14]
        h = pool(h)
        
        // =====================================================================
        // Step 4: Flatten
        // =====================================================================
        // Reshape from [N, 8, 14, 14] to [N, 1568]
        // The fully connected layer expects a 1D feature vector.
        let batchSize = h.shape[0]
        h = h.reshaped([batchSize, CNN_FLATTEN_SIZE])
        
        // =====================================================================
        // Step 5: Fully Connected Layer
        // =====================================================================
        // Map the 1568-dim feature vector to 10 class logits.
        // Output: [N, 10]
        h = fc(h)
        
        return h
    }
}

// =============================================================================
// MARK: - Training Functions
// =============================================================================

/// Computes cross-entropy loss for classification
///
/// Cross-entropy loss measures how different the predicted probability
/// distribution is from the true distribution (one-hot labels).
///
/// Formula: L = -sum(y * log(softmax(logits)))
///
/// For classification, this reduces to: L = -log(p_correct)
/// where p_correct is the predicted probability of the true class.
///
/// - Parameters:
///   - model: The CNN model
///   - images: Input images [N, 1, 28, 28]
///   - labels: True labels [N] (integers 0-9)
/// - Returns: Scalar loss value (lower is better)
public func cnnLoss(model: CNNModel, images: MLXArray, labels: MLXArray) -> MLXArray {
    // Forward pass: get logits
    let logits = model(images)
    
    // Cross-entropy loss with mean reduction
    // This function internally applies softmax for numerical stability
    return crossEntropy(logits: logits, targets: labels, reduction: .mean)
}

/// Computes accuracy on a batch
///
/// - Parameters:
///   - model: The CNN model
///   - images: Input images [N, 1, 28, 28]
///   - labels: True labels [N]
/// - Returns: Accuracy as a Float (0.0 to 1.0)
public func cnnAccuracy(model: CNNModel, images: MLXArray, labels: MLXArray) -> Float {
    // Get predictions
    let logits = model(images)
    
    // argmax gives the predicted class (highest logit)
    let predictions = argMax(logits, axis: 1)
    
    // Compare predictions to labels
    let correct = predictions .== labels
    
    // Mean of boolean array gives accuracy
    return mean(correct).item(Float.self)
}

/// Trains the CNN model for one epoch
///
/// An epoch is one complete pass through the training data.
///
/// - Parameters:
///   - model: The CNN model to train
///   - optimizer: SGD or other optimizer
///   - trainImages: Training images [N, 784]
///   - trainLabels: Training labels [N]
///   - batchSize: Number of samples per batch
/// - Returns: Average loss for the epoch
public func trainCNNEpoch(
    model: CNNModel,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    batchSize: Int
) -> Float {
    let n = trainImages.shape[0]
    var totalLoss: Float = 0
    var batchCount = 0
    
    // -------------------------------------------------------------------------
    // Create the loss-and-gradient function
    // -------------------------------------------------------------------------
    // valueAndGrad creates a function that computes BOTH:
    // 1. The loss value (for monitoring)
    // 2. The gradients (for weight updates)
    //
    // This is MLX's automatic differentiation magic!
    // No need to write backward passes manually.
    let lossAndGrad = valueAndGrad(model: model, cnnLoss)
    
    // -------------------------------------------------------------------------
    // Shuffle indices for SGD
    // -------------------------------------------------------------------------
    var indices = Array(0..<n)
    indices.shuffle()
    
    // -------------------------------------------------------------------------
    // Training loop over batches
    // -------------------------------------------------------------------------
    var start = 0
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)
        
        // Get batch data
        // Reshape to [N, 1, 28, 28] for CNN (add channel dimension)
        let batchImages = trainImages[idxArray].reshaped([-1, 1, 28, 28])
        let batchLabels = trainLabels[idxArray]
        
        // Compute loss and gradients
        let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)
        
        // Update model parameters using the optimizer
        // This applies: weights = weights - lr * gradients
        optimizer.update(model: model, gradients: grads)
        
        // Evaluate to ensure computation happens
        // MLX uses lazy evaluation, so we need to force it here
        eval(model, optimizer)
        
        totalLoss += loss.item(Float.self)
        batchCount += 1
        start = end
    }
    
    return totalLoss / Float(batchCount)
}

// =============================================================================
// MARK: - Compiled Training (Performance Optimized)
// =============================================================================

/// For faster training, use the compiled version of this function!
///
/// MLX's compile() API can fuse the forward pass, backward pass, and optimizer
/// update into a single optimized GPU kernel, providing significant speedup.
///
/// ## Compiled Training Function
/// ```swift
/// trainCNNEpochCompiled(
///     model: model,
///     optimizer: optimizer,
///     trainImages: trainImages,
///     trainLabels: trainLabels,
///     batchSize: batchSize
/// )
/// ```
///
/// **Location**: `Sources/MNISTMLX/CompiledTraining.swift`
///
/// ## Performance Benefits
/// - 1.5-2.5x faster training on MNIST-sized models
/// - Reduced memory bandwidth usage (no intermediate tensors)
/// - Better GPU utilization through kernel fusion
/// - Lower kernel launch overhead
///
/// ## Why CNNs Benefit from Compilation
/// CNNs perform many sequential operations that can be fused:
/// - Conv2d → ReLU → MaxPool (all fused into one kernel!)
/// - Linear transformations
/// - Loss computation → Backward pass → Parameter update
///
/// Each of these operations normally launches separate GPU kernels.
/// Compilation fuses them, eliminating intermediate memory allocations.
///
/// ## When to Use Compiled Training
/// - ✓ Training for many epochs (amortizes compilation cost)
/// - ✓ Larger models with many operations
/// - ✓ Production training pipelines
/// - ✗ Debugging (uncompiled is easier to debug)
/// - ✗ Single-epoch experiments
///
/// ## Example: Switching to Compiled Training
/// ```swift
/// // Uncompiled (easier to debug)
/// let loss = trainCNNEpoch(
///     model: model,
///     optimizer: optimizer,
///     trainImages: trainImages,
///     trainLabels: trainLabels,
///     batchSize: 128
/// )
///
/// // Compiled (faster, same results)
/// let loss = trainCNNEpochCompiled(
///     model: model,
///     optimizer: optimizer,
///     trainImages: trainImages,
///     trainLabels: trainLabels,
///     batchSize: 128
/// )
/// ```
///
/// See `CompiledTraining.swift` for implementation details and more information
/// about MLX's compilation API.
