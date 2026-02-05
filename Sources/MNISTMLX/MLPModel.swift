// ============================================================================
// MLPModel.swift - Multi-Layer Perceptron for MNIST using MLX Swift
// ============================================================================
//
// This file implements a simple MLP (Multi-Layer Perceptron) for classifying
// MNIST digits using Apple's MLX Swift framework.
//
// ARCHITECTURE:
//   Input: [N, 784] - Batch of flattened images
//     ↓
//   Linear: 784→512 (hidden layer)
//     ↓
//   ReLU activation
//     ↓
//   Linear: 512→10 (output layer)
//     ↓
//   Output: logits for 10 classes
//
// WHAT IS AN MLP?
//   An MLP is the simplest type of neural network (also called a "feedforward"
//   or "fully connected" network). Each neuron in one layer connects to ALL
//   neurons in the next layer.
//
//   While simpler than CNNs, MLPs can still achieve ~97% on MNIST!
//
// WHY USE AN MLP?
//   - Simple to understand and implement
//   - Good baseline for comparison
//   - Fast training (fewer operations than CNN)
//
// LIMITATIONS:
//   - No spatial awareness (treats image as flat vector)
//   - Many parameters (784 × 512 = 401,408 just for layer 1!)
//   - Doesn't generalize as well to transformed images
//
// ============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

// =============================================================================
// MARK: - MLP Architecture Constants
// =============================================================================

/// Number of input features (28 × 28 = 784 pixels)
private let MLP_INPUT_SIZE = 784

/// Number of hidden layer neurons
/// More neurons = more capacity but also more parameters and slower training
/// 512 is a good balance for MNIST
private let MLP_HIDDEN_SIZE = 512

/// Number of output classes (digits 0-9)
private let MLP_OUTPUT_SIZE = 10

// =============================================================================
// MARK: - MLP Model Definition
// =============================================================================

/// Multi-Layer Perceptron for MNIST classification
///
/// This is the simplest architecture that can learn non-linear patterns:
/// - Linear layer 1: Projects input to hidden representation
/// - ReLU: Adds non-linearity
/// - Linear layer 2: Maps hidden to output classes
///
/// ## Mathematical View
/// ```
/// hidden = ReLU(input @ W1 + b1)
/// logits = hidden @ W2 + b2
/// ```
///
/// ## Why ReLU?
/// ReLU (Rectified Linear Unit) = max(0, x)
/// - Simple and fast to compute
/// - Helps with vanishing gradient problem
/// - Introduces non-linearity (without it, the network is just linear!)
///
/// ## Example Usage
/// ```swift
/// let model = MLPModel()
/// let images = MLXArray(...)  // [32, 784]
/// let logits = model(images)  // [32, 10]
/// ```
public class MLPModel: Module {
    // -------------------------------------------------------------------------
    // MARK: - Layers
    // -------------------------------------------------------------------------
    
    /// First linear layer (hidden layer)
    ///
    /// Linear layers compute: output = input @ weights + biases
    /// - Input: 784 features (flattened 28×28 image)
    /// - Output: 512 hidden activations
    ///
    /// The weights matrix has shape [784, 512] = 401,408 parameters!
    @ModuleInfo(key: "hidden") var hidden: Linear
    
    /// Second linear layer (output layer)
    ///
    /// Maps the 512-dimensional hidden representation to 10 class logits.
    /// - Input: 512 hidden features
    /// - Output: 10 logits (one per digit class)
    @ModuleInfo(key: "output") var output: Linear
    
    // -------------------------------------------------------------------------
    // MARK: - Initialization
    // -------------------------------------------------------------------------
    
    /// Creates a new MLP model with randomly initialized weights
    ///
    /// MLX uses Xavier/Glorot initialization by default:
    /// - Weights are initialized from a uniform distribution
    /// - Range is scaled based on layer input/output sizes
    /// - This helps maintain similar variance across layers
    public override init() {
        // Hidden layer: 784 → 512
        _hidden = ModuleInfo(
            wrappedValue: Linear(MLP_INPUT_SIZE, MLP_HIDDEN_SIZE),
            key: "hidden"
        )
        
        // Output layer: 512 → 10
        _output = ModuleInfo(
            wrappedValue: Linear(MLP_HIDDEN_SIZE, MLP_OUTPUT_SIZE),
            key: "output"
        )
    }
    
    /// Initialize with custom hidden size
    ///
    /// - Parameter hiddenSize: Number of neurons in the hidden layer
    public init(hiddenSize: Int) {
        _hidden = ModuleInfo(
            wrappedValue: Linear(MLP_INPUT_SIZE, hiddenSize),
            key: "hidden"
        )
        _output = ModuleInfo(
            wrappedValue: Linear(hiddenSize, MLP_OUTPUT_SIZE),
            key: "output"
        )
    }
    
    // -------------------------------------------------------------------------
    // MARK: - Forward Pass
    // -------------------------------------------------------------------------
    
    /// Forward pass: computes class logits from input images
    ///
    /// - Parameter x: Input images of shape [N, 784] (flattened)
    /// - Returns: Logits of shape [N, 10]
    ///
    /// ## Step-by-Step Computation
    /// 1. Linear transform: z1 = x @ W1 + b1
    /// 2. Activation: a1 = ReLU(z1) = max(0, z1)
    /// 3. Linear transform: logits = a1 @ W2 + b2
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // =====================================================================
        // Step 1: Hidden Layer
        // =====================================================================
        // Linear transformation: projects 784-dim input to 512-dim hidden space.
        // Mathematically: z = x @ W + b where W is [784, 512] and b is [512]
        var h = hidden(x)
        
        // =====================================================================
        // Step 2: ReLU Activation
        // =====================================================================
        // ReLU(z) = max(0, z)
        //
        // WHY NON-LINEARITY IS ESSENTIAL:
        // Without ReLU (or another activation), the network would be:
        //   y = (x @ W1 + b1) @ W2 + b2
        //     = x @ (W1 @ W2) + (b1 @ W2 + b2)
        //     = x @ W' + b'
        //
        // This is just another linear transformation! No matter how many layers,
        // without non-linearities the network can only learn linear functions.
        //
        // ReLU breaks this linearity by "zeroing out" negative values.
        h = relu(h)
        
        // =====================================================================
        // Step 3: Output Layer
        // =====================================================================
        // Maps the 512-dim hidden representation to 10 class scores.
        // These are called "logits" (unnormalized log-probabilities).
        h = output(h)
        
        return h
    }
}

// =============================================================================
// MARK: - Training Functions
// =============================================================================

/// Computes cross-entropy loss for MLP
///
/// - Parameters:
///   - model: The MLP model
///   - images: Input images [N, 784] (already flattened)
///   - labels: True labels [N]
/// - Returns: Scalar loss value
public func mlpLoss(model: MLPModel, images: MLXArray, labels: MLXArray) -> MLXArray {
    let logits = model(images)
    return crossEntropy(logits: logits, targets: labels, reduction: .mean)
}

/// Computes accuracy on a batch
///
/// - Parameters:
///   - model: The MLP model
///   - images: Input images [N, 784]
///   - labels: True labels [N]
/// - Returns: Accuracy as a Float (0.0 to 1.0)
public func mlpAccuracy(model: MLPModel, images: MLXArray, labels: MLXArray) -> Float {
    let logits = model(images)
    let predictions = argMax(logits, axis: 1)
    let correct = predictions .== labels
    return mean(correct).item(Float.self)
}

/// Trains the MLP model for one epoch
///
/// - Parameters:
///   - model: The MLP model to train
///   - optimizer: SGD or other optimizer
///   - trainImages: Training images [N, 784]
///   - trainLabels: Training labels [N]
///   - batchSize: Number of samples per batch
/// - Returns: Average loss for the epoch
public func trainMLPEpoch(
    model: MLPModel,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    batchSize: Int
) -> Float {
    let n = trainImages.shape[0]
    var totalLoss: Float = 0
    var batchCount = 0
    
    // -------------------------------------------------------------------------
    // Automatic Differentiation Setup
    // -------------------------------------------------------------------------
    // valueAndGrad is the magic of modern ML frameworks!
    //
    // In the original mnist_mlp.swift, we had to write:
    // - fcBackward: 35 lines to compute gradients manually
    // - reluBackward: gradient masking for ReLU
    // - Manual chain rule application
    //
    // MLX does all of this automatically by tracing the forward pass!
    let lossAndGrad = valueAndGrad(model: model, mlpLoss)
    
    // -------------------------------------------------------------------------
    // Shuffle for Stochastic Gradient Descent
    // -------------------------------------------------------------------------
    // SGD theory says we should present samples in random order each epoch.
    // This helps the optimizer escape local minima and generalizes better.
    var indices = Array(0..<n)
    indices.shuffle()
    
    // -------------------------------------------------------------------------
    // Mini-batch Training Loop
    // -------------------------------------------------------------------------
    var start = 0
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)
        
        // Get batch data (no reshape needed for MLP - already flat)
        let batchImages = trainImages[idxArray]
        let batchLabels = trainLabels[idxArray]
        
        // =====================================================================
        // The Training Step (Loss + Gradients + Update)
        // =====================================================================
        // This single line replaces hundreds of lines in the original code!
        //
        // 1. Forward pass: compute predictions
        // 2. Loss: compare predictions to labels
        // 3. Backward pass: compute gradients via chain rule
        let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)
        
        // Update weights: w = w - lr * grad
        optimizer.update(model: model, gradients: grads)
        
        // Force evaluation (MLX is lazy)
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
/// trainMLPEpochCompiled(
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
/// - Reduced memory bandwidth usage
/// - Better GPU utilization through kernel fusion
/// - Lower kernel launch overhead
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
/// let loss = trainMLPEpoch(
///     model: model,
///     optimizer: optimizer,
///     trainImages: trainImages,
///     trainLabels: trainLabels,
///     batchSize: 128
/// )
///
/// // Compiled (faster, same results)
/// let loss = trainMLPEpochCompiled(
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
