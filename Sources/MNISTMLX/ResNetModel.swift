// ============================================================================
// ResNetModel.swift - Residual Neural Network for MNIST using MLX Swift
// ============================================================================
//
// This file implements a simplified ResNet (Residual Network) for classifying
// MNIST digits using Apple's MLX Swift framework.
//
// ARCHITECTURE:
//   Input: [N, 1, 28, 28] - Batch of grayscale images
//     ↓
//   Initial Conv2d: 1→16 channels, 3×3 kernel, padding=1 → [N, 16, 28, 28]
//     ↓
//   ResidualBlock × N: Skip connections enable deep networks
//     ↓
//   GlobalAveragePool: [N, C, H, W] → [N, C]
//     ↓
//   Linear: C→10 → [N, 10]
//     ↓
//   Output: logits for 10 classes
//
// WHY RESIDUAL CONNECTIONS?
//   - Enable training of very deep networks (100+ layers)
//   - Solve vanishing gradient problem through identity shortcuts
//   - Allow gradients to flow directly through skip connections
//   - Learn residual mappings F(x) rather than direct mappings H(x)
//   - Identity mapping: H(x) = F(x) + x, where F(x) learns the residual
//
// VANISHING GRADIENT PROBLEM:
//   In deep networks, gradients can become exponentially small during
//   backpropagation, making it hard to train early layers. ResNet's skip
//   connections provide a direct path for gradients to flow backwards,
//   bypassing potentially problematic layers.
//
// ============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

// =============================================================================
// MARK: - ResNet Architecture Constants
// =============================================================================

/// Number of channels after initial convolution
///
/// This determines the base width of the network. All residual blocks in this
/// simplified implementation maintain this channel dimension.
///
/// We use 16 for MNIST (sufficient for simple digits). Production ResNets use:
/// - ResNet-18/34: starts at 64 channels
/// - ResNet-50/101/152: starts at 64, uses bottleneck blocks
///
/// ## Impact on Model Capacity
///
/// More channels = more features the network can learn, but also:
/// - More parameters (quadratic growth in convolution weights)
/// - More computation (each conv layer does outputChannels × inputChannels operations)
/// - Higher memory usage
///
/// For MNIST (28×28 grayscale digits), 16 channels provides plenty of capacity.
private let RESNET_INITIAL_CHANNELS = 16

/// Convolution kernel size (3×3 is standard for residual blocks)
///
/// 3×3 kernels are the sweet spot for CNNs:
/// - Large enough to capture local patterns (edges, corners, textures)
/// - Small enough to keep parameter count manageable
/// - Can build large receptive fields by stacking multiple layers
///
/// Receptive field growth: Stacking N layers of 3×3 convolutions gives an
/// effective receptive field of (2N+1) × (2N+1). With 3 residual blocks
/// (6 conv layers), we can see patterns up to 13×13 pixels.
private let RESNET_KERNEL_SIZE = 3

// =============================================================================
// MARK: - Residual Block Definition
// =============================================================================

/// Residual Block: The fundamental building block of ResNet
///
/// A residual block learns a residual mapping F(x) and adds it to the input:
///   output = F(x) + x
///
/// This is equivalent to learning:
///   output = F(x) + identity(x)
///
/// ## Architecture of F(x):
/// ```
/// Input x
///   ↓
/// Conv2d (3×3)
///   ↓
/// BatchNorm
///   ↓
/// ReLU
///   ↓
/// Conv2d (3×3)
///   ↓
/// BatchNorm
///   ↓
/// Add skip connection: + x
///   ↓
/// ReLU
///   ↓
/// Output
/// ```
///
/// ## Dimension Matching
///
/// When input and output channels differ, we need a projection shortcut:
/// - If inputChannels == outputChannels: use identity skip connection
/// - If inputChannels != outputChannels: use 1×1 convolution to project x
///
/// ## Why This Works
///
/// 1. **Gradient Flow**: Gradients can flow through skip connection (∂output/∂x = 1 + ∂F/∂x)
/// 2. **Identity Mapping**: If F(x) ≈ 0, block learns identity (no transformation needed)
/// 3. **Easier Optimization**: Learning residuals F(x) is easier than learning H(x) directly
///
/// ## Example Usage
/// ```swift
/// let block = ResidualBlock(inputChannels: 16, outputChannels: 16)
/// let x = MLXArray(...)  // [N, 16, 28, 28]
/// let output = block(x)  // [N, 16, 28, 28]
/// ```
public class ResidualBlock: Module {
    // -------------------------------------------------------------------------
    // MARK: - Layers
    // -------------------------------------------------------------------------

    /// First convolutional layer in the residual path
    ///
    /// Conv2d parameters:
    /// - kernelSize: 3×3 (standard for residual blocks)
    /// - stride: 1 (preserve spatial dimensions)
    /// - padding: 1 (keep same spatial size)
    @ModuleInfo(key: "conv1") var conv1: Conv2d

    /// Batch normalization after first convolution
    ///
    /// BatchNorm normalizes activations to have mean=0, variance=1.
    /// Benefits:
    /// - Stabilizes training by reducing internal covariate shift
    /// - Allows higher learning rates
    /// - Acts as regularization (slight noise from batch statistics)
    @ModuleInfo(key: "bn1") var bn1: BatchNorm

    /// Second convolutional layer in the residual path
    ///
    /// This completes the F(x) transformation before adding the skip connection.
    @ModuleInfo(key: "conv2") var conv2: Conv2d

    /// Batch normalization after second convolution
    ///
    /// Applied before the skip connection addition.
    @ModuleInfo(key: "bn2") var bn2: BatchNorm

    /// Optional projection shortcut for dimension matching
    ///
    /// When input and output channels differ, we need to project the skip
    /// connection to match dimensions. This uses a 1×1 convolution.
    ///
    /// Example: If input is [N, 16, 28, 28] but output should be [N, 32, 28, 28],
    /// the skip connection needs to be projected from 16 to 32 channels.
    @ModuleInfo(key: "shortcut") var shortcut: Conv2d?

    // -------------------------------------------------------------------------
    // MARK: - Initialization
    // -------------------------------------------------------------------------

    /// Creates a new residual block
    ///
    /// - Parameters:
    ///   - inputChannels: Number of input channels
    ///   - outputChannels: Number of output channels
    ///
    /// If inputChannels != outputChannels, a 1×1 projection convolution is
    /// created for the skip connection to enable dimension matching.
    ///
    /// ## Dimension Matching Problem
    ///
    /// The skip connection requires adding the input to the output:
    ///   output = F(x) + x
    ///
    /// This only works if F(x) and x have the same shape!
    ///
    /// When channel dimensions differ, we need a projection:
    ///   output = F(x) + projection(x)
    ///
    /// ## Two Types of Skip Connections
    ///
    /// 1. **Identity shortcut** (inputChannels == outputChannels):
    ///    - Direct addition: output = F(x) + x
    ///    - No parameters
    ///    - Maximum gradient flow
    ///    - Preferred when possible
    ///
    /// 2. **Projection shortcut** (inputChannels != outputChannels):
    ///    - 1×1 convolution projects x to match F(x) dimensions
    ///    - Adds learnable parameters
    ///    - Still provides gradient highway (better than no skip connection)
    ///    - Used when changing channel dimensions between network stages
    public init(inputChannels: Int, outputChannels: Int) {
        // =====================================================================
        // First Convolutional Layer
        // =====================================================================
        // Transform input channels to output channels using 3×3 kernel.
        //
        // This is where the channel dimension change happens in the residual
        // path. If inputChannels != outputChannels, this layer learns how to
        // map between different feature spaces.
        //
        // Parameters:
        // - kernelSize: 3×3 (captures local spatial context)
        // - stride: 1 (preserve spatial dimensions)
        // - padding: 1 (28×28 → 28×28, same spatial size)
        _conv1 = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: IntOrPair(RESNET_KERNEL_SIZE),
                stride: IntOrPair(1),
                padding: IntOrPair(1)
            ),
            key: "conv1"
        )

        // =====================================================================
        // First Batch Normalization
        // =====================================================================
        // Normalize the outputChannels feature maps.
        // Essential for stable training in deep networks.
        _bn1 = ModuleInfo(
            wrappedValue: BatchNorm(featureCount: outputChannels),
            key: "bn1"
        )

        // =====================================================================
        // Second Convolutional Layer
        // =====================================================================
        // Refine features while maintaining channel dimension.
        //
        // Both input and output are outputChannels (no dimension change here).
        // This completes the residual function F(x).
        _conv2 = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: outputChannels,
                outputChannels: outputChannels,
                kernelSize: IntOrPair(RESNET_KERNEL_SIZE),
                stride: IntOrPair(1),
                padding: IntOrPair(1)
            ),
            key: "conv2"
        )

        // =====================================================================
        // Second Batch Normalization
        // =====================================================================
        // Normalize before adding the skip connection.
        // This ensures F(x) has controlled magnitude.
        _bn2 = ModuleInfo(
            wrappedValue: BatchNorm(featureCount: outputChannels),
            key: "bn2"
        )

        // =====================================================================
        // Skip Connection: Dimension Matching
        // =====================================================================
        // Determine whether we need a projection shortcut.

        if inputChannels != outputChannels {
            // =================================================================
            // Case 1: Projection Shortcut (Dimension Matching Required)
            // =================================================================
            // Use 1×1 convolution to match dimensions.
            //
            // Example scenario: inputChannels=16, outputChannels=32
            // - Residual path F(x): [N, 16, H, W] → [N, 32, H, W]
            // - Identity path x: [N, 16, H, W]
            // - Problem: Can't add [N, 32, H, W] + [N, 16, H, W]
            // - Solution: Project identity: [N, 16, H, W] → [N, 32, H, W]
            //
            // 1×1 Convolution properties:
            // - kernelSize: 1×1 (pointwise, no spatial mixing)
            // - Operates independently on each spatial location
            // - Learns a linear combination of input channels
            // - Essentially a learned projection matrix per pixel
            // - padding: 0 (1×1 kernel doesn't need padding)
            //
            // This projection shortcut is also trained via backpropagation,
            // so it learns the optimal way to map input to output space.
            _shortcut = ModuleInfo(
                wrappedValue: Conv2d(
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    kernelSize: IntOrPair(1),
                    stride: IntOrPair(1),
                    padding: IntOrPair(0)
                ),
                key: "shortcut"
            )
        } else {
            // =================================================================
            // Case 2: Identity Shortcut (No Transformation Needed)
            // =================================================================
            // Input and output dimensions match, so we can use direct addition.
            //
            // This is the "pure" identity mapping from the ResNet paper:
            //   output = F(x) + x
            //
            // Advantages:
            // - No additional parameters
            // - Maximum gradient flow (no learned transformation to attenuate gradients)
            // - Simpler and more elegant
            //
            // This is why ResNets often keep channel dimensions constant within
            // a stage, only changing between stages (e.g., 64→128→256→512).
            _shortcut = ModuleInfo(wrappedValue: nil, key: "shortcut")
        }

        super.init()
    }

    // -------------------------------------------------------------------------
    // MARK: - Forward Pass
    // -------------------------------------------------------------------------

    /// Forward pass through the residual block
    ///
    /// Computes: output = F(x) + x, where F(x) is the residual function.
    ///
    /// - Parameter x: Input tensor of shape [N, inputChannels, H, W]
    /// - Returns: Output tensor of shape [N, outputChannels, H, W]
    ///
    /// ## The Skip Connection Mechanism
    ///
    /// The key innovation of ResNet is the skip connection (also called shortcut):
    ///   output = F(x) + x
    ///
    /// Where:
    /// - F(x) is the residual mapping learned by the block's layers
    /// - x is the identity (input passed through unchanged)
    ///
    /// ## Why Skip Connections Solve Vanishing Gradients
    ///
    /// During backpropagation, the gradient flows as:
    ///   ∂loss/∂x = ∂loss/∂output × (1 + ∂F/∂x)
    ///
    /// The "+1" term is crucial:
    /// - Even if ∂F/∂x → 0 (gradients vanish through the residual path)
    /// - The gradient still flows through the identity path with magnitude 1
    /// - This allows training of very deep networks (100+ layers)
    ///
    /// In traditional deep networks without skip connections:
    ///   ∂loss/∂x = ∂loss/∂output × (∂F₁/∂x × ∂F₂/∂F₁ × ... × ∂Fₙ/∂Fₙ₋₁)
    ///
    /// Each multiplication can reduce gradient magnitude, leading to exponential
    /// decay (vanishing gradients) or growth (exploding gradients).
    ///
    /// ## Identity Mapping
    ///
    /// When the optimal transformation is close to identity (output ≈ input),
    /// the network can simply learn F(x) ≈ 0, making the block a no-op.
    /// This is easier than learning H(x) = x directly through weight matrices.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // =====================================================================
        // Save Identity for Skip Connection
        // =====================================================================
        // Store the input for later addition. This creates the "shortcut path"
        // that allows gradients to bypass the residual function F(x).
        let identity = x

        // =====================================================================
        // Residual Path F(x): First Transformation Block
        // =====================================================================
        // The residual function F(x) learns what to ADD to the input,
        // rather than learning the full output directly.

        // First convolution: extract features with 3×3 kernel
        // This learns local patterns while preserving spatial structure
        var out = conv1(x)

        // Batch normalization: normalize activations
        // - Mean = 0, variance = 1 across the batch
        // - Reduces internal covariate shift
        // - Allows higher learning rates
        // - Critical for training deep networks
        out = bn1(out)

        // ReLU activation: introduce non-linearity
        // ReLU(x) = max(0, x)
        // - Allows the network to learn complex, non-linear functions
        // - Computationally efficient (just a threshold operation)
        out = relu(out)

        // =====================================================================
        // Residual Path F(x): Second Transformation Block
        // =====================================================================

        // Second convolution: further refine features
        // This completes the residual function F(x) = conv2(relu(bn1(conv1(x))))
        out = conv2(out)

        // Batch normalization before skip connection addition
        // Normalizing here ensures the residual F(x) has controlled magnitude
        out = bn2(out)

        // =====================================================================
        // Skip Connection: The Heart of ResNet
        // =====================================================================
        // Add the identity to the residual: output = F(x) + x
        //
        // This single addition operation is what makes ResNets trainable at
        // extreme depths. It provides a highway for gradients to flow backwards.

        // Check if we need dimension matching
        if let shortcut = shortcut {
            // ================================================================
            // Case 1: Dimensions Don't Match
            // ================================================================
            // When inputChannels != outputChannels, we need to project the
            // identity to match the output dimensions.
            //
            // Example: If F(x) has shape [N, 32, H, W] but identity has
            // shape [N, 16, H, W], we can't add them directly.
            //
            // Solution: Use 1×1 convolution to project identity from 16→32 channels
            // - 1×1 kernel: pointwise operation (no spatial mixing)
            // - Linear transformation per spatial location
            // - Learnable projection (not just zero-padding)
            //
            // This projection shortcut is also trained via backpropagation.
            out = out + shortcut(identity)
        } else {
            // ================================================================
            // Case 2: Dimensions Match (Identity Mapping)
            // ================================================================
            // When inputChannels == outputChannels, we can directly add:
            //   output = F(x) + x
            //
            // This is the "pure" identity mapping from the original ResNet paper.
            // No learnable parameters in the skip connection—just addition.
            //
            // Benefits:
            // - Maximum gradient flow (no transformation in the shortcut)
            // - Fewer parameters
            // - Easier optimization (identity is always available)
            out = out + identity
        }

        // =====================================================================
        // Final Activation
        // =====================================================================

        // ReLU after the skip connection addition
        //
        // Architecture note: The original ResNet paper applies ReLU AFTER
        // addition. Later variants (ResNet-v2) apply it BEFORE for improved
        // gradient flow, but we follow the original design here.
        //
        // This activation operates on the combined features: relu(F(x) + x)
        out = relu(out)

        return out
    }
}

// =============================================================================
// MARK: - ResNet Model Definition
// =============================================================================

/// Residual Neural Network for MNIST classification
///
/// This model implements a simplified ResNet architecture using residual blocks
/// with skip connections. The architecture enables training deeper networks by
/// solving the vanishing gradient problem.
///
/// ## Architecture Details
///
/// The architecture follows the ResNet pattern:
/// 1. **Initial Convolution**: Extract initial features from input
/// 2. **Residual Blocks**: Stack of blocks with skip connections
/// 3. **Global Average Pooling**: Reduce spatial dimensions
/// 4. **Linear**: Map features to class predictions
///
/// ## Why ResNet for MNIST?
///
/// While MNIST doesn't require a deep network, this implementation demonstrates:
/// - How residual connections enable deeper architectures
/// - Batch normalization for training stability
/// - Global average pooling as alternative to flattening
///
/// ## Example Usage
/// ```swift
/// let model = ResNetModel(numBlocks: 3)
/// let images = MLXArray(...)  // [32, 1, 28, 28]
/// let logits = model(images)  // [32, 10]
/// ```
public class ResNetModel: Module {
    // -------------------------------------------------------------------------
    // MARK: - Layers
    // -------------------------------------------------------------------------

    /// Initial convolutional layer: 1 input channel → 16 output channels
    ///
    /// This layer transforms grayscale MNIST images into feature maps.
    /// Unlike the residual blocks, this doesn't have a skip connection
    /// because it changes the channel dimension significantly (1→16).
    @ModuleInfo(key: "conv1") var conv1: Conv2d

    /// Batch normalization after initial convolution
    ///
    /// Normalizes the initial feature maps for stable training.
    @ModuleInfo(key: "bn1") var bn1: BatchNorm

    /// Stack of residual blocks
    ///
    /// These blocks form the core of the network, each with skip connections
    /// that enable gradient flow through the entire network.
    ///
    /// We use an array to store multiple blocks, allowing flexible depth.
    @ModuleInfo(key: "layers") var layers: [ResidualBlock]

    /// Final linear layer: maps from channels to class logits
    ///
    /// After global average pooling reduces spatial dimensions to 1×1,
    /// this layer maps the channel features to 10 class scores.
    @ModuleInfo(key: "fc") var fc: Linear

    // -------------------------------------------------------------------------
    // MARK: - Initialization
    // -------------------------------------------------------------------------

    /// Creates a new ResNet model with specified number of residual blocks
    ///
    /// - Parameter numBlocks: Number of residual blocks to stack (default: 3)
    ///
    /// For MNIST, 3 blocks is sufficient. Deeper networks (e.g., ResNet-50)
    /// would use many more blocks with varying channel dimensions.
    ///
    /// ## Weight Initialization
    ///
    /// MLX uses Xavier/Glorot initialization by default for Conv2d and Linear layers:
    /// - Weights drawn from uniform distribution with variance scaled by fan-in/fan-out
    /// - Helps maintain gradient magnitude through layers
    /// - Critical for training deep networks
    ///
    /// ## Architecture Comparison
    ///
    /// This simplified ResNet uses uniform channel dimensions:
    /// - All residual blocks: 16 channels
    /// - No downsampling (maintains 28×28 spatial size)
    ///
    /// Production ResNets (e.g., ResNet-50) use a different structure:
    /// - Progressive channel increases: 64 → 128 → 256 → 512
    /// - Spatial downsampling via stride-2 convolutions
    /// - Bottleneck blocks (1×1 → 3×3 → 1×1) for efficiency
    public init(numBlocks: Int = 3) {
        // =====================================================================
        // Initial Convolution Layer
        // =====================================================================
        // First layer transforms grayscale input to feature maps.
        //
        // Parameters:
        // - inputChannels: 1 (grayscale MNIST images)
        // - outputChannels: 16 (base network width)
        // - kernelSize: 3×3 (standard for feature extraction)
        // - stride: 1 (preserve spatial dimensions)
        // - padding: 1 (keep 28×28 → 28×28)
        //
        // This layer doesn't use a skip connection because it's doing
        // significant dimensionality change (1→16 channels).
        _conv1 = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: 1,
                outputChannels: RESNET_INITIAL_CHANNELS,
                kernelSize: IntOrPair(RESNET_KERNEL_SIZE),
                stride: IntOrPair(1),
                padding: IntOrPair(1)
            ),
            key: "conv1"
        )

        // =====================================================================
        // Batch Normalization after Initial Conv
        // =====================================================================
        // Normalizes the 16 feature maps to have mean=0, variance=1.
        //
        // BatchNorm parameters:
        // - featureCount: 16 (number of channels to normalize)
        // - Learnable parameters: gamma (scale) and beta (shift) per channel
        //
        // Benefits:
        // - Reduces internal covariate shift
        // - Allows higher learning rates
        // - Acts as regularization
        // - Essential for training deep ResNets
        _bn1 = ModuleInfo(
            wrappedValue: BatchNorm(featureCount: RESNET_INITIAL_CHANNELS),
            key: "bn1"
        )

        // =====================================================================
        // Stack of Residual Blocks
        // =====================================================================
        // Create an array of residual blocks, each with skip connections.
        //
        // All blocks maintain the same dimensions:
        // - inputChannels: 16
        // - outputChannels: 16
        // - Spatial size: 28×28 (no downsampling)
        //
        // Since input and output channels match, these blocks use identity
        // skip connections (no projection shortcut needed).
        //
        // Design note: In deeper ResNets (ResNet-34, ResNet-50), the network
        // is divided into stages with increasing channels:
        //   Stage 1: 64 channels
        //   Stage 2: 128 channels (first block needs projection shortcut)
        //   Stage 3: 256 channels (first block needs projection shortcut)
        //   Stage 4: 512 channels (first block needs projection shortcut)
        //
        // We keep it simple here with uniform 16 channels throughout.
        var blockArray: [ResidualBlock] = []
        for _ in 0..<numBlocks {
            blockArray.append(
                ResidualBlock(
                    inputChannels: RESNET_INITIAL_CHANNELS,
                    outputChannels: RESNET_INITIAL_CHANNELS
                )
            )
        }

        // Store blocks in an array that MLX can track
        _layers = ModuleInfo(wrappedValue: blockArray, key: "layers")

        // =====================================================================
        // Final Classification Layer
        // =====================================================================
        // Linear layer maps from feature vector to class logits.
        //
        // After global average pooling reduces [N, 16, 28, 28] to [N, 16],
        // this layer produces [N, 10] class scores.
        //
        // Parameters:
        // - Input size: 16 (number of channels after pooling)
        // - Output size: 10 (number of digit classes: 0-9)
        // - Trainable parameters: 16×10 + 10 = 170 (weights + biases)
        //
        // This is much smaller than if we flattened [N, 16, 28, 28] → [N, 12544]
        // and used a dense layer, which would require 12544×10 = 125,440 parameters!
        _fc = ModuleInfo(
            wrappedValue: Linear(RESNET_INITIAL_CHANNELS, 10),
            key: "fc"
        )

        super.init()
    }

    // -------------------------------------------------------------------------
    // MARK: - Forward Pass
    // -------------------------------------------------------------------------

    /// Forward pass: computes class logits from input images
    ///
    /// This method defines the data flow through the ResNet architecture.
    /// Skip connections in residual blocks allow gradients to flow backwards
    /// efficiently during training, solving the vanishing gradient problem.
    ///
    /// - Parameter x: Input images of shape [N, 1, 28, 28]
    /// - Returns: Logits of shape [N, 10] (unnormalized class scores)
    ///
    /// ## Architecture Flow
    /// ```
    /// Input [N, 1, 28, 28]
    ///   ↓ conv1 + bn1 + relu
    /// Features [N, 16, 28, 28]
    ///   ↓ residual blocks (with skip connections)
    /// Features [N, 16, 28, 28]
    ///   ↓ global average pool
    /// Features [N, 16]
    ///   ↓ fc (linear)
    /// Logits [N, 10]
    /// ```
    ///
    /// ## How Skip Connections Enable Deep Networks
    ///
    /// Each residual block computes: h = F(h) + h
    ///
    /// This creates multiple gradient paths during backpropagation:
    /// 1. Through the residual function F (learns transformations)
    /// 2. Through the identity shortcut (provides direct gradient highway)
    ///
    /// The gradient at the input becomes:
    ///   ∂loss/∂x = ∂loss/∂output × (1 + ∂F₁/∂x) × (1 + ∂F₂/∂F₁) × ...
    ///
    /// The "+1" terms prevent gradient vanishing, unlike traditional networks:
    ///   ∂loss/∂x = ∂loss/∂output × ∂F₁/∂x × ∂F₂/∂F₁ × ...
    ///   (can vanish if any ∂F/∂x << 1)
    ///
    /// ## Note on Logits vs Probabilities
    /// We return raw logits (not softmax) because:
    /// 1. Cross-entropy loss includes softmax internally (numerically stable)
    /// 2. At test time, argmax on logits gives same result as on probabilities
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // =====================================================================
        // Step 1: Initial Convolution Layer
        // =====================================================================
        // Transform grayscale images to feature maps.
        // This layer extracts low-level features (edges, corners, textures).
        //
        // Unlike residual blocks, this layer doesn't have a skip connection
        // because it dramatically changes dimensions (1 channel → 16 channels),
        // and we want to learn a meaningful initial representation.
        //
        // Input:  [N, 1, 28, 28] (grayscale images)
        // Output: [N, 16, 28, 28] (16 feature maps)
        var h = conv1(x)

        // Batch normalization: stabilize training
        // Normalizes activations to mean=0, std=1 across the batch
        h = bn1(h)

        // ReLU activation: introduce non-linearity
        // ReLU(x) = max(0, x)
        // Enables learning of complex patterns
        h = relu(h)

        // =====================================================================
        // Step 2: Stack of Residual Blocks
        // =====================================================================
        // Pass through multiple residual blocks, each with skip connections.
        // Shape remains [N, 16, 28, 28] throughout (no downsampling).
        //
        // Each block computes: h_new = F(h_old) + h_old
        //
        // WHY THIS WORKS:
        // - If the optimal mapping is identity (h_new = h_old), the block
        //   can learn F(h) ≈ 0, which is easier than learning identity through
        //   weight matrices
        // - If the optimal mapping requires transformation, F(h) can learn it
        // - Gradients always have a direct path backwards through the identity
        //
        // GRADIENT FLOW:
        // During backpropagation, gradients flow through TWO paths per block:
        // 1. Through the skip connection (identity): unchanged magnitude
        // 2. Through the residual layers F: may be amplified or reduced
        //
        // This dual-path design prevents vanishing gradients in deep networks.
        for block in layers {
            h = block(h)
        }

        // =====================================================================
        // Step 3: Global Average Pooling
        // =====================================================================
        // Reduce spatial dimensions to a single value per channel.
        // This is a modern alternative to flattening + dense layers.
        //
        // Operation: For each of the 16 channels, average all 28×28 pixels
        //   output[n, c] = mean(input[n, c, :, :])
        //
        // mean(h, axes: [2, 3]) averages over:
        // - axis 2: height (28 pixels)
        // - axis 3: width (28 pixels)
        //
        // Input:  [N, 16, 28, 28] = N × 16 × 784 values
        // Output: [N, 16] = N × 16 values
        //
        // ADVANTAGES over flattening:
        // - Fewer parameters: 16×10 vs 784×10 (if we flattened 28×28)
        // - Translation invariance: averaging makes position less important
        // - Regularization: less prone to overfitting
        // - Forces each channel to learn a meaningful feature (since the
        //   entire spatial extent is averaged into a single value)
        //
        // This technique was popularized by Network-in-Network (NIN) and
        // is now standard in modern CNNs and ResNets.
        h = mean(h, axes: [2, 3])

        // =====================================================================
        // Step 4: Classification Layer
        // =====================================================================
        // Map the 16-dimensional feature vector to 10 class logits.
        //
        // Linear layer computes: output = h @ weights + bias
        // - Input:  [N, 16] feature vector per image
        // - Weights: [16, 10] weight matrix
        // - Bias:   [10] bias vector
        // - Output: [N, 10] logits (unnormalized class scores)
        //
        // Each output logit represents the "score" for one digit class (0-9).
        // Higher score = model is more confident that digit belongs to that class.
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
///   - model: The ResNet model
///   - images: Input images [N, 1, 28, 28]
///   - labels: True labels [N] (integers 0-9)
/// - Returns: Scalar loss value (lower is better)
public func resnetLoss(model: ResNetModel, images: MLXArray, labels: MLXArray) -> MLXArray {
    // Forward pass: get logits
    let logits = model(images)

    // Cross-entropy loss with mean reduction
    // This function internally applies softmax for numerical stability
    return crossEntropy(logits: logits, targets: labels, reduction: .mean)
}

/// Computes accuracy on a batch
///
/// - Parameters:
///   - model: The ResNet model
///   - images: Input images [N, 1, 28, 28]
///   - labels: True labels [N]
/// - Returns: Accuracy as a Float (0.0 to 1.0)
public func resnetAccuracy(model: ResNetModel, images: MLXArray, labels: MLXArray) -> Float {
    // Get predictions
    let logits = model(images)

    // argmax gives the predicted class (highest logit)
    let predictions = argMax(logits, axis: 1)

    // Compare predictions to labels
    let correct = predictions .== labels

    // Mean of boolean array gives accuracy
    return mean(correct).item(Float.self)
}

/// Trains the ResNet model for one epoch
///
/// An epoch is one complete pass through the training data.
/// This function performs mini-batch SGD with automatic differentiation.
///
/// - Parameters:
///   - model: The ResNet model to train
///   - optimizer: SGD or other optimizer
///   - trainImages: Training images [N, 784]
///   - trainLabels: Training labels [N]
///   - batchSize: Number of samples per batch
/// - Returns: Average loss for the epoch
///
/// ## How Skip Connections Improve Training
///
/// During backpropagation, gradients flow through multiple paths:
///
/// Traditional network (no skip connections):
/// ```
/// Loss → Layer N → Layer N-1 → ... → Layer 1 → Input
/// ```
/// Gradients must flow through ALL layers sequentially.
/// Each layer can reduce gradient magnitude, leading to vanishing gradients.
///
/// ResNet (with skip connections):
/// ```
/// Loss → Block N (F + identity) → Block N-1 (F + identity) → ... → Input
///        ↓                       ↓
///        gradient path 1         gradient path 2
///        (through F)             (through skip)
/// ```
/// Gradients flow through BOTH the residual function AND the identity.
/// The identity path provides a "gradient highway" with no attenuation.
///
/// Mathematically, for a single residual block:
///   y = F(x) + x
///   ∂loss/∂x = ∂loss/∂y × (∂F/∂x + 1)
///
/// The "+1" term ensures gradients never vanish completely, enabling
/// training of networks with 100+ layers.
public func trainResNetEpoch(
    model: ResNetModel,
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
    //
    // For ResNet, this will automatically:
    // - Compute forward pass through all residual blocks
    // - Backpropagate gradients through both residual paths and skip connections
    // - Accumulate gradients correctly at skip connection addition points
    let lossAndGrad = valueAndGrad(model: model, resnetLoss)

    // -------------------------------------------------------------------------
    // Shuffle indices for SGD
    // -------------------------------------------------------------------------
    var indices = Array(0..<n)
    indices.shuffle()

    // -------------------------------------------------------------------------
    // Progress Bar Setup
    // -------------------------------------------------------------------------
    let totalBatches = (n + batchSize - 1) / batchSize
    let progressBar = ProgressBar(totalBatches: totalBatches)
    progressBar.start()

    // -------------------------------------------------------------------------
    // Training loop over batches
    // -------------------------------------------------------------------------
    var start = 0
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)

        // Get batch data
        // Reshape to [N, 1, 28, 28] for ResNet (add channel dimension)
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

        let lossValue = loss.item(Float.self)
        totalLoss += lossValue
        batchCount += 1

        // Update progress bar
        progressBar.update(batch: batchCount, loss: lossValue)

        start = end
    }

    // Finish progress bar
    progressBar.finish()

    return totalLoss / Float(batchCount)
}
