// ============================================================================
// AttentionModel.swift - Transformer-Style Attention for MNIST using MLX Swift
// ============================================================================
//
// This file implements a Vision Transformer (ViT) style model for MNIST using
// self-attention. While overkill for MNIST, this demonstrates how attention
// works for educational purposes.
//
// ARCHITECTURE:
//   Input: [N, 784] - Batch of flattened images
//     ↓
//   Patchify: Split into 7×7 grid of 4×4 patches → 49 tokens
//     ↓
//   Patch Embedding: Linear projection + positional embeddings → [N, 49, D]
//     ↓
//   Self-Attention: Q/K/V projections → attention scores → weighted sum
//     ↓
//   Feed-Forward: MLP per token (D → FF → D)
//     ↓
//   Mean Pooling: Average over tokens → [N, D]
//     ↓
//   Classifier: Linear → [N, 10]
//
// WHAT IS ATTENTION?
//   Attention allows each token to "look at" all other tokens and decide
//   which ones are relevant. For a digit image:
//   - A patch containing part of a "7" might attend to other patches
//     to figure out if it's a "7" or a "1"
//   - The attention weights are learned, not hardcoded
//
// KEY EQUATIONS:
//   Q = tokens @ W_Q  (Query: "what am I looking for?")
//   K = tokens @ W_K  (Key: "what do I contain?")
//   V = tokens @ W_V  (Value: "what information do I provide?")
//
//   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
//
// ============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

// =============================================================================
// MARK: - Attention Architecture Constants
// =============================================================================

/// Patch size (4×4 pixels per patch)
private let ATTN_PATCH_SIZE = 4

/// Number of patches in each dimension (28 / 4 = 7)
private let ATTN_GRID_SIZE = 7

/// Total number of tokens (patches) = 7 × 7 = 49
private let ATTN_SEQ_LEN = ATTN_GRID_SIZE * ATTN_GRID_SIZE

/// Dimension of each patch (4 × 4 = 16 pixels)
private let ATTN_PATCH_DIM = ATTN_PATCH_SIZE * ATTN_PATCH_SIZE

/// Model dimension (embedding size)
/// This is the hidden dimension used throughout the transformer
private let ATTN_D_MODEL = 32

/// Feed-forward hidden dimension
/// Typically 2-4x the model dimension
private let ATTN_FF_DIM = 64

// =============================================================================
// MARK: - Attention Model Definition
// =============================================================================

/// Vision Transformer-style model for MNIST classification
///
/// This model treats an image as a sequence of patches, similar to how
/// language models treat text as a sequence of tokens.
///
/// ## Why Attention for Images?
/// - Can capture long-range dependencies (patches far apart can interact)
/// - No inductive bias about locality (learns what's relevant)
/// - Same architecture works for many modalities (text, images, audio)
///
/// ## Comparison to CNN
/// - CNN: Local features → hierarchical composition
/// - Attention: All patches can interact directly from layer 1
///
/// ## Example Usage
/// ```swift
/// let model = AttentionModel()
/// let images = MLXArray(...)  // [32, 784]
/// let logits = model(images)  // [32, 10]
/// ```
public class AttentionModel: Module {
    // -------------------------------------------------------------------------
    // MARK: - Embedding Layers
    // -------------------------------------------------------------------------
    
    /// Projects 4×4 patches (16 pixels) to model dimension (32)
    ///
    /// This is like creating a "vocabulary" where each patch pattern
    /// gets mapped to a learned embedding vector.
    @ModuleInfo(key: "patch_embed") var patchEmbed: Linear
    
    /// Learnable positional embeddings
    ///
    /// Since attention is permutation-invariant, we need to tell the model
    /// WHERE each patch came from. These embeddings are learned during training.
    ///
    /// Shape: [49, 32] - one embedding per patch position
    var posEmbeddings: MLXArray
    
    // -------------------------------------------------------------------------
    // MARK: - Attention Layers
    // -------------------------------------------------------------------------
    
    /// Query projection: what is each token looking for?
    @ModuleInfo(key: "wq") var wQ: Linear
    
    /// Key projection: what does each token contain?
    @ModuleInfo(key: "wk") var wK: Linear
    
    /// Value projection: what information does each token provide?
    @ModuleInfo(key: "wv") var wV: Linear
    
    // -------------------------------------------------------------------------
    // MARK: - Feed-Forward Layers
    // -------------------------------------------------------------------------
    
    /// First feed-forward layer: D → FF
    @ModuleInfo(key: "ff1") var ff1: Linear
    
    /// Second feed-forward layer: FF → D
    @ModuleInfo(key: "ff2") var ff2: Linear
    
    // -------------------------------------------------------------------------
    // MARK: - Classification Head
    // -------------------------------------------------------------------------
    
    /// Maps pooled representation to class logits
    @ModuleInfo(key: "classifier") var classifier: Linear
    
    // -------------------------------------------------------------------------
    // MARK: - Initialization
    // -------------------------------------------------------------------------
    
    public override init() {
        // Patch embedding: 16 → 32
        _patchEmbed = ModuleInfo(
            wrappedValue: Linear(ATTN_PATCH_DIM, ATTN_D_MODEL),
            key: "patch_embed"
        )
        
        // Positional embeddings: learned, initialized small
        // We use normal distribution scaled down to not dominate early training
        posEmbeddings = MLXRandom.normal([ATTN_SEQ_LEN, ATTN_D_MODEL]) * 0.02
        
        // Q, K, V projections: 32 → 32
        _wQ = ModuleInfo(wrappedValue: Linear(ATTN_D_MODEL, ATTN_D_MODEL), key: "wq")
        _wK = ModuleInfo(wrappedValue: Linear(ATTN_D_MODEL, ATTN_D_MODEL), key: "wk")
        _wV = ModuleInfo(wrappedValue: Linear(ATTN_D_MODEL, ATTN_D_MODEL), key: "wv")
        
        // Feed-forward network: 32 → 64 → 32
        _ff1 = ModuleInfo(wrappedValue: Linear(ATTN_D_MODEL, ATTN_FF_DIM), key: "ff1")
        _ff2 = ModuleInfo(wrappedValue: Linear(ATTN_FF_DIM, ATTN_D_MODEL), key: "ff2")
        
        // Classification head: 32 → 10
        _classifier = ModuleInfo(wrappedValue: Linear(ATTN_D_MODEL, 10), key: "classifier")
    }
    
    // -------------------------------------------------------------------------
    // MARK: - Forward Pass
    // -------------------------------------------------------------------------
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.shape[0]
        
        // =====================================================================
        // Step 1: Patchify the Image
        // =====================================================================
        // Split 28×28 image into a 7×7 grid of 4×4 patches.
        // Each patch becomes a "token" in the sequence.
        //
        // Input:  [N, 784]
        // Output: [N, 49, 16] (49 patches, each with 16 pixels)
        //
        // We reshape from flat image to patches using view operations.
        let patches = patchifyImages(x, batchSize: batchSize)
        
        // =====================================================================
        // Step 2: Patch Embedding + Positional Encoding
        // =====================================================================
        // Project each patch to model dimension and add positional information.
        //
        // patches: [N, 49, 16]
        // embedded: [N, 49, 32]
        var tokens = patchEmbed(patches)
        
        // Add positional embeddings
        // posEmbeddings has shape [49, 32], broadcasts to [N, 49, 32]
        tokens = tokens + posEmbeddings
        
        // Apply ReLU activation (not typical in ViT, but matches original code)
        tokens = relu(tokens)
        
        // =====================================================================
        // Step 3: Self-Attention
        // =====================================================================
        // Each token attends to all other tokens to gather information.
        //
        // Q, K, V all have shape [N, 49, 32]
        // Attention scores: [N, 49, 49] (each token to every other token)
        // Output: [N, 49, 32] (weighted combination of values)
        //
        // The attention formula:
        //   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        //
        tokens = selfAttention(tokens, batchSize: batchSize)
        
        // =====================================================================
        // Step 4: Feed-Forward Network
        // =====================================================================
        // Per-token MLP: D → FF → D
        // This allows each token to process its information independently.
        //
        // FF networks add more parameters and non-linearity.
        var h = ff1(tokens)
        h = relu(h)
        tokens = ff2(h)
        
        // =====================================================================
        // Step 5: Mean Pooling over Tokens
        // =====================================================================
        // We need a single vector per image for classification.
        // The simplest approach: average all token representations.
        //
        // tokens: [N, 49, 32]
        // pooled: [N, 32]
        //
        // Alternative: Use a special [CLS] token (like BERT)
        let pooled = mean(tokens, axis: 1)
        
        // =====================================================================
        // Step 6: Classification
        // =====================================================================
        // Map the pooled representation to class logits.
        //
        // pooled: [N, 32]
        // logits: [N, 10]
        let logits = classifier(pooled)
        
        return logits
    }
    
    // -------------------------------------------------------------------------
    // MARK: - Helper Functions
    // -------------------------------------------------------------------------
    
    /// Converts flat images to patches
    ///
    /// - Parameters:
    ///   - x: Flat images [N, 784]
    ///   - batchSize: Number of images in batch
    /// - Returns: Patches [N, 49, 16]
    private func patchifyImages(_ x: MLXArray, batchSize: Int) -> MLXArray {
        // Reshape to 2D image: [N, 784] → [N, 28, 28]
        let images = x.reshaped([batchSize, 28, 28])
        
        // Extract patches using unfold-like operations
        // We need to extract each 4×4 patch from the 7×7 grid
        //
        // This is done by reshaping and transposing:
        // [N, 28, 28] → [N, 7, 4, 7, 4] → [N, 7, 7, 4, 4] → [N, 49, 16]
        
        // Step 1: Split rows into 7 groups of 4
        var reshaped = images.reshaped([batchSize, ATTN_GRID_SIZE, ATTN_PATCH_SIZE, 28])
        
        // Step 2: Split cols into 7 groups of 4
        reshaped = reshaped.reshaped([batchSize, ATTN_GRID_SIZE, ATTN_PATCH_SIZE, ATTN_GRID_SIZE, ATTN_PATCH_SIZE])
        
        // Step 3: Reorder axes to group patches
        // [N, 7, 4, 7, 4] → [N, 7, 7, 4, 4]
        reshaped = reshaped.transposed(0, 1, 3, 2, 4)
        
        // Step 4: Flatten to [N, 49, 16]
        let patches = reshaped.reshaped([batchSize, ATTN_SEQ_LEN, ATTN_PATCH_DIM])
        
        return patches
    }
    
    /// Computes self-attention
    ///
    /// - Parameters:
    ///   - tokens: Input tokens [N, 49, 32]
    ///   - batchSize: Number of samples
    /// - Returns: Attended tokens [N, 49, 32]
    private func selfAttention(_ tokens: MLXArray, batchSize: Int) -> MLXArray {
        // Compute Q, K, V projections
        // Each is [N, 49, 32]
        let q = wQ(tokens)
        let k = wK(tokens)
        let v = wV(tokens)
        
        // Compute attention scores: Q @ K^T
        // [N, 49, 32] @ [N, 32, 49] → [N, 49, 49]
        //
        // We use matmul with transposed K
        let kT = k.transposed(0, 2, 1)  // [N, 32, 49]
        var scores = matmul(q, kT)       // [N, 49, 49]
        
        // Scale by sqrt(d_k) for stable gradients
        // Without scaling, large dot products → sharp softmax → vanishing gradients
        let scale = Float(1.0 / sqrt(Float(ATTN_D_MODEL)))
        scores = scores * scale
        
        // Apply softmax to get attention weights
        // Each row sums to 1 (probability distribution over keys)
        let attnWeights = softmax(scores, axis: -1)  // [N, 49, 49]
        
        // Weighted sum of values
        // [N, 49, 49] @ [N, 49, 32] → [N, 49, 32]
        let output = matmul(attnWeights, v)
        
        return output
    }
}

// =============================================================================
// MARK: - Training Functions
// =============================================================================

/// Computes cross-entropy loss for Attention model
public func attentionLoss(model: AttentionModel, images: MLXArray, labels: MLXArray) -> MLXArray {
    let logits = model(images)
    return crossEntropy(logits: logits, targets: labels, reduction: .mean)
}

/// Computes accuracy on a batch
public func attentionAccuracy(model: AttentionModel, images: MLXArray, labels: MLXArray) -> Float {
    let logits = model(images)
    let predictions = argMax(logits, axis: 1)
    let correct = predictions .== labels
    return mean(correct).item(Float.self)
}

/// Trains the Attention model for one epoch
public func trainAttentionEpoch(
    model: AttentionModel,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    batchSize: Int
) -> Float {
    let n = trainImages.shape[0]
    var totalLoss: Float = 0
    var batchCount = 0
    
    // Create loss-and-gradient function
    let lossAndGrad = valueAndGrad(model: model, attentionLoss)
    
    // Shuffle indices
    var indices = Array(0..<n)
    indices.shuffle()
    
    // Training loop
    var start = 0
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)
        
        let batchImages = trainImages[idxArray]
        let batchLabels = trainLabels[idxArray]
        
        // Compute loss and gradients
        let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)
        
        // Update parameters
        optimizer.update(model: model, gradients: grads)
        
        // Force evaluation
        eval(model, optimizer)
        
        totalLoss += loss.item(Float.self)
        batchCount += 1
        start = end
    }
    
    return totalLoss / Float(batchCount)
}
