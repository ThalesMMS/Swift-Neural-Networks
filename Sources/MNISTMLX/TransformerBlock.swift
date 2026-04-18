// ============================================================================
// TransformerBlock.swift - Transformer Encoder Block for MNIST using MLX Swift
// ============================================================================
//
// Owns the reusable encoder block: LayerNorm, multi-head attention, feed-forward
// network, and residual composition.
//
// ============================================================================

import Foundation
import MLX
import MLXNN

// =============================================================================
// MARK: - Layer Normalization Wrapper
// =============================================================================

/// Helper function to create a LayerNorm module with standard transformer settings
///
/// This is a convenience function that wraps MLXNN's LayerNorm with
/// transformer-specific defaults.
///
/// - Parameter dimensions: The feature dimension to normalize (typically d_model)
/// - Returns: A configured LayerNorm module
///
/// ## Usage Example
/// ```swift
/// let norm = createLayerNorm(dimensions: 32)
/// let x = MLXArray(...)  // [batch, seq_len, d_model]
/// let normalized = norm(x)  // [batch, seq_len, d_model]
/// ```
///
/// ## What This Does
/// ```
/// Input:  [batch, seq_len, d_model]
///   ↓
/// Compute mean & variance over d_model dimension
///   ↓
/// Normalize: (x - mean) / sqrt(variance + eps)
///   ↓
/// Scale & shift: normalized * gamma + beta
///   ↓
/// Output: [batch, seq_len, d_model]
/// ```
///
/// ## Parameters Explained
/// - **dimensions**: The size of the last dimension (d_model)
///   - For d_model=32: gamma and beta are [32] vectors
///   - Applied to each token independently
///
/// - **eps**: Numerical stability constant (1e-5)
///   - Prevents division by zero when variance ≈ 0
///   - Standard value across all transformer implementations
///
/// - **affine**: Enable learnable scale (gamma)
///   - Always true for transformers
///   - Allows network to learn optimal scaling
///
/// - **bias**: Enable learnable shift (beta)
///   - Always true for transformers
///   - Allows network to learn optimal offset
///
/// ## Why a Helper Function?
/// - Centralizes LayerNorm configuration
/// - Ensures consistent settings across all transformer blocks
/// - Makes it easy to experiment with different settings
/// - Documents the standard transformer configuration
fileprivate func createLayerNorm(dimensions: Int) -> LayerNorm {
    return LayerNorm(
        dimensions: dimensions,
        eps: TRANSFORMER_LAYER_NORM_EPS,
        affine: true,  // Include learnable scale (gamma)
        bias: true     // Include learnable shift (beta)
    )
}

// =============================================================================
// MARK: - Transformer Block (Skeleton)
// =============================================================================

/// Transformer Encoder Block
///
/// This class will implement a complete transformer encoder block with:
/// - Multi-head self-attention
/// - Feed-forward network
/// - Layer normalization (Pre-LN style)
/// - Residual connections
///
/// Architecture:
/// ```
/// Input x [N, SeqLen, D]
///   ↓
/// ┌─────────────────────────────────────┐
/// │ x1 = LayerNorm(x)                   │
/// │ x2 = MultiHeadAttention(x1) + x     │  ← First residual
/// │ x3 = LayerNorm(x2)                  │
/// │ x4 = FeedForward(x3) + x2           │  ← Second residual
/// └─────────────────────────────────────┘
///   ↓
/// Output x4 [N, SeqLen, D]
/// ```
///
/// ## Implementation Status
/// - [x] LayerNorm wrapper (subtask-2-1)
/// - [x] Multi-head attention (subtask-3-1)
/// - [x] Complete transformer block (subtask-4-1)
/// - [ ] Full transformer model (subtask-5-1)
///
/// The transformer block is now fully implemented with attention, FFN, and residuals.
public class TransformerBlock: Module {
    // -------------------------------------------------------------------------
    // MARK: - Layers
    // -------------------------------------------------------------------------

    /// Layer normalization before attention
    ///
    /// Normalizes input before multi-head attention for training stability.
    /// In Pre-LN architecture, this comes BEFORE the attention operation.
    @ModuleInfo(key: "norm1") var norm1: LayerNorm

    /// Layer normalization before feed-forward network
    ///
    /// Normalizes input before FFN for training stability.
    /// In Pre-LN architecture, this comes BEFORE the FFN operation.
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    /// Multi-head self-attention mechanism
    ///
    /// Allows tokens to attend to each other and exchange information.
    /// Uses multiple parallel attention heads for richer representations.
    @ModuleInfo(key: "attention") var attention: MultiHeadAttention

    /// First feed-forward layer: d_model → ff_dim
    ///
    /// Expands the representation to a higher dimension (ff_dim).
    /// This expansion allows the network to learn more complex transformations.
    ///
    /// Followed by ReLU activation for non-linearity.
    @ModuleInfo(key: "ffn1") var ffn1: Linear

    /// Second feed-forward layer: ff_dim → d_model
    ///
    /// Projects back down to the model dimension.
    /// This completes the position-wise feed-forward transformation:
    ///   FFN(x) = W2 @ ReLU(W1 @ x + b1) + b2
    ///
    /// The two-layer FFN with ReLU allows learning non-linear transformations
    /// that are applied independently to each token (position-wise).
    @ModuleInfo(key: "ffn2") var ffn2: Linear

    // -------------------------------------------------------------------------
    // MARK: - Properties
    // -------------------------------------------------------------------------

    /// Model dimension (d_model)
    let dModel: Int

    /// Number of attention heads
    let numHeads: Int

    /// Feed-forward hidden dimension
    let ffDim: Int

    // -------------------------------------------------------------------------
    // MARK: - Initialization
    // -------------------------------------------------------------------------

    /// Creates a new transformer encoder block
    ///
    /// - Parameters:
    ///   - dModel: Model dimension (default: 32 for MNIST)
    ///   - numHeads: Number of attention heads (default: 4)
    ///   - ffDim: Feed-forward hidden dimension (default: 64)
    ///
    /// ## Example
    /// ```swift
    /// let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
    /// let x = MLXArray(...)  // [batch, seq_len, d_model]
    /// let output = block(x)  // [batch, seq_len, d_model]
    /// ```
    ///
    /// ## Requirements
    /// - dModel must be divisible by numHeads
    /// - Each head will have dimension dModel/numHeads
    public init(
        dModel: Int = TRANSFORMER_D_MODEL,
        numHeads: Int = TRANSFORMER_NUM_HEADS,
        ffDim: Int = TRANSFORMER_FF_DIM
    ) {
        // Validate that d_model is divisible by num_heads
        precondition(
            dModel % numHeads == 0,
            "dModel (\(dModel)) must be divisible by numHeads (\(numHeads))"
        )

        self.dModel = dModel
        self.numHeads = numHeads
        self.ffDim = ffDim

        // Initialize layer normalization modules
        _norm1 = ModuleInfo(
            wrappedValue: createLayerNorm(dimensions: dModel),
            key: "norm1"
        )
        _norm2 = ModuleInfo(
            wrappedValue: createLayerNorm(dimensions: dModel),
            key: "norm2"
        )

        // Initialize multi-head attention
        _attention = ModuleInfo(
            wrappedValue: MultiHeadAttention(dModel: dModel, numHeads: numHeads),
            key: "attention"
        )

        // Initialize feed-forward network
        _ffn1 = ModuleInfo(
            wrappedValue: Linear(dModel, ffDim),
            key: "ffn1"
        )
        _ffn2 = ModuleInfo(
            wrappedValue: Linear(ffDim, dModel),
            key: "ffn2"
        )
    }

    // -------------------------------------------------------------------------
    // MARK: - Forward Pass
    // -------------------------------------------------------------------------

    /// Forward pass through the transformer block
    ///
    /// - Parameter x: Input tensor [batch, seq_len, d_model]
    /// - Returns: Output tensor [batch, seq_len, d_model]
    ///
    /// ## Architecture (Pre-LN Transformer)
    /// ```
    /// x1 = norm1(x)                  // Normalize input
    /// x2 = attention(x1) + x         // Multi-head attention + residual
    /// x3 = norm2(x2)                 // Normalize before FFN
    /// x4 = ffn(x3) + x2              // Feed-forward + residual
    /// return x4
    /// ```
    ///
    /// ## Why Pre-LN (Layer Norm before sublayer)?
    /// - More stable training (gradients flow through normalized activations)
    /// - Allows higher learning rates
    /// - Better for deep networks (>12 layers)
    /// - Used by GPT-2, GPT-3, modern transformers
    ///
    /// ## Residual Connections
    /// Both the attention and FFN use residual connections:
    /// - output = sublayer(norm(x)) + x
    /// - Enables gradient flow through deep networks
    /// - Allows learning identity mappings (if sublayer ≈ 0)
    ///
    /// ## Feed-Forward Network
    /// Position-wise FFN applied to each token independently:
    /// - FFN(x) = W2 @ ReLU(W1 @ x + b1) + b2
    /// - Expands to ff_dim (64), then back to d_model (32)
    /// - Same weights applied to every position/token
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // ====================================================================
        // FIRST SUB-LAYER: Multi-Head Attention with Residual Connection
        // ====================================================================

        // Step 1: Apply layer normalization before attention (Pre-LN)
        let normalized1 = norm1(x)

        // Step 2: Apply multi-head self-attention
        let attended = attention(normalized1)

        // Step 3: Add residual connection (skip connection)
        // This allows gradients to flow directly through the block
        let x2 = attended + x

        // ====================================================================
        // SECOND SUB-LAYER: Feed-Forward Network with Residual Connection
        // ====================================================================

        // Step 4: Apply layer normalization before FFN (Pre-LN)
        let normalized2 = norm2(x2)

        // Step 5: Apply position-wise feed-forward network
        // FFN(x) = W2(ReLU(W1(x)))
        // This is applied to each token independently
        let ffnHidden = ffn1(normalized2)       // [batch, seq_len, ff_dim]
        let ffnActivated = relu(ffnHidden)       // Apply ReLU activation
        let ffnOutput = ffn2(ffnActivated)       // [batch, seq_len, d_model]

        // Step 6: Add residual connection (skip connection)
        let output = ffnOutput + x2

        return output
    }
}
