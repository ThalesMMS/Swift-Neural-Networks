// ============================================================================
// MultiHeadAttention.swift - Multi-Head Self-Attention for MNIST Transformer
// ============================================================================

import Foundation
import MLX
import MLXNN

// =============================================================================
// MARK: - Multi-Head Attention
// =============================================================================

/// Multi-Head Self-Attention
///
/// Multi-head attention splits the model dimension into multiple parallel
/// attention heads, allowing the model to attend to information from different
/// representation subspaces.
///
/// ## Architecture
/// ```
/// Input: [N, SeqLen, D]
///   ↓
/// Q, K, V = Linear projections [N, SeqLen, D]
///   ↓
/// Split into H heads: [N, SeqLen, H, D/H]
///   ↓
/// Transpose: [N, H, SeqLen, D/H]
///   ↓
/// Attention per head: softmax(Q@K^T / sqrt(d_k)) @ V
///   ↓
/// Concat heads: [N, SeqLen, D]
///   ↓
/// Output projection: [N, SeqLen, D]
/// ```
///
/// ## Why Multiple Heads?
/// - Different heads can learn different attention patterns
/// - One head might focus on local context, another on global
/// - Increases model capacity without increasing d_model
/// - Empirically works better than single large head
///
/// ## Example
/// With d_model=32, num_heads=4:
/// - Each head has dimension 32/4 = 8
/// - 4 parallel attention operations
/// - Results concatenated back to dimension 32
public class MultiHeadAttention: Module {
    // -------------------------------------------------------------------------
    // MARK: - Properties
    // -------------------------------------------------------------------------

    /// Model dimension (d_model)
    let dModel: Int

    /// Number of attention heads
    let numHeads: Int

    /// Dimension per head (d_model / num_heads)
    let headDim: Int

    // -------------------------------------------------------------------------
    // MARK: - Layers
    // -------------------------------------------------------------------------

    /// Query projection: D → D
    @ModuleInfo(key: "wq") var wQ: Linear

    /// Key projection: D → D
    @ModuleInfo(key: "wk") var wK: Linear

    /// Value projection: D → D
    @ModuleInfo(key: "wv") var wV: Linear

    /// Output projection: D → D
    ///
    /// After concatenating heads, we project back to d_model.
    /// This allows the model to learn how to combine information
    /// from different heads.
    @ModuleInfo(key: "wo") var wO: Linear

    // -------------------------------------------------------------------------
    // MARK: - Initialization
    // -------------------------------------------------------------------------

    /// Creates a multi-head attention module
    ///
    /// - Parameters:
    ///   - dModel: Model dimension (must be divisible by numHeads)
    ///   - numHeads: Number of parallel attention heads
    ///
    /// ## Example
    /// ```swift
    /// let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
    /// let x = MLXArray(...)  // [batch, seq_len, 32]
    /// let output = attn(x)   // [batch, seq_len, 32]
    /// ```
    public init(dModel: Int, numHeads: Int) {
        precondition(
            dModel % numHeads == 0,
            "dModel (\(dModel)) must be divisible by numHeads (\(numHeads))"
        )

        self.dModel = dModel
        self.numHeads = numHeads
        self.headDim = dModel / numHeads

        // Initialize Q, K, V, O projections
        _wQ = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "wq")
        _wK = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "wk")
        _wV = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "wv")
        _wO = ModuleInfo(wrappedValue: Linear(dModel, dModel), key: "wo")
    }

    // -------------------------------------------------------------------------
    // MARK: - Forward Pass
    // -------------------------------------------------------------------------

    /// Forward pass through multi-head attention
    ///
    /// - Parameter x: Input tensor [batch, seq_len, d_model]
    /// - Returns: Output tensor [batch, seq_len, d_model]
    ///
    /// ## Implementation Steps
    /// 1. Project to Q, K, V
    /// 2. Split into multiple heads
    /// 3. Compute scaled dot-product attention per head
    /// 4. Concatenate heads
    /// 5. Project output
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]

        // Step 1: Project to Q, K, V
        // Each: [batch, seq_len, d_model]
        let q = wQ(x)
        let k = wK(x)
        let v = wV(x)

        // Step 2: Split into heads
        // Reshape: [batch, seq_len, d_model] → [batch, seq_len, num_heads, head_dim]
        // Then transpose: [batch, num_heads, seq_len, head_dim]
        let qHeads = splitHeads(q, batchSize: batchSize, seqLen: seqLen)
        let kHeads = splitHeads(k, batchSize: batchSize, seqLen: seqLen)
        let vHeads = splitHeads(v, batchSize: batchSize, seqLen: seqLen)

        // Step 3: Scaled dot-product attention
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        // [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        // → [batch, num_heads, seq_len, seq_len]
        let kT = kHeads.transposed(0, 1, 3, 2)  // Transpose last two dims
        var scores = matmul(qHeads, kT)

        // Scale by sqrt(head_dim) for stable gradients
        let scale = Float(1.0 / sqrt(Float(headDim)))
        scores = scores * scale

        // Apply softmax to get attention weights
        let attnWeights = softmax(scores, axis: -1)  // [batch, num_heads, seq_len, seq_len]

        // Weighted sum of values
        // [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        // → [batch, num_heads, seq_len, head_dim]
        let attnOutput = matmul(attnWeights, vHeads)

        // Step 4: Concatenate heads
        // [batch, num_heads, seq_len, head_dim] → [batch, seq_len, d_model]
        let concatenated = concatenateHeads(attnOutput, batchSize: batchSize, seqLen: seqLen)

        // Step 5: Output projection
        let output = wO(concatenated)

        return output
    }

    // -------------------------------------------------------------------------
    // MARK: - Helper Functions
    // -------------------------------------------------------------------------

    /// Splits input into multiple attention heads
    ///
    /// - Parameters:
    ///   - x: Input tensor [batch, seq_len, d_model]
    ///   - batchSize: Batch size
    ///   - seqLen: Sequence length
    /// - Returns: Tensor with shape [batch, num_heads, seq_len, head_dim]
    ///
    /// ## Transformation
    /// ```
    /// [batch, seq_len, d_model]
    ///   ↓ reshape
    /// [batch, seq_len, num_heads, head_dim]
    ///   ↓ transpose
    /// [batch, num_heads, seq_len, head_dim]
    /// ```
    private func splitHeads(_ x: MLXArray, batchSize: Int, seqLen: Int) -> MLXArray {
        // Reshape: [batch, seq_len, d_model] → [batch, seq_len, num_heads, head_dim]
        let reshaped = x.reshaped([batchSize, seqLen, numHeads, headDim])

        // Transpose: [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]
        let transposed = reshaped.transposed(0, 2, 1, 3)

        return transposed
    }

    /// Concatenates multiple attention heads back into single tensor
    ///
    /// - Parameters:
    ///   - x: Input tensor [batch, num_heads, seq_len, head_dim]
    ///   - batchSize: Batch size
    ///   - seqLen: Sequence length
    /// - Returns: Tensor with shape [batch, seq_len, d_model]
    ///
    /// ## Transformation
    /// ```
    /// [batch, num_heads, seq_len, head_dim]
    ///   ↓ transpose
    /// [batch, seq_len, num_heads, head_dim]
    ///   ↓ reshape
    /// [batch, seq_len, d_model]
    /// ```
    private func concatenateHeads(_ x: MLXArray, batchSize: Int, seqLen: Int) -> MLXArray {
        // Transpose: [batch, num_heads, seq_len, head_dim] → [batch, seq_len, num_heads, head_dim]
        let transposed = x.transposed(0, 2, 1, 3)

        // Reshape: [batch, seq_len, num_heads, head_dim] → [batch, seq_len, d_model]
        let concatenated = transposed.reshaped([batchSize, seqLen, dModel])

        return concatenated
    }
}

