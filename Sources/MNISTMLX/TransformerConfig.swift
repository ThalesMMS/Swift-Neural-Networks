// ============================================================================
// TransformerConfig.swift - Transformer Architecture Constants
// ============================================================================

// =============================================================================
// MARK: - Transformer Architecture Constants
// =============================================================================

/// Model dimension (embedding size, d_model)
///
/// This is the fundamental dimension throughout the transformer:
/// - Patch embeddings are projected to this dimension
/// - Attention operates in this dimension
/// - Feed-forward network input/output dimension
/// - Positional embeddings have this dimension
///
/// For MNIST with 49 tokens (7×7 patches), we use d_model=32:
/// - Sufficient capacity to represent patch relationships
/// - Lightweight for educational purposes
/// - Follows dModel ≈ sqrt(vocab_size × seq_len) heuristic
///
/// Production transformers use much larger values:
/// - BERT-Base: 768
/// - GPT-2: 768, 1024, 1280, 1600 (Small/Medium/Large/XL)
/// - GPT-3: up to 12,288
public let TRANSFORMER_D_MODEL = 32

/// Number of attention heads
///
/// Multi-head attention splits d_model into multiple parallel attention
/// operations. Each head has dimension d_model/num_heads.
///
/// Requirements:
/// - d_model must be divisible by num_heads
/// - More heads = more parallel attention patterns
/// - Typical values: 8, 12, 16 for large models
///
/// For d_model=32, we use 4 heads:
/// - Each head has dimension 32/4 = 8
/// - Provides 4 different attention patterns
/// - Reasonable parallelism for small model
public let TRANSFORMER_NUM_HEADS = 4

/// Feed-forward network hidden dimension
///
/// The FFN expands from d_model → ff_dim → d_model.
/// Standard practice: ff_dim = 4 × d_model
///
/// For d_model=32:
/// - ff_dim = 64 (2× expansion, lightweight for MNIST)
/// - Full-scale would be 128 (4×), but MNIST doesn't need it
///
/// Production transformers:
/// - BERT: ff_dim = 4 × d_model = 3072 (for d_model=768)
/// - GPT-2: ff_dim = 4 × d_model
public let TRANSFORMER_FF_DIM = 64

/// Dropout rate for regularization
///
/// Dropout randomly zeros elements during training to prevent overfitting.
/// Applied after:
/// - Attention weights
/// - Attention output
/// - Feed-forward output
///
/// Typical values:
/// - 0.1 for large datasets (ImageNet, large text corpora)
/// - 0.0-0.1 for small datasets like MNIST
/// - 0.0 for this educational implementation (simpler to understand)
public let TRANSFORMER_DROPOUT: Float = 0.0

/// Layer normalization epsilon
///
/// Small constant added to variance for numerical stability.
/// Standard value: 1e-5 (works well in practice)
///
/// Too small: risk of division by zero
/// Too large: reduces normalization effectiveness
public let TRANSFORMER_LAYER_NORM_EPS: Float = 1e-5
