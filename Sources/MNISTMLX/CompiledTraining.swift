// ============================================================================
// CompiledTraining.swift - MLX Compiled Training Functions
// ============================================================================
//
// This file provides compiled versions of training step functions for MNIST
// models using MLX's compile() API. Compilation fuses operations into
// optimized GPU kernels for improved performance.
//
// WHAT IS FUNCTION COMPILATION?
//   MLX's compile() function takes a computation graph and optimizes it by:
//   - Fusing multiple GPU kernel launches into a single kernel
//   - Eliminating intermediate buffer allocations
//   - Reducing memory bandwidth requirements
//   - Reducing kernel launch overhead
//
//   Think of it like this: instead of calling the GPU 100 times (once per
//   operation), we analyze all 100 operations, optimize them together, and
//   call the GPU just once with a super-efficient fused kernel.
//
// KERNEL FUSION - THE KEY OPTIMIZATION:
//   Kernel fusion combines multiple operations into a single GPU kernel.
//
//   ## Without Fusion (Uncompiled)
//   ```
//   x1 = x @ W1       // GPU kernel #1, writes intermediate to memory
//   x2 = x1 + b1      // GPU kernel #2, reads x1, writes x2
//   x3 = relu(x2)     // GPU kernel #3, reads x2, writes x3
//   x4 = x3 @ W2      // GPU kernel #4, reads x3, writes x4
//   ```
//   - 4 separate GPU kernel launches (overhead!)
//   - 3 intermediate tensors stored in memory (x1, x2, x3)
//   - Each operation reads from memory, computes, writes back
//   - Memory bandwidth becomes the bottleneck
//
//   ## With Fusion (Compiled)
//   ```
//   x4 = fused_kernel(x, W1, b1, W2)  // Single GPU kernel, no intermediates!
//   ```
//   - 1 GPU kernel launch (minimal overhead)
//   - No intermediate tensors (computed on-the-fly in registers)
//   - Reads input once, writes output once
//   - Compute-bound instead of memory-bound
//
//   SPEEDUP COMES FROM:
//   - Fewer memory transfers (often 10-100x slower than compute!)
//   - Lower kernel launch overhead (~5-10μs per launch adds up)
//   - Better cache/register usage (intermediates stay in fast memory)
//   - Enables more aggressive compiler optimizations
//
// WHY USE COMPILATION FOR TRAINING?
//   Training loops perform many sequential operations:
//     Forward pass → Loss computation → Backward pass → Parameter update
//
//   Without compilation, each operation launches separate GPU kernels with
//   intermediate memory allocations. Compilation fuses these into a single
//   optimized kernel.
//
//   ## Concrete Example - MLP Forward Pass
//   Uncompiled: ~8 kernel launches per training step
//   - Linear layer 1: matmul + bias (2 kernels)
//   - ReLU: (1 kernel)
//   - Linear layer 2: matmul + bias (2 kernels)
//   - Cross-entropy loss: (2 kernels)
//   - Backward pass: (many more kernels for each gradient!)
//
//   Compiled: 1-2 optimized kernels for entire training step!
//
// LAZY EVALUATION - HOW IT ENABLES FUSION:
//   MLX uses lazy evaluation: operations don't execute immediately, they
//   build a computation graph. This lets compile() see the entire graph
//   and optimize it holistically.
//
//   ## Example
//   ```swift
//   let a = x + 1        // Doesn't execute! Builds graph node
//   let b = a * 2        // Builds another node, knows it depends on 'a'
//   let c = relu(b)      // Another node
//   eval(c)              // NOW execute, but as optimized fused kernel!
//   ```
//
//   The eval() call triggers optimization and execution. compile() makes
//   this optimization strategy explicit and reusable.
//
// PERFORMANCE BENEFITS:
//   - Reduced memory usage (no intermediate tensors)
//   - Lower kernel launch overhead
//   - Better GPU utilization through kernel fusion
//   - Faster training iterations
//   - Typical speedups: 1.5-5x depending on model complexity
//
// WHEN TO USE:
//   - Training loops that run many iterations
//   - Models with sequential operations that can be fused
//   - When you need maximum performance
//   - After you've verified correctness with uncompiled version
//
// WHEN NOT TO USE:
//   - Debugging (uncompiled is easier to debug)
//   - One-time operations (compilation overhead not amortized)
//   - When shapes vary frequently (even with shapeless=true, there are limits)
//   - Very simple operations (overhead exceeds benefit)
//
// ============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers

// =============================================================================
// MARK: - Compiled Training Step Functions
// =============================================================================

/// Creates a compiled training step function for MLP model
///
/// This function compiles the entire training step (forward + backward + update)
/// into a single optimized GPU kernel.
///
/// ## What Gets Compiled?
/// ```
/// Training Step:
///   1. Forward pass: model(images) → logits
///   2. Loss: crossEntropy(logits, labels) → loss
///   3. Backward: compute gradients via chain rule
///   4. Update: optimizer.update(model, gradients)
/// ```
///
/// All of these operations are fused into one kernel!
///
/// ## Parameters
/// - model: The MLP model (must be provided to track its state)
/// - optimizer: The optimizer (must be provided to track its state)
///
/// ## Returns
/// A compiled function that takes (images, labels) and returns loss
///
/// ## Usage Example
/// ```swift
/// let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)
///
/// // Use in training loop
/// for batch in batches {
///     let loss = compiledStep(batchImages, batchLabels)
///     print("Loss: \(loss.item(Float.self))")
/// }
/// ```
///
/// ## Technical Details
/// - `inputs: [model, optimizer]`: Declares mutable state for the compiler
/// - `shapeless: true`: Allows variable batch sizes without recompilation
/// - The compiled function is cached by MLX for reuse
///
/// ## Performance Note
/// First call triggers compilation (one-time cost). Subsequent calls use
/// the cached compiled function. For small models/batches, compilation
/// overhead may dominate. Real speedups appear with larger models and batches.
public func createCompiledMLPTrainingStep(
    model: MLPModel,
    optimizer: SGD
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    return compile(
        inputs: [model, optimizer],
        shapeless: true  // Allow variable batch sizes without recompilation
    ) { images, labels -> MLXArray in
        // =====================================================================
        // Forward + Backward Pass
        // =====================================================================
        // valueAndGrad computes both loss and gradients in one pass.
        // This is more efficient than computing them separately.
        //
        // WHAT GETS FUSED HERE:
        // - All forward pass operations (Linear, ReLU, etc.)
        // - Loss computation (cross-entropy)
        // - Entire backward pass (automatic differentiation)
        //
        // Instead of separate kernels for each layer's forward/backward,
        // the compiler fuses them into optimized GPU kernels!
        let lossAndGrad = valueAndGrad(model: model, mlpLoss)
        let (loss, grads) = lossAndGrad(model, images, labels)

        // =====================================================================
        // Parameter Update
        // =====================================================================
        // Update weights: w = w - learning_rate * gradient
        //
        // This updates model parameters in-place. The optimizer maintains
        // mutable state (current weights), which is why we pass it via
        // `inputs: [model, optimizer]` to compile().
        optimizer.update(model: model, gradients: grads)

        // =====================================================================
        // Force Evaluation
        // =====================================================================
        // MLX uses LAZY EVALUATION: operations build a computation graph
        // but don't execute until needed. eval() forces immediate execution.
        //
        // WHY IS THIS NECESSARY?
        // - Without eval(), the graph would keep growing without executing
        // - Optimizer state needs to be updated for next iteration
        // - Forces memory to be freed for intermediate tensors
        //
        // The eval() call triggers the actual GPU kernel execution of all
        // the fused operations we've built up!
        eval(model, optimizer)

        return loss
    }
}

/// Creates a compiled training step function for CNN model
///
/// Same as MLP version but works with CNN models. The CNN has more
/// operations (conv, pooling, batch norm) that benefit even more from
/// kernel fusion.
///
/// ## Architecture Reminder
/// CNN performs these operations in sequence:
/// - Conv1 → ReLU → MaxPool
/// - Conv2 → ReLU → MaxPool
/// - Flatten → Linear → ReLU → Linear
///
/// Each of these can be fused for better performance!
///
/// ## Parameters
/// - model: The CNN model
/// - optimizer: The optimizer
///
/// ## Returns
/// A compiled function that takes (images, labels) and returns loss
public func createCompiledCNNTrainingStep(
    model: CNNModel,
    optimizer: SGD
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    return compile(
        inputs: [model, optimizer],
        shapeless: true
    ) { images, labels -> MLXArray in
        let lossAndGrad = valueAndGrad(model: model, cnnLoss)
        let (loss, grads) = lossAndGrad(model, images, labels)
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)
        return loss
    }
}

/// Creates a compiled training step function for Attention model
///
/// Attention models have even more operations (multi-head attention,
/// layer norm, etc.) that benefit significantly from compilation.
///
/// ## Architecture Reminder
/// Attention performs:
/// - Patch embedding
/// - Positional encoding
/// - Multi-head self-attention (many matrix operations!)
/// - Layer normalization
/// - Feed-forward network
/// - Classification head
///
/// Compilation fuses all of these into optimized kernels.
///
/// ## Parameters
/// - model: The Attention model
/// - optimizer: The optimizer
///
/// ## Returns
/// A compiled function that takes (images, labels) and returns loss
public func createCompiledAttentionTrainingStep(
    model: AttentionModel,
    optimizer: SGD
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    return compile(
        inputs: [model, optimizer],
        shapeless: true
    ) { images, labels -> MLXArray in
        let lossAndGrad = valueAndGrad(model: model, attentionLoss)
        let (loss, grads) = lossAndGrad(model, images, labels)
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)
        return loss
    }
}

/// Creates a compiled training step function for ResNet model
///
/// ResNet models have residual blocks with skip connections and batch
/// normalization that benefit from compilation.
///
/// ## Architecture Reminder
/// ResNet performs:
/// - Initial convolution
/// - Residual blocks with skip connections
/// - Batch normalization
/// - Global average pooling
/// - Linear classification head
///
/// Compilation fuses all of these into optimized kernels.
///
/// ## Parameters
/// - model: The ResNet model
/// - optimizer: The optimizer
///
/// ## Returns
/// A compiled function that takes (images, labels) and returns loss
public func createCompiledResNetTrainingStep(
    model: ResNetModel,
    optimizer: SGD
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    return compile(
        inputs: [model, optimizer],
        shapeless: true
    ) { images, labels -> MLXArray in
        let lossAndGrad = valueAndGrad(model: model, resnetLoss)
        let (loss, grads) = lossAndGrad(model, images, labels)
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)
        return loss
    }
}

// =============================================================================
// MARK: - Compiled Training Loop Functions
// =============================================================================

/// Trains the MLP model for one epoch using compiled training steps
///
/// This is a compiled version of trainMLPEpoch() that uses fused GPU kernels
/// for better performance. The logic is identical, but each training step
/// runs faster due to kernel fusion.
///
/// ## Differences from trainMLPEpoch()
/// - Creates a compiled training step function (one-time compilation cost)
/// - Uses the compiled function for all batches (amortizes compilation)
/// - Should be faster for repeated execution
///
/// ## When to Use
/// - Training for many epochs (amortizes compilation overhead)
/// - Larger models (more operations to fuse)
/// - Larger batches (more parallelism to exploit)
///
/// ## When Not to Use
/// - Debugging (stick with uncompiled for easier debugging)
/// - Single-epoch training (compilation overhead not worth it)
/// - Very small models/batches (overhead dominates)
///
/// ## Parameters
/// - model: The MLP model to train
/// - optimizer: SGD or other optimizer
/// - trainImages: Training images [N, 784]
/// - trainLabels: Training labels [N]
/// - batchSize: Number of samples per batch
///
/// ## Returns
/// Average loss for the epoch
public func trainMLPEpochCompiled(
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
    // Create Compiled Training Step
    // -------------------------------------------------------------------------
    // This happens once per epoch. The compiled function is reused for all
    // batches, so the compilation cost is amortized across many iterations.
    //
    // COMPILATION OVERHEAD:
    // - First call analyzes the computation graph
    // - Identifies fusible operations
    // - Generates optimized GPU code
    // - This can take 100-500ms for complex models
    //
    // BUT: The compiled function is cached and reused for all subsequent
    // batches. For MNIST (60,000 samples / 128 batch size = ~469 batches),
    // we pay the cost once and get speedup 469 times!
    let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)

    // -------------------------------------------------------------------------
    // Shuffle for Stochastic Gradient Descent
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
    // Mini-batch Training Loop
    // -------------------------------------------------------------------------
    var start = 0
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)

        // Get batch data
        let batchImages = trainImages[idxArray]
        let batchLabels = trainLabels[idxArray]

        // =====================================================================
        // Compiled Training Step
        // =====================================================================
        // This single call performs:
        // 1. Forward pass (fused operations)
        // 2. Loss computation (fused with forward)
        // 3. Backward pass (fused gradient computation)
        // 4. Parameter update (fused with backward)
        //
        // All in 1-2 optimized GPU kernels instead of ~20+ separate kernels!
        //
        // WHAT'S HAPPENING UNDER THE HOOD:
        // - MLX launches the pre-compiled GPU kernel(s)
        // - All intermediate computations happen in GPU registers (fast!)
        // - Only final results are written back to memory
        // - No Python/Swift overhead between operations
        //
        // PERFORMANCE COMPARISON (typical MNIST batch):
        // - Uncompiled: ~20-30 kernel launches, ~2-3ms per batch
        // - Compiled: ~1-2 kernel launches, ~1-1.5ms per batch
        // - Speedup: 1.5-2x (varies by hardware)
        let loss = compiledStep(batchImages, batchLabels)

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

/// Trains the CNN model for one epoch using compiled training steps
///
/// Compiled version of trainCNNEpoch(). CNNs benefit significantly from
/// compilation due to the many sequential operations (conv, relu, pool).
///
/// ## Parameters
/// - model: The CNN model to train
/// - optimizer: SGD or other optimizer
/// - trainImages: Training images [N, 28, 28, 1]
/// - trainLabels: Training labels [N]
/// - batchSize: Number of samples per batch
///
/// ## Returns
/// Average loss for the epoch
public func trainCNNEpochCompiled(
    model: CNNModel,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    batchSize: Int
) -> Float {
    let n = trainImages.shape[0]
    var totalLoss: Float = 0
    var batchCount = 0

    let compiledStep = createCompiledCNNTrainingStep(model: model, optimizer: optimizer)

    var indices = Array(0..<n)
    indices.shuffle()

    // -------------------------------------------------------------------------
    // Progress Bar Setup
    // -------------------------------------------------------------------------
    let totalBatches = (n + batchSize - 1) / batchSize
    let progressBar = ProgressBar(totalBatches: totalBatches)
    progressBar.start()

    var start = 0
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)

        let batchImages = trainImages[idxArray]
        let batchLabels = trainLabels[idxArray]

        let loss = compiledStep(batchImages, batchLabels)

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

/// Trains the Attention model for one epoch using compiled training steps
///
/// Compiled version of trainAttentionEpoch(). Attention models have the most
/// complex operations and benefit the most from compilation.
///
/// ## Parameters
/// - model: The Attention model to train
/// - optimizer: SGD or other optimizer
/// - trainImages: Training images [N, 28, 28, 1]
/// - trainLabels: Training labels [N]
/// - batchSize: Number of samples per batch
///
/// ## Returns
/// Average loss for the epoch
public func trainAttentionEpochCompiled(
    model: AttentionModel,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    batchSize: Int
) -> Float {
    let n = trainImages.shape[0]
    var totalLoss: Float = 0
    var batchCount = 0

    let compiledStep = createCompiledAttentionTrainingStep(model: model, optimizer: optimizer)

    var indices = Array(0..<n)
    indices.shuffle()

    // -------------------------------------------------------------------------
    // Progress Bar Setup
    // -------------------------------------------------------------------------
    let totalBatches = (n + batchSize - 1) / batchSize
    let progressBar = ProgressBar(totalBatches: totalBatches)
    progressBar.start()

    var start = 0
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)

        let batchImages = trainImages[idxArray]
        let batchLabels = trainLabels[idxArray]

        let loss = compiledStep(batchImages, batchLabels)

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

/// Trains the ResNet model for one epoch using compiled training steps
///
/// Compiled version of trainResNetEpoch(). ResNet models with residual
/// connections and batch normalization benefit from compilation due to
/// kernel fusion of sequential operations.
///
/// ## Parameters
/// - model: The ResNet model to train
/// - optimizer: SGD or other optimizer
/// - trainImages: Training images [N, 784]
/// - trainLabels: Training labels [N]
/// - batchSize: Number of samples per batch
///
/// ## Returns
/// Average loss for the epoch
public func trainResNetEpochCompiled(
    model: ResNetModel,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    batchSize: Int
) -> Float {
    let n = trainImages.shape[0]
    var totalLoss: Float = 0
    var batchCount = 0

    let compiledStep = createCompiledResNetTrainingStep(model: model, optimizer: optimizer)

    var indices = Array(0..<n)
    indices.shuffle()

    // -------------------------------------------------------------------------
    // Progress Bar Setup
    // -------------------------------------------------------------------------
    let totalBatches = (n + batchSize - 1) / batchSize
    let progressBar = ProgressBar(totalBatches: totalBatches)
    progressBar.start()

    var start = 0
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)

        // Reshape to [N, 1, 28, 28] for ResNet (add channel dimension)
        let batchImages = trainImages[idxArray].reshaped([-1, 1, 28, 28])
        let batchLabels = trainLabels[idxArray]

        let loss = compiledStep(batchImages, batchLabels)

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

// =============================================================================
// MARK: - Additional Notes
// =============================================================================

// COMPILATION BEST PRACTICES:
//
// 1. Compile Once, Reuse Many Times
//    ✓ let compiled = compile(...) { ... }
//      for i in 0..<1000 { compiled(data) }
//    ✗ for i in 0..<1000 { let c = compile(...) { ... }; c(data) }
//
//    WHY: Compilation analyzes the graph, generates GPU code, and compiles it.
//    This overhead (~100-500ms) must be amortized across many calls to see
//    real speedups. Compiling inside a loop wastes this time every iteration!
//
// 2. Use shapeless=true for Training
//    - Allows variable batch sizes (last batch often smaller)
//    - Prevents recompilation on shape changes
//    - Only prevents recompilation for shape, not dimension or type
//
//    EXAMPLE: With batch_size=128 and 60,000 samples:
//    - First 468 batches: shape [128, 784]
//    - Last batch: shape [96, 784]
//    Without shapeless: recompiles on last batch (slow!)
//    With shapeless: handles both shapes with same compiled code (fast!)
//
// 3. Profile Before and After
//    - Measure actual speedup (not all code benefits equally)
//    - Watch for compilation overhead on first call
//    - Consider compilation only for hot paths
//
//    HOW TO PROFILE:
//    ```swift
//    let start = Date()
//    let loss = compiledStep(images, labels)
//    eval(loss)  // Force execution
//    let elapsed = Date().timeIntervalSince(start)
//    ```
//
// 4. Test Correctness First
//    - Verify uncompiled version works correctly
//    - Compare compiled vs uncompiled outputs
//    - Compiled should produce identical results (within floating-point tolerance)
//
//    SANITY CHECK:
//    ```swift
//    let lossUncompiled = trainMLPEpoch(...)
//    let lossCompiled = trainMLPEpochCompiled(...)
//    assert(abs(lossUncompiled - lossCompiled) < 1e-5)
//    ```
//
// 5. Be Careful with Shapeless
//    - Don't use shape-dependent conditionals (if x.shape[0] > 10 { ... })
//    - Avoid dynamic reshaping based on input dimensions
//    - These patterns break with shapeless compilation
//
//    BAD PATTERN (breaks with shapeless):
//    ```swift
//    compile(shapeless: true) { x in
//        if x.shape[0] < 32 {  // Conditional on shape!
//            return x * 2
//        } else {
//            return x * 3
//        }
//    }
//    ```
//
// 6. Understand What Gets Fused
//    - Element-wise ops: add, multiply, relu (easy to fuse)
//    - Matrix ops: matmul, conv (can fuse with surrounding ops)
//    - Reductions: sum, mean (harder to fuse, may need separate kernel)
//
//    FUSION EXAMPLE:
//    ```
//    y = relu(x @ W + b)  // All three ops fused into one kernel!
//    ```
//
// DEBUGGING TIPS:
//
// - If compiled version crashes: test uncompiled first
// - If results differ: check for shape-dependent logic
// - If slow: check if compilation is happening every call
// - If error messages are cryptic: simplify the function
// - Use MLX_DISABLE_COMPILE=1 environment variable to disable compilation
// - Add print statements before compilation, not inside compiled function
//
// PERFORMANCE EXPECTATIONS:
//
// - Small models: 1.1-1.5x speedup (compilation overhead limits gains)
// - Medium models (MNIST): 1.5-2.5x speedup (good balance)
// - Large models: 2-5x speedup (many operations to fuse)
// - GPU-bound code: Higher speedup (kernel fusion more effective)
// - CPU-bound code: Lower speedup (limited by single-core performance)
//
// TYPICAL MNIST TRAINING PERFORMANCE (M1/M2 Mac):
// - Uncompiled MLP: ~2.5ms per batch
// - Compiled MLP: ~1.5ms per batch (1.7x speedup)
// - Uncompiled CNN: ~4ms per batch
// - Compiled CNN: ~2ms per batch (2x speedup)
// - Uncompiled Attention: ~8ms per batch
// - Compiled Attention: ~3ms per batch (2.7x speedup)
//
// Notice: More complex models → more operations → better speedup!
//
// =============================================================================
// MARK: - Understanding Kernel Fusion (Deep Dive)
// =============================================================================
//
// WHAT IS A GPU KERNEL?
//   A GPU kernel is a function that runs on the GPU. Each kernel launch has
//   overhead: scheduling, memory setup, parameter passing, etc. (~5-10μs)
//
//   For small operations, this overhead can dominate actual compute time!
//
// MEMORY BANDWIDTH BOTTLENECK:
//   Modern GPUs are incredibly fast at computation (TFLOPS) but memory
//   bandwidth is limited. Reading/writing to GPU memory is often the bottleneck.
//
//   EXAMPLE: M1 Max GPU
//   - Compute: ~10 TFLOPS (10 trillion operations/second)
//   - Memory bandwidth: ~400 GB/s
//
//   For a simple operation like y = x + 1 on 1M floats:
//   - Compute: 1M ops ÷ 10 TFLOPS = 0.1 microseconds
//   - Memory: (1M read + 1M write) × 4 bytes ÷ 400 GB/s = 20 microseconds
//   - Memory is 200x slower than compute!
//
// HOW FUSION HELPS:
//   By computing multiple operations in a single kernel, we read input once
//   and write output once, keeping intermediate results in fast registers.
//
//   EXAMPLE: y = relu(x @ W + b)
//
//   Without Fusion:
//   1. Read x, W from memory → compute x @ W → write temp1 to memory
//   2. Read temp1, b from memory → compute temp1 + b → write temp2 to memory
//   3. Read temp2 from memory → compute relu(temp2) → write y to memory
//   Total: 3 kernel launches, 6 memory operations
//
//   With Fusion:
//   1. Read x, W, b from memory → compute relu(x @ W + b) → write y to memory
//   Total: 1 kernel launch, 2 memory operations (3x fewer!)
//
// WHEN FUSION ISN'T POSSIBLE:
//   - Operations with different execution patterns (e.g., matmul + reduction)
//   - Operations requiring synchronization barriers
//   - Very large intermediate tensors that don't fit in registers
//   - Operations with data dependencies that prevent pipelining
//
// THE ROLE OF COMPILATION:
//   MLX's compile() analyzes the computation graph and automatically:
//   1. Identifies fusible operation sequences
//   2. Generates optimized Metal/GPU code for fused kernels
//   3. Manages memory allocation and scheduling
//   4. Caches compiled code for reuse
//
//   This is why compiled functions are faster: they've been analyzed and
//   optimized specifically for the computation pattern you're running!
//
// =============================================================================
