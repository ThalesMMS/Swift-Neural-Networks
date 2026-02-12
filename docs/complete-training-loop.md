# Complete Training Loop Walkthrough: MLP for MNIST

This document provides a comprehensive, step-by-step walkthrough of the complete training process for our Multi-Layer Perceptron (MLP) network. We'll cover everything from data preparation through multiple epochs of training, including evaluation.

## What is a Training Loop?

The **training loop** is the heart of machine learning. It's the iterative process that adjusts the network's parameters (weights and biases) to minimize prediction errors.

**Core concept:**
1. Show the network examples (forward pass)
2. Measure how wrong it is (loss computation)
3. Calculate how to improve (backward pass / gradients)
4. Make small improvements (weight updates)
5. Repeat thousands of times until it learns

---

## Complete Training Loop Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE TRAINING LOOP                        │
└─────────────────────────────────────────────────────────────────┘

INITIALIZATION (Once)
  ↓
  • Create model with random weights
  • Create optimizer (SGD with learning rate)
  • Load training and test datasets

┌─────────────────────────────────────────────────────────────────┐
│ EPOCH LOOP (Outer Loop) - Repeat for N epochs                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. DATA SHUFFLING                                        │   │
│  │    • Shuffle training data indices                       │   │
│  │    • Prevents learning order-dependent patterns          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. MINI-BATCH LOOP (Inner Loop)                          │   │
│  │                                                          │   │
│  │    FOR each batch in shuffled dataset:                  │   │
│  │                                                          │   │
│  │    ┌─────────────────────────────────────────────────┐  │   │
│  │    │ a) Get Batch                                    │  │   │
│  │    │    • Extract batch_size samples                 │  │   │
│  │    │    • Images: [batch_size, 784]                  │  │   │
│  │    │    • Labels: [batch_size]                       │  │   │
│  │    └─────────────────────────────────────────────────┘  │   │
│  │               ↓                                         │   │
│  │    ┌─────────────────────────────────────────────────┐  │   │
│  │    │ b) Forward Pass                                 │  │   │
│  │    │    • Input → Hidden → ReLU → Output             │  │   │
│  │    │    • Produces predictions (logits)              │  │   │
│  │    └─────────────────────────────────────────────────┘  │   │
│  │               ↓                                         │   │
│  │    ┌─────────────────────────────────────────────────┐  │   │
│  │    │ c) Loss Computation                             │  │   │
│  │    │    • Compare predictions to true labels         │  │   │
│  │    │    • Cross-entropy loss (single number)         │  │   │
│  │    └─────────────────────────────────────────────────┘  │   │
│  │               ↓                                         │   │
│  │    ┌─────────────────────────────────────────────────┐  │   │
│  │    │ d) Backward Pass                                │  │   │
│  │    │    • Compute gradients (∂Loss/∂W, ∂Loss/∂b)    │  │   │
│  │    │    • Automatic differentiation (chain rule)     │  │   │
│  │    └─────────────────────────────────────────────────┘  │   │
│  │               ↓                                         │   │
│  │    ┌─────────────────────────────────────────────────┐  │   │
│  │    │ e) Weight Update                                │  │   │
│  │    │    • W = W - learning_rate × gradient           │  │   │
│  │    │    • Optimizer updates all parameters           │  │   │
│  │    └─────────────────────────────────────────────────┘  │   │
│  │               ↓                                         │   │
│  │    ┌─────────────────────────────────────────────────┐  │   │
│  │    │ f) Progress Tracking                            │  │   │
│  │    │    • Accumulate batch loss                      │  │   │
│  │    │    • Update progress bar                        │  │   │
│  │    └─────────────────────────────────────────────────┘  │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. EVALUATION PHASE                                      │   │
│  │    • Test on validation/test set                         │   │
│  │    • NO weight updates (inference only)                  │   │
│  │    • Compute accuracy                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. EPOCH SUMMARY                                         │   │
│  │    • Print epoch number, avg loss, accuracy              │   │
│  │    • Check if we should stop (early stopping)            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
  ↓
  Repeat for next epoch until:
  • Reached max epochs
  • Loss stops improving
  • Accuracy satisfactory
```

---

## 1. Initialization Phase

Before training begins, we need to set up all the components:

### Step 1.1: Create Model

```swift
// Create MLP with random initial weights
let model = MLPModel()

// Architecture:
//   Input: 784 (28×28 pixels)
//   Hidden: 512 neurons
//   Output: 10 classes (digits 0-9)
//
// Total parameters: ~407,000
```

**Why random initialization?**
- All zeros would make all neurons learn the same thing
- Random breaks symmetry, allowing diverse feature learning
- MLX uses Xavier/Glorot initialization for stable gradients

### Step 1.2: Create Optimizer

```swift
// Stochastic Gradient Descent with learning rate
let learningRate: Float = 0.01
let optimizer = SGD(learningRate: learningRate)
```

**What is the learning rate?**
- Controls how big each weight update step is
- Too large: unstable, overshoots minimum
- Too small: slow convergence, gets stuck
- 0.01 is a good starting point for MNIST

### Step 1.3: Load Dataset

```swift
// Load MNIST data
let mnistData = loadMNIST()

// Training data
let trainImages = mnistData.trainImages  // [60000, 784]
let trainLabels = mnistData.trainLabels  // [60000]

// Test data
let testImages = mnistData.testImages    // [10000, 784]
let testLabels = mnistData.testLabels    // [10000]
```

### Step 1.4: Set Hyperparameters

```swift
let epochs = 10          // Number of complete passes through dataset
let batchSize = 128      // Samples per mini-batch

// Derived values
let numSamples = 60_000  // MNIST training set size
let batchesPerEpoch = (numSamples + batchSize - 1) / batchSize  // ~469 batches
```

**Hyperparameters explained:**
- **Epochs**: How many times to see the entire dataset
- **Batch size**: Samples processed before updating weights
  - Larger: more stable gradients, more memory
  - Smaller: noisier gradients, faster iterations

---

## 2. Outer Loop: Epochs

An **epoch** is one complete pass through the entire training dataset.

### Why Multiple Epochs?

One pass isn't enough to learn complex patterns. The network needs to see each example multiple times:
- Epoch 1: Learn basic patterns (edges, shapes)
- Epoch 2-5: Learn combinations (curves forming digits)
- Epoch 6-10: Fine-tune and generalize

### Epoch Loop Structure

```swift
// =============================================================================
// TRAINING LOOP: Multiple Epochs
// =============================================================================

for epoch in 1...epochs {
    print("Epoch \(epoch)/\(epochs)")

    // -------------------------------------------------------------------------
    // TRAINING PHASE: Update weights on training data
    // -------------------------------------------------------------------------
    let avgLoss = trainMLPEpoch(
        model: model,
        optimizer: optimizer,
        trainImages: trainImages,
        trainLabels: trainLabels,
        batchSize: batchSize
    )

    // -------------------------------------------------------------------------
    // EVALUATION PHASE: Test without updating weights
    // -------------------------------------------------------------------------
    let accuracy = evaluateModel(
        model: model,
        testImages: testImages,
        testLabels: testLabels
    )

    // -------------------------------------------------------------------------
    // EPOCH SUMMARY
    // -------------------------------------------------------------------------
    print("Epoch \(epoch) - Loss: \(avgLoss), Accuracy: \(accuracy * 100)%")
}
```

---

## 3. Data Shuffling

Before processing mini-batches each epoch, we **shuffle** the training data:

```swift
// Create indices for all training samples
var indices = Array(0..<numSamples)  // [0, 1, 2, ..., 59999]

// Randomly shuffle
indices.shuffle()  // [42357, 891, 13452, ..., 5623]
```

### Why Shuffle?

**Without shuffling:**
```
Batch 1: [0, 0, 0, 0, ...]  ← All zeros
Batch 2: [0, 0, 0, 0, ...]  ← Still all zeros
Batch 3: [1, 1, 1, 1, ...]  ← Now all ones
```
The model would:
- Learn zeros, forget it when seeing ones
- Learn ones, forget it when seeing twos
- Never converge properly

**With shuffling:**
```
Batch 1: [7, 2, 1, 0, 9, 4, ...]  ← Mixed digits
Batch 2: [3, 5, 1, 7, 2, 8, ...]  ← Different mix
Batch 3: [0, 4, 6, 9, 1, 3, ...]  ← Another mix
```
The model sees diverse examples in each batch, leading to:
- Better generalization
- Faster convergence
- Escapes local minima

**Visual representation:**
```
Before Shuffling (sorted by class):
┌────────────────────────────────────────────────┐
│ 0000000...111111...222222...333333...999999    │
└────────────────────────────────────────────────┘
  ↓ shuffle()
After Shuffling (random order):
┌────────────────────────────────────────────────┐
│ 7,2,1,0,9,4,3,5,1,7,2,8,0,4,6,9,1,3,5,8,2...  │
└────────────────────────────────────────────────┘
```

---

## 4. Inner Loop: Mini-Batch Iteration

The **mini-batch loop** iterates through the shuffled dataset in chunks.

### Complete Mini-Batch Loop

```swift
// =============================================================================
// MINI-BATCH TRAINING LOOP
// =============================================================================

func trainMLPEpoch(
    model: MLPModel,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    batchSize: Int
) -> Float {
    let numSamples = trainImages.shape[0]  // 60,000
    var totalLoss: Float = 0
    var batchCount = 0

    // -------------------------------------------------------------------------
    // Step 1: SHUFFLE DATA
    // -------------------------------------------------------------------------
    var indices = Array(0..<numSamples)
    indices.shuffle()

    // -------------------------------------------------------------------------
    // Step 2: Setup Automatic Differentiation
    // -------------------------------------------------------------------------
    // This function computes both loss value AND gradients automatically!
    let lossAndGrad = valueAndGrad(model: model, mlpLoss)

    // -------------------------------------------------------------------------
    // Step 3: Progress Tracking
    // -------------------------------------------------------------------------
    let totalBatches = (numSamples + batchSize - 1) / batchSize
    let progressBar = ProgressBar(totalBatches: totalBatches)
    progressBar.start()

    // -------------------------------------------------------------------------
    // Step 4: ITERATE THROUGH MINI-BATCHES
    // -------------------------------------------------------------------------
    var start = 0
    while start < numSamples {
        // ---------------------------------------------------------------------
        // 4a. Extract Batch Indices
        // ---------------------------------------------------------------------
        let end = min(start + batchSize, numSamples)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)

        // ---------------------------------------------------------------------
        // 4b. Get Batch Data
        // ---------------------------------------------------------------------
        // Use indices to extract batch from shuffled data
        let batchImages = trainImages[idxArray]  // [batch_size, 784]
        let batchLabels = trainLabels[idxArray]  // [batch_size]

        // =====================================================================
        // 4c. THE TRAINING STEP: Forward → Loss → Backward → Update
        // =====================================================================

        // FORWARD PASS + LOSS + BACKWARD PASS (all automatic!)
        // This single line:
        //   1. Runs forward pass: images → hidden → ReLU → output → logits
        //   2. Computes loss: cross_entropy(logits, labels)
        //   3. Backward pass: computes all gradients via chain rule
        let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)

        // WEIGHT UPDATE: w = w - learning_rate × gradient
        optimizer.update(model: model, gradients: grads)

        // Force evaluation (MLX uses lazy evaluation)
        eval(model, optimizer)

        // ---------------------------------------------------------------------
        // 4d. Track Progress
        // ---------------------------------------------------------------------
        let lossValue = loss.item(Float.self)
        totalLoss += lossValue
        batchCount += 1

        // Update progress bar
        progressBar.update(batch: batchCount, loss: lossValue)

        // Move to next batch
        start = end
    }

    // -------------------------------------------------------------------------
    // Step 5: Finish Epoch
    // -------------------------------------------------------------------------
    progressBar.finish()

    // Return average loss across all batches
    return totalLoss / Float(batchCount)
}
```

### Mini-Batch Processing Timeline

```
Epoch 1 (60,000 samples, batch_size=128):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Batch 1:  [0-127]     → Forward → Loss → Backward → Update
Batch 2:  [128-255]   → Forward → Loss → Backward → Update
Batch 3:  [256-383]   → Forward → Loss → Backward → Update
...
Batch 468: [59904-60031] → Forward → Loss → Backward → Update
Batch 469: [60032-59999] → Forward → Loss → Backward → Update (last batch, 96 samples)

Total: 469 weight updates per epoch
```

---

## 5. The Training Cycle: Forward → Loss → Backward → Update

This is the **core of machine learning** - the four-step process that makes learning happen.

### Detailed Breakdown

```
┌──────────────────────────────────────────────────────────────┐
│         THE TRAINING CYCLE (Per Mini-Batch)                  │
└──────────────────────────────────────────────────────────────┘

INPUT: Batch of images [128, 784] and labels [128]

┌──────────────────────────────────────────────────────────────┐
│ STEP 1: FORWARD PASS                                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Images [128, 784]                                           │
│       ↓                                                      │
│  Hidden Layer: z1 = images @ W1 + b1                         │
│       ↓                                                      │
│  Activation: a1 = ReLU(z1)                                   │
│       ↓                                                      │
│  Output Layer: logits = a1 @ W2 + b2                         │
│       ↓                                                      │
│  Logits [128, 10]                                            │
│                                                              │
│  Example for one sample:                                     │
│    True label: 7                                             │
│    Predicted logits: [-1.2, 0.3, 4.8, -0.5, 1.1, ...]       │
│                                   ↑                          │
│                            Wrongly predicts "2"              │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: LOSS COMPUTATION                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Cross-Entropy Loss:                                         │
│                                                              │
│    loss = -log(softmax(logits)[true_class])                 │
│                                                              │
│  For each sample:                                            │
│    1. Convert logits to probabilities (softmax)              │
│    2. Look at probability for true class                     │
│    3. Take negative log                                      │
│                                                              │
│  Example:                                                    │
│    Logits for "7": [-1.2, 0.3, 4.8, ..., 2.1, ...]          │
│                                           ↑                  │
│                                      class 7: 2.1            │
│                                                              │
│    After softmax: [0.01, 0.03, 0.65, ..., 0.18, ...]        │
│                                            ↑                 │
│                                   P(class=7) = 0.18          │
│                                                              │
│    Loss = -log(0.18) = 1.71  (high because confidence low)  │
│                                                              │
│  Batch average: loss = mean(all 128 sample losses)          │
│                      = scalar value (e.g., 0.85)            │
│                                                              │
│  This single number measures "how wrong" the model is!       │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: BACKWARD PASS (Backpropagation)                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Compute gradients: ∂Loss/∂W1, ∂Loss/∂b1, ∂Loss/∂W2, ...   │
│                                                              │
│  Chain rule (automatic differentiation):                    │
│    Loss ← Softmax ← Output ← ReLU ← Hidden ← Input          │
│      ↓       ↓        ↓       ↓       ↓                     │
│    Gradients flow backward through each operation           │
│                                                              │
│  Gradient tells us:                                          │
│    • Which direction to change each weight                   │
│    • How much it affects the loss                            │
│                                                              │
│  Example gradient for one weight:                            │
│    ∂Loss/∂W1[234,67] = -0.042                               │
│                         ↑                                    │
│                   Negative means: increase this weight       │
│                   to reduce loss                             │
│                                                              │
│  Output: Gradient tensors matching parameter shapes         │
│    grads.W1: [784, 512]  (same shape as W1)                 │
│    grads.b1: [512]       (same shape as b1)                 │
│    grads.W2: [512, 10]   (same shape as W2)                 │
│    grads.b2: [10]        (same shape as b2)                 │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: WEIGHT UPDATE (Gradient Descent)                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  For each parameter:                                         │
│    new_weight = old_weight - learning_rate × gradient        │
│                                                              │
│  Example (learning_rate = 0.01):                             │
│                                                              │
│    W1[234,67] = 0.523 - 0.01 × (-0.042)                     │
│               = 0.523 + 0.00042                              │
│               = 0.52342                                      │
│                                                              │
│    The weight increased slightly (gradient was negative)     │
│                                                              │
│  This happens for ALL ~407,000 parameters simultaneously!    │
│                                                              │
│  After update:                                               │
│    • Weights have changed slightly                           │
│    • Model should perform slightly better                    │
│    • Repeat for next batch to keep improving                 │
└──────────────────────────────────────────────────────────────┘
```

### Code for Training Cycle

```swift
// This is what happens inside the mini-batch loop:

// STEP 1-3: Forward pass + Loss + Backward pass (automatic!)
let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)
// Returns:
//   loss: Scalar MLXArray (e.g., 0.85)
//   grads: Dictionary of gradients for each parameter

// STEP 4: Weight update
optimizer.update(model: model, gradients: grads)
// Updates all weights: w = w - lr × grad

// Force evaluation (MLX is lazy)
eval(model, optimizer)
```

**What `valueAndGrad` does:**
```swift
// Under the hood (conceptual):
func valueAndGrad(model, lossFunc) {
    return { model, images, labels in
        // 1. FORWARD PASS (with gradient tracking)
        let logits = model(images)

        // 2. LOSS COMPUTATION
        let loss = crossEntropy(logits, labels)

        // 3. BACKWARD PASS (automatic chain rule)
        let grads = computeGradients(loss, model.parameters)

        return (loss, grads)
    }
}
```

---

## 6. Progress Tracking

During training, we track:
1. **Per-batch loss**: How well we're doing on current batch
2. **Average epoch loss**: Overall performance this epoch
3. **Progress bar**: Visual feedback

### Progress Bar Example

```
Epoch 1/10
[=====================================>            ] 75% (351/469) Loss: 0.234
```

### Implementation

```swift
class ProgressBar {
    let totalBatches: Int
    var currentBatch: Int = 0

    func update(batch: Int, loss: Float) {
        currentBatch = batch
        let percent = Int((Float(batch) / Float(totalBatches)) * 100)

        // Print progress
        print("\r[\(progressString(percent))] \(percent)% (\(batch)/\(totalBatches)) Loss: \(String(format: "%.3f", loss))", terminator: "")
    }

    func finish() {
        print("")  // New line after progress bar
    }
}
```

### Tracking Over Time

```
Epoch 1: Batch 100, Loss: 0.856
Epoch 1: Batch 200, Loss: 0.432
Epoch 1: Batch 300, Loss: 0.287
Epoch 1: Batch 400, Loss: 0.198
Epoch 1: Average Loss: 0.321

Epoch 2: Batch 100, Loss: 0.156
Epoch 2: Batch 200, Loss: 0.134
...

Notice: Loss decreases both within epoch and across epochs!
```

---

## 7. Evaluation Phase

After each training epoch, we evaluate on the **test set** to see how well the model generalizes.

### Key Differences: Training vs Evaluation

```
┌─────────────────────────┬─────────────────────────┐
│      TRAINING           │      EVALUATION         │
├─────────────────────────┼─────────────────────────┤
│ Use training data       │ Use test data           │
│ Update weights          │ NO weight updates       │
│ Compute gradients       │ No gradients needed     │
│ Track loss              │ Track accuracy          │
│ Mini-batches            │ Can use full dataset    │
│ Data shuffled           │ Order doesn't matter    │
└─────────────────────────┴─────────────────────────┘
```

### Evaluation Function

```swift
func evaluateModel(
    model: MLPModel,
    testImages: MLXArray,
    testLabels: MLXArray
) -> Float {
    // -------------------------------------------------------------------------
    // Forward pass ONLY (no gradient computation needed)
    // -------------------------------------------------------------------------
    let logits = model(testImages)  // [10000, 10]

    // -------------------------------------------------------------------------
    // Get predictions
    // -------------------------------------------------------------------------
    // Find class with highest logit value
    let predictions = argMax(logits, axis: 1)  // [10000]

    // -------------------------------------------------------------------------
    // Compute accuracy
    // -------------------------------------------------------------------------
    // Count how many predictions match true labels
    let correct = predictions .== testLabels  // [10000] of booleans
    let accuracy = mean(correct).item(Float.self)  // Single value: 0.0 to 1.0

    return accuracy
}
```

### Evaluation Example

```swift
// After epoch 1
let accuracy = evaluateModel(model, testImages, testLabels)
// accuracy = 0.9234 (92.34% correct)

print("Test Accuracy: \(accuracy * 100)%")
// Output: Test Accuracy: 92.34%
```

### What Accuracy Means

```
Test set: 10,000 images

Predictions:
┌──────────┬─────────┬────────────┬────────┐
│  Image   │  True   │ Predicted  │ Result │
├──────────┼─────────┼────────────┼────────┤
│   img1   │    7    │     7      │   ✓    │
│   img2   │    2    │     2      │   ✓    │
│   img3   │    1    │     7      │   ✗    │
│   img4   │    0    │     0      │   ✓    │
│   ...    │   ...   │    ...     │  ...   │
│  img10000│    4    │     4      │   ✓    │
└──────────┴─────────┴────────────┴────────┘

Correct: 9,234 out of 10,000
Accuracy: 9,234 / 10,000 = 0.9234 = 92.34%
```

---

## 8. Complete Training Example with Full Annotations

Here's a complete, runnable training loop with line-by-line explanations:

```swift
// =============================================================================
// COMPLETE TRAINING LOOP EXAMPLE
// =============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers

func trainMNIST() {
    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    // Step 1: Create model with random weights
    let model = MLPModel()
    print("Model created with \(model.parameters().count) parameter tensors")

    // Step 2: Create optimizer
    let learningRate: Float = 0.01
    let optimizer = SGD(learningRate: learningRate)
    print("Optimizer: SGD with learning rate \(learningRate)")

    // Step 3: Load MNIST dataset
    print("Loading MNIST dataset...")
    let mnist = loadMNIST()
    let trainImages = mnist.trainImages  // [60000, 784]
    let trainLabels = mnist.trainLabels  // [60000]
    let testImages = mnist.testImages    // [10000, 784]
    let testLabels = mnist.testLabels    // [10000]
    print("Dataset loaded: 60,000 training samples, 10,000 test samples")

    // Step 4: Set hyperparameters
    let epochs = 10
    let batchSize = 128
    print("Training for \(epochs) epochs with batch size \(batchSize)")
    print("")

    // =========================================================================
    // TRAINING LOOP: EPOCHS
    // =========================================================================

    for epoch in 1...epochs {
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("Epoch \(epoch)/\(epochs)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        // =====================================================================
        // TRAINING PHASE: One complete pass through training data
        // =====================================================================

        let epochStartTime = Date()

        // Train for one epoch (see detailed function below)
        let avgLoss = trainMLPEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: batchSize
        )

        let epochDuration = Date().timeIntervalSince(epochStartTime)

        // =====================================================================
        // EVALUATION PHASE: Test on held-out data
        // =====================================================================

        print("Evaluating on test set...")
        let accuracy = evaluateModel(
            model: model,
            testImages: testImages,
            testLabels: testLabels
        )

        // =====================================================================
        // EPOCH SUMMARY
        // =====================================================================

        print("")
        print("Epoch \(epoch) Summary:")
        print("  Average Loss: \(String(format: "%.4f", avgLoss))")
        print("  Test Accuracy: \(String(format: "%.2f", accuracy * 100))%")
        print("  Duration: \(String(format: "%.1f", epochDuration))s")
        print("")

        // =====================================================================
        // EARLY STOPPING (Optional)
        // =====================================================================

        if accuracy > 0.98 {
            print("🎉 Reached 98% accuracy! Stopping early.")
            break
        }
    }

    // =========================================================================
    // TRAINING COMPLETE
    // =========================================================================

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Training complete!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}

// =============================================================================
// TRAINING EPOCH FUNCTION: Inner loop with mini-batches
// =============================================================================

func trainMLPEpoch(
    model: MLPModel,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    batchSize: Int
) -> Float {

    let numSamples = trainImages.shape[0]  // 60,000
    var totalLoss: Float = 0
    var batchCount = 0

    // =========================================================================
    // STEP 1: SHUFFLE DATA
    // =========================================================================
    // Create random permutation of indices
    var indices = Array(0..<numSamples)
    indices.shuffle()

    print("  Data shuffled: presenting samples in random order")

    // =========================================================================
    // STEP 2: SETUP AUTOMATIC DIFFERENTIATION
    // =========================================================================
    // Create function that computes loss AND gradients
    let lossAndGrad = valueAndGrad(model: model, mlpLoss)

    print("  Automatic differentiation enabled")

    // =========================================================================
    // STEP 3: PROGRESS BAR SETUP
    // =========================================================================
    let totalBatches = (numSamples + batchSize - 1) / batchSize
    let progressBar = ProgressBar(totalBatches: totalBatches)

    print("  Training on \(totalBatches) mini-batches:")
    progressBar.start()

    // =========================================================================
    // STEP 4: MINI-BATCH LOOP
    // =========================================================================
    var start = 0
    while start < numSamples {

        // ---------------------------------------------------------------------
        // 4a. Get batch indices
        // ---------------------------------------------------------------------
        let end = min(start + batchSize, numSamples)
        let batchIndices = Array(indices[start..<end]).map { Int32($0) }
        let idxArray = MLXArray(batchIndices)

        // ---------------------------------------------------------------------
        // 4b. Extract batch data using shuffled indices
        // ---------------------------------------------------------------------
        let batchImages = trainImages[idxArray]  // [batch_size, 784]
        let batchLabels = trainLabels[idxArray]  // [batch_size]

        // =====================================================================
        // 4c. THE TRAINING CYCLE: Forward → Loss → Backward → Update
        // =====================================================================

        // FORWARD PASS + LOSS COMPUTATION + BACKWARD PASS
        // This computes:
        //   1. predictions = model(batchImages)
        //   2. loss = crossEntropy(predictions, batchLabels)
        //   3. gradients = backprop(loss)
        let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)

        // WEIGHT UPDATE
        // For each parameter: w = w - learning_rate × gradient
        optimizer.update(model: model, gradients: grads)

        // Force evaluation (MLX is lazy - doesn't compute until needed)
        eval(model, optimizer)

        // ---------------------------------------------------------------------
        // 4d. Track progress
        // ---------------------------------------------------------------------
        let lossValue = loss.item(Float.self)
        totalLoss += lossValue
        batchCount += 1

        // Update progress bar
        progressBar.update(batch: batchCount, loss: lossValue)

        // Move to next batch
        start = end
    }

    // =========================================================================
    // STEP 5: FINISH EPOCH
    // =========================================================================
    progressBar.finish()

    // Return average loss across all batches
    return totalLoss / Float(batchCount)
}

// =============================================================================
// EVALUATION FUNCTION: Test accuracy without training
// =============================================================================

func evaluateModel(
    model: MLPModel,
    testImages: MLXArray,
    testLabels: MLXArray
) -> Float {

    // Forward pass (no gradient computation needed)
    let logits = model(testImages)

    // Get predicted classes (argmax)
    let predictions = argMax(logits, axis: 1)

    // Count correct predictions
    let correct = predictions .== testLabels

    // Return accuracy (fraction correct)
    return mean(correct).item(Float.self)
}

// =============================================================================
// RUN TRAINING
// =============================================================================

trainMNIST()
```

### Expected Output

```
Model created with 4 parameter tensors
Optimizer: SGD with learning rate 0.01
Loading MNIST dataset...
Dataset loaded: 60,000 training samples, 10,000 test samples
Training for 10 epochs with batch size 128

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 1/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Data shuffled: presenting samples in random order
  Automatic differentiation enabled
  Training on 469 mini-batches:
[=====================================>] 100% (469/469) Loss: 0.234
Evaluating on test set...

Epoch 1 Summary:
  Average Loss: 0.3214
  Test Accuracy: 91.23%
  Duration: 12.3s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 2/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Data shuffled: presenting samples in random order
  Automatic differentiation enabled
  Training on 469 mini-batches:
[=====================================>] 100% (469/469) Loss: 0.156
Evaluating on test set...

Epoch 2 Summary:
  Average Loss: 0.1876
  Test Accuracy: 94.56%
  Duration: 11.8s

...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 8/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Data shuffled: presenting samples in random order
  Automatic differentiation enabled
  Training on 469 mini-batches:
[=====================================>] 100% (469/469) Loss: 0.021
Evaluating on test set...

Epoch 8 Summary:
  Average Loss: 0.0312
  Test Accuracy: 98.12%
  Duration: 11.5s

🎉 Reached 98% accuracy! Stopping early.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 9. Training Progression Visualization

### How Loss Decreases

```
Loss Over Time:

1.0 ┤
    │●
0.9 ┤ ●
    │  ●
0.8 ┤   ●
    │    ●
0.7 ┤     ●
    │      ●
0.6 ┤       ●
    │        ●●
0.5 ┤          ●●
    │            ●●
0.4 ┤              ●●●
    │                 ●●●
0.3 ┤                    ●●●●
    │                        ●●●●●
0.2 ┤                             ●●●●●●
    │                                   ●●●●●●●
0.1 ┤                                          ●●●●●●
    │                                                ●●●●
0.0 ┤
    └┬────┬────┬────┬────┬────┬────┬────┬────┬────┬
     0    1    2    3    4    5    6    7    8    9
                         Epoch

Note: Rapid improvement early, then gradual refinement
```

### How Accuracy Increases

```
Accuracy Over Time:

100%┤                                          ●●●●●
    │                                    ●●●●●●
 95%┤                              ●●●●●●
    │                        ●●●●●●
 90%┤                  ●●●●●●
    │            ●●●●●●
 85%┤      ●●●●●●
    │  ●●●●
 80%┤●●
    │
 75%┤
    │
 70%┤
    └┬────┬────┬────┬────┬────┬────┬────┬────┬────┬
     0    1    2    3    4    5    6    7    8    9
                         Epoch

Typical MNIST MLP progression:
  Epoch 1:  80-85%  (learns basic patterns)
  Epoch 2:  90-92%  (refines features)
  Epoch 5:  95-96%  (good generalization)
  Epoch 10: 97-98%  (near-optimal for MLP)
```

---

## 10. Key Takeaways

### Essential Concepts

1. **Two Nested Loops**:
   - Outer: Epochs (complete passes through dataset)
   - Inner: Mini-batches (chunks of data)

2. **Data Shuffling**:
   - Shuffle once per epoch
   - Prevents order-dependent learning
   - Helps generalization

3. **The Training Cycle** (per mini-batch):
   ```
   Forward Pass → Loss → Backward Pass → Update Weights
   ```

4. **Progress Tracking**:
   - Monitor loss (training performance)
   - Monitor accuracy (test performance)
   - Watch for overfitting

5. **Evaluation**:
   - Always test on held-out data
   - No gradient computation needed
   - Measures generalization ability

### Why This Works

**Stochastic Gradient Descent (SGD):**
- Update weights after each mini-batch (not entire dataset)
- Noisy gradients help escape local minima
- Much faster than computing on full dataset

**Multiple Epochs:**
- Network needs to see examples multiple times
- Early epochs: learn basic patterns
- Later epochs: fine-tune and generalize

**Batching:**
- Efficient GPU utilization
- Stable gradient estimates
- Memory efficient

---

## 11. Common Questions

### Q: Why use mini-batches instead of full dataset?

**Full dataset (Batch Gradient Descent):**
- Pros: Stable, smooth convergence
- Cons: Slow, requires lots of memory, can get stuck

**Single sample (Online/Stochastic):**
- Pros: Fast updates, escapes local minima
- Cons: Very noisy, unstable

**Mini-batches (best of both):**
- Pros: Good gradient estimates, efficient, GPU-friendly
- Cons: One more hyperparameter to tune

### Q: How do I choose batch size?

**Common values:** 32, 64, 128, 256

**Larger batches (256+):**
- More stable gradients
- Better GPU utilization
- Requires more memory
- May generalize worse

**Smaller batches (32-64):**
- More noise → helps escape local minima
- Less memory
- More updates per epoch
- May generalize better

**For MNIST:** 128 is a good balance

### Q: How many epochs should I train?

Train until:
- Loss stops decreasing
- Accuracy plateaus
- Validation accuracy starts decreasing (overfitting)

**For MNIST MLP:**
- 5-10 epochs usually sufficient
- More won't hurt but won't help much

### Q: What if loss increases?

Possible causes:
- Learning rate too high (most common)
- Bug in code
- Gradient explosion
- Bad initialization

**Solution:** Reduce learning rate by 10x

### Q: What's the difference between loss and accuracy?

**Loss:**
- Continuous measure of error
- What the network optimizes
- Can be any positive value
- Lower is better

**Accuracy:**
- Discrete measure (correct/incorrect)
- Easy to interpret
- Bounded 0-100%
- Higher is better

---

## 12. References

### Code Locations

- **Full MLP Model**: `Sources/MNISTMLX/MLPModel.swift`
- **Training Function**: `trainMLPEpoch()` in MLPModel.swift
- **Simple Example**: `mlp_simple.swift` (educational XOR example)

### Related Documentation

- **Forward Pass**: `docs/forward-pass.md`
- **Loss Computation**: `docs/loss-computation.md`
- **Backpropagation**: `docs/backpropagation.md`
- **Weight Updates**: `docs/weight-updates.md`
- **MLP Overview**: `docs/mlp-walkthrough.md`

---

## Summary

The **complete training loop** brings together all the pieces of neural network learning:

1. **Initialization**: Create model, optimizer, load data
2. **Epochs**: Multiple complete passes through dataset
3. **Shuffling**: Randomize order each epoch
4. **Mini-batches**: Process data in chunks
5. **Training Cycle**: Forward → Loss → Backward → Update
6. **Progress Tracking**: Monitor performance
7. **Evaluation**: Test on held-out data
8. **Iteration**: Repeat until converged

This is the foundation of how neural networks learn - a simple but powerful algorithm that enables models to learn complex patterns from data!
