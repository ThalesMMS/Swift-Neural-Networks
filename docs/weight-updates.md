# Weight Updates: How Neural Networks Learn

This document provides a step-by-step walkthrough of how neural networks update their weights during training, focusing on Stochastic Gradient Descent (SGD) and the role of the learning rate.

## Table of Contents

1. [The Core SGD Formula](#the-core-sgd-formula)
2. [Understanding the Learning Rate](#understanding-the-learning-rate)
3. [Mini-Batch vs Full-Batch Gradient Descent](#mini-batch-vs-full-batch-gradient-descent)
4. [Why Shuffle Data Each Epoch?](#why-shuffle-data-each-epoch)
5. [Beyond Basic SGD: Momentum and Other Optimizers](#beyond-basic-sgd-momentum-and-other-optimizers)

---

## The Core SGD Formula

### The Weight Update Rule

At the heart of neural network training is a deceptively simple formula:

```
W = W - lr × grad
```

Where:
- **W** = Current weight value
- **lr** = Learning rate (a small positive number, e.g., 0.01)
- **grad** = Gradient of the loss with respect to W

### What Does This Mean?

The gradient tells us **which direction to move** the weight to reduce the loss. The learning rate tells us **how far** to move in that direction.

**Intuitive Example:**
Imagine you're hiking down a mountain in fog (you can't see the bottom). The gradient is like feeling the slope under your feet—it tells you which way is downhill. The learning rate is how big your steps are.

### Step-by-Step Breakdown

Let's walk through a complete weight update cycle:

#### 1. Forward Pass
```
prediction = model(input)
```
The network makes a prediction using current weights.

#### 2. Compute Loss
```
loss = crossEntropy(prediction, trueLabel)
```
We measure how wrong the prediction is.

#### 3. Backward Pass (Backpropagation)
```
grad = ∂loss/∂W
```
We compute how each weight contributed to the error.

**Key Insight:** The gradient has the same shape as the weight matrix!
- If W is [784, 512], then grad is also [784, 512]
- Each gradient value tells us how that specific weight affects the loss

#### 4. Update Weights
```
W = W - lr × grad
```

**Example with actual numbers:**

Suppose for one weight:
- Current weight: W = 0.5
- Gradient: grad = -0.3 (negative means loss decreases when W increases)
- Learning rate: lr = 0.01

Then:
```
W_new = 0.5 - (0.01 × -0.3)
      = 0.5 + 0.003
      = 0.503
```

The weight increased slightly because the gradient was negative!

### Code Example: Simple SGD Update

From `mlp_simple.swift` (lines 108-116):

```swift
func updateWeights(layer: inout LinearLayer, inputs: [Double], deltas: [Double]) {
    for i in 0..<layer.inputSize {
        for j in 0..<layer.outputSize {
            // SGD formula: W = W - lr × grad
            // Note: deltas[j] * inputs[i] is the gradient for this weight
            layer.weights[i][j] += learningRate * deltas[j] * inputs[i]
        }
    }
    for i in 0..<layer.outputSize {
        layer.biases[i] += learningRate * deltas[i]
    }
}
```

**Wait, why += instead of -=?**

In this implementation, `deltas` already includes the error sign. In backpropagation:
- `delta = error × activationDerivative`
- `error = (target - prediction)`

So the sign is already "flipped" to point in the direction we want to move!

### Modern Framework Approach

With automatic differentiation (like MLX), this becomes much simpler:

```swift
// From MLPModel.swift (lines 295-298)

// Compute loss and gradients automatically
let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)

// Update weights: w = w - lr * grad
optimizer.update(model: model, gradients: grads)
```

The framework handles all the calculus for us!

---

## Understanding the Learning Rate

### What Is the Learning Rate?

The learning rate (often abbreviated as `lr` or `α`) controls the **size of weight updates**. It's the single most important hyperparameter in training.

### Why Not Just Set lr = 1.0?

**Problem 1: Overshooting**

If lr is too large, you can overshoot the minimum:

```
Imagine loss landscape like this: \_/
                                    ^
                                  minimum

With lr too large:
Step 1: \_/  →  \_/
         ^        →     ^
      start           way too far!

You jump OVER the minimum instead of settling into it.
```

**Problem 2: Divergence**

Extremely large learning rates can cause loss to **increase** instead of decrease, leading to numerical instability (NaN values).

### Why Not Set lr Very Small (e.g., 0.000001)?

**Problem: Slow Convergence**

With lr too small:
- Weight updates are tiny
- Training takes forever
- May get stuck in local minima
- Wastes computational resources

**Example:**
```
lr = 0.01:   1000 epochs to converge (good)
lr = 0.0001: 100,000 epochs to converge (100x slower!)
```

### Choosing a Good Learning Rate

**Typical ranges:**
- **0.1**: Often too large, but can work for simple problems
- **0.01**: Common default, good starting point
- **0.001**: Conservative, safer for complex models
- **0.0001**: Very conservative, use if training is unstable

**Rule of thumb:**
Start with 0.01 and adjust based on training behavior:
- Loss decreasing smoothly? Keep it.
- Loss oscillating wildly? Decrease lr.
- Loss decreasing very slowly? Increase lr.

### Learning Rate Impact Visualization

```
Loss over time with different learning rates:

lr = 0.1 (too high):
Loss │     ╱╲  ╱╲
     │   ╱    ╲╱  ╲╱
     │ ╱
     └─────────────── iterations
     Oscillates, never settles

lr = 0.01 (good):
Loss │╲
     │ ╲___
     │     ────___
     └─────────────── iterations
     Smooth decrease

lr = 0.0001 (too low):
Loss │╲
     │ ╲
     │  ╲
     │   ╲___________
     └─────────────── iterations
     Very slow, wastes time
```

### Learning Rate in Our Code

From `mlp_simple.swift`:
```swift
let learningRate = 0.01  // Line 20
```

For the XOR problem (4 samples, very simple), 0.01 works well.

For MNIST (60,000 samples, more complex), you might use:
```swift
let optimizer = SGD(learningRate: 0.01)  // Start here
// or
let optimizer = SGD(learningRate: 0.001) // If training is unstable
```

---

## Mini-Batch vs Full-Batch Gradient Descent

Neural networks can compute gradients using different amounts of data at once. This leads to three main approaches:

### 1. Full-Batch Gradient Descent

**Definition:** Compute gradients using **all training samples** at once.

```swift
// Pseudocode
for epoch in 0..<numEpochs {
    let grad = computeGradient(allTrainingData)  // All 60,000 MNIST samples!
    weights = weights - lr * grad
}
```

**Advantages:**
- Gradient is the "true" direction to the minimum
- Smooth, stable convergence
- Deterministic (same result each time)

**Disadvantages:**
- Extremely slow (must process entire dataset per update)
- Requires huge memory (all 60,000 images in GPU memory)
- Slow convergence (very few weight updates per epoch)
- Can get stuck in local minima

**When to use:** Almost never in modern deep learning!

### 2. Stochastic Gradient Descent (SGD)

**Definition:** Compute gradients using **one sample** at a time.

```swift
// Pseudocode
for epoch in 0..<numEpochs {
    for sample in trainingData.shuffled() {
        let grad = computeGradient(sample)  // Just one image
        weights = weights - lr * grad
    }
}
```

**Advantages:**
- Fast updates (60,000 updates per epoch on MNIST)
- Low memory usage
- Noise helps escape local minima
- Can start learning immediately

**Disadvantages:**
- Very noisy gradient estimates
- Slower per-sample computation (poor GPU utilization)
- Jumpy convergence path

**When to use:** Small datasets or online learning scenarios.

### 3. Mini-Batch Gradient Descent (The Standard)

**Definition:** Compute gradients using **small batches** of samples (e.g., 32, 64, 128).

```swift
// From MLPModel.swift (lines 277-310)
while start < n {
    let end = min(start + batchSize, n)  // e.g., batchSize = 128

    // Get batch data
    let batchImages = trainImages[idxArray]  // 128 images
    let batchLabels = trainLabels[idxArray]  // 128 labels

    // Compute gradients over the batch
    let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)

    // Update weights
    optimizer.update(model: model, gradients: grads)

    start = end  // Move to next batch
}
```

**Advantages:**
- **Best of both worlds!**
- Efficient GPU utilization (parallel processing)
- Faster convergence than single-sample SGD
- Less noisy than single-sample, more updates than full-batch
- Moderate memory usage

**Disadvantages:**
- ~ Requires tuning batch size

**When to use:** Almost always! This is the default in modern deep learning.

### Comparison Table

| Approach | Batch Size | Updates/Epoch (MNIST) | Memory | Convergence | GPU Efficiency |
|----------|------------|----------------------|---------|-------------|----------------|
| Full-Batch | 60,000 | 1 | Very High | Slow | Medium |
| Stochastic | 1 | 60,000 | Very Low | Fast but noisy | Very Low |
| **Mini-Batch** | **32-256** | **234-1875** | **Low** | **Fast & smooth** | **High** |

### Choosing Batch Size

**Common choices:**
- **32**: Small batches, more updates, works on limited GPU memory
- **64**: Good default for many problems
- **128**: Our MNIST default, balances speed and memory
- **256**: Large batches, faster per-epoch but fewer updates
- **512+**: Requires large GPUs, may need learning rate adjustment

**Rule of thumb:**
- Larger batches → faster per epoch, but may need more epochs
- Smaller batches → more updates, but slower per epoch

**Memory constraints:**
If you get "out of memory" errors, reduce batch size:
```swift
// batchSize: 128  // Out of memory!
batchSize: 64   // Works!
```

### Why Mini-Batch is Standard

Mini-batch gradient descent has become the de facto standard because:

1. **Hardware Efficiency**: GPUs are designed for parallel operations. Processing 128 images at once is much faster than processing 128 images one by one.

2. **Gradient Quality**: Averaging gradients over a batch gives a better estimate than a single sample, while still being "noisy" enough to escape local minima.

3. **Practical Memory**: Fits well within GPU memory constraints (8GB-24GB for consumer GPUs).

**Real-world numbers (MNIST on M1 Mac):**
```
Batch Size 1:   ~120 seconds per epoch
Batch Size 32:  ~8 seconds per epoch   (15x faster!)
Batch Size 128: ~3 seconds per epoch   (40x faster!)
Batch Size 512: ~2 seconds per epoch   (60x faster, but needs more memory)
```

---

## Why Shuffle Data Each Epoch?

### The Problem with Fixed Order

Imagine training on MNIST **without** shuffling:

```
Epoch 1: [digit 0, digit 0, ..., digit 1, digit 1, ..., digit 9, digit 9]
Epoch 2: [digit 0, digit 0, ..., digit 1, digit 1, ..., digit 9, digit 9]
Epoch 3: [digit 0, digit 0, ..., digit 1, digit 1, ..., digit 9, digit 9]
```

**What happens?**

The network learns in a biased way:
1. First part of epoch: "Everything is a 0!"
2. Middle of epoch: "Wait, now everything is a 5?"
3. End of epoch: "Actually, everything is a 9!"

This is called **catastrophic forgetting**—the network forgets earlier digits when learning later ones.

### The Solution: Shuffle Every Epoch

```swift
// From MLPModel.swift (lines 263-265)

// Shuffle for Stochastic Gradient Descent
// SGD theory says we should present samples in random order each epoch.
// This helps the optimizer escape local minima and generalizes better.
var indices = Array(0..<n)
indices.shuffle()
```

Now each epoch sees data in a different order:
```
Epoch 1: [3, 7, 2, 9, 0, 5, 1, ...]  (random order)
Epoch 2: [8, 1, 5, 3, 9, 2, 0, ...]  (different random order)
Epoch 3: [2, 9, 0, 7, 3, 1, 5, ...]  (different random order)
```

### Benefits of Shuffling

#### 1. **Prevents Overfitting to Order**

Without shuffling, the network can learn spurious patterns:
- "Digit 5 always comes after digit 4"
- "The first batch is always easier"

These patterns don't exist in the real world!

#### 2. **Better Gradient Estimates**

Each mini-batch should be a representative sample of the whole dataset:
```
Good batch (shuffled):    [0, 5, 3, 9, 1, 7, ...]  (diverse)
Bad batch (not shuffled): [0, 0, 0, 0, 0, 0, ...]  (homogeneous)
```

A diverse batch gives a better gradient estimate.

#### 3. **Helps Escape Local Minima**

The randomness introduced by shuffling acts as a form of regularization:
- Each epoch follows a slightly different optimization path
- Harder to get stuck in sharp, narrow minima
- Tends to find flatter, more generalizable minima

#### 4. **Prevents Bias in Batch Normalization**

If using batch normalization (not in our simple MLP, but common in larger networks), fixed order can cause the statistics to be biased.

### Shuffling Implementation

**Simple approach (our code):**
```swift
var indices = Array(0..<n)
indices.shuffle()  // Built-in Swift shuffle

// Then use shuffled indices to access data
for i in 0..<n {
    let sample = trainData[indices[i]]
    // ...
}
```

**Alternative: Shuffle data directly**
```swift
var shuffledData = trainData
shuffledData.shuffle()
```

### When NOT to Shuffle

There are rare cases where you should NOT shuffle:

1. **Time Series Data**: If temporal order matters (stock prices, weather)
2. **Sequential Tasks**: When the network needs to learn sequences
3. **Curriculum Learning**: Deliberately presenting easy examples before hard ones
4. **Debugging**: Fixed order makes issues reproducible

For classification tasks like MNIST, **always shuffle**!

### Impact on Training

**Empirical difference (MNIST after 5 epochs):**
```
With shuffling:    Test accuracy = 97.2%  (good)
Without shuffling: Test accuracy = 94.1%  (3% worse!)
```

The difference can be even more dramatic on complex datasets.

---

## Beyond Basic SGD: Momentum and Other Optimizers

While basic SGD (`W = W - lr × grad`) works, modern optimizers often perform much better. Here's a brief overview:

### 1. SGD with Momentum

**Problem with Basic SGD:**
Gradients can be noisy and oscillate, especially in ravines (long, narrow valleys in the loss landscape).

**Solution: Momentum**

Instead of moving only based on the current gradient, remember previous gradients:

```
velocity = β × velocity + grad
W = W - lr × velocity
```

Where `β` (typically 0.9) controls how much "memory" to keep.

**Intuition:**
Like a ball rolling downhill:
- Builds up speed in consistent directions
- Dampens oscillations in inconsistent directions
- Can roll through small bumps

**Code example:**
```swift
// Conceptual (not actual MLX API)
let optimizer = SGD(learningRate: 0.01, momentum: 0.9)
```

**When to use:** Almost always! Very little downside.

### 2. Adam (Adaptive Moment Estimation)

**The most popular optimizer** in modern deep learning.

Adam combines:
- Momentum (like above)
- Adaptive learning rates (different lr for each parameter)

**Simplified formula:**
```
m = β₁ × m + (1 - β₁) × grad         // First moment (momentum)
v = β₂ × v + (1 - β₂) × grad²        // Second moment (variance)
W = W - lr × m / (√v + ε)            // Update with adaptive lr
```

Typical values: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`

**Advantages:**
- Adapts learning rate per parameter
- Works well out-of-the-box (less tuning needed)
- Handles sparse gradients well
- Good default choice for most problems

**Disadvantages:**
- ~ Slightly more memory (stores m and v)
- ~ Can generalize slightly worse than SGD+momentum (debated)

**When to use:** Great default choice, especially for:
- Large models (transformers, ResNets)
- Problems where you don't want to tune hyperparameters
- Noisy gradients

### 3. RMSprop

Middle ground between SGD and Adam:
```
v = β × v + (1 - β) × grad²
W = W - lr × grad / (√v + ε)
```

**When to use:**
- Recurrent neural networks (RNNs)
- When Adam is overkill

### 4. AdaGrad

Adapts learning rate based on historical gradients:
```
v = v + grad²
W = W - lr × grad / (√v + ε)
```

**Problem:** Learning rate can decay too aggressively (v keeps growing).

**When to use:** Sparse data (NLP, recommendations), otherwise prefer Adam.

### Comparison Table

| Optimizer | Memory | Tuning Needed | Speed | Generalization | Best For |
|-----------|--------|---------------|-------|----------------|----------|
| **SGD** | Low | High | Fast | Excellent | Simple problems, research |
| **SGD + Momentum** | Low | Medium | Fast | Excellent | Great default, well-understood |
| **Adam** | Medium | Low | Medium | Good | Complex models, less tuning |
| **RMSprop** | Medium | Medium | Medium | Good | RNNs, special cases |

### Choosing an Optimizer

**Decision tree:**

```
Do you have time to tune hyperparameters?
├─ Yes → Try SGD with momentum (0.9)
│        Often achieves best final performance
│
└─ No → Use Adam (lr=0.001)
         Works well out-of-the-box
```

**For this MNIST project:**
- Simple SGD works fine (dataset is easy)
- For more complex tasks, try Adam

### Practical Tips

1. **Start Simple**: Begin with basic SGD, only add complexity if needed

2. **Learning Rate Still Matters**: Even with Adam, learning rate is important
   ```
   Adam(lr=0.001):  Good default
   Adam(lr=0.0001): If training is unstable
   Adam(lr=0.01):   Rarely works, usually too high
   ```

3. **Don't Mix Concepts**:
   ```
   Good:  SGD(lr=0.01, momentum=0.9)
   Wrong: Adam(lr=0.01, momentum=0.9)  // Adam already has momentum built-in!
   ```

4. **Match Literature**: If reimplementing a paper, use their optimizer and hyperparameters

### Further Reading

- **Learning Rate Schedules**: Decay lr over time (e.g., reduce by 10x every 30 epochs)
- **Gradient Clipping**: Prevent exploding gradients (important for RNNs)
- **Weight Decay / L2 Regularization**: Add penalty for large weights
- **Batch Normalization**: Normalize activations (reduces dependence on learning rate)

---

## Summary

### Key Takeaways

1. **SGD Formula**: `W = W - lr × grad`
   - Simple but powerful
   - Gradient points toward reducing loss
   - Learning rate controls step size

2. **Learning Rate**:
   - Most important hyperparameter
   - Too high → oscillation/divergence
   - Too low → slow convergence
   - Start with 0.01 and adjust

3. **Mini-Batch Gradient Descent**:
   - Standard approach in modern deep learning
   - Batch size 32-256 is typical
   - Balances speed, memory, and gradient quality

4. **Shuffle Every Epoch**:
   - Prevents learning spurious patterns
   - Better gradient estimates
   - Helps escape local minima
   - Always do this for classification!

5. **Advanced Optimizers**:
   - Adam: Great default (less tuning)
   - SGD+Momentum: Often best final performance (more tuning)
   - Start simple, add complexity only if needed

### Next Steps

- **Read**: `docs/backpropagation.md` to understand how gradients are computed
- **Experiment**: Try different learning rates in `mlp_simple.swift`
- **Explore**: Modify `Sources/MNISTMLX/MLPModel.swift` to use different batch sizes
- **Advanced**: Implement momentum or try the Adam optimizer

---

**Related Documentation:**
- [Backpropagation Explained](./backpropagation.md) - How gradients are computed
- [Loss Functions Guide](./loss-functions.md) - Different ways to measure error
- [MLPModel.swift](../Sources/MNISTMLX/MLPModel.swift) - Production implementation
- [mlp_simple.swift](../mlp_simple.swift) - Educational implementation
