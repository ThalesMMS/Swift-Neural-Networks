# Loss Computation Walkthrough: Softmax and Cross-Entropy

This document provides a detailed explanation of how we measure and compute loss in neural networks for classification tasks, specifically focusing on softmax and cross-entropy loss used in our MNIST digit classifier.

## What is Loss?

**Loss** (also called "cost" or "error") is a numerical measure of how wrong the network's predictions are. During training, we compute the loss to understand how far our predictions are from the true labels, then use this information to adjust the network's weights.

**The Training Loop:**
```
1. Forward pass → Get predictions (logits)
2. Compute loss → Measure how wrong we are
3. Backward pass → Compute gradients
4. Update weights → Improve the model
5. Repeat
```

---

## The Problem: From Logits to Probabilities

After the forward pass, our network outputs **logits** - unnormalized scores for each class:

```
Network output (logits):
┌────────┬────────┐
│ Class  │ Logit  │
├────────┼────────┤
│   0    │  -1.2  │
│   1    │   0.3  │
│   2    │   4.8  │  ← Highest score
│   3    │  -0.5  │
│   4    │   1.1  │
│   5    │  -2.3  │
│   6    │   0.8  │
│   7    │   2.1  │
│   8    │  -1.8  │
│   9    │   0.2  │
└────────┴────────┘
```

**Problems with using logits directly:**
- They can be negative
- They can be arbitrarily large or small
- They don't sum to 1
- They're hard to interpret as confidence scores

**What we need:**
- Valid probabilities (between 0 and 1)
- All probabilities sum to 1
- Higher logits → higher probabilities
- Differentiable (for backpropagation)

**Solution:** The **Softmax** function!

---

## Softmax: Converting Logits to Probabilities

### Mathematical Formula

The softmax function converts a vector of logits into a probability distribution:

```
              exp(z_i)
softmax(z)_i = ─────────────────
               Σ exp(z_j)
               j=0 to 9

Where:
  z_i    = logit for class i
  exp()  = exponential function (e^x)
  Σ      = sum over all classes
```

**In plain English:**
1. Take the exponential of each logit
2. Sum all the exponentials
3. Divide each exponential by the sum

This ensures:
- All outputs are positive (exp() is always > 0)
- All outputs sum to 1 (we divide by the sum)
- Relative ordering is preserved (higher logits → higher probabilities)

---

### Step-by-Step Softmax Computation

Let's work through a concrete example with our logits from above:

**Step 1: Compute exponentials**
```
Logits:    [-1.2,  0.3,  4.8, -0.5,  1.1, -2.3,  0.8,  2.1, -1.8,  0.2]
            ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
exp():     [0.30, 1.35, 121.5, 0.61, 3.00, 0.10, 2.23, 8.17, 0.17, 1.22]
```

**Step 2: Sum all exponentials**
```
Sum = 0.30 + 1.35 + 121.5 + 0.61 + 3.00 + 0.10 + 2.23 + 8.17 + 0.17 + 1.22
    = 138.65
```

**Step 3: Divide each exponential by the sum**
```
Probabilities:
┌────────┬────────┬─────────┬──────────────┐
│ Class  │ Logit  │ exp()   │ Probability  │
├────────┼────────┼─────────┼──────────────┤
│   0    │  -1.2  │   0.30  │ 0.002 (0.2%) │
│   1    │   0.3  │   1.35  │ 0.010 (1.0%) │
│   2    │   4.8  │ 121.50  │ 0.876 (87.6%)│ ← Highest!
│   3    │  -0.5  │   0.61  │ 0.004 (0.4%) │
│   4    │   1.1  │   3.00  │ 0.022 (2.2%) │
│   5    │  -2.3  │   0.10  │ 0.001 (0.1%) │
│   6    │   0.8  │   2.23  │ 0.016 (1.6%) │
│   7    │   2.1  │   8.17  │ 0.059 (5.9%) │
│   8    │  -1.8  │   0.17  │ 0.001 (0.1%) │
│   9    │   0.2  │   1.22  │ 0.009 (0.9%) │
├────────┴────────┴─────────┼──────────────┤
│                    TOTAL: │ 1.000 (100%) │ OK
└───────────────────────────┴──────────────┘
```

**Key observations:**
- The highest logit (4.8 for class 2) becomes the highest probability (87.6%)
- Negative logits still get non-zero probabilities (but very small)
- All probabilities sum to exactly 1.0
- The network is 87.6% confident the digit is a "2"

---

### Why Exponential?

The exponential function has special properties that make softmax work well:

**1. Always positive:**
```
exp(x) > 0  for all x
```

**2. Monotonic:**
Higher input → higher output (preserves ranking)
```
If x > y, then exp(x) > exp(y)
```

**3. Amplifies differences:**
Small differences in logits become larger differences in probabilities
```
Graph of exp(x):
     exp(x)
       ↑
  20.0 │                           ╱
       │                        ╱
  16.0 │                     ╱
       │                  ╱
  12.0 │               ╱
       │            ╱
   8.0 │         ╱
       │      ╱
   4.0 │   ╱
       │ ╱
   0.0 ├────────────────────────→ x
       -2  -1   0   1   2   3

Notice: The curve gets steeper as x increases
```

**Example of amplification:**
```
Logits:     [2.0,  3.0]  ← Difference: 1.0
exp():      [7.39, 20.09]
Softmax:    [0.27, 0.73] ← Difference: 0.46 (amplified!)

Logits:     [1.0,  5.0]  ← Difference: 4.0
exp():      [2.72, 148.4]
Softmax:    [0.02, 0.98] ← Difference: 0.96 (highly confident!)
```

The exponential makes the network's confidence explicit and differentiable.

---

### Softmax Intuition: Temperature Analogy

Softmax can be seen as a "soft" version of max (argmax):

```
Hard max (argmax):        Soft max (softmax):
┌────────┬────────┐      ┌────────┬────────┐
│ Class  │ Output │      │ Class  │ Output │
├────────┼────────┤      ├────────┼────────┤
│   0    │   0    │      │   0    │  0.002 │
│   1    │   0    │      │   1    │  0.010 │
│   2    │   1    │ ←    │   2    │  0.876 │ ← Mostly here
│   3    │   0    │      │   3    │  0.004 │
│   4    │   0    │      │   4    │  0.022 │
│  ...   │  ...   │      │  ...   │  ...   │
└────────┴────────┘      └────────┴────────┘
  Winner takes all       Smooth distribution
  (discrete)              (differentiable)
```

**Why "soft"?**
- Argmax picks one winner (all-or-nothing)
- Softmax gives the top class most probability, but others get non-zero
- This smoothness is essential for gradient-based learning!

---

## Cross-Entropy Loss: Measuring Prediction Quality

Now that we have probabilities from softmax, we need to measure how good they are. This is where **cross-entropy loss** comes in.

### The Goal

We want a loss function that:
- Is **low** when predictions match the true label
- Is **high** when predictions are wrong
- Is **differentiable** (for backpropagation)
- **Encourages** high confidence on the correct class

---

### Mathematical Formula

For a single sample:

```
L = -log(p_correct)

Where:
  p_correct = predicted probability for the true class
  log()     = natural logarithm
```

For a batch of N samples:

```
         1   N-1
L_batch = ─ × Σ  -log(p_correct[i])
         N   i=0

This is the average loss across the batch.
```

**Alternative formulation (one-hot encoding):**
```
         K-1
L = -    Σ   y_k × log(p_k)
        k=0

Where:
  y_k = 1 if k is the true class, 0 otherwise (one-hot)
  p_k = predicted probability for class k
  K   = number of classes (10 for MNIST)
```

Since only one y_k is 1, this simplifies to -log(p_correct).

---

### Why Negative Log?

The negative log has perfect properties for a loss function:

**Properties of -log(p) where p ∈ [0, 1]:**

```
Graph of -log(p):
    Loss
      ↑
  10.0│╲
      │ ╲
   8.0│  ╲
      │   ╲
   6.0│    ╲
      │     ╲
   4.0│      ╲
      │       ╲
   2.0│        ╲_
      │           ──___
   0.0│                 ─────────→ p (probability)
      └────────────────────────
      0.0  0.2  0.4  0.6  0.8  1.0

Key properties:
  * p -> 1  (correct): loss -> 0  Good!
  * p -> 0  (wrong):   loss -> infinity  Heavily penalized!
  • Always positive
  • Smooth and differentiable
```

**Numerical examples:**
```
┌─────────────────┬──────────────┐
│ Probability     │ Loss (-log)  │
├─────────────────┼──────────────┤
│ 1.00 (perfect)  │ 0.00         │ + No loss
│ 0.99            │ 0.01         │ + Tiny loss
│ 0.90            │ 0.11         │ + Small loss
│ 0.50            │ 0.69         │ Warning: Medium loss
│ 0.10            │ 2.30         │ - High loss
│ 0.01            │ 4.61         │ -- Very high loss
│ 0.001           │ 6.91         │ --- Extreme loss
└─────────────────┴──────────────┘
```

**Why this works so well:**
1. **Exponential penalty** for wrong predictions (not linear)
2. **Infinite penalty** for being completely wrong (p=0)
3. **Zero loss** only at perfect confidence (p=1)
4. **Smooth gradient** for optimization

---

### Cross-Entropy Example

Let's compute the loss for our example where the true label is "2":

**Predicted probabilities (from softmax):**
```
┌────────┬──────────────┐
│ Class  │ Probability  │
├────────┼──────────────┤
│   0    │ 0.002        │
│   1    │ 0.010        │
│   2    │ 0.876        │ ← TRUE LABEL
│   3    │ 0.004        │
│   4    │ 0.022        │
│   5    │ 0.001        │
│   6    │ 0.016        │
│   7    │ 0.059        │
│   8    │ 0.001        │
│   9    │ 0.009        │
└────────┴──────────────┘
```

**Loss computation:**
```
p_correct = 0.876  (probability for class 2)

L = -log(0.876)
  = -log(0.876)
  = -(-0.132)
  = 0.132
```

**This is a LOW loss** (good!) because the network predicted 87.6% confidence on the correct class.

---

### Comparing Good vs Bad Predictions

**Scenario 1: Good prediction (correct class = 2)**
```
Probabilities: [0.002, 0.010, 0.876, 0.004, 0.022, ...]
                                 ↑
                           True label: 2

p_correct = 0.876
Loss = -log(0.876) = 0.132  Low loss (good)
```

**Scenario 2: Confident but WRONG prediction (correct class = 2)**
```
Probabilities: [0.002, 0.010, 0.010, 0.004, 0.850, ...]
                                 ↑              ↑
                           True label: 2   Wrong class 4

p_correct = 0.010
Loss = -log(0.010) = 4.605  High loss!
```

**Scenario 3: Uncertain prediction (correct class = 2)**
```
Probabilities: [0.10, 0.10, 0.15, 0.10, 0.10, 0.10, 0.10, 0.15, 0.05, 0.05]
                               ↑
                         True label: 2

p_correct = 0.15
Loss = -log(0.15) = 1.897  Medium-high loss
```

---

## Why Cross-Entropy for Classification?

Cross-entropy has become the standard loss for classification tasks. Here's why:

### 1. Information-Theoretic Interpretation

Cross-entropy measures the **difference between two probability distributions**:
- **True distribution**: One-hot encoded label (all 0s except 1 for correct class)
- **Predicted distribution**: Softmax output

```
True distribution (label = 2):
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

Predicted distribution:
[0.002, 0.010, 0.876, 0.004, 0.022, 0.001, 0.016, 0.059, 0.001, 0.009]

Cross-entropy measures: How many extra bits do we need because
we predicted the wrong distribution?
```

---

### 2. Maximum Likelihood Estimation

Minimizing cross-entropy is equivalent to **maximum likelihood estimation** - we're finding the model that makes the observed data most likely.

```
Mathematically:
  Minimizing -log(p_correct)
  = Maximizing log(p_correct)
  = Maximizing likelihood
```

---

### 3. Perfect Gradient Properties

The combination of softmax + cross-entropy has a beautiful mathematical property:

**Gradient of loss w.r.t. logits:**
```
∂L/∂z_i = p_i - y_i

Where:
  p_i = predicted probability (softmax output)
  y_i = true label (1 for correct class, 0 otherwise)
```

**Example:**
```
If true class = 2:
  Predicted: [0.002, 0.010, 0.876, 0.004, ...]
  True:      [0,     0,     1,     0,     ...]
  Gradient:  [0.002, 0.010, -0.124, 0.004, ...]
                              ↑
                   Negative! Increase logit for class 2

For wrong classes, gradient is positive (decrease logits)
For correct class, gradient is negative (increase logits)
```

This gradient is **simple, clean, and efficient** to compute!

---

### 4. Compared to Alternatives

**Mean Squared Error (MSE) - NOT recommended for classification:**
```
L_MSE = Σ (p_i - y_i)²

Problems:
  • Doesn't push for high confidence (gradient saturates)
  • Treats all errors equally
  • Slower convergence
  • Less interpretable for probabilities
```

**Example comparison:**
```
True label: class 2
Predicted: [0.01, 0.01, 0.50, 0.01, ...]  (uncertain)

Cross-entropy: -log(0.50) = 0.693
MSE: (1-0.50)² + other terms ≈ 0.29

After update, predicted: [0.01, 0.01, 0.90, 0.01, ...]  (better!)

Cross-entropy: -log(0.90) = 0.105  (much lower! clear signal)
MSE: (1-0.90)² + other terms ≈ 0.02  (smaller change, weaker signal)
```

Cross-entropy gives **stronger learning signals** especially when the model is uncertain!

---

## Complete Pipeline: Logits → Softmax → Loss

Here's the complete data flow with our running example:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────┘

Step 1: Network outputs LOGITS (unnormalized scores)
─────────────────────────────────────────────────────────────────────
┌─────┬───────┐
│Class│ Logit │
├─────┼───────┤
│  0  │  -1.2 │
│  1  │   0.3 │
│  2  │   4.8 │ ← Highest logit
│  3  │  -0.5 │
│  4  │   1.1 │
│  5  │  -2.3 │
│  6  │   0.8 │
│  7  │   2.1 │
│  8  │  -1.8 │
│  9  │   0.2 │
└─────┴───────┘
       │
       │ SOFTMAX: exp(z_i) / Σ exp(z_j)
       ↓

Step 2: SOFTMAX converts to probabilities
─────────────────────────────────────────────────────────────────────
┌─────┬─────────┬──────┐
│Class│  exp()  │ Prob │
├─────┼─────────┼──────┤
│  0  │   0.30  │ 0.00 │
│  1  │   1.35  │ 0.01 │
│  2  │ 121.50  │ 0.88 │ ← Highest probability (87.6%)
│  3  │   0.61  │ 0.00 │
│  4  │   3.00  │ 0.02 │
│  5  │   0.10  │ 0.00 │
│  6  │   2.23  │ 0.02 │
│  7  │   8.17  │ 0.06 │
│  8  │   0.17  │ 0.00 │
│  9  │   1.22  │ 0.01 │
├─────┼─────────┼──────┤
│ SUM │ 138.65  │ 1.00 │ OK
└─────┴─────────┴──────┘
       │
       │ TRUE LABEL: Class 2
       ↓

Step 3: CROSS-ENTROPY computes loss
─────────────────────────────────────────────────────────────────────
p_correct = probability[2] = 0.876

Loss = -log(p_correct)
     = -log(0.876)
     = 0.132

┌──────────────────────────────────┐
│  Final Loss: 0.132               │
│                                  │
│  Interpretation:                 │
│  • LOW loss (good prediction!)   │
│  • Network is 87.6% confident    │
│  • Room for improvement to p=1.0 │
└──────────────────────────────────┘
```

---

## Numerical Example: Complete 10-Class Scenario

Let's work through a complete example with all 10 MNIST classes:

### Example 1: Correct Prediction

**Input:** Image of digit "7"
**True Label:** Class 7

**Network Logits:**
```
[-0.5, -1.2, 0.3, -0.8, 0.1, -1.5, 0.2, 3.8, -0.3, 0.5]
                                           ↑
                                    Highest for class 7 (correct)
```

**Step 1: Compute exponentials**
```
exp(logits):
[0.61, 0.30, 1.35, 0.45, 1.11, 0.22, 1.22, 44.7, 0.74, 1.65]
Sum = 52.35
```

**Step 2: Softmax probabilities**
```
┌────────┬────────┬─────────┐
│ Class  │ Logit  │  Prob   │
├────────┼────────┼─────────┤
│   0    │  -0.5  │  0.012  │
│   1    │  -1.2  │  0.006  │
│   2    │   0.3  │  0.026  │
│   3    │  -0.8  │  0.009  │
│   4    │   0.1  │  0.021  │
│   5    │  -1.5  │  0.004  │
│   6    │   0.2  │  0.023  │
│   7    │   3.8  │  0.854  │ <- TRUE LABEL (correct)
│   8    │  -0.3  │  0.014  │
│   9    │   0.5  │  0.032  │
└────────┴────────┴─────────┘
```

**Step 3: Cross-entropy loss**
```
p_correct = 0.854
Loss = -log(0.854) = 0.158

LOW loss - good prediction!
```

---

### Example 2: Wrong Prediction

**Input:** Image of digit "3"
**True Label:** Class 3

**Network Logits:**
```
[-0.2, 0.1, 0.8, -0.5, 0.3, 2.9, 0.4, -1.0, 1.2, 0.6]
                             ↑                   ↑
                       True: 3           Predicted: 5 (wrong)
```

**Step 1 & 2: Softmax**
```
┌────────┬────────┬─────────┐
│ Class  │ Logit  │  Prob   │
├────────┼────────┼─────────┤
│   0    │  -0.2  │  0.030  │
│   1    │   0.1  │  0.040  │
│   2    │   0.8  │  0.081  │
│   3    │  -0.5  │  0.022  │ ← TRUE LABEL
│   4    │   0.3  │  0.049  │
│   5    │   2.9  │  0.659  │ ← WRONG (highest)
│   6    │   0.4  │  0.054  │
│   7    │  -1.0  │  0.013  │
│   8    │   1.2  │  0.120  │
│   9    │   0.6  │  0.066  │
└────────┴────────┴─────────┘
```

**Step 3: Cross-entropy loss**
```
p_correct = 0.022  (only 2.2% confidence on true class!)
Loss = -log(0.022) = 3.817

HIGH loss - wrong prediction!
```

**What happens during backpropagation:**
- Gradient for class 3: 0.022 - 1 = **-0.978** (strongly increase)
- Gradient for class 5: 0.659 - 0 = **+0.659** (strongly decrease)
- This will push the network to correct its mistake!

---

## Visual Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    COMPLETE FLOW DIAGRAM                        │
└────────────────────────────────────────────────────────────────┘

FORWARD PASS                    LOSS COMPUTATION
─────────────                   ────────────────

Input Image                     Network Logits
    │                          (unnormalized)
    ↓                                │
Hidden Layers                        │
    │                                ↓
    ↓                          ┌──────────┐
Output Layer               →   │ Softmax  │
    │                          └──────────┘
    ↓                                │
Logits (z)                           │
[z₀, z₁, ..., z₉]                    ↓
                              Probabilities (p)
                              [p₀, p₁, ..., p₉]
                              • All positive
                              • Sum to 1.0
                                     │
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                         ↓                       ↓
                   True Label (y)         Cross-Entropy
                   [0,0,1,0,...]           Loss = -log(p_correct)
                         │                       │
                         └───────────┬───────────┘
                                     │
                                     ↓
                              Single Number
                              (How wrong we are)
                                     │
                                     ↓
                              BACKPROPAGATION
                              (Update weights)
```

---

## Code Reference

In our MLX implementation (`Sources/MNISTMLX/MLPModel.swift`):

```swift
/// Computes cross-entropy loss for MLP
func lossFunction(_ model: MLPModel, _ inputs: MLXArray, _ targets: MLXArray) -> MLXArray {
    // Forward pass: get logits [N, 10]
    let logits = model(inputs)

    // Compute softmax cross-entropy loss
    // MLX efficiently combines softmax + cross-entropy
    let loss = crossEntropy(
        logits: logits,
        targets: targets,
        reduction: .mean  // Average over batch
    )

    return loss
}
```

**Note:** MLX (and most frameworks) combine softmax and cross-entropy into a single operation for numerical stability and efficiency. Computing them separately can lead to overflow/underflow issues with exp().

---

## Key Takeaways

### Softmax
1. **Converts logits to probabilities** using exp(z_i) / Σ exp(z_j)
2. **Properties**: All positive, sum to 1, differentiable
3. **Amplifies differences**: Exponential makes confident predictions more explicit
4. **"Soft" max**: Smooth, differentiable version of argmax

### Cross-Entropy Loss
1. **Formula**: L = -log(p_correct)
2. **Perfect properties**:
   - Low when correct (p → 1, loss → 0)
   - High when wrong (p → 0, loss → ∞)
   - Smooth gradients for optimization
3. **Why it works**: Information theory, maximum likelihood, clean gradients

### Why This Combination?
- **Mathematically elegant**: Gradients simplify beautifully (∂L/∂z = p - y)
- **Empirically effective**: Industry standard for classification
- **Interpretable**: Loss directly relates to prediction confidence
- **Numerical stability**: Can be computed efficiently in log-space

---

## Next Steps

Now that you understand loss computation, you're ready to learn about:
- **Backpropagation**: How gradients flow backward through the network
- **Gradient Descent**: How we use gradients to update weights
- **Training Loop**: Putting it all together

For the complete forward pass, see `docs/forward-pass.md`.
For the full MLP architecture, see `docs/mlp-walkthrough.md`.
