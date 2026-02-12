# Forward Pass Walkthrough: MLP for MNIST

This document provides a step-by-step walkthrough of the forward pass (inference) in our Multi-Layer Perceptron (MLP) network for MNIST digit classification.

## What is a Forward Pass?

The **forward pass** is the process of taking input data and propagating it through the network layers to produce an output (prediction). During training, this output is compared to the true label to compute the loss, which is then used for backpropagation. During inference, the forward pass is all we need to make predictions.

---

## Network Architecture Overview

Our MLP has the following architecture:

```
Input Layer    → Hidden Layer    → Activation → Output Layer
[N, 784]         [N, 512]          [N, 512]      [N, 10]
   ↓                ↓                  ↓             ↓
Flattened      Linear Transform     ReLU        Logits
Images         W1@x + b1                        (Class scores)
```

**Key Components:**
- **Input**: Flattened 28×28 MNIST images → 784 features per image
- **Hidden Layer**: 784 → 512 linear transformation
- **Activation**: ReLU (Rectified Linear Unit)
- **Output Layer**: 512 → 10 linear transformation (one score per digit class)

---

## Step-by-Step Forward Pass

### Step 1: Input Layer - Shape [N, 784]

**What happens here:**
- We receive a batch of N grayscale MNIST images
- Each image is 28×28 pixels
- Images are flattened into 784-dimensional vectors

**Mathematical representation:**
```
x ∈ ℝ^(N×784)
```

**Example with batch size N=32:**
```
Input shape: [32, 784]
           ↓
    32 images, each represented as 784 pixel values
```

**Why flatten?**
MLPs (unlike CNNs) don't have spatial awareness. They treat the image as a simple vector of features. Each of the 784 input neurons represents one pixel value (0.0 to 1.0 for normalized data).

**Visual representation:**
```
Original Image (28×28):        Flattened Vector (784):
┌─────────────────┐            [0.0, 0.0, ..., 0.8, 0.9, 1.0, ...]
│ □ □ □ ... □ □ □ │              ↑                          ↑
│ □ ■ ■ ... ■ □ □ │            pixel 0                  pixel 783
│ □ ■ ■ ... ■ □ □ │
│ ... (28 rows)   │
└─────────────────┘
```

---

### Step 2: Hidden Layer - Matrix Multiplication W1@x + b1

**What happens here:**
- Linear transformation from 784 input features to 512 hidden features
- Each of the 512 hidden neurons computes a weighted sum of all 784 inputs

**Mathematical representation:**
```
z₁ = x @ W₁ + b₁

Where:
  x  ∈ ℝ^(N×784)   - Input batch
  W₁ ∈ ℝ^(784×512) - Weight matrix (401,408 parameters!)
  b₁ ∈ ℝ^(512)     - Bias vector
  z₁ ∈ ℝ^(N×512)   - Pre-activation hidden layer output
```

**Shape transformation:**
```
[N, 784] @ [784, 512] + [512] = [N, 512]
                                  ↑
                        Broadcasting adds bias to each sample
```

**What does this do?**
Each of the 512 hidden neurons learns to detect different patterns in the input. For example:
- Neuron 1 might activate for "diagonal edges"
- Neuron 2 might activate for "circular shapes"
- Neuron 3 might activate for "vertical lines"
- ...and so on

**Detailed computation for ONE hidden neuron:**
```
z₁[i,j] = Σ(k=0 to 783) x[i,k] × W₁[k,j] + b₁[j]

For batch sample i, hidden neuron j:
  - Multiply each of 784 input pixels by corresponding weight
  - Sum all 784 products
  - Add bias term
```

**Data flow diagram:**
```
Input Batch                Weight Matrix             Hidden Layer (pre-activation)
[N, 784]                   [784, 512]                [N, 512]
┌─────────┐                ┌──────────┐              ┌─────────┐
│ sample1 │                │          │              │ hidden1 │
│ sample2 │      @         │ weights  │      +b₁     │ hidden2 │
│   ...   │  ────────→     │          │  ─────────→  │   ...   │
│ sampleN │                │          │              │ hiddenN │
└─────────┘                └──────────┘              └─────────┘
  784 per                   784×512                    512 per
  sample                    connections                sample
```

---

### Step 3: Activation Functions - Introducing Non-Linearity

**What happens here:**
- Apply an activation function to introduce non-linearity
- Our MLP uses ReLU (Rectified Linear Unit)
- We'll also compare with Sigmoid to understand why ReLU is preferred

---

#### Why Non-Linearity is Essential

Without activation functions, the network would only learn linear transformations:
```
Without activation:
  output = (x @ W₁ + b₁) @ W₂ + b₂
         = x @ (W₁ @ W₂) + (b₁ @ W₂ + b₂)
         = x @ W' + b'

This is just ANOTHER linear transformation!
```

**No matter how many layers** you stack, without non-linearities the entire network collapses to a single linear transformation. This means:
- The network can only learn linear decision boundaries
- Can't learn complex patterns like curves, circles, XOR
- Essentially becomes no better than logistic regression

**Activation functions break this linearity** by introducing non-linear transformations, allowing the network to learn arbitrarily complex patterns.

---

#### ReLU (Rectified Linear Unit) - Our Choice

**Mathematical Formula:**
```
ReLU(z) = max(0, z) = {
  z   if z > 0
  0   if z ≤ 0
}

Element-wise operation:
  a₁[i,j] = max(0, z₁[i,j])  for all i, j
```

**Implementation in our network:**
```
a₁ = ReLU(z₁) = max(0, z₁)
```

**ReLU Graph:**
```
     Output
       ↑
   4.0 │                    ╱
       │                  ╱
   3.0 │                ╱
       │              ╱
   2.0 │            ╱
       │          ╱
   1.0 │        ╱
       │      ╱
   0.0 │────╱─────────────────→ Input
       │  ╱
  -1.0 │╱
       │
       -3  -2  -1  0   1   2   3

Key properties:
  • Slope = 0 for x < 0 (dead neurons)
  • Slope = 1 for x > 0 (identity)
  • Non-differentiable at x = 0
```

**ReLU Behavior on Sample Data:**
```
Input (z):  ... -2.0  -1.0   0.0   1.0   2.0 ...
            ─────┬─────┬─────┬─────┬─────┬─────
                 │     │     │     │     │
ReLU(z):    ...  0.0   0.0   0.0   1.0   2.0 ...
                 │     │     │     │     │
                 └─────┴─────┘     └─────┘
                 Negative → 0    Positive unchanged
```

**Visual representation:**
```
Before ReLU [N, 512]:          After ReLU [N, 512]:
┌─────────────────────┐         ┌─────────────────────┐
│ -2.1  0.5  -0.3 ... │         │  0.0  0.5   0.0 ... │
│  1.2 -1.0   2.3 ... │   →     │  1.2  0.0   2.3 ... │
│ -0.1  3.4  -5.6 ... │         │  0.0  3.4   0.0 ... │
│  ...                │         │  ...                │
└─────────────────────┘         └─────────────────────┘
   Pre-activation                  Activated
   (z₁)                            (a₁)
```

**Why we use ReLU:**
1. **Computationally efficient**: Just a max(0, x) operation
2. **No vanishing gradient**: Gradient is 1 for positive values
3. **Sparse activation**: About 50% of neurons are "off" (zero)
4. **Biologically inspired**: Similar to neuron firing patterns
5. **Empirically effective**: Works very well in practice

**Limitations of ReLU:**
- **Dying ReLU problem**: Neurons can get stuck outputting 0
- **Not zero-centered**: All outputs are ≥ 0
- **Unbounded**: No upper limit on activation

---

#### Sigmoid - Alternative Activation (For Comparison)

**Mathematical Formula:**
```
Sigmoid(z) = σ(z) = 1 / (1 + e^(-z))

Range: (0, 1)
```

**Derivative (useful for backprop):**
```
σ'(z) = σ(z) × (1 - σ(z))
```

**Sigmoid Graph:**
```
     Output
       ↑
   1.0 │        ─────────────
       │      ╱
   0.8 │     ╱
       │    │
   0.6 │   │
       │  │
   0.4 │  │
       │   │
   0.2 │    ╲
       │      ╲_
   0.0 │        ─────────────→ Input
       │
       -6  -4  -2  0   2   4   6

Key properties:
  • Output range: (0, 1)
  • S-shaped curve
  • Smooth everywhere
  • Saturates at both ends
```

**Sigmoid Behavior on Sample Data:**
```
Input (z):     -6.0   -2.0    0.0    2.0    6.0
               ──┬─────┬──────┬──────┬──────┬──
                 │     │      │      │      │
Sigmoid(z):    0.002  0.119  0.500  0.881  0.998
                 │     │      │      │      │
                 └─────┴──────┴──────┴──────┘
                Near 0      0.5       Near 1
```

**Why Sigmoid was popular historically:**
- Output range (0, 1) can be interpreted as probabilities
- Smooth and differentiable everywhere
- Used extensively in classical neural networks
- Still used in output layer for binary classification

**Why we DON'T use Sigmoid in hidden layers:**
1. **Vanishing gradient problem**:
   - For large |z|, gradient ≈ 0
   - Severely slows down learning
   - Worse with deep networks
2. **Not zero-centered**: Outputs always positive
3. **Expensive computation**: Requires exp() operation
4. **Saturation**: Neurons can get "stuck" in flat regions

---

#### Side-by-Side Comparison

```
┌─────────────────────┬─────────────────────┐
│      ReLU           │     Sigmoid         │
├─────────────────────┼─────────────────────┤
│  ReLU(z) = max(0,z) │  σ(z) = 1/(1+e^-z) │
├─────────────────────┼─────────────────────┤
│  Range: [0, ∞)      │  Range: (0, 1)      │
├─────────────────────┼─────────────────────┤
│  Gradient:          │  Gradient:          │
│    1 if z > 0       │    σ(z)(1-σ(z))     │
│    0 if z ≤ 0       │    max ≈ 0.25       │
├─────────────────────┼─────────────────────┤
│  Computation: Fast  │  Computation: Slow  │
│    (just compare)   │    (exponential)    │
├─────────────────────┼─────────────────────┤
│  Vanishing gradient:│  Vanishing gradient:│
│    No (for z > 0)   │    Yes (saturates)  │
├─────────────────────┼─────────────────────┤
│  Sparsity:          │  Sparsity:          │
│    Yes (~50%)       │    No (always > 0)  │
├─────────────────────┼─────────────────────┤
│  Use case:          │  Use case:          │
│    Hidden layers    │    Output (binary)  │
│    (modern default) │    classification   │
└─────────────────────┴─────────────────────┘
```

**Visual Comparison:**
```
    ReLU vs Sigmoid

     1.0 ┤           ╱╱╱╱          ──────────── Sigmoid
         │         ╱╱
     0.8 ┤       ╱╱              ╱╱
         │     ╱╱              ╱╱
     0.6 ┤   ╱╱              ╱╱
         │  ╱              ╱╱
     0.4 ┤ ╱            ╱╱         ╱╱╱╱╱╱╱╱╱ ReLU
         │            ╱╱         ╱╱
     0.2 ┤          ╱╱        ╱╱
         │        ╱╱       ╱╱
     0.0 ┤──────╱╱──────╱╱────────────
         └─────┴──────┴──────┴─────
         -3   -2   -1    0    1    2    3

Notice:
  • ReLU: Simple, piecewise linear
  • Sigmoid: Smooth curve, bounded
  • ReLU: Steeper gradient for positive values
  • Sigmoid: Gradient vanishes at extremes
```

---

#### Example: mlp_simple.swift (Sigmoid Implementation)

Our educational `mlp_simple.swift` example uses Sigmoid to demonstrate classical backpropagation:

```swift
// Sigmoid activation function
func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}

// Sigmoid derivative for backprop
func sigmoidDerivative(_ x: Double) -> Double {
    return x * (1.0 - x)  // Assumes x = sigmoid(z)
}
```

This is used for learning XOR (a classic non-linear problem), demonstrating that activation functions enable learning non-linear patterns.

---

#### Effect on Learning (ReLU in Our MLP)

**Sparsity:** About 50% of neurons are typically "off" (zero)
```
Hidden layer activations:
[0.0, 2.3, 0.0, 1.5, 0.0, 0.0, 3.1, 0.0, ...]
 ↑         ↑              ↑
Dead    Active         Active
```

**Selective activation:** Only relevant features fire for each input
- Different neurons activate for different digit patterns
- Creates efficient, sparse representations

**Gradient flow:** Helps avoid vanishing gradients
- For positive activations: gradient = 1 (perfect flow)
- For negative: gradient = 0 (may cause dead neurons, but no saturation)

---

### Step 4: Output Layer Computation - Final Classification

**What happens here:**
- Linear transformation from 512 hidden features to 10 class scores (logits)
- One score for each digit class (0-9)

**Mathematical representation:**
```
logits = a₁ @ W₂ + b₂

Where:
  a₁     ∈ ℝ^(N×512)  - Activated hidden layer
  W₂     ∈ ℝ^(512×10) - Output weight matrix (5,120 parameters)
  b₂     ∈ ℝ^(10)     - Output bias vector
  logits ∈ ℝ^(N×10)   - Final class scores (unnormalized)
```

**Shape transformation:**
```
[N, 512] @ [512, 10] + [10] = [N, 10]
```

**What are logits?**
Logits are **unnormalized log-probabilities**. Higher values indicate higher confidence for that class. They are NOT probabilities yet!

**Example output for one image:**
```
Logits for sample i:
┌────────┬────────┐
│ Class  │ Logit  │
├────────┼────────┤
│   0    │  -1.2  │
│   1    │   0.3  │
│   2    │   4.8  │  ← Highest score (predicted class = 2)
│   3    │  -0.5  │
│   4    │   1.1  │
│   5    │  -2.3  │
│   6    │   0.8  │
│   7    │   2.1  │
│   8    │  -1.8  │
│   9    │   0.2  │
└────────┴────────┘
```

**Converting logits to probabilities (softmax):**
While not part of the forward pass in our model, during training/inference we often apply softmax:
```
probability[i] = exp(logit[i]) / Σ(j=0 to 9) exp(logit[j])
```

This normalizes logits into probabilities that sum to 1.0.

---

## Complete Data Flow Diagram

Here's the complete forward pass with shapes at each step:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FORWARD PASS                                  │
└──────────────────────────────────────────────────────────────────────┘

Step 1: Input Layer
─────────────────────────────────────────────────────────────────────
Input Images (batch)
Shape: [N, 784]
│
│  Example: N=32 (batch size)
│  Each sample: 784 pixel values (normalized 0.0-1.0)
│
└─→ Ready for processing

Step 2: Hidden Layer (Linear Transform)
─────────────────────────────────────────────────────────────────────
z₁ = x @ W₁ + b₁

    [N, 784]  @  [784, 512]  +  [512]
       ↓            ↓             ↓
    Input       Weights        Bias
                (401,408       (512
                params)        params)
         ↓
    [N, 512] ← Pre-activation hidden layer

Step 3: ReLU Activation (Non-linearity)
─────────────────────────────────────────────────────────────────────
a₁ = ReLU(z₁) = max(0, z₁)

    [N, 512]  →  [N, 512]
       ↓             ↓
  Pre-activation  Activated
  (some negative)  (all ≥ 0)

Step 4: Output Layer (Classification)
─────────────────────────────────────────────────────────────────────
logits = a₁ @ W₂ + b₂

    [N, 512]  @  [512, 10]  +  [10]
       ↓           ↓            ↓
   Activated    Weights       Bias
                (5,120        (10
                params)       params)
         ↓
    [N, 10] ← Final logits (class scores)
```

---

## Summary: Shape Transformations

| Step | Operation | Input Shape | Output Shape | Parameters |
|------|-----------|-------------|--------------|------------|
| 1 | Input | — | [N, 784] | 0 |
| 2 | Linear (hidden) | [N, 784] | [N, 512] | 401,408 + 512 = 401,920 |
| 3 | ReLU | [N, 512] | [N, 512] | 0 (no learnable params) |
| 4 | Linear (output) | [N, 512] | [N, 10] | 5,120 + 10 = 5,130 |
| **Total** | | | | **407,050 parameters** |

---

## Key Takeaways

1. **Input Layer [N, 784]**: Flattened MNIST images, one row per sample in the batch

2. **Hidden Layer (W1@x+b1)**:
   - Linear transformation: 784 → 512
   - Each hidden neuron sees ALL input pixels
   - Learns to detect patterns/features
   - Shape: [N, 784] @ [784, 512] → [N, 512]

3. **ReLU Activation**:
   - Non-linearity is ESSENTIAL for learning complex patterns
   - ReLU(z) = max(0, z) - simple but effective
   - Enables the network to learn beyond linear transformations

4. **Output Layer**:
   - Maps 512 hidden features to 10 class scores
   - Higher logit = higher confidence for that digit
   - Shape: [N, 512] @ [512, 10] → [N, 10]

5. **Batch Processing**: The network processes N samples in parallel (vectorized operations)

---

## Code Reference

See the implementation in `Sources/MNISTMLX/MLPModel.swift`:

```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Step 2: Hidden layer transformation
    var h = hidden(x)  // [N, 784] → [N, 512]

    // Step 3: ReLU activation
    h = relu(h)        // [N, 512] → [N, 512]

    // Step 4: Output layer
    h = output(h)      // [N, 512] → [N, 10]

    return h  // Return logits
}
```

This compact code represents the entire forward pass described in this document!

---

## Next Steps

- **Backward Pass**: Learn how gradients flow backward through the network
- **Loss Functions**: Understand how we measure prediction quality
- **Optimization**: See how SGD updates the weights using gradients

For more details on the MLP architecture, see `docs/mlp-walkthrough.md`.
