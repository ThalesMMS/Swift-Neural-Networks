# MLP Walkthrough: Neural Networks from First Principles

## What You'll Learn

This walkthrough teaches you **exactly how a neural network works** by walking through every step of training a Multi-Layer Perceptron (MLP) on the MNIST digit dataset.

By the end, you'll understand:
- ✅ What happens when an image enters the network
- ✅ How matrix multiplications transform data
- ✅ Why activation functions create non-linearity
- ✅ How loss measures prediction quality
- ✅ How gradients flow backward through layers
- ✅ How weights get updated to improve performance

**No black boxes. No hand-waving. Just clear explanations with code.**

---

## The MLP Architecture

Our MNIST classifier uses a simple 3-layer architecture:

```
Input Layer    Hidden Layer    Output Layer
[784 pixels] → [512 neurons] → [10 classes]
```

**In detail:**

1. **Input Layer:** 784 pixels (28×28 grayscale image flattened)
2. **Hidden Layer:** 512 neurons with ReLU activation
3. **Output Layer:** 10 neurons (one per digit 0-9) with Softmax

### Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-LAYER PERCEPTRON (MLP)                          │
│                      MNIST Digit Classification                          │
└─────────────────────────────────────────────────────────────────────────┘

INPUT IMAGE (28×28 pixels)
        ┃
        ┃ Flatten
        ▼
┌───────────────┐
│  Input Layer  │     784 features (28×28 pixels flattened)
│   [N, 784]    │     N = batch size
└───────┬───────┘     Each pixel: normalized value 0.0-1.0
        │
        │ Matrix Multiply: x @ W1 + b1
        │   W1: [784, 512] - 401,408 parameters
        │   b1: [512]      - 512 parameters
        ▼
┌───────────────┐
│ Hidden Layer  │     512 neurons (pre-activation)
│   [N, 512]    │     z1 = x @ W1 + b1
└───────┬───────┘     Linear transformation
        │
        │ Activation: ReLU(z1) = max(0, z1)
        │   • Introduces non-linearity
        │   • Enables learning complex patterns
        │   • ~50% sparsity (zeros out negative values)
        ▼
┌───────────────┐
│   Activated   │     512 neurons (post-activation)
│   [N, 512]    │     a1 = ReLU(z1)
└───────┬───────┘     All values ≥ 0
        │
        │ Matrix Multiply: a1 @ W2 + b2
        │   W2: [512, 10]  - 5,120 parameters
        │   b2: [10]       - 10 parameters
        ▼
┌───────────────┐
│ Output Layer  │     10 class scores (logits)
│   [N, 10]     │     logits = a1 @ W2 + b2
└───────┬───────┘     One score per digit (0-9)
        │
        │ Softmax: exp(logit[i]) / Σexp(logit[j])
        │   • Converts logits to probabilities
        │   • All probabilities sum to 1.0
        │   • Applied during loss computation
        ▼
┌───────────────┐
│ Probabilities │     10 probability values
│   [N, 10]     │     p = softmax(logits)
└───────────────┘     Predicted class = argmax(p)

═══════════════════════════════════════════════════════════════════════════

PARAMETER SUMMARY:
  Layer 1 (Input → Hidden):  784 × 512 + 512 = 401,920 parameters
  Layer 2 (Hidden → Output): 512 × 10  + 10  = 5,130 parameters
  ─────────────────────────────────────────────────────────────
  TOTAL:                                       407,050 parameters

═══════════════════════════════════════════════════════════════════════════
```

### Detailed Layer Connectivity

```
        Input (784)              Hidden (512)           Output (10)
             ┃                        ┃                     ┃
             ┃                        ┃                     ┃
    ┌────────────────┐       ┌────────────────┐    ┌──────────────┐
    │   Pixel 0      │       │   Neuron 0     │    │   Class 0    │
    │   Pixel 1      │       │   Neuron 1     │    │   Class 1    │
    │   Pixel 2      │──W1──→│   Neuron 2     │──W2│   Class 2    │
    │      ...       │  +b1  │      ...       │ +b2│     ...      │
    │   Pixel 783    │       │   Neuron 511   │    │   Class 9    │
    └────────────────┘       └────────────────┘    └──────────────┘
                                     │                     │
                                   ReLU                 Softmax
                                     ↓                     ↓
                              (Non-linearity)      (Probabilities)
```

### Data Flow with Tensor Shapes

```
┌──────────────────────────────────────────────────────────────────┐
│                        FORWARD PASS DATA FLOW                     │
└──────────────────────────────────────────────────────────────────┘

Step 1: Input
─────────────────────────────────────────────────────────────────
  Input:  [N, 28, 28]    ← Batch of grayscale images
    ↓
  Flatten
    ↓
  x:      [N, 784]       ← Flattened pixel values


Step 2: Hidden Layer (Linear Transform)
─────────────────────────────────────────────────────────────────
  x:      [N, 784]
  W1:     [784, 512]     ← Learned weights
  b1:     [512]          ← Learned biases
    ↓
  z1 = x @ W1 + b1
    ↓
  z1:     [N, 512]       ← Pre-activation


Step 3: Activation (Non-linearity)
─────────────────────────────────────────────────────────────────
  z1:     [N, 512]
    ↓
  a1 = ReLU(z1) = max(0, z1)
    ↓
  a1:     [N, 512]       ← Post-activation (all ≥ 0)


Step 4: Output Layer (Classification)
─────────────────────────────────────────────────────────────────
  a1:     [N, 512]
  W2:     [512, 10]      ← Learned weights
  b2:     [10]           ← Learned biases
    ↓
  logits = a1 @ W2 + b2
    ↓
  logits: [N, 10]        ← Class scores (unnormalized)


Step 5: Softmax (Probability Distribution)
─────────────────────────────────────────────────────────────────
  logits: [N, 10]
    ↓
  probs = softmax(logits)
    ↓
  probs:  [N, 10]        ← Probabilities (sum to 1.0)
    ↓
  prediction = argmax(probs, axis=1)
    ↓
  prediction: [N]        ← Predicted digit (0-9)
```

**Total parameters:** 784×512 + 512 + 512×10 + 10 = **407,050 learnable weights**

---

## The Training Pipeline

Training a neural network involves repeating this cycle:

```
┌──────────────┐
│ 1. Forward   │ - Pass input through network to get predictions
│    Pass      │   (Input → Hidden → Output)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. Compute   │ - Measure how wrong the predictions are
│    Loss      │   (Cross-entropy between prediction & truth)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. Backward  │ - Calculate how to adjust weights
│    Pass      │   (Compute gradients via backpropagation)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. Update    │ - Adjust weights to reduce loss
│    Weights   │   (Gradient descent: W = W - lr*grad)
└──────┬───────┘
       │
       └──────────────┐
                      │
          Repeat for all batches in epoch
```

**Each section below dives deep into one of these steps.**

---

## Section 1: Forward Pass

**What happens:** Input data flows through the network to produce predictions.

### Steps:

1. **Input Layer → Hidden Layer**
   ```swift
   // Matrix multiplication + bias
   hidden = input @ W1 + b1    // Shape: [N, 784] @ [784, 512] → [N, 512]
   ```

2. **Apply Activation**
   ```swift
   // ReLU: max(0, x)
   hidden = relu(hidden)       // Shape stays [N, 512]
   ```

3. **Hidden Layer → Output Layer**
   ```swift
   // Matrix multiplication + bias
   logits = hidden @ W2 + b2   // Shape: [N, 512] @ [512, 10] → [N, 10]
   ```

4. **Apply Softmax (implicitly in loss)**
   ```swift
   // Convert logits to probabilities
   probabilities = softmax(logits)  // Shape: [N, 10], each row sums to 1
   ```

**Why ReLU?** Without activation functions, the network is just nested linear transformations, which collapse to a single linear function. ReLU adds non-linearity, allowing the network to learn complex patterns.

**📖 Read the detailed explanation:** [Forward Pass Documentation](forward-pass.md)

---

## Section 2: Loss Computation

**What happens:** We measure how wrong our predictions are.

### Cross-Entropy Loss

For classification, we use **cross-entropy loss**:

```
Loss = -log(p_correct_class)
```

Where `p_correct_class` is the predicted probability for the true class.

**Example:**
- True label: 5 (digit "5")
- Predicted probabilities: [0.01, 0.02, 0.05, 0.10, 0.15, **0.50**, 0.05, 0.04, 0.05, 0.03]
- Loss = -log(0.50) ≈ 0.693

**Good predictions** (high probability for correct class) → **Low loss**
**Bad predictions** (low probability for correct class) → **High loss**

**Why cross-entropy?** It penalizes confident wrong predictions heavily and encourages the model to be accurate and confident.

**📖 Read the detailed explanation:** [Loss Computation Documentation](loss-computation.md)

---

## Section 3: Backpropagation

**What happens:** We calculate how each weight contributed to the loss, using the chain rule.

### Gradient Flow

Gradients flow **backward** through the network:

```
Loss
  ↓
∂Loss/∂logits (from softmax + cross-entropy)
  ↓
∂Loss/∂W2, ∂Loss/∂b2 (output layer gradients)
  ↓
∂Loss/∂hidden (before ReLU)
  ↓
∂Loss/∂hidden_pre_relu (apply ReLU mask)
  ↓
∂Loss/∂W1, ∂Loss/∂b1 (input layer gradients)
```

### Key Insights

1. **Chain Rule:** Gradients multiply as we go backward
   ```
   ∂Loss/∂W1 = ∂Loss/∂hidden × ∂hidden/∂W1
   ```

2. **ReLU Backward:** Gradient passes through only where input was > 0
   ```swift
   grad_hidden = grad_output * (hidden > 0)  // Masking
   ```

3. **Matrix Transpose:** Appears in backprop to match dimensions
   ```swift
   ∂Loss/∂W2 = hidden.T @ ∂Loss/∂logits
   ```

**Manual vs Automatic:** Modern frameworks (like MLX) compute gradients automatically. Understanding manual backprop helps you debug and design custom architectures.

**📖 Read the detailed explanation:** [Backpropagation Documentation](backpropagation.md)

---

## Section 4: Weight Updates

**What happens:** We adjust weights in the direction that reduces loss.

### Stochastic Gradient Descent (SGD)

The update rule is simple:

```swift
W = W - learning_rate * gradient
```

**Parameters:**
- **Learning rate** (e.g., 0.01): How big a step to take
  - Too large → Overshooting, unstable training
  - Too small → Slow convergence

- **Mini-batch:** Update weights using a subset of data (e.g., 64 images)
  - Faster than full-batch (all data)
  - More stable than single example (SGD)

**Example:**
```swift
// If gradient is [0.5, -0.2, 0.3] and learning_rate is 0.01
W_new = W_old - 0.01 * [0.5, -0.2, 0.3]
      = W_old - [0.005, -0.002, 0.003]
```

**Why "Stochastic"?** We use random mini-batches instead of the full dataset, adding noise that can help escape local minima.

**📖 Read the detailed explanation:** [Weight Updates Documentation](weight-updates.md)

---

## Section 5: Complete Training Loop

**What happens:** We tie all the pieces together into a training loop.

### Pseudocode

```swift
for epoch in 1...num_epochs {
    // Shuffle data each epoch
    shuffle(training_data)

    // Process in mini-batches
    for batch in training_data.batches(size: batch_size) {
        // 1. Forward pass
        predictions = model.forward(batch.images)

        // 2. Compute loss
        loss = crossEntropy(predictions, batch.labels)

        // 3. Backward pass (compute gradients)
        gradients = model.backward(loss)

        // 4. Update weights
        optimizer.update(model.weights, gradients)
    }

    // Evaluate on validation set
    accuracy = evaluate(model, validation_data)
    print("Epoch \(epoch): Loss = \(loss), Accuracy = \(accuracy)%")
}
```

### Key Concepts

1. **Epochs:** Complete passes through the dataset
2. **Mini-batches:** Subsets of data processed together (vectorization speedup)
3. **Shuffling:** Prevents the model from learning data order
4. **Evaluation:** Check performance on held-out data

**Typical MNIST Results:**
- Epoch 1: ~90% accuracy
- Epoch 5: ~97% accuracy
- Epoch 10: ~98% accuracy

**📖 Read the detailed explanation:** [Complete Training Loop Documentation](complete-training-loop.md)

---

## Code References

### Where to Find the Code

This walkthrough references two implementations:

#### 1. Simple Educational Example: [`mlp_simple.swift`](../mlp_simple.swift)

**Characteristics:**
- 218 lines of pure Swift
- Learns XOR function (2→4→1 network)
- Manual backpropagation with clear variable names
- No external dependencies

**Best for:** First-time learners

**Run it:**
```bash
swift mlp_simple.swift
```

**Expected output:**
```
Training simple MLP to learn XOR...
Epoch 0, Loss: 0.6931
Epoch 1000, Loss: 0.0123
...
Final predictions close to [0, 1, 1, 0]
```

#### 2. Production MNIST MLP: [`Sources/MNISTMLX/MLPModel.swift`](../Sources/MNISTMLX/MLPModel.swift)

**Characteristics:**
- ~80 lines using automatic differentiation (MLX framework)
- MNIST digit classification (784→512→10)
- GPU-accelerated by default
- Production-quality code

**Best for:** Understanding modern ML workflows

**Run it:**
```bash
swift build
swift run MNISTMLX --model mlp --epochs 10
```

**Expected output:**
```
Epoch 1: Loss: 0.2341, Accuracy: 93.2%
Epoch 2: Loss: 0.1245, Accuracy: 96.1%
...
Epoch 10: Loss: 0.0456, Accuracy: 98.3%
```

### Code Reading Guide

**If you want to understand manual backprop:**
1. Read [`mlp_simple.swift`](../mlp_simple.swift) - Start here!
2. Read [`mnist_mlp.swift`](../mnist_mlp.swift) - Full MNIST with manual gradients
3. Compare with [`Sources/MNISTMLX/MLPModel.swift`](../Sources/MNISTMLX/MLPModel.swift) - See auto-diff

**If you want to build production models:**
1. Read [`Sources/MNISTMLX/MLPModel.swift`](../Sources/MNISTMLX/MLPModel.swift) - Modern approach
2. Skim [`mnist_mlp.swift`](../mnist_mlp.swift) - Understand what MLX is doing under the hood
3. Explore other models in `Sources/MNISTMLX/` - CNN, Attention, ResNet

---

## Next Steps

### 1. Read the Detailed Sections

Work through each section in order:
1. **[Forward Pass](forward-pass.md)** - How data flows through the network
2. **[Loss Computation](loss-computation.md)** - Measuring prediction quality
3. **[Backpropagation](backpropagation.md)** - Computing gradients via chain rule
4. **[Weight Updates](weight-updates.md)** - Stochastic gradient descent
5. **[Complete Training Loop](complete-training-loop.md)** - Tying it all together

### 2. Run the Code

Get hands-on experience:
```bash
# Simple XOR example
swift mlp_simple.swift

# Full MNIST MLP
swift run MNISTMLX --model mlp --epochs 10

# MNIST CNN (better accuracy)
swift run MNISTMLX --model cnn --epochs 5
```

### 3. Experiment

Modify the code to deepen understanding:
- Change learning rate (try 0.001, 0.01, 0.1)
- Adjust hidden layer size (256, 512, 1024)
- Add more layers
- Try different activation functions

### 4. Explore Advanced Topics

Once you understand MLPs:
- **Convolutional Neural Networks (CNNs):** Spatial pattern recognition
- **Attention Mechanisms:** Transformers and self-attention
- **Residual Networks (ResNets):** Skip connections for deep networks

See the [main README](../README.md) for all available implementations.

---

## Conceptual Roadmap

Here's what each section teaches:

| Section | Core Question Answered |
|---------|------------------------|
| **Forward Pass** | How does input data become predictions? |
| **Loss Computation** | How do we measure prediction quality? |
| **Backpropagation** | How do we calculate weight adjustments? |
| **Weight Updates** | How do weights improve over time? |
| **Training Loop** | How does everything fit together? |

**By the end, you'll know:**
- What every line of an MLP implementation does
- Why neural networks need non-linearity
- How gradients flow through computational graphs
- Why automatic differentiation is powerful
- How to debug training issues

---

## Philosophy of This Walkthrough

### Principles

1. **Intuition over formalism:** We explain *why*, not just *what*
2. **Code over math:** Executable examples, not just equations
3. **Concrete over abstract:** Real numbers, not just symbols
4. **Complete over partial:** End-to-end understanding, no gaps

### What Makes This Different

- **Line-by-line code explanations** - Every operation is annotated
- **Tensor shape tracking** - Know the dimensions at each step
- **Numerical examples** - See actual matrices and calculations
- **Gradient flow diagrams** - Visualize backpropagation
- **Manual vs auto-diff comparison** - Understand what frameworks do

**Goal:** After this walkthrough, neural networks won't feel like magic anymore.

---

## Related Documentation

- **[Main Documentation Index](README.md)** - All documentation sections
- **[Learning Guide](../LEARNING_GUIDE.md)** - Complete repository navigation
- **[Educational Implementations](../educational/README.md)** - Standalone examples

---

## Questions?

As you go through this walkthrough, keep these questions in mind:

- **Forward Pass:** What shape is this tensor? Why this matrix size?
- **Loss:** Why does low probability cause high loss?
- **Backprop:** Where does the transpose come from? Why multiply gradients?
- **Updates:** What happens if learning rate is too high/low?

The detailed sections answer all of these!

---

---

## Navigation

- **⬅️ Back:** [Documentation Home](README.md)
- **➡️ Next:** [Forward Pass](forward-pass.md)

### All Sections
1. **MLP Walkthrough Overview** (you are here)
2. [Forward Pass](forward-pass.md)
3. [Loss Computation](loss-computation.md)
4. [Backpropagation](backpropagation.md)
5. [Weight Updates](weight-updates.md)
6. [Complete Training Loop](complete-training-loop.md)

### Related Resources
- **[Learning Guide](../LEARNING_GUIDE.md)** - Complete repository navigation and learning paths
- **[Educational Examples](../educational/README.md)** - Standalone educational implementations
- **[Main README](../README.md)** - Project overview and quickstart

---

**Happy learning!** 🧠
