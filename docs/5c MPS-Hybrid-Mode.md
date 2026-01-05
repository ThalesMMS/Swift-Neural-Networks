# MPS Hybrid Mode

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/README.md)
> * [mlp_simple.swift](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mlp_simple.swift)
> * [mnist_cnn.swift](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_cnn.swift)
> * [mnist_mlp.swift](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift)

## Purpose and Scope

This page documents the MPS (Metal Performance Shaders) hybrid GPU acceleration backend for the MNIST MLP implementation. This backend combines GPU-accelerated matrix multiplication via `MPSMatrixMultiplication` with custom Metal compute kernels for element-wise operations, using shared CPU/GPU memory buffers to eliminate data transfer overhead.

For information about backend selection and the `GemmEngine` protocol abstraction, see [Backend Selection](#5.1). For the pure CPU implementation, see [Accelerate Framework (CPU)](#5.2). For the full GPU graph-based approach, see [MPSGraph Full GPU Mode](#5.4).

**Sources:** [mnist_mlp.swift L1-L2223](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1-L2223)

---

## Architecture Overview

The MPS hybrid mode splits neural network operations between CPU and GPU, using shared memory buffers to avoid explicit data transfers. Matrix multiplications (GEMM) are accelerated using `MPSMatrixMultiplication`, while element-wise operations are handled by custom Metal compute kernels.

### Operational Split

```mermaid
flowchart TD

SHUFFLE["Data Shuffling (Fisher-Yates)"]
BATCH["Batch Gathering (gatherBatchToPointer)"]
LOSS_READ["Loss Accumulation (CPU reads shared buffer)"]
WEIGHT_COPY["Weight Copy Back (final sync to NeuralNetwork)"]
GEMM["Matrix Multiplication (MPSMatrixMultiplication)"]
KERNELS["Element-wise Kernels (add_bias, relu, softmax, etc)"]
INPUTS["batchInputs: MpsBuffer"]
LABELS["batchLabels: MpsBufferU8"]
A1["a1: MpsBuffer (hidden activations)"]
A2["a2: MpsBuffer (output logits)"]
WEIGHTS["w1, b1, w2, b2 (MpsBuffer)"]
GRADS["gradW1, gradB1, gradW2, gradB2 (MpsBuffer)"]

BATCH -.-> INPUTS
BATCH -.-> LABELS
INPUTS -.-> GEMM
WEIGHTS -.-> GEMM
GEMM -.-> A1
A1 -.-> KERNELS
KERNELS -.-> A2
A2 -.-> LOSS_READ
GRADS -.-> KERNELS
KERNELS -.-> WEIGHTS
WEIGHTS -.-> WEIGHT_COPY

subgraph subGraph2 ["Shared Memory (storageModeShared)"]
    INPUTS
    LABELS
    A1
    A2
    WEIGHTS
    GRADS
end

subgraph subGraph1 ["GPU Device"]
    GEMM
    KERNELS
end

subgraph subGraph0 ["CPU Host"]
    SHUFFLE
    BATCH
    LOSS_READ
    WEIGHT_COPY
    SHUFFLE -.-> BATCH
end
```

**Sources:** [mnist_mlp.swift L1519-L1767](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1519-L1767)

 [mnist_mlp.swift L883-L989](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L883-L989)

 [mnist_mlp.swift L563-L620](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L563-L620)

---

## Shared Memory Model

### MpsBuffer Class

The `MpsBuffer` class wraps `MTLBuffer` objects created with `storageModeShared`, providing zero-copy access from both CPU and GPU.

```mermaid
flowchart TD

BUF["buffer: MTLBuffer (storageModeShared)"]
CNT["count: Int (number of Float elements)"]
PTR["pointer: UnsafeMutablePointer<Float> (CPU-accessible)"]
CPU["CPU Code (Swift)"]
GPU["GPU Kernels (Metal)"]

CPU -.->|"direct access"| PTR
GPU -.->|"direct access"| BUF

subgraph subGraph0 ["MpsBuffer Structure"]
    BUF
    CNT
    PTR
    PTR -.->|"same memory"| BUF
end
```

**Key characteristics:**

| Feature | Description |
| --- | --- |
| Storage Mode | `storageModeShared` - CPU and GPU access same physical memory |
| Initialization | Created with `device.makeBuffer(length:options:)` |
| CPU Access | Via `pointer` property (typed `UnsafeMutablePointer<Float>`) |
| GPU Access | Via `buffer` property (typed `MTLBuffer`) |
| Allocation Strategy | Persistent across batches - allocated once per training session |

**Implementation details:**

The `MpsBuffer` class [mnist_mlp.swift L563-L600](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L563-L600)

 provides:

* Initialization with optional initial data
* `update(from:count:)` method for CPU-to-buffer copy
* `copy(to:)` method for buffer-to-CPU copy
* Direct pointer access for zero-copy operations

For label data, `MpsBufferU8` [mnist_mlp.swift L603-L620](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L603-L620)

 provides identical functionality with `UInt8` element type.

**Sources:** [mnist_mlp.swift L563-L600](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L563-L600)

 [mnist_mlp.swift L603-L620](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L603-L620)

---

## GEMM Operations

### MpsGemmEngine Class

The `MpsGemmEngine` [mnist_mlp.swift L883-L989](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L883-L989)

 encapsulates GPU-accelerated matrix multiplication using `MPSMatrixMultiplication`.

```mermaid
flowchart TD

DEVICE["device: MTLDevice"]
QUEUE["commandQueue: MTLCommandQueue"]
ENCODE["encodeGemm() (encode operation)"]
GEMM["gemm() (create + commit buffer)"]
CMD_BUF["MTLCommandBuffer"]
MAT_DESC["MPSMatrixDescriptor (rows, cols, rowBytes)"]
MPS_MAT["MPSMatrix (wraps MTLBuffer)"]
MPS_OP["MPSMatrixMultiplication (transposeLeft, transposeRight)"]
GPU["GPU Execution"]

ENCODE -.-> MAT_DESC
GEMM -.->|"encode(commandBuffer:)"| CMD_BUF

subgraph subGraph1 ["Operation Flow"]
    CMD_BUF
    MAT_DESC
    MPS_MAT
    MPS_OP
end

subgraph MpsGemmEngine ["MpsGemmEngine"]
    DEVICE
    QUEUE
    ENCODE
    GEMM
end
```

**GEMM operation signature:**

The `encodeGemm` method [mnist_mlp.swift L902-L955](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L902-L955)

 performs:

```
C = alpha * op(A) * op(B) + beta * C
```

Where:

* `op(A)` = A or A^T (controlled by `transposeA`)
* `op(B)` = B or B^T (controlled by `transposeB`)
* `alpha`, `beta` are scalar coefficients

**Matrix descriptor configuration:**

```mermaid
flowchart TD

C_ROWS["m (fixed)"]
C_COLS["n (fixed)"]
C_DESC["MPSMatrixDescriptor (rows, columns, rowBytes)"]
B_ROWS["bRows = transposeB ? n : k"]
B_COLS["bCols = transposeB ? k : n"]
B_DESC["MPSMatrixDescriptor (rows, columns, rowBytes)"]
A_ROWS["aRows = transposeA ? k : m"]
A_COLS["aCols = transposeA ? m : k"]
A_DESC["MPSMatrixDescriptor (rows, columns, rowBytes)"]

subgraph subGraph2 ["Matrix C (Result)"]
    C_ROWS
    C_COLS
    C_DESC
end

subgraph subGraph1 ["Matrix B"]
    B_ROWS
    B_COLS
    B_DESC
end

subgraph subGraph0 ["Matrix A"]
    A_ROWS
    A_COLS
    A_DESC
end
```

**Key GEMM use cases in training:**

| Operation | Dimensions | Purpose |
| --- | --- | --- |
| Hidden Forward | `[batch × 784] × [784 × 512]` | Input to hidden layer |
| Output Forward | `[batch × 512] × [512 × 10]` | Hidden to output layer |
| Output Gradient | `[512 × batch]^T × [batch × 10]` | Weight gradients (dW2) |
| Hidden Backprop | `[batch × 10] × [10 × 512]^T` | Error propagation (dZ1) |
| Hidden Gradient | `[784 × batch]^T × [batch × 512]` | Weight gradients (dW1) |

**Sources:** [mnist_mlp.swift L883-L989](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L883-L989)

 [mnist_mlp.swift L902-L955](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L902-L955)

---

## Custom Metal Kernels

### MpsKernels Class

The `MpsKernels` class [mnist_mlp.swift L623-L880](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L623-L880)

 compiles and manages seven compute kernels for element-wise operations that complement GEMM operations.

```mermaid
flowchart TD

K1["add_bias (broadcast bias to matrix)"]
K2["relu_inplace (max(0, x))"]
K3["relu_grad (zero grad if x ≤ 0)"]
K4["softmax_rows (stable softmax per row)"]
K5["sum_rows (column reduction)"]
K6["delta_and_loss (xent + grad)"]
K7["sgd_update (w -= lr * grad)"]
SOURCE["Metal Source Code (inline string)"]
LIBRARY["MTLLibrary"]
FUNCTION["MTLFunction (per kernel name)"]
PSO["MTLComputePipelineState"]

subgraph subGraph1 ["Kernel Inventory"]
    K1
    K2
    K3
    K4
    K5
    K6
    K7
end

subgraph subGraph0 ["Kernel Compilation"]
    SOURCE
    LIBRARY
    FUNCTION
    PSO
    SOURCE -.->|"makeLibrary(source:)"| LIBRARY
    LIBRARY -.->|"makeFunction(name:)"| FUNCTION
    FUNCTION -.->|"makeComputePipelineState()"| PSO
end
```

### Kernel Details

#### 1. add_bias Kernel

[mnist_mlp.swift L637-L646](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L637-L646)

```
data[row, col] += bias[col]
```

**Purpose:** Broadcast bias vector to each row of a matrix (post-GEMM bias addition).

**Dispatch:** 1D threadgroup covering `rows × cols` elements.

**Sources:** [mnist_mlp.swift L637-L646](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L637-L646)

 [mnist_mlp.swift L785-L794](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L785-L794)

---

#### 2. relu_inplace Kernel

[mnist_mlp.swift L648-L654](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L648-L654)

```
data[i] = max(0, data[i])
```

**Purpose:** Apply ReLU activation in-place.

**Dispatch:** 1D threadgroup covering `count` elements.

**Sources:** [mnist_mlp.swift L648-L654](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L648-L654)

 [mnist_mlp.swift L796-L802](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L796-L802)

---

#### 3. relu_grad Kernel

[mnist_mlp.swift L656-L664](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L656-L664)

```
if activations[i] <= 0:
    grads[i] = 0
```

**Purpose:** Backpropagate through ReLU by zeroing gradients where activation was negative.

**Dispatch:** 1D threadgroup covering `count` elements.

**Sources:** [mnist_mlp.swift L656-L664](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L656-L664)

 [mnist_mlp.swift L804-L811](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L804-L811)

---

#### 4. softmax_rows Kernel

[mnist_mlp.swift L666-L687](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L666-L687)

```
for each row:
    max_val = max(row)
    exp_sum = sum(exp(row - max_val))
    row = exp(row - max_val) / exp_sum
```

**Purpose:** Numerically stable softmax per matrix row.

**Dispatch:** 1D threadgroup with one thread per row (each thread processes an entire row).

**Sources:** [mnist_mlp.swift L666-L687](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L666-L687)

 [mnist_mlp.swift L813-L821](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L813-L821)

---

#### 5. sum_rows Kernel

[mnist_mlp.swift L689-L701](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L689-L701)

```
for each column:
    out[col] = scale * sum(data[:, col])
```

**Purpose:** Column-wise reduction (used for bias gradients).

**Dispatch:** 1D threadgroup with one thread per column (each thread reduces one column).

**Sources:** [mnist_mlp.swift L689-L701](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L689-L701)

 [mnist_mlp.swift L823-L841](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L823-L841)

---

#### 6. delta_and_loss Kernel

[mnist_mlp.swift L703-L721](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L703-L721)

```python
for each sample in batch:
    label = labels[sample]
    loss[sample] = -log(outputs[sample, label])
    for each class:
        delta[sample, class] = outputs[sample, class]
        if class == label:
            delta[sample, class] -= 1
```

**Purpose:** Compute cross-entropy loss and softmax gradient (combined for efficiency).

**Dispatch:** 1D threadgroup with one thread per sample (each thread processes one row).

**Sources:** [mnist_mlp.swift L703-L721](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L703-L721)

 [mnist_mlp.swift L843-L862](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L843-L862)

---

#### 7. sgd_update Kernel

[mnist_mlp.swift L723-L730](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L723-L730)

```
weights[i] -= learning_rate * grads[i]
```

**Purpose:** Apply SGD weight update.

**Dispatch:** 1D threadgroup covering `count` weight elements.

**Sources:** [mnist_mlp.swift L723-L730](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L723-L730)

 [mnist_mlp.swift L864-L879](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L864-L879)

---

### Kernel Dispatch Pattern

All kernels use a common dispatch helper [mnist_mlp.swift L768-L783](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L768-L783)

:

```mermaid
flowchart TD

PIPELINE["MTLComputePipelineState"]
ENCODER["MTLComputeCommandEncoder"]
THREADS["MTLSize (width: count)"]
GROUP["MTLSize (width: min(width, count))"]
WIDTH["Optimal Width"]
ARGS["Kernel Arguments"]

PIPELINE -.->|"threadExecutionWidth"| WIDTH
ENCODER -.->|"setComputePipelineState"| PIPELINE
ENCODER -.->|"setBuffer / setBytes"| ARGS
ENCODER -.->|"dispatchThreads"| THREADS
THREADS -.-> GROUP
```

**Sources:** [mnist_mlp.swift L768-L783](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L768-L783)

---

## Training Workflow

### trainMps Function

The `trainMps` function [mnist_mlp.swift L1519-L1767](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1519-L1767)

 orchestrates hybrid CPU/GPU training using persistent shared buffers and batched command encoding.

### Buffer Allocation Strategy

```mermaid
flowchart TD

INPUTS["batchInputs (batchSize × 784)"]
LABELS["batchLabels (batchSize)"]
ACTIVATIONS["a1 (hidden), a2 (output) (batch × layer_size)"]
DELTAS["dZ1, dZ2 (batch × layer_size)"]
GRADIENTS["gradW1, gradB1, gradW2, gradB2 (layer dimensions)"]
WEIGHTS["w1, b1, w2, b2 (layer dimensions)"]
LOSS["loss (batchSize)"]
GATHER["CPU: gatherBatchToPointer (copy shuffled data)"]
ENCODE["GPU: Encode all ops (single command buffer)"]
COMMIT["GPU: commit() waitUntilCompleted()"]
ACCUMULATE["CPU: Sum loss values (read shared buffer)"]

INPUTS -.-> GATHER
LABELS -.-> GATHER
WEIGHTS -.-> ENCODE
ENCODE -.-> ACTIVATIONS
ENCODE -.-> DELTAS
ENCODE -.-> GRADIENTS
ENCODE -.-> LOSS
COMMIT -.-> LOSS
LOSS -.-> ACCUMULATE

subgraph subGraph1 ["Per-Batch Operations"]
    GATHER
    ENCODE
    COMMIT
    ACCUMULATE
    GATHER -.-> ENCODE
    ENCODE -.-> COMMIT
end

subgraph subGraph0 ["Persistent Buffers (Once Per Training)"]
    INPUTS
    LABELS
    ACTIVATIONS
    DELTAS
    GRADIENTS
    WEIGHTS
    LOSS
end
```

### Training Loop Structure

```mermaid
flowchart TD

START["Start Epoch"]
SHUFFLE["Fisher-Yates Shuffle (CPU)"]
BATCH_LOOP["For each batch"]
GATHER["Gather batch to shared buffers (CPU: gatherBatchToPointer)"]
CMD["Create MTLCommandBuffer"]
F1["GEMM: X·W1 → a1"]
F2["Kernel: a1 += b1"]
F3["Kernel: ReLU(a1)"]
F4["GEMM: a1·W2 → a2"]
F5["Kernel: a2 += b2"]
F6["Kernel: Softmax(a2)"]
F7["Kernel: delta_and_loss"]
B1["GEMM: a1^T·dZ2 → gradW2"]
B2["Kernel: sum_rows(dZ2) → gradB2"]
B3["GEMM: dZ2·W2^T → dZ1"]
B4["Kernel: relu_grad(a1, dZ1)"]
B5["GEMM: X^T·dZ1 → gradW1"]
B6["Kernel: sum_rows(dZ1) → gradB1"]
U1["Kernel: sgd_update(w2, gradW2)"]
U2["Kernel: sgd_update(b2, gradB2)"]
U3["Kernel: sgd_update(w1, gradW1)"]
U4["Kernel: sgd_update(b1, gradB1)"]
COMMIT["commandBuffer.commit() waitUntilCompleted()"]
READ["CPU: Read loss buffer Accumulate totalLoss"]
NEXT["Next batch"]
LOG["Log epoch stats (loss, time)"]
SYNC["Copy weights back to NeuralNetwork (w1.copy(to:) etc)"]

START -.-> SHUFFLE
SHUFFLE -.-> BATCH_LOOP
READ -.-> NEXT
NEXT -.-> BATCH_LOOP
NEXT -.-> LOG
LOG -.-> SYNC

subgraph subGraph3 ["Batch Processing"]
    GATHER
    CMD
    COMMIT
    READ
    GATHER -.-> CMD
    CMD -.-> F1
    F7 -.-> B1
    B6 -.-> U1
    U4 -.-> COMMIT
    COMMIT -.-> READ

subgraph subGraph2 ["Weight Update Encoding"]
    U1
    U2
    U3
    U4
    U1 -.->|"epoch complete"| U2
    U2 -.-> U3
    U3 -.-> U4
end

subgraph subGraph1 ["Backward Pass Encoding"]
    B1
    B2
    B3
    B4
    B5
    B6
    B1 -.-> B2
    B2 -.-> B3
    B3 -.-> B4
    B4 -.-> B5
    B5 -.-> B6
end

subgraph subGraph0 ["Forward Pass Encoding"]
    F1
    F2
    F3
    F4
    F5
    F6
    F7
    F1 -.-> F2
    F2 -.->|"more batches"| F3
    F3 -.-> F4
    F4 -.-> F5
    F5 -.-> F6
    F6 -.-> F7
end
end
```

### Key Implementation Details

**Batch gathering:** [mnist_mlp.swift L1579-L1588](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1579-L1588)

```yaml
gatherBatchToPointer(
    images: images,
    labels: labels,
    indices: indices,
    start: batchStart,
    count: batchCount,
    inputSize: numInputs,
    outInputs: batchInputs.pointer,
    outLabels: batchLabels.pointer
)
```

Uses direct pointer access to shared buffers - no explicit GPU upload required.

**Command buffer pattern:** [mnist_mlp.swift L1590-L1592](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1590-L1592)

```javascript
guard let commandBuffer = engine.commandQueue.makeCommandBuffer() else {
    continue
}
```

Single command buffer encodes all operations for one batch (forward + backward + update).

**Loss accumulation:** [mnist_mlp.swift L1735-L1740](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1735-L1740)

```javascript
var batchLoss: Float = 0.0
let lossPtr = loss.pointer
for i in 0..<batchCount {
    batchLoss += lossPtr[i]
}
totalLoss += batchLoss
```

Reads loss values directly from shared buffer after GPU completion.

**Weight synchronization:** [mnist_mlp.swift L1753-L1766](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1753-L1766)

```
w1.copy(to: &hiddenWeights)
b1.copy(to: &hiddenBiases)
w2.copy(to: &outputWeights)
b2.copy(to: &outputBiases)

nn.hidden.weights = hiddenWeights
nn.hidden.biases = hiddenBiases
nn.output.weights = outputWeights
nn.output.biases = outputBiases
```

Final synchronization copies GPU-updated weights back to `NeuralNetwork` struct for testing and serialization.

**Sources:** [mnist_mlp.swift L1519-L1767](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1519-L1767)

 [mnist_mlp.swift L1280-L1300](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1280-L1300)

 [mnist_mlp.swift L1579-L1741](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1579-L1741)

---

## Testing Workflow

### testMps Function

The `testMps` function [mnist_mlp.swift L1770-L1859](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1770-L1859)

 performs GPU-accelerated inference without computing gradients or updating weights.

```mermaid
flowchart TD

TEST_IN["batchInputs (batchSize × 784)"]
TEST_A1["a1 (batchSize × 512)"]
TEST_A2["a2 (batchSize × 10)"]
TEST_W["w1, b1, w2, b2 (model weights)"]
COPY["CPU: Copy test batch (contiguous, no shuffle)"]
CMD["Create CommandBuffer"]
T1["GEMM: X·W1 → a1"]
T2["Kernel: a1 += b1"]
T3["Kernel: ReLU(a1)"]
T4["GEMM: a1·W2 → a2"]
T5["Kernel: a2 += b2"]
COMMIT["commit + wait"]
ARGMAX["CPU: Argmax over logits (no softmax needed)"]
COUNT["Accumulate correct count"]
ACCURACY["Print accuracy percentage"]

COPY -.-> TEST_IN
COMMIT -.-> TEST_A2
COUNT -.-> ACCURACY

subgraph subGraph2 ["Batch Loop"]
    COPY
    CMD
    COMMIT
    ARGMAX
    COUNT
    CMD -.-> T1
    T5 -.-> COMMIT
    ARGMAX -.-> COUNT

subgraph subGraph1 ["Forward Pass Only"]
    T1
    T2
    T3
    T4
    T5
    T1 -.-> T2
    T2 -.-> T3
    T3 -.-> T4
    T4 -.-> T5
end
end

subgraph subGraph0 ["Inference Buffers"]
    TEST_IN
    TEST_A1
    TEST_A2
    TEST_W
end
```

### Simplifications vs Training

| Aspect | Training | Testing |
| --- | --- | --- |
| Data Order | Shuffled per epoch | Sequential |
| Buffers | Activations, deltas, gradients | Activations only |
| Operations | Forward + backward + update | Forward only |
| Output Processing | Softmax for loss | Argmax on logits (no softmax) |
| Synchronization | Per batch (loss accumulation) | Per batch (prediction check) |

**Key code differences:**

**No softmax required:** [mnist_mlp.swift L1838-L1853](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1838-L1853)

```javascript
let logitsPtr = a2.pointer
for i in 0..<batchCount {
    let base = i * numOutputs
    var maxVal = logitsPtr[base]
    var maxIdx = 0
    for o in 1..<numOutputs {
        let v = logitsPtr[base + o]
        if v > maxVal {
            maxVal = v
            maxIdx = o
        }
    }
    if UInt8(maxIdx) == labels[batchStart + i] {
        correct += 1
    }
}
```

Argmax on logits is sufficient for classification - softmax is monotonic and doesn't change argmax.

**Contiguous batch copy:** [mnist_mlp.swift L1796-L1798](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1796-L1798)

```javascript
let src = imagesBase.advanced(by: batchStart * numInputs)
batchInputs.pointer.update(from: src, count: batchCount * numInputs)
```

No shuffling or indirect indexing needed during testing.

**Sources:** [mnist_mlp.swift L1770-L1859](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1770-L1859)

 [mnist_mlp.swift L1796-L1853](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1796-L1853)

---

## Performance Characteristics

### Memory Efficiency

```mermaid
flowchart TD

G_FIXED["Fixed batch size (graph compilation)"]
G_COPY["Implicit GPU transfers (feed/fetch)"]
C_ALLOC["Per-batch allocation (Swift arrays)"]
C_COPY["Memory copies (array operations)"]
H_ALLOC["One-time allocation (training start)"]
H_ZERO["Zero-copy access (storageModeShared)"]
H_PERSIST["Persistent buffers (entire training)"]

subgraph MPSGraph ["MPSGraph"]
    G_FIXED
    G_COPY
end

subgraph subGraph1 ["CPU Backend"]
    C_ALLOC
    C_COPY
end

subgraph subGraph0 ["MPS Hybrid"]
    H_ALLOC
    H_ZERO
    H_PERSIST
end
```

**MPS Hybrid advantages:**

| Feature | Benefit |
| --- | --- |
| Shared Memory | No CPU↔GPU transfers - both sides access same physical memory |
| Persistent Buffers | Allocated once at training start, reused for all batches |
| Zero-Copy Reads | Loss accumulation and prediction checking read GPU results directly |
| Flexible Batch Size | Handles variable-size last batch without recompilation |

### Operational Overhead

**Per-batch overhead components:**

1. **CPU operations:** * Fisher-Yates shuffle: O(n) per epoch * Batch gathering: O(batch_size × 784) memory copy * Loss accumulation: O(batch_size) shared buffer read
2. **GPU operations:** * Command buffer creation: ~μs * Kernel dispatch: ~μs per kernel (7 kernels per batch) * GEMM encoding: ~μs per operation (6 GEMMs per batch) * GPU execution: depends on batch size and hardware
3. **Synchronization:** * `waitUntilCompleted()`: blocks until GPU finishes * No explicit transfers due to shared memory

### Comparison with Other Backends

| Characteristic | CPU | MPS Hybrid | MPSGraph |
| --- | --- | --- | --- |
| Matrix Operations | vDSP (Accelerate) | MPSMatrixMultiplication | Auto-optimized graph |
| Element-wise Ops | Swift loops | Custom Metal kernels | Built-in graph ops |
| Memory Model | Swift arrays | Shared MTLBuffer | Graph-managed tensors |
| Data Transfers | In-memory only | Zero-copy | Implicit feed/fetch |
| Batch Flexibility | Full | Full | Fixed (graph compiled) |
| Gradient Computation | Manual backprop | Manual encoding | Automatic differentiation |
| Typical Speedup | 1× (baseline) | 3-5× | 5-10× |

**Sources:** [mnist_mlp.swift L1519-L1767](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1519-L1767)

 [mnist_mlp.swift L883-L989](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L883-L989)

 [mnist_mlp.swift L563-L620](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L563-L620)

---

## Backend Activation

The MPS hybrid backend is selected via command-line flag:

```
./mnist_mlp_swift --mps
```

**Backend selection logic:** [mnist_mlp.swift L2098-L2099](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L2098-L2099)

 [mnist_mlp.swift L1001-L1012](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1001-L1012)

```javascript
let useMPS = CommandLine.arguments.contains("--mps") || useMpsGraph
let backend = selectGemmBackend(useMPS: useMPS)
```

The `selectGemmBackend` function attempts to create `MpsGemmEngine`, falling back to `CpuGemmEngine` if Metal is unavailable.

**Training dispatch:** [mnist_mlp.swift L2138-L2160](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L2138-L2160)

```javascript
switch backend {
case .cpu(let cpu):
    train(nn: &nn, images: trainImages, labels: trainLabels,
          numSamples: trainImages.count / numInputs,
          engine: cpu, rng: &rng)
case .mps(let mps):
    trainMps(nn: &nn, images: trainImages, labels: trainLabels,
             numSamples: trainImages.count / numInputs,
             engine: mps, rng: &rng)
}
```

**Sources:** [mnist_mlp.swift L1001-L1012](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L1001-L1012)

 [mnist_mlp.swift L2094-L2160](https://github.com/ThalesMMS/Swift-Neural-Networks/blob/3a1c4fc2/mnist_mlp.swift#L2094-L2160)

Refresh this wiki

Last indexed: 5 January 2026 ([3a1c4f](https://github.com/ThalesMMS/Swift-Neural-Networks/commit/3a1c4fc2))

### On this page

* [MPS Hybrid Mode](#5.3-mps-hybrid-mode)
* [Purpose and Scope](#5.3-purpose-and-scope)
* [Architecture Overview](#5.3-architecture-overview)
* [Operational Split](#5.3-operational-split)
* [Shared Memory Model](#5.3-shared-memory-model)
* [MpsBuffer Class](#5.3-mpsbuffer-class)
* [GEMM Operations](#5.3-gemm-operations)
* [MpsGemmEngine Class](#5.3-mpsgemmengine-class)
* [Custom Metal Kernels](#5.3-custom-metal-kernels)
* [MpsKernels Class](#5.3-mpskernels-class)
* [Kernel Details](#5.3-kernel-details)
* [Kernel Dispatch Pattern](#5.3-kernel-dispatch-pattern)
* [Training Workflow](#5.3-training-workflow)
* [trainMps Function](#5.3-trainmps-function)
* [Buffer Allocation Strategy](#5.3-buffer-allocation-strategy)
* [Training Loop Structure](#5.3-training-loop-structure)
* [Key Implementation Details](#5.3-key-implementation-details)
* [Testing Workflow](#5.3-testing-workflow)
* [testMps Function](#5.3-testmps-function)
* [Simplifications vs Training](#5.3-simplifications-vs-training)
* [Performance Characteristics](#5.3-performance-characteristics)
* [Memory Efficiency](#5.3-memory-efficiency)
* [Operational Overhead](#5.3-operational-overhead)
* [Comparison with Other Backends](#5.3-comparison-with-other-backends)
* [Backend Activation](#5.3-backend-activation)

Ask Devin about Swift-Neural-Networks