# Neural Network Models (MLP, CNN, Attention)

Authors: Antonio Neto and Thales Matheus

This project implements several small neural nets for two problems:

- MNIST digit classification (MLP, CNN, and single-head self-attention + FFN)
- XOR toy example (2->4->1)

Current code is in Rust and Swift. The design and binary model format are inspired by https://github.com/djbyrne/mlp.c.

## Contents

Source code:

- `mnist_mlp.rs`, `mnist_cnn.rs`, `mnist_attention_pool.rs`, `mlp_simple.rs` (Rust)
- `mnist_mlp.swift`, `mnist_cnn.swift`, `mnist_attention_pool.swift`, `mlp_simple.swift` (Swift)
- `Cargo.toml` / `Cargo.lock` (Rust build config)

Scripts:

- `digit_recognizer.py` (draw digits and run inference with a saved model)
- `plot_comparison.py` (plots training loss from `logs/`)
- `requirements.txt` (Python dependencies)

Data and outputs:

- `data/` (MNIST IDX files)
- `logs/` (training loss logs)
- `mnist_model.bin` (saved model)
- `logs/training_loss_cnn.txt` (CNN loss log)
- `logs/training_loss_attention_mnist.txt` (attention loss log)

## MNIST MLP model

Architecture:

- Input: 784 neurons (28x28 pixels)
- Hidden: 512 neurons (ReLU)
- Output: 10 neurons (Softmax)

Default training parameters (Swift and Rust):

- Learning rate: 0.01
- Batch size: 64
- Epochs: 10

Expected accuracy: ~94-97% depending on backend and hyperparameters.

Rust MNIST notes:

- Uses `f32` tensors and batched GEMM via BLAS for speed.
- On macOS the default BLAS backend is Accelerate (via `blas-src`).
- Threading is controlled by `VECLIB_MAXIMUM_THREADS` when using Accelerate.

## MNIST CNN model

Architecture:

- Input: 28x28 image
- Conv: 8 filters (3x3) + ReLU
- MaxPool: 2x2
- FC: 1568 -> 10

Default training parameters:

- Learning rate: 0.01
- Batch size: 32
- Epochs: 3

## MNIST attention model

Architecture:

- 4x4 patches => 49 tokens
- Token projection + position embeddings + ReLU
- Self-attention (1 head, Q/K/V, 49x49 scores)
- Feed-forward MLP per token (D -> FF -> D)
- Mean-pool tokens -> 10 classes

Default training parameters:

- D model: 16
- FF dim: 32
- Learning rate: 0.01
- Batch size: 32
- Epochs: 5

## XOR model

Architecture:

- Input: 2 neurons
- Hidden: 4 neurons (Sigmoid)
- Output: 1 neuron (Sigmoid)

Training uses 1,000,000 epochs by default.

## Swift GPU optimizations

`mnist_mlp.swift` includes GPU paths for faster training and testing:

- MPS GEMM with shared CPU/GPU buffers
- Custom Metal kernels for bias add, ReLU, softmax, reductions, loss, and SGD
- MPSGraph to run forward, loss, gradients, and updates fully on GPU
- GPU testing path that uses argmax on logits (no softmax needed)

Note: `--mpsgraph` uses a fixed batch size; leftover samples are dropped to keep the graph static.

## Build and run

### Swift

Build:

```
swiftc -O mnist_mlp.swift -o mnist_mlp_swift
swiftc -O mlp_simple.swift -o mlp_simple_swift
swiftc -O mnist_cnn.swift -o mnist_cnn_swift
swiftc -O mnist_attention_pool.swift -o mnist_attention_pool_swift
```

Run MNIST MLP:

```
./mnist_mlp_swift --mps
./mnist_mlp_swift --mpsgraph
```

Run MNIST CNN:

```
./mnist_cnn_swift
```

Run MNIST attention:

```
./mnist_attention_pool_swift
```

Run XOR:

```
./mlp_simple_swift
```

Swift MLP options (`mnist_mlp_swift`):

```
--mps          use MPS GEMM + Metal kernels
--mpsgraph     use MPSGraph (train and test on GPU)
--batch N      batch size (default: 64)
--hidden N     hidden layer size (default: 512)
--epochs N     epochs (default: 10)
--lr F         learning rate (default: 0.01)
--seed N       RNG seed (default: 1)
--help         print usage
```

Swift attention options (`mnist_attention_pool_swift`):

```
--batch N      batch size (default: 32)
--epochs N     epochs (default: 5)
--lr F         learning rate (default: 0.01)
--seed N       RNG seed (default: 1)
--help         print usage
```

Note: `mnist_cnn_swift` uses fixed defaults and has no CLI flags.

### Rust

Build:

```
cargo build --release
```

Run MNIST MLP:

```
cargo run --release --bin mnist_mlp
```

Run XOR:

```
cargo run --release --bin mlp_simple
```

Run MNIST CNN (standalone, no external crates):

```
rustc -O mnist_cnn.rs -o mnist_cnn
./mnist_cnn
```

Run MNIST attention (standalone, no external crates):

```
rustc -O mnist_attention_pool.rs -o mnist_attention_pool
./mnist_attention_pool
```

Performance tips:

```
RUSTFLAGS="-C target-cpu=native" VECLIB_MAXIMUM_THREADS=8 cargo run --release --bin mnist_mlp
```

Linux/Windows note: the Rust MNIST build is configured for Accelerate on macOS. For other platforms, swap the BLAS backend in `Cargo.toml` (e.g., OpenBLAS) and ensure the library is installed.

## MNIST dataset

Expected files under `data/`:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Download from:

- https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- http://yann.lecun.com/exdb/mnist/

## Visualization

To plot training curves:

```
python plot_comparison.py
```

## Digit recognizer UI

The drawing app loads `mnist_model.bin` and runs inference:

```
python digit_recognizer.py
```

Install dependencies:

```
pip install -r requirements.txt
```

## References

- https://github.com/djbyrne/mlp.c
- http://yann.lecun.com/exdb/mnist/
