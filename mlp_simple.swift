// ============================================================================
// EDUCATIONAL REFERENCE: Simple MLP implementation for learning purposes
//
// This is a minimal educational example demonstrating the fundamentals of
// neural networks. It trains a small multi-layer perceptron to learn the
// XOR function using manual backpropagation with sigmoid activation.
//
// For production use, see: swift run MNISTMLX
// For learning progression, see: LEARNING_GUIDE.md
// ============================================================================

import Foundation
import MNISTCommon

// Small MLP to learn XOR (educational example).
let numInputs = 2
let numHidden = 4
let numOutputs = 1
let numSamples = 4
let learningRate = 0.01
let epochs = 1_000_000

// Sigmoid activation function.
func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}

// Sigmoid derivative assuming x = sigmoid(z).
func sigmoidDerivative(_ x: Double) -> Double {
    return x * (1.0 - x)
}

// Dense layer: weights (input x output) and biases (output).
struct LinearLayer {
    let inputSize: Int
    let outputSize: Int
    var weights: [[Double]]
    var biases: [Double]
}

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    var hidden: LinearLayer
    var output: LinearLayer
}

// Initialize weights and biases with small random values.
func initializeLayer(inputSize: Int, outputSize: Int, rng: inout SimpleRng) -> LinearLayer {
    var weights = Array(repeating: Array(repeating: 0.0, count: outputSize), count: inputSize)
    for i in 0..<inputSize {
        for j in 0..<outputSize {
            weights[i][j] = rng.uniform(-0.5, 0.5)
        }
    }

    var biases = Array(repeating: 0.0, count: outputSize)
    for i in 0..<outputSize {
        biases[i] = rng.uniform(-0.5, 0.5)
    }

    return LinearLayer(inputSize: inputSize, outputSize: outputSize, weights: weights, biases: biases)
}

// Create the full network with fixed XOR sizes.
func initializeNetwork(rng: inout SimpleRng) -> NeuralNetwork {
    rng.reseedFromTime()
    let hidden = initializeLayer(inputSize: numInputs, outputSize: numHidden, rng: &rng)
    let output = initializeLayer(inputSize: numHidden, outputSize: numOutputs, rng: &rng)
    return NeuralNetwork(hidden: hidden, output: output)
}

// Layer forward: z = W*x + b, followed by sigmoid.
func forward(layer: LinearLayer, inputs: [Double], outputs: inout [Double]) {
    for i in 0..<layer.outputSize {
        var activation = layer.biases[i]
        for j in 0..<layer.inputSize {
            activation += inputs[j] * layer.weights[j][i]
        }
        outputs[i] = sigmoid(activation)
    }
}

// Backprop: compute deltas for output and hidden layers.
func backward(
    nn: NeuralNetwork,
    hiddenOutputs: [Double],
    outputOutputs: [Double],
    errors: [Double],
    deltaHidden: inout [Double],
    deltaOutput: inout [Double]
) {
    for i in 0..<nn.output.outputSize {
        // delta_out = error * activation derivative.
        deltaOutput[i] = errors[i] * sigmoidDerivative(outputOutputs[i])
    }

    for i in 0..<nn.hidden.outputSize {
        // Error backpropagated from output to hidden layer.
        var error = 0.0
        for j in 0..<nn.output.outputSize {
            error += deltaOutput[j] * nn.output.weights[i][j]
        }
        deltaHidden[i] = error * sigmoidDerivative(hiddenOutputs[i])
    }
}

// Update weights and biases with gradient descent (SGD).
func updateWeights(layer: inout LinearLayer, inputs: [Double], deltas: [Double]) {
    for i in 0..<layer.inputSize {
        for j in 0..<layer.outputSize {
            layer.weights[i][j] += learningRate * deltas[j] * inputs[i]
        }
    }
    for i in 0..<layer.outputSize {
        layer.biases[i] += learningRate * deltas[i]
    }
}

// Training with mean squared error per sample.
func train(nn: inout NeuralNetwork, inputs: [[Double]], expected: [[Double]]) {
    var deltaHidden = Array(repeating: 0.0, count: numHidden)
    var deltaOutput = Array(repeating: 0.0, count: numOutputs)
    var errors = Array(repeating: 0.0, count: numOutputs)

    for epoch in 0..<epochs {
        var totalErrors = 0.0
        for sample in 0..<numSamples {
            var hiddenOutputs = Array(repeating: 0.0, count: numHidden)
            var outputOutputs = Array(repeating: 0.0, count: numOutputs)

            // Forward pass.
            forward(layer: nn.hidden, inputs: inputs[sample], outputs: &hiddenOutputs)
            forward(layer: nn.output, inputs: hiddenOutputs, outputs: &outputOutputs)

            // Per-output error and MSE accumulation.
            for i in 0..<numOutputs {
                errors[i] = expected[sample][i] - outputOutputs[i]
                totalErrors += errors[i] * errors[i]
            }

            // Backprop and parameter updates.
            backward(
                nn: nn,
                hiddenOutputs: hiddenOutputs,
                outputOutputs: outputOutputs,
                errors: errors,
                deltaHidden: &deltaHidden,
                deltaOutput: &deltaOutput
            )

            updateWeights(layer: &nn.output, inputs: hiddenOutputs, deltas: deltaOutput)
            updateWeights(layer: &nn.hidden, inputs: inputs[sample], deltas: deltaHidden)
        }

        // Average loss per epoch, printed every 1000 epochs.
        let loss = totalErrors / Double(numSamples)
        if (epoch + 1) % 1000 == 0 {
            print("Epoch \(epoch + 1), Error: \(String(format: "%.6f", loss))")
        }
    }
}

// Simple evaluation on XOR samples.
func test(nn: NeuralNetwork, inputs: [[Double]], expected: [[Double]]) {
    print("\nTesting the trained network:")
    for sample in 0..<numSamples {
        var hiddenOutputs = Array(repeating: 0.0, count: numHidden)
        var outputOutputs = Array(repeating: 0.0, count: numOutputs)

        // Forward pass to get the prediction.
        forward(layer: nn.hidden, inputs: inputs[sample], outputs: &hiddenOutputs)
        forward(layer: nn.output, inputs: hiddenOutputs, outputs: &outputOutputs)

        print(
            String(
                format: "Input: %.1f, %.1f, Expected Output: %.1f, Predicted Output: %.3f",
                inputs[sample][0],
                inputs[sample][1],
                expected[sample][0],
                outputOutputs[0]
            )
        )
    }
}

func main() {
    // Fixed initial seed for partial reproducibility.
    var rng = SimpleRng(seed: 42)

    // XOR dataset (binary inputs and expected outputs).
    let inputs: [[Double]] = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    let expected: [[Double]] = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]

    // Training and testing in the same process.
    var nn = initializeNetwork(rng: &rng)
    train(nn: &nn, inputs: inputs, expected: expected)
    test(nn: nn, inputs: inputs, expected: expected)
}

main()
