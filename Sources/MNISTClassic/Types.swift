import Foundation

// Activation types used in the network.
enum ActivationType {
    case sigmoid
    case relu
    case softmax
}

// Dense layer: weights (input x output), biases, and activation.
struct DenseLayer {
    let inputSize: Int
    let outputSize: Int
    var weights: [Float] // row-major
    var biases: [Float]
    let activation: ActivationType
}

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    var hidden: DenseLayer
    var output: DenseLayer
}
