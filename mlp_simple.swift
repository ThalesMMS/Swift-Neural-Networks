import Foundation

// Pequena MLP para aprender XOR (exemplo didatico).
let numInputs = 2
let numHidden = 4
let numOutputs = 1
let numSamples = 4
let learningRate = 0.01
let epochs = 1_000_000

// RNG simples para evitar dependencias externas (nao criptografico).
struct SimpleRng {
    private var state: UInt64

    // Seed explicita (se zero, usa um valor fixo).
    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9e3779b97f4a7c15 : seed
    }

    // Re-seed baseado no tempo atual.
    mutating func reseedFromTime() {
        let nanos = UInt64(Date().timeIntervalSince1970 * 1_000_000_000)
        state = nanos == 0 ? 0x9e3779b97f4a7c15 : nanos
    }

    // Xorshift basico para gerar u32.
    mutating func nextUInt32() -> UInt32 {
        var x = state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        state = x
        return UInt32(truncatingIfNeeded: x >> 32)
    }

    // Converte para [0, 1).
    mutating func nextDouble() -> Double {
        return Double(nextUInt32()) / Double(UInt32.max)
    }

    // Amostra uniforme em [low, high).
    mutating func uniform(_ low: Double, _ high: Double) -> Double {
        return low + (high - low) * nextDouble()
    }
}

// Funcao de ativacao sigmoid.
func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}

// Derivada da sigmoid, assumindo x = sigmoid(z).
func sigmoidDerivative(_ x: Double) -> Double {
    return x * (1.0 - x)
}

// Camada densa: pesos (input x output) e vies (output).
struct LinearLayer {
    let inputSize: Int
    let outputSize: Int
    var weights: [[Double]]
    var biases: [Double]
}

// Rede com uma camada escondida e uma de saida.
struct NeuralNetwork {
    var hidden: LinearLayer
    var output: LinearLayer
}

// Inicializa pesos e vies com valores pequenos aleatorios.
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

// Cria a rede completa com tamanhos fixos do XOR.
func initializeNetwork(rng: inout SimpleRng) -> NeuralNetwork {
    rng.reseedFromTime()
    let hidden = initializeLayer(inputSize: numInputs, outputSize: numHidden, rng: &rng)
    let output = initializeLayer(inputSize: numHidden, outputSize: numOutputs, rng: &rng)
    return NeuralNetwork(hidden: hidden, output: output)
}

// Forward de uma camada: z = W*x + b, seguido de sigmoid.
func forward(layer: LinearLayer, inputs: [Double], outputs: inout [Double]) {
    for i in 0..<layer.outputSize {
        var activation = layer.biases[i]
        for j in 0..<layer.inputSize {
            activation += inputs[j] * layer.weights[j][i]
        }
        outputs[i] = sigmoid(activation)
    }
}

// Backprop: calcula deltas da saida e da camada escondida.
func backward(
    nn: NeuralNetwork,
    hiddenOutputs: [Double],
    outputOutputs: [Double],
    errors: [Double],
    deltaHidden: inout [Double],
    deltaOutput: inout [Double]
) {
    for i in 0..<nn.output.outputSize {
        // delta_out = erro * derivada da ativacao.
        deltaOutput[i] = errors[i] * sigmoidDerivative(outputOutputs[i])
    }

    for i in 0..<nn.hidden.outputSize {
        // Erro propagado da saida para a camada escondida.
        var error = 0.0
        for j in 0..<nn.output.outputSize {
            error += deltaOutput[j] * nn.output.weights[i][j]
        }
        deltaHidden[i] = error * sigmoidDerivative(hiddenOutputs[i])
    }
}

// Atualiza pesos e vies com gradiente descendente (SGD).
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

// Treino com erro quadratico medio por amostra.
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

            // Erro por saida e acumulacao do MSE.
            for i in 0..<numOutputs {
                errors[i] = expected[sample][i] - outputOutputs[i]
                totalErrors += errors[i] * errors[i]
            }

            // Backprop e atualizacao dos parametros.
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

        // Loss media por epoca, exibida a cada 1000 epocas.
        let loss = totalErrors / Double(numSamples)
        if (epoch + 1) % 1000 == 0 {
            print("Epoch \(epoch + 1), Error: \(String(format: "%.6f", loss))")
        }
    }
}

// Avaliacao simples sobre as amostras do XOR.
func test(nn: NeuralNetwork, inputs: [[Double]], expected: [[Double]]) {
    print("\nTesting the trained network:")
    for sample in 0..<numSamples {
        var hiddenOutputs = Array(repeating: 0.0, count: numHidden)
        var outputOutputs = Array(repeating: 0.0, count: numOutputs)

        // Forward pass para obter a predicao.
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
    // Seed inicial fixa para reproducibilidade parcial.
    var rng = SimpleRng(seed: 42)

    // Dataset XOR (entrada binaria e saida esperada).
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

    // Treino e teste no mesmo processo.
    var nn = initializeNetwork(rng: &rng)
    train(nn: &nn, inputs: inputs, expected: expected)
    test(nn: nn, inputs: inputs, expected: expected)
}

main()
