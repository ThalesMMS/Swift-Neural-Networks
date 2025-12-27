use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use rayon::prelude::*;

// MLP sequencial para MNIST (port em Rust para estudo).
const NUM_INPUTS: usize = 784;
const NUM_HIDDEN: usize = 512;
const NUM_OUTPUTS: usize = 10;
const TRAIN_SAMPLES: usize = 60000;
const TEST_SAMPLES: usize = 10000;
// Hiperparametros do treino.
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 64;

// RNG simples para reproducibilidade sem crates externas.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    // Seed explicita (se zero, usa um valor fixo).
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    // Re-seed baseado no tempo atual.
    fn reseed_from_time(&mut self) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state = if nanos == 0 { 0x9e3779b97f4a7c15 } else { nanos };
    }

    // Xorshift basico para gerar u32.
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    // Converte para [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }

    // Amostra uniforme em [low, high).
    fn gen_range_f64(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    // Amostra inteira em [0, upper).
    fn gen_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper
        }
    }
}

// Tipos de ativacao usados na rede.
#[derive(Clone, Copy, PartialEq)]
enum ActivationType {
    Sigmoid,
    Relu,
    Softmax,
}

// Camada densa: pesos (input x output), vies e tipo de ativacao.
struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activation: ActivationType,
}

// Rede com uma camada escondida e uma camada de saida.
struct NeuralNetwork {
    hidden_layer: LinearLayer,
    output_layer: LinearLayer,
}

// Gradientes acumulados para um minibatch (armazenados de forma contigua).
struct Gradients {
    hidden_weights: Vec<f64>,
    hidden_biases: Vec<f64>,
    output_weights: Vec<f64>,
    output_biases: Vec<f64>,
}

impl Gradients {
    fn zeros() -> Self {
        Self {
            hidden_weights: vec![0.0; NUM_INPUTS * NUM_HIDDEN],
            hidden_biases: vec![0.0; NUM_HIDDEN],
            output_weights: vec![0.0; NUM_HIDDEN * NUM_OUTPUTS],
            output_biases: vec![0.0; NUM_OUTPUTS],
        }
    }

    fn add_sample(
        &mut self,
        inputs: &[f64],
        hidden_outputs: &[f64],
        delta_hidden: &[f64],
        delta_output: &[f64],
    ) {
        for i in 0..NUM_INPUTS {
            let input_value = inputs[i];
            let row_offset = i * NUM_HIDDEN;
            for j in 0..NUM_HIDDEN {
                self.hidden_weights[row_offset + j] += delta_hidden[j] * input_value;
            }
        }
        for j in 0..NUM_HIDDEN {
            self.hidden_biases[j] += delta_hidden[j];
        }

        for i in 0..NUM_HIDDEN {
            let hidden_value = hidden_outputs[i];
            let row_offset = i * NUM_OUTPUTS;
            for j in 0..NUM_OUTPUTS {
                self.output_weights[row_offset + j] += delta_output[j] * hidden_value;
            }
        }
        for j in 0..NUM_OUTPUTS {
            self.output_biases[j] += delta_output[j];
        }
    }

    fn add_inplace(&mut self, other: &Gradients) {
        for (value, other_value) in self.hidden_weights.iter_mut().zip(&other.hidden_weights) {
            *value += other_value;
        }
        for (value, other_value) in self.hidden_biases.iter_mut().zip(&other.hidden_biases) {
            *value += other_value;
        }

        for (value, other_value) in self.output_weights.iter_mut().zip(&other.output_weights) {
            *value += other_value;
        }
        for (value, other_value) in self.output_biases.iter_mut().zip(&other.output_biases) {
            *value += other_value;
        }
    }
}

// Sigmoid e sua derivada (usada em backprop).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

// ReLU e derivada.
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// Softmax numericamente estavel: subtrai o max antes do exp.
fn softmax_inplace(outputs: &mut [f64]) {
    let mut max = outputs[0];
    for &value in outputs.iter().skip(1) {
        if value > max {
            max = value;
        }
    }

    let mut sum = 0.0;
    for value in outputs.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }

    for value in outputs.iter_mut() {
        *value /= sum;
    }
}

// Inicializa uma camada com Xavier e vies zeros.
fn initialize_layer(
    input_size: usize,
    output_size: usize,
    activation: ActivationType,
    rng: &mut SimpleRng,
) -> LinearLayer {
    let mut weights = vec![vec![0.0; output_size]; input_size];
    // Xavier: limite depende do fan_in + fan_out.
    let limit = (6.0 / (input_size + output_size) as f64).sqrt();
    for i in 0..input_size {
        for j in 0..output_size {
            weights[i][j] = rng.gen_range_f64(-limit, limit);
        }
    }

    LinearLayer {
        input_size,
        output_size,
        weights,
        biases: vec![0.0; output_size],
        activation,
    }
}

// Construcao da rede 784 -> 512 -> 10.
fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    rng.reseed_from_time();
    let hidden_layer = initialize_layer(NUM_INPUTS, NUM_HIDDEN, ActivationType::Relu, rng);
    let output_layer = initialize_layer(NUM_HIDDEN, NUM_OUTPUTS, ActivationType::Softmax, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

// Salva o modelo em binario (int + doubles, endianness nativa).
fn save_model(nn: &NeuralNetwork, filename: &str) {
    let file = File::create(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {} for writing model", filename);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    // Helpers para escrever em bytes.
    let write_i32 = |writer: &mut BufWriter<File>, value: i32| {
        writer.write_all(&value.to_ne_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };
    let write_f64 = |writer: &mut BufWriter<File>, value: f64| {
        writer.write_all(&value.to_ne_bytes()).unwrap_or_else(|_| {
            eprintln!("Failed writing model data");
            process::exit(1);
        });
    };

    // Dimensoes das camadas.
    write_i32(&mut writer, nn.hidden_layer.input_size as i32);
    write_i32(&mut writer, nn.hidden_layer.output_size as i32);
    write_i32(&mut writer, nn.output_layer.output_size as i32);

    // Pesos e vies da camada escondida.
    for i in 0..nn.hidden_layer.input_size {
        for value in &nn.hidden_layer.weights[i] {
            write_f64(&mut writer, *value);
        }
    }
    for value in &nn.hidden_layer.biases {
        write_f64(&mut writer, *value);
    }

    // Pesos e vies da camada de saida.
    for i in 0..nn.output_layer.input_size {
        for value in &nn.output_layer.weights[i] {
            write_f64(&mut writer, *value);
        }
    }
    for value in &nn.output_layer.biases {
        write_f64(&mut writer, *value);
    }

    println!("Model saved to {}", filename);
}

// Forward de uma camada: z = W*x + b, depois ativacao.
fn linear_layer_forward(layer: &LinearLayer, inputs: &[f64], outputs: &mut [f64]) {
    for i in 0..layer.output_size {
        let mut activation_sum = layer.biases[i];
        for j in 0..layer.input_size {
            activation_sum += inputs[j] * layer.weights[j][i];
        }
        outputs[i] = activation_sum;
    }

    // Aplica a funcao de ativacao da camada.
    match layer.activation {
        ActivationType::Sigmoid => {
            for value in outputs.iter_mut() {
                *value = sigmoid(*value);
            }
        }
        ActivationType::Relu => {
            for value in outputs.iter_mut() {
                *value = relu(*value);
            }
        }
        ActivationType::Softmax => {
            softmax_inplace(outputs);
        }
    }
}

// Forward completo (entrada -> escondida -> saida).
fn forward(
    nn: &NeuralNetwork,
    inputs: &[Vec<f64>],
    idx: usize,
    hidden_outputs: &mut [f64],
    output_outputs: &mut [f64],
) {
    linear_layer_forward(&nn.hidden_layer, &inputs[idx], hidden_outputs);
    linear_layer_forward(&nn.output_layer, hidden_outputs, output_outputs);
}

// Backprop: calcula deltas (sem atualizar pesos/vies).
fn compute_deltas(
    nn: &NeuralNetwork,
    hidden_outputs: &[f64],
    output_outputs: &[f64],
    expected_outputs: &[f64],
    delta_hidden: &mut [f64],
    delta_output: &mut [f64],
) {
    // Softmax + cross entropy: delta_out = y_pred - y_true.
    for i in 0..NUM_OUTPUTS {
        delta_output[i] = output_outputs[i] - expected_outputs[i];
    }

    // Erro retropropagado para camada escondida.
    for i in 0..NUM_HIDDEN {
        let mut error = 0.0;
        for j in 0..NUM_OUTPUTS {
            error += delta_output[j] * nn.output_layer.weights[i][j];
        }

        // Derivada depende da ativacao escolhida.
        let activation_derivative = if nn.hidden_layer.activation == ActivationType::Sigmoid {
            sigmoid_derivative(hidden_outputs[i])
        } else {
            relu_derivative(hidden_outputs[i])
        };
        delta_hidden[i] = error * activation_derivative;
    }

}

// Aplica gradientes acumulados (media do minibatch).
fn apply_gradients(nn: &mut NeuralNetwork, grads: &Gradients, scale: f64) {
    let lr = LEARNING_RATE * scale;

    for i in 0..nn.hidden_layer.input_size {
        let row_offset = i * nn.hidden_layer.output_size;
        for j in 0..nn.hidden_layer.output_size {
            nn.hidden_layer.weights[i][j] -= lr * grads.hidden_weights[row_offset + j];
        }
    }
    for j in 0..nn.hidden_layer.output_size {
        nn.hidden_layer.biases[j] -= lr * grads.hidden_biases[j];
    }

    for i in 0..nn.output_layer.input_size {
        let row_offset = i * nn.output_layer.output_size;
        for j in 0..nn.output_layer.output_size {
            nn.output_layer.weights[i][j] -= lr * grads.output_weights[row_offset + j];
        }
    }
    for j in 0..nn.output_layer.output_size {
        nn.output_layer.biases[j] -= lr * grads.output_biases[j];
    }
}

// Cross entropy com epsilon para evitar log(0).
fn cross_entropy_loss(predicted: &[f64], expected: &[f64]) -> f64 {
    let mut loss = 0.0;
    for i in 0..predicted.len() {
        loss -= expected[i] * (predicted[i] + 1e-9).ln();
    }
    loss
}

// Converte um rotulo em vetor one-hot.
fn one_hot_encode(label: usize, vector: &mut [f64]) {
    for value in vector.iter_mut() {
        *value = 0.0;
    }
    if label < vector.len() {
        vector[label] = 1.0;
    }
}

// Treino com embaralhamento e minibatches.
fn train(
    nn: &mut NeuralNetwork,
    inputs: &mut Vec<Vec<f64>>,
    labels: &mut Vec<usize>,
    num_samples: usize,
    rng: &mut SimpleRng,
) {
    // Log de loss/tempo por epoca.
    let file = File::create("./logs/training_loss_c.txt").unwrap_or_else(|_| {
        eprintln!("Could not open file for writing training loss.");
        process::exit(1);
    });
    let mut loss_file = BufWriter::new(file);

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;
        let start_time = Instant::now();

        // Embaralha imagens e rotulos em sincronia.
        for i in 0..num_samples {
            let j = rng.gen_usize(num_samples);
            inputs.swap(i, j);
            labels.swap(i, j);
        }

        let inputs_ref = &*inputs;
        let labels_ref = &*labels;

        // Loop por minibatch (gradientes acumulados em paralelo).
        for batch_start in (0..num_samples).step_by(BATCH_SIZE) {
            let mut batch_end = batch_start + BATCH_SIZE;
            if batch_end > num_samples {
                batch_end = num_samples;
            }

            let (batch_loss, batch_grads) = {
                let nn_ref = &*nn;
                (batch_start..batch_end)
                    .into_par_iter()
                    .fold(
                        || (0.0, Gradients::zeros()),
                        |(mut loss, mut grads), idx| {
                            let input = &inputs_ref[idx];
                            let mut hidden_outputs = [0.0; NUM_HIDDEN];
                            let mut output_outputs = [0.0; NUM_OUTPUTS];
                            let mut expected_output = [0.0; NUM_OUTPUTS];
                            let mut delta_hidden = [0.0; NUM_HIDDEN];
                            let mut delta_output = [0.0; NUM_OUTPUTS];

                            // One-hot do rotulo e forward pass.
                            one_hot_encode(labels_ref[idx], &mut expected_output);
                            forward(nn_ref, inputs_ref, idx, &mut hidden_outputs, &mut output_outputs);

                            // Loss para monitorar convergencia.
                            loss += cross_entropy_loss(&output_outputs, &expected_output);

                            // Backprop apenas para deltas e gradientes.
                            compute_deltas(
                                nn_ref,
                                &hidden_outputs,
                                &output_outputs,
                                &expected_output,
                                &mut delta_hidden,
                                &mut delta_output,
                            );
                            grads.add_sample(input, &hidden_outputs, &delta_hidden, &delta_output);

                            (loss, grads)
                        },
                    )
                    .reduce_with(|(loss_a, mut grads_a), (loss_b, grads_b)| {
                        grads_a.add_inplace(&grads_b);
                        (loss_a + loss_b, grads_a)
                    })
                    .unwrap_or_else(|| (0.0, Gradients::zeros()))
            };

            let batch_len = (batch_end - batch_start) as f64;
            apply_gradients(nn, &batch_grads, 1.0 / batch_len);
            total_loss += batch_loss;
        }

        let duration = start_time.elapsed().as_secs_f64();
        let average_loss = total_loss / num_samples as f64;
        println!(
            "Epoch {}, Loss: {:.6} Time: {:.6}",
            epoch + 1,
            average_loss,
            duration
        );
        writeln!(loss_file, "{},{},{}", epoch + 1, average_loss, duration).unwrap_or_else(|_| {
            eprintln!("Failed writing training loss data.");
            process::exit(1);
        });
    }
}

// Avalia acuracia no conjunto de teste.
fn test(nn: &NeuralNetwork, inputs: &[Vec<f64>], labels: &[usize], num_samples: usize) {
    let correct_predictions: usize = (0..num_samples)
        .into_par_iter()
        .map(|idx| {
            let mut hidden_outputs = [0.0; NUM_HIDDEN];
            let mut output_outputs = [0.0; NUM_OUTPUTS];

            // Forward para obter as probabilidades.
            linear_layer_forward(&nn.hidden_layer, &inputs[idx], &mut hidden_outputs);
            linear_layer_forward(&nn.output_layer, &hidden_outputs, &mut output_outputs);

            // Argmax da saida (classe prevista).
            let mut predicted_label = 0;
            let mut max_prob = output_outputs[0];
            for i in 1..NUM_OUTPUTS {
                if output_outputs[i] > max_prob {
                    max_prob = output_outputs[i];
                    predicted_label = i;
                }
            }

            if predicted_label == labels[idx] { 1 } else { 0 }
        })
        .sum();

    let accuracy = correct_predictions as f64 / num_samples as f64 * 100.0;
    println!("Test Accuracy: {:.2}%", accuracy);
}

// MNIST IDX usa big-endian nos inteiros.
fn read_be_u32(file: &mut File) -> u32 {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap_or_else(|_| {
        eprintln!("Failed reading MNIST header.");
        process::exit(1);
    });
    u32::from_be_bytes(buf)
}

// Le imagens IDX e normaliza para [0, 1].
fn read_mnist_images(filename: &str, num_images: usize) -> Vec<Vec<f64>> {
    let mut file = File::open(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {}", filename);
        process::exit(1);
    });

    let _magic_number = read_be_u32(&mut file);
    let _number_of_images = read_be_u32(&mut file);
    let rows = read_be_u32(&mut file) as usize;
    let cols = read_be_u32(&mut file) as usize;
    let image_size = rows * cols;

    // Lemos em um buffer para reduzir overhead de IO.
    let mut images = Vec::with_capacity(num_images);
    let mut buffer = vec![0u8; image_size];

    for _ in 0..num_images {
        file.read_exact(&mut buffer).unwrap_or_else(|_| {
            eprintln!("Failed reading MNIST image data.");
            process::exit(1);
        });
        // Cada pixel vira f64 normalizado.
        let mut image = Vec::with_capacity(image_size);
        for &pixel in &buffer {
            image.push(pixel as f64 / 255.0);
        }
        images.push(image);
    }

    images
}

// Le rotulos IDX (0-9).
fn read_mnist_labels(filename: &str, num_labels: usize) -> Vec<usize> {
    let mut file = File::open(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {}", filename);
        process::exit(1);
    });

    let _magic_number = read_be_u32(&mut file);
    let _number_of_labels = read_be_u32(&mut file);

    let mut labels = Vec::with_capacity(num_labels);
    let mut buffer = vec![0u8; num_labels];
    file.read_exact(&mut buffer).unwrap_or_else(|_| {
        eprintln!("Failed reading MNIST label data.");
        process::exit(1);
    });
    for label in buffer {
        labels.push(label as usize);
    }

    labels
}

fn main() {
    // Cronometro total do programa.
    let program_start = Instant::now();

    // Le dados de treino.
    println!("Loading training data...");
    let load_start = Instant::now();
    let mut train_images = read_mnist_images("./data/train-images.idx3-ubyte", TRAIN_SAMPLES);
    let mut train_labels = read_mnist_labels("./data/train-labels.idx1-ubyte", TRAIN_SAMPLES);

    // Le dados de teste.
    println!("Loading test data...");
    let test_images = read_mnist_images("./data/t10k-images.idx3-ubyte", TEST_SAMPLES);
    let test_labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte", TEST_SAMPLES);
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Data loading time: {:.2} seconds", load_time);

    // Inicializa a rede com pesos aleatorios.
    println!("Initializing neural network...");
    let mut rng = SimpleRng::new(1);
    let mut nn = initialize_network(&mut rng);

    // Treina a rede.
    println!("Training neural network...");
    let train_start = Instant::now();
    train(
        &mut nn,
        &mut train_images,
        &mut train_labels,
        TRAIN_SAMPLES,
        &mut rng,
    );
    let train_time = train_start.elapsed().as_secs_f64();
    println!("Total training time: {:.2} seconds", train_time);

    // Avalia no teste.
    println!("Testing neural network...");
    let test_start = Instant::now();
    test(&nn, &test_images, &test_labels, TEST_SAMPLES);
    let test_time = test_start.elapsed().as_secs_f64();
    println!("Testing time: {:.2} seconds", test_time);

    // Salva o modelo para uso futuro.
    println!("Saving model...");
    save_model(&nn, "mnist_model.bin");

    // Resumo de tempos.
    let total_time = program_start.elapsed().as_secs_f64();
    println!("\n=== Performance Summary ===");
    println!("Data loading time: {:.2} seconds", load_time);
    println!("Total training time: {:.2} seconds", train_time);
    println!("Testing time: {:.2} seconds", test_time);
    println!("Total program time: {:.2} seconds", total_time);
    println!("========================");
}
