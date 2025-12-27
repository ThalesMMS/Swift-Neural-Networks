use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::process;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const NUM_INPUTS: usize = 784;
const NUM_HIDDEN: usize = 512;
const NUM_OUTPUTS: usize = 10;
const TRAIN_SAMPLES: usize = 60000;
const TEST_SAMPLES: usize = 10000;
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 64;

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    fn reseed_from_time(&mut self) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state = if nanos == 0 { 0x9e3779b97f4a7c15 } else { nanos };
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }

    fn gen_range_f64(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    fn gen_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ActivationType {
    Sigmoid,
    Relu,
    Softmax,
}

struct LinearLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activation: ActivationType,
}

struct NeuralNetwork {
    hidden_layer: LinearLayer,
    output_layer: LinearLayer,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

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

fn initialize_layer(
    input_size: usize,
    output_size: usize,
    activation: ActivationType,
    rng: &mut SimpleRng,
) -> LinearLayer {
    let mut weights = vec![vec![0.0; output_size]; input_size];
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

fn initialize_network(rng: &mut SimpleRng) -> NeuralNetwork {
    rng.reseed_from_time();
    let hidden_layer = initialize_layer(NUM_INPUTS, NUM_HIDDEN, ActivationType::Relu, rng);
    let output_layer = initialize_layer(NUM_HIDDEN, NUM_OUTPUTS, ActivationType::Softmax, rng);

    NeuralNetwork {
        hidden_layer,
        output_layer,
    }
}

fn save_model(nn: &NeuralNetwork, filename: &str) {
    let file = File::create(filename).unwrap_or_else(|_| {
        eprintln!("Could not open file {} for writing model", filename);
        process::exit(1);
    });
    let mut writer = BufWriter::new(file);

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

    write_i32(&mut writer, nn.hidden_layer.input_size as i32);
    write_i32(&mut writer, nn.hidden_layer.output_size as i32);
    write_i32(&mut writer, nn.output_layer.output_size as i32);

    for i in 0..nn.hidden_layer.input_size {
        for value in &nn.hidden_layer.weights[i] {
            write_f64(&mut writer, *value);
        }
    }
    for value in &nn.hidden_layer.biases {
        write_f64(&mut writer, *value);
    }

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

fn linear_layer_forward(layer: &LinearLayer, inputs: &[f64], outputs: &mut [f64]) {
    for i in 0..layer.output_size {
        let mut activation_sum = layer.biases[i];
        for j in 0..layer.input_size {
            activation_sum += inputs[j] * layer.weights[j][i];
        }
        outputs[i] = activation_sum;
    }

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

fn backward(
    nn: &mut NeuralNetwork,
    inputs: &[f64],
    hidden_outputs: &[f64],
    output_outputs: &[f64],
    expected_outputs: &[f64],
    delta_hidden: &mut [f64],
    delta_output: &mut [f64],
) {
    for i in 0..NUM_OUTPUTS {
        delta_output[i] = output_outputs[i] - expected_outputs[i];
    }

    for i in 0..NUM_HIDDEN {
        let mut error = 0.0;
        for j in 0..NUM_OUTPUTS {
            error += delta_output[j] * nn.output_layer.weights[i][j];
        }

        let activation_derivative = if nn.hidden_layer.activation == ActivationType::Sigmoid {
            sigmoid_derivative(hidden_outputs[i])
        } else {
            relu_derivative(hidden_outputs[i])
        };
        delta_hidden[i] = error * activation_derivative;
    }

    update_weights_biases(&mut nn.output_layer, hidden_outputs, delta_output);
    update_weights_biases(&mut nn.hidden_layer, inputs, delta_hidden);
}

fn update_weights_biases(layer: &mut LinearLayer, inputs: &[f64], deltas: &[f64]) {
    for i in 0..layer.input_size {
        for j in 0..layer.output_size {
            layer.weights[i][j] -= LEARNING_RATE * deltas[j] * inputs[i];
        }
    }

    for i in 0..layer.output_size {
        layer.biases[i] -= LEARNING_RATE * deltas[i];
    }
}

fn cross_entropy_loss(predicted: &[f64], expected: &[f64]) -> f64 {
    let mut loss = 0.0;
    for i in 0..predicted.len() {
        loss -= expected[i] * (predicted[i] + 1e-9).ln();
    }
    loss
}

fn one_hot_encode(label: usize, vector: &mut [f64]) {
    for value in vector.iter_mut() {
        *value = 0.0;
    }
    if label < vector.len() {
        vector[label] = 1.0;
    }
}

fn train(
    nn: &mut NeuralNetwork,
    inputs: &mut Vec<Vec<f64>>,
    labels: &mut Vec<usize>,
    num_samples: usize,
    rng: &mut SimpleRng,
) {
    let file = File::create("./logs/training_loss_c.txt").unwrap_or_else(|_| {
        eprintln!("Could not open file for writing training loss.");
        process::exit(1);
    });
    let mut loss_file = BufWriter::new(file);

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;
        let start_time = Instant::now();

        for i in 0..num_samples {
            let j = rng.gen_usize(num_samples);
            inputs.swap(i, j);
            labels.swap(i, j);
        }

        for batch_start in (0..num_samples).step_by(BATCH_SIZE) {
            let mut batch_end = batch_start + BATCH_SIZE;
            if batch_end > num_samples {
                batch_end = num_samples;
            }

            for idx in batch_start..batch_end {
                let mut hidden_outputs = [0.0; NUM_HIDDEN];
                let mut output_outputs = [0.0; NUM_OUTPUTS];
                let mut expected_output = [0.0; NUM_OUTPUTS];

                one_hot_encode(labels[idx], &mut expected_output);
                forward(nn, inputs, idx, &mut hidden_outputs, &mut output_outputs);

                let loss = cross_entropy_loss(&output_outputs, &expected_output);
                total_loss += loss;

                let mut delta_hidden = [0.0; NUM_HIDDEN];
                let mut delta_output = [0.0; NUM_OUTPUTS];
                backward(
                    nn,
                    &inputs[idx],
                    &hidden_outputs,
                    &output_outputs,
                    &expected_output,
                    &mut delta_hidden,
                    &mut delta_output,
                );
            }
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

fn test(nn: &NeuralNetwork, inputs: &[Vec<f64>], labels: &[usize], num_samples: usize) {
    let mut correct_predictions = 0;

    for idx in 0..num_samples {
        let mut hidden_outputs = [0.0; NUM_HIDDEN];
        let mut output_outputs = [0.0; NUM_OUTPUTS];

        linear_layer_forward(&nn.hidden_layer, &inputs[idx], &mut hidden_outputs);
        linear_layer_forward(&nn.output_layer, &hidden_outputs, &mut output_outputs);

        let mut predicted_label = 0;
        let mut max_prob = output_outputs[0];
        for i in 1..NUM_OUTPUTS {
            if output_outputs[i] > max_prob {
                max_prob = output_outputs[i];
                predicted_label = i;
            }
        }

        if predicted_label == labels[idx] {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f64 / num_samples as f64 * 100.0;
    println!("Test Accuracy: {:.2}%", accuracy);
}

fn read_be_u32(file: &mut File) -> u32 {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap_or_else(|_| {
        eprintln!("Failed reading MNIST header.");
        process::exit(1);
    });
    u32::from_be_bytes(buf)
}

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

    let mut images = Vec::with_capacity(num_images);
    let mut buffer = vec![0u8; image_size];

    for _ in 0..num_images {
        file.read_exact(&mut buffer).unwrap_or_else(|_| {
            eprintln!("Failed reading MNIST image data.");
            process::exit(1);
        });
        let mut image = Vec::with_capacity(image_size);
        for &pixel in &buffer {
            image.push(pixel as f64 / 255.0);
        }
        images.push(image);
    }

    images
}

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
    let program_start = Instant::now();

    println!("Loading training data...");
    let load_start = Instant::now();
    let mut train_images = read_mnist_images("./data/train-images.idx3-ubyte", TRAIN_SAMPLES);
    let mut train_labels = read_mnist_labels("./data/train-labels.idx1-ubyte", TRAIN_SAMPLES);

    println!("Loading test data...");
    let mut test_images = read_mnist_images("./data/t10k-images.idx3-ubyte", TEST_SAMPLES);
    let mut test_labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte", TEST_SAMPLES);
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Data loading time: {:.2} seconds", load_time);

    println!("Initializing neural network...");
    let mut rng = SimpleRng::new(1);
    let mut nn = initialize_network(&mut rng);

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

    println!("Testing neural network...");
    let test_start = Instant::now();
    test(&nn, &test_images, &test_labels, TEST_SAMPLES);
    let test_time = test_start.elapsed().as_secs_f64();
    println!("Testing time: {:.2} seconds", test_time);

    println!("Saving model...");
    save_model(&nn, "mnist_model.bin");

    let total_time = program_start.elapsed().as_secs_f64();
    println!("\n=== Performance Summary ===");
    println!("Data loading time: {:.2} seconds", load_time);
    println!("Total training time: {:.2} seconds", train_time);
    println!("Testing time: {:.2} seconds", test_time);
    println!("Total program time: {:.2} seconds", total_time);
    println!("========================");
}
