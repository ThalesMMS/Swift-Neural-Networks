/*
 * =============================================================================
 * mnist_mlp_openmp_cpu.c - OpenMP CPU Parallel Implementation
 * =============================================================================
 *
 * DESCRIÇÃO:
 * Implementação paralela para CPU utilizando OpenMP de uma rede neural MLP
 * para classificação do dataset MNIST.
 *
 * PARALELIZAÇÃO:
 * - Forward propagation: #pragma omp parallel for (linhas ~329-339)
 * - Activation functions: #pragma omp parallel for (linhas ~343-363)
 * - Backward propagation: #pragma omp parallel for (linhas ~377-399)
 * - Weight updates: #pragma omp parallel for collapse(2) (linhas ~417-427)
 *
 * MUDANÇAS DA VERSÃO SEQUENCIAL:
 * 1. Adição de diretivas #pragma omp parallel for em loops computacionais
 * 2. Medição de tempo: clock() → omp_get_wtime() (para tempo real wall-clock)
 * 3. Thread control: usa variável de ambiente OMP_NUM_THREADS (padrão: 4)
 * 4. Log files: criados com número de threads no nome para evitar sobrescrever
 *
 * TEMPOS DE EXECUÇÃO (10 ÉPOCAS):
 *
 * AMBIENTE: Intel Core i7-15th Gen, NVIDIA RTX 3050 Ti, Windows
 *
 * Versão Sequencial (baseline):
 *   Tempo total: 337.11 segundos
 *   Tempo/época: 33.71 segundos
 *
 * OpenMP CPU (esta versão - 4 threads padrão):
 *   Tempo total: 354.36 segundos
 *   Tempo/época: 35.44 segundos
 *   Speedup: 0.95× (5% mais lento que sequential!)
 *
 * NOTA: Esta versão OpenMP CPU apresentou desempenho PIOR que a versão
 * sequencial com 4 threads. Possíveis causas:
 * - Overhead de criação/sincronização de threads
 * - Contenção de memória e cache thrashing
 * - Problema não possui paralelismo suficiente para 4 threads
 * - False sharing entre threads
 *
 * Resultados com diferentes números de threads (vs sequential baseline 33.71s):
 *   1 thread:    42.79s/época (Speedup: 0.79×) - 27% mais lento!
 *   2 threads:   37.73s/época (Speedup: 0.89×) - 12% mais lento
 *   4 threads:   35.44s/época (Speedup: 0.95×) - 5% mais lento
 *   8 threads:   49.94s/época (Speedup: 0.68×) - 48% mais lento!
 *   16 threads:  71.44s/época (Speedup: 0.47×) - 112% mais lento!!
 *   32 threads: 102.83s/época (Speedup: 0.33×) - 205% mais lento!!!
 *
 * ANÁLISE: NEGATIVE SCALING - Performance PIORA com mais threads!
 * - 1-4 threads: Overhead moderado (~5-27% perda)
 * - 8+ threads: Colapso de performance devido a contenção massiva
 * - Causa raiz: Problema possui paralelismo insuficiente + overhead domina
 *
 * COMPILAÇÃO:
 *   gcc -fopenmp -O3 -o mnist_mlp_openmp_cpu mnist_mlp_openmp_cpu.c -lm
 *
 * EXECUÇÃO:
 *   export OMP_NUM_THREADS=4
 *   ./mnist_mlp_openmp_cpu
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define NUM_INPUTS 784      // 28x28 pixels
#define NUM_HIDDEN 512      // Number of hidden neurons
#define NUM_OUTPUTS 10      // Digits 0-9
#define TRAIN_SAMPLES 60000 // Number of training samples
#define TEST_SAMPLES 10000  // Number of test samples
#define LEARNING_RATE 0.01  // Learning rate
#define EPOCHS 10           // Number of training epochs
#define BATCH_SIZE 64       // Mini-batch size

// Activation Types
typedef enum
{
    SIGMOID,
    RELU,
    SOFTMAX
} ActivationType;

// Data structures
typedef struct
{
    int input_size;
    int output_size;
    double **weights;
    double *biases;
    ActivationType activation;
} LinearLayer;

typedef struct
{
    LinearLayer hidden_layer;
    LinearLayer output_layer;
} NeuralNetwork;

// Layer prototypes
void initialize_layer(LinearLayer *layer, int input_size, int output_size, ActivationType activation);
void free_layer(LinearLayer *layer);
void initialize_network(NeuralNetwork *nn);
void free_network(NeuralNetwork *nn);
void save_model(NeuralNetwork *nn, const char *filename);
void linear_layer_forward(LinearLayer *layer, double inputs[], double outputs[]);

// Activation function prototypes
double sigmoid(double x);
double sigmoid_derivative(double x);
double relu(double x);
double relu_derivative(double x);
void softmax(double inputs[], int size, double outputs[]);

// Neural network prototypes
void forward(NeuralNetwork *nn, double **inputs, int idx, double hidden_outputs[NUM_HIDDEN], double output_outputs[NUM_OUTPUTS]);
void backward(NeuralNetwork *nn, double inputs[], double hidden_outputs[], double output_outputs[], double expected_outputs[], double delta_hidden[], double delta_output[]);
double cross_entropy_loss(double predicted[], double expected[], int num_outputs);
void update_weights_biases(LinearLayer *layer, double inputs[], double deltas[]);
void train(NeuralNetwork *nn, double **inputs, int *labels, int num_samples);
void test(NeuralNetwork *nn, double **inputs, int *labels, int num_samples);

// Data prototypes
void read_mnist_images(const char *filename, double **images, int num_images);
void read_mnist_labels(const char *filename, int *labels, int num_labels);
int reverse_int(int i);
void one_hot_encode(int label, double *vector, int size);

int main()
{
    // Set OpenMP thread count from environment variable or default to 4
    char *env_threads = getenv("OMP_NUM_THREADS");
    int num_threads = env_threads ? atoi(env_threads) : 4;
    omp_set_num_threads(num_threads);

    printf("Running with %d OpenMP threads\n", omp_get_max_threads());

    // Allocate memory for training data
    double **train_images = (double **)malloc(TRAIN_SAMPLES * sizeof(double *));
    int *train_labels = (int *)malloc(TRAIN_SAMPLES * sizeof(int));

    // Read training data
    printf("Loading training data...\n");
    read_mnist_images("./data/train-images.idx3-ubyte", train_images, TRAIN_SAMPLES);
    read_mnist_labels("./data/train-labels.idx1-ubyte", train_labels, TRAIN_SAMPLES);

    // Allocate memory for test data
    double **test_images = (double **)malloc(TEST_SAMPLES * sizeof(double *));
    int *test_labels = (int *)malloc(TEST_SAMPLES * sizeof(int));

    // Read test data
    printf("Loading test data...\n");
    read_mnist_images("./data/t10k-images.idx3-ubyte", test_images, TEST_SAMPLES);
    read_mnist_labels("./data/t10k-labels.idx1-ubyte", test_labels, TEST_SAMPLES);

    // Initialize neural network
    printf("Initializing neural network...\n");
    NeuralNetwork nn;
    initialize_network(&nn);

    // Train the neural network
    printf("Training neural network...\n");
    train(&nn, train_images, train_labels, TRAIN_SAMPLES);

    // Test the neural network
    printf("Testing neural network...\n");
    test(&nn, test_images, test_labels, TEST_SAMPLES);

    // Save the trained model
    printf("Saving model...\n");
    save_model(&nn, "mnist_model_cpu.bin");

    // Free training data
    for (int i = 0; i < TRAIN_SAMPLES; i++)
    {
        free(train_images[i]);
    }
    free(train_images);
    free(train_labels);

    // Free test data
    for (int i = 0; i < TEST_SAMPLES; i++)
    {
        free(test_images[i]);
    }
    free(test_images);
    free(test_labels);

    // Free neural network
    free_network(&nn);

    return 0;
}

// Activation function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of activation function
double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

// relu activation function
double relu(double x)
{
    return x > 0 ? x : 0;
}

// Derivative of relu activation function
double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}

// Softmax function
void softmax(double inputs[], int size, double outputs[])
{
    // softmax as defined here: https://en.wikipedia.org/wiki/Softmax_function

    // Find the maximum value across all input values
    double max = inputs[0];
    for (int i = 1; i < size; i++)
    {
        if (inputs[i] > max)
            max = inputs[i];
    }

    // Compute the exponentials of the input values
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        outputs[i] = exp(inputs[i] - max);
        sum += outputs[i];
    }

    // Normalize the output values
    for (int i = 0; i < size; i++)
    {
        outputs[i] /= sum;
    }
}

// Initialize a layer
void initialize_layer(LinearLayer *layer, int input_size, int output_size, ActivationType activation)
{
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;

    // Allocate memory for weights
    layer->weights = (double **)malloc(input_size * sizeof(double *));
    for (int i = 0; i < input_size; i++)
    {
        layer->weights[i] = (double *)malloc(output_size * sizeof(double));
    }

    // Allocate memory for biases
    layer->biases = (double *)malloc(output_size * sizeof(double));

    // Xavier Initialization
    double limit = sqrt(6.0 / (input_size + output_size));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < input_size; i++)
        for (int j = 0; j < output_size; j++)
            layer->weights[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;

#pragma omp parallel for
    for (int i = 0; i < output_size; i++)
        layer->biases[i] = 0.0;
}

// Free memory allocated for a layer
void free_layer(LinearLayer *layer)
{
    for (int i = 0; i < layer->input_size; i++)
    {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
}

// Initialize the neural network
void initialize_network(NeuralNetwork *nn)
{
    srand(time(NULL));
    initialize_layer(&nn->hidden_layer, NUM_INPUTS, NUM_HIDDEN, RELU);
    initialize_layer(&nn->output_layer, NUM_HIDDEN, NUM_OUTPUTS, SOFTMAX);
}

// Free memory allocated for the neural network
void free_network(NeuralNetwork *nn)
{
    free_layer(&nn->hidden_layer);
    free_layer(&nn->output_layer);
}

// Save model weights to file
void save_model(NeuralNetwork *nn, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Could not open file %s for writing model\n", filename);
        exit(1);
    }

    // Write layer dimensions
    fwrite(&nn->hidden_layer.input_size, sizeof(int), 1, fp);
    fwrite(&nn->hidden_layer.output_size, sizeof(int), 1, fp);
    fwrite(&nn->output_layer.output_size, sizeof(int), 1, fp);

    // Write hidden layer weights
    for (int i = 0; i < nn->hidden_layer.input_size; i++)
    {
        fwrite(nn->hidden_layer.weights[i], sizeof(double), nn->hidden_layer.output_size, fp);
    }

    // Write hidden layer biases
    fwrite(nn->hidden_layer.biases, sizeof(double), nn->hidden_layer.output_size, fp);

    // Write output layer weights
    for (int i = 0; i < nn->output_layer.input_size; i++)
    {
        fwrite(nn->output_layer.weights[i], sizeof(double), nn->output_layer.output_size, fp);
    }

    // Write output layer biases
    fwrite(nn->output_layer.biases, sizeof(double), nn->output_layer.output_size, fp);

    fclose(fp);
    printf("Model saved to %s\n", filename);
}

// Forward propagation for a single layer
void linear_layer_forward(LinearLayer *layer, double inputs[], double outputs[])
{
// For each neuron in the layer compute the weighted sum of inputs and add the bias to the activation
#pragma omp parallel for
    for (int i = 0; i < layer->output_size; i++)
    {
        double activation_sum = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++)
        {
            // z = W * x + b^{(1)}
            activation_sum += inputs[j] * layer->weights[j][i];
        }
        outputs[i] = activation_sum; // Pre-activation value
    }

    // Apply activation function
    switch (layer->activation)
    {
    case SIGMOID:
#pragma omp parallel for
        for (int i = 0; i < layer->output_size; i++)
        {
            outputs[i] = sigmoid(outputs[i]);
        }
        break;
    case RELU:
#pragma omp parallel for
        for (int i = 0; i < layer->output_size; i++)
        {
            outputs[i] = relu(outputs[i]);
        }
        break;
    case SOFTMAX:
        softmax(outputs, layer->output_size, outputs);
        break;
    }
}

void forward(NeuralNetwork *nn, double **inputs, int idx, double hidden_outputs[NUM_HIDDEN], double output_outputs[NUM_OUTPUTS])
{
    // defines the network forward pass
    linear_layer_forward(&nn->hidden_layer, inputs[idx], hidden_outputs);
    linear_layer_forward(&nn->output_layer, hidden_outputs, output_outputs);
}

// Backpropagation
void backward(NeuralNetwork *nn, double inputs[], double hidden_outputs[], double output_outputs[], double expected_outputs[], double delta_hidden[], double delta_output[])
{
// Output layer delta
#pragma omp parallel for
    for (int i = 0; i < NUM_OUTPUTS; i++)
    {
        // For softmax and cross-entropy
        delta_output[i] = output_outputs[i] - expected_outputs[i];
    }

// Hidden layer delta
#pragma omp parallel for
    for (int i = 0; i < NUM_HIDDEN; i++)
    {
        double error = 0.0;
        for (int j = 0; j < NUM_OUTPUTS; j++)
        {
            error += delta_output[j] * nn->output_layer.weights[i][j];
        }

        // check the activation
        double activation_derivative = 0.0;
        if (nn->hidden_layer.activation == SIGMOID)
        {
            activation_derivative = sigmoid_derivative(hidden_outputs[i]);
        }
        else
        {
            activation_derivative = relu_derivative(hidden_outputs[i]);
        }
        delta_hidden[i] = error * activation_derivative;
    }

    // Update weights and biases
    update_weights_biases(&nn->output_layer, hidden_outputs, delta_output);
    update_weights_biases(&nn->hidden_layer, inputs, delta_hidden);
}

// Update weights and biases for a layer
void update_weights_biases(LinearLayer *layer, double inputs[], double deltas[])
{
// Update weights
#pragma omp parallel for collapse(2)
    for (int i = 0; i < layer->input_size; i++)
    {
        for (int j = 0; j < layer->output_size; j++)
        {
            layer->weights[i][j] -= LEARNING_RATE * deltas[j] * inputs[i];
        }
    }

// Update biases
#pragma omp parallel for
    for (int i = 0; i < layer->output_size; i++)
    {
        layer->biases[i] -= LEARNING_RATE * deltas[i];
    }
}

// Cross-Entropy Loss Function
double cross_entropy_loss(double predicted[], double expected[], int num_outputs)
{
    double loss = 0.0;
    for (int i = 0; i < num_outputs; i++)
    {
        // Add a small epsilon to prevent log(0)
        loss -= expected[i] * log(predicted[i] + 1e-9);
    }

    return loss;
}

// Training function
void train(NeuralNetwork *nn, double **inputs, int *labels, int num_samples)
{
    // Create log filename with thread count
    int num_threads = omp_get_max_threads();
    char log_filename[256];
    snprintf(log_filename, sizeof(log_filename), "./logs/training_loss_openmp_cpu_%dthreads.txt", num_threads);

    // Open file to log training loss
    FILE *loss_file = fopen(log_filename, "w");
    if (!loss_file)
    {
        printf("Could not open file for writing training loss.\n");
        exit(1);
    }
    printf("Logging to: %s\n", log_filename);

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;
        double start_time = omp_get_wtime();

        // Shuffle the dataset
        for (int i = 0; i < num_samples; i++)
        {
            int j = rand() % num_samples;
            // Swap images
            double *temp_image = inputs[i];
            inputs[i] = inputs[j];
            inputs[j] = temp_image;

            // Swap labels
            int temp_label = labels[i];
            labels[i] = labels[j];
            labels[j] = temp_label;
        }

        // Mini-batch training
        for (int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE)
        {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > num_samples)
                batch_end = num_samples;

            for (int idx = batch_start; idx < batch_end; idx++)
            {
                double hidden_outputs[NUM_HIDDEN];
                double output_outputs[NUM_OUTPUTS];
                double expected_output[NUM_OUTPUTS];

                // One-hot encode the label
                one_hot_encode(labels[idx], expected_output, NUM_OUTPUTS);

                // Forward propagation
                forward(nn, inputs, idx, hidden_outputs, output_outputs);

                // Cross Entropy Loss
                double loss = cross_entropy_loss(output_outputs, expected_output, NUM_OUTPUTS);
                total_loss += loss;

                // Backpropagation
                double delta_hidden[NUM_HIDDEN];
                double delta_output[NUM_OUTPUTS];
                backward(nn, inputs[idx], hidden_outputs, output_outputs, expected_output, delta_hidden, delta_output);
            }
        }

        double end_time = omp_get_wtime();
        double duration = end_time - start_time;
        double average_loss = total_loss / num_samples;
        printf("Epoch %d, Loss: %f Time: %f\n", epoch + 1, average_loss, duration);
        fprintf(loss_file, "%d,%f,%f\n", epoch + 1, average_loss, duration); // Log the metrics
    }

    fclose(loss_file); // Close the loss file
}

// Testing function
void test(NeuralNetwork *nn, double **inputs, int *labels, int num_samples)
{
    int correct_predictions = 0;

    for (int idx = 0; idx < num_samples; idx++)
    {
        double hidden_outputs[NUM_HIDDEN];
        double output_outputs[NUM_OUTPUTS];

        // Forward propagation
        linear_layer_forward(&nn->hidden_layer, inputs[idx], hidden_outputs);
        linear_layer_forward(&nn->output_layer, hidden_outputs, output_outputs);

        // Get the predicted label
        int predicted_label = 0;
        double max_prob = output_outputs[0];
        for (int i = 1; i < NUM_OUTPUTS; i++)
        {
            if (output_outputs[i] > max_prob)
            {
                max_prob = output_outputs[i];
                predicted_label = i;
            }
        }

        if (predicted_label == labels[idx])
        {
            correct_predictions++;
        }
    }

    double accuracy = (double)correct_predictions / num_samples * 100.0;
    printf("Test Accuracy: %.2f%%\n", accuracy);
}

// Read MNIST images
void read_mnist_images(const char *filename, double **images, int num_images)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Could not open file %s\n", filename);
        exit(1);
    }
    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = reverse_int(magic_number);

    fread(&number_of_images, sizeof(int), 1, fp);
    number_of_images = reverse_int(number_of_images);

    fread(&rows, sizeof(int), 1, fp);
    rows = reverse_int(rows);

    fread(&cols, sizeof(int), 1, fp);
    cols = reverse_int(cols);

    for (int i = 0; i < num_images; ++i)
    {
        images[i] = (double *)malloc(rows * cols * sizeof(double));
        for (int r = 0; r < rows * cols; ++r)
        {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, fp);
            images[i][r] = pixel / 255.0; // Normalize pixel values
        }
    }
    fclose(fp);
}

// Read MNIST labels
void read_mnist_labels(const char *filename, int *labels, int num_labels)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Could not open file %s\n", filename);
        exit(1);
    }
    int magic_number = 0;
    int number_of_labels = 0;

    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = reverse_int(magic_number);

    fread(&number_of_labels, sizeof(int), 1, fp);
    number_of_labels = reverse_int(number_of_labels);

    for (int i = 0; i < num_labels; ++i)
    {
        unsigned char label = 0;
        fread(&label, sizeof(unsigned char), 1, fp);
        labels[i] = (int)label;
    }
    fclose(fp);
}

// Reverse integer byte order
int reverse_int(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// One-hot encode labels
void one_hot_encode(int label, double *vector, int size)
{
    for (int i = 0; i < size; i++)
    {
        vector[i] = 0.0;
    }
    vector[label] = 1.0;
}