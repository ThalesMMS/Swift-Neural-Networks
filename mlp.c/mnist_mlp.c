/*
 * =============================================================================
 * mnist_mlp.c - Sequential CPU Implementation (Baseline)
 * =============================================================================
 *
 * DESCRIÇÃO:
 * Implementação sequencial de uma rede neural MLP para classificação do
 * dataset MNIST. Serve como baseline para comparação com versões paralelas.
 *
 * ARQUITETURA:
 * - Camada de entrada: 784 neurônios (28×28 pixels)
 * - Camada oculta: 512 neurônios (ReLU)
 * - Camada de saída: 10 neurônios (Softmax)
 * - Taxa de aprendizado: 0.01
 * - Batch size: 64 amostras
 *
 * TEMPOS DE EXECUÇÃO (10 ÉPOCAS):
 *
 * AMBIENTE: Intel Core i7-15th Gen, NVIDIA RTX 3050 Ti, Windows
 *
 * Sequential CPU (baseline - esta versão):
 *   Tempo total: 337.11 segundos
 *   Tempo/época: 33.71 segundos
 *   Acurácia final: ~97.4%
 *
 * COMPILAÇÃO:
 *   gcc -O3 -o mnist_mlp mnist_mlp.c -lm
 *
 * EXECUÇÃO:
 *   ./mnist_mlp
 *
 * FONTE ORIGINAL:
 * Extraido de https://github.com/djbyrne/mlp.c
 * Aumentado a camada oculta de 128 para 512 neurônios
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

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
    // Start overall timing
    clock_t program_start = clock();

    // Allocate memory for training data
    double **train_images = (double **)malloc(TRAIN_SAMPLES * sizeof(double *));
    int *train_labels = (int *)malloc(TRAIN_SAMPLES * sizeof(int));

    // Read training data
    printf("Loading training data...\n");
    clock_t load_start = clock();
    read_mnist_images("./data/train-images.idx3-ubyte", train_images, TRAIN_SAMPLES);
    read_mnist_labels("./data/train-labels.idx1-ubyte", train_labels, TRAIN_SAMPLES);

    // Allocate memory for test data
    double **test_images = (double **)malloc(TEST_SAMPLES * sizeof(double *));
    int *test_labels = (int *)malloc(TEST_SAMPLES * sizeof(int));

    // Read test data
    printf("Loading test data...\n");
    read_mnist_images("./data/t10k-images.idx3-ubyte", test_images, TEST_SAMPLES);
    read_mnist_labels("./data/t10k-labels.idx1-ubyte", test_labels, TEST_SAMPLES);
    clock_t load_end = clock();
    double load_time = (double)(load_end - load_start) / CLOCKS_PER_SEC;
    printf("Data loading time: %.2f seconds\n", load_time);

    // Initialize neural network
    printf("Initializing neural network...\n");
    NeuralNetwork nn;
    initialize_network(&nn);

    // Train the neural network
    printf("Training neural network...\n");
    clock_t train_start = clock();
    train(&nn, train_images, train_labels, TRAIN_SAMPLES);
    clock_t train_end = clock();
    double train_time = (double)(train_end - train_start) / CLOCKS_PER_SEC;
    printf("Total training time: %.2f seconds\n", train_time);

    // Test the neural network
    printf("Testing neural network...\n");
    clock_t test_start = clock();
    test(&nn, test_images, test_labels, TEST_SAMPLES);
    clock_t test_end = clock();
    double test_time = (double)(test_end - test_start) / CLOCKS_PER_SEC;
    printf("Testing time: %.2f seconds\n", test_time);

    // Save the trained model
    printf("Saving model...\n");
    save_model(&nn, "mnist_model.bin");

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

    // Print total program execution time
    clock_t program_end = clock();
    double total_time = (double)(program_end - program_start) / CLOCKS_PER_SEC;
    printf("\n=== Performance Summary ===\n");
    printf("Data loading time: %.2f seconds\n", load_time);
    printf("Total training time: %.2f seconds\n", train_time);
    printf("Testing time: %.2f seconds\n", test_time);
    printf("Total program time: %.2f seconds\n", total_time);
    printf("========================\n");

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
    for (int i = 0; i < input_size; i++)
        for (int j = 0; j < output_size; j++)
            layer->weights[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;

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
        for (int i = 0; i < layer->output_size; i++)
        {
            outputs[i] = sigmoid(outputs[i]);
        }
        break;
    case RELU:
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
    for (int i = 0; i < NUM_OUTPUTS; i++)
    {
        // For softmax and cross-entropy
        delta_output[i] = output_outputs[i] - expected_outputs[i];
    }

    // Hidden layer delta
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
    for (int i = 0; i < layer->input_size; i++)
    {
        for (int j = 0; j < layer->output_size; j++)
        {
            layer->weights[i][j] -= LEARNING_RATE * deltas[j] * inputs[i];
        }
    }

    // Update biases
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
    // Open file to log training loss
    FILE *loss_file = fopen("./logs/training_loss_c.txt", "w");
    if (!loss_file)
    {
        printf("Could not open file for writing training loss.\n");
        exit(1);
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;
        float start_time = clock();

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

        double end_time = clock();
        double duration = (end_time - start_time) / CLOCKS_PER_SEC;
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