#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>  // Include OpenMP header
#include <stdio.h>

#include "../include/mnist_file.h"
#include "../include/neural_network.h"

// Convert a pixel value from 0-255 to one from 0 to 1
#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

// Returns a random value between 0 and 1
#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))

/**
 * Initialise the weights and bias vectors with values between 0 and 1
 */
void neural_network_random_weights(neural_network_t * network)
{
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] = RAND_FLOAT();

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] = RAND_FLOAT();
        }
    }
}

/**
 * Calculate the softmax vector from the activations. This uses a more
 * numerically stable algorithm that normalises the activations to prevent
 * large exponents.
 */
void neural_network_softmax(float * activations, int length)
{
    int i;
    float sum = 0, max = activations[0];

    for (i = 1; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0; i < length; i++) {
        activations[i] = exp(activations[i] - max);
        sum += activations[i];
    }

   for (i = 0; i < length; i++) {
        activations[i] /= sum;
    }
}

/**
 * Use the weights and bias vector to forward propogate through the neural
 * network and calculate the activations.
 */
void neural_network_hypothesis(mnist_image_t * image, neural_network_t * network, float activations[MNIST_LABELS])
{
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = network->b[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            activations[i] += network->W[i][j] * PIXEL_SCALE(image->pixels[j]);
        }
    }

    neural_network_softmax(activations, MNIST_LABELS);
}


/**
 * Update the gradients for this step of gradient descent using the gradient
 * contributions from a single training example (image).
 */
float neural_network_gradient_update(mnist_image_t * image, neural_network_t * network, neural_network_gradient_t * gradient, uint8_t label)
{
    float activations[MNIST_LABELS];
    float b_grad, W_grad;
    int i, j;

    // First forward propagate through the network to calculate activations
    neural_network_hypothesis(image, network, activations);

    for (i = 0; i < MNIST_LABELS; i++) {
        // This is the gradient for a softmax bias input
        b_grad = (i == label) ? activations[i] - 1 : activations[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            // The gradient for the neuron weight is the bias multiplied by the input weight
            W_grad = b_grad * PIXEL_SCALE(image->pixels[j]);

            // Update the weight gradient
            gradient->W_grad[i][j] += W_grad;
        }

        // Update the bias gradient
        gradient->b_grad[i] += b_grad;
    }

    // Cross entropy loss
    return 0.0f - log(activations[label]);
}

/**
 * Run one step of gradient descent and update the neural network.
 */
float neural_network_training_step(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate)
{
    neural_network_gradient_t gradient;
    float total_loss = 0.0f;
    int i, j;

    // Zero initialise gradient for weights and bias vector
    memset(&gradient, 0, sizeof(neural_network_gradient_t));

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0; i < dataset->size; i++) {
        total_loss += neural_network_gradient_update(&dataset->images[i], network, &gradient, dataset->labels[i]);
    }

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] -= learning_rate * gradient.b_grad[i] / ((float) dataset->size);

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] -= learning_rate * gradient.W_grad[i][j] / ((float) dataset->size);
        }
    }

    return total_loss;
}

// Parallel version of the training step
float neural_network_training_step_parallel(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate)
{

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the number of processes

    int local_size = dataset->size / size;  // Divide dataset evenly among processes
    int start_idx = rank * local_size;
    int end_idx = (rank == size - 1) ? dataset->size : start_idx + local_size;

    neural_network_gradient_t local_gradient;
    neural_network_gradient_t global_gradient;
    float local_loss = 0.0f;
    float global_loss = 0.0f;

    // Zero initialize gradients for weights and bias
    memset(&local_gradient, 0, sizeof(neural_network_gradient_t));
    memset(&global_gradient, 0, sizeof(neural_network_gradient_t));

    // Each process calculates gradients and loss on its subset of data
    #pragma omp parallel for reduction(+:local_loss)
    for (int i = start_idx; i < end_idx; i++) {
        float temp_loss = neural_network_gradient_update(
            &dataset->images[i], network, &local_gradient, dataset->labels[i]
        );

        #pragma omp critical
        local_loss += temp_loss;
    }


    // Reduce the gradients and loss across all processes
    MPI_Reduce(&local_loss, &global_loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_gradient.b_grad, global_gradient.b_grad, MNIST_LABELS, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    for (int i = 0; i < MNIST_LABELS; i++) {
        MPI_Reduce(local_gradient.W_grad[i], global_gradient.W_grad[i], MNIST_IMAGE_SIZE, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // The root process updates the network weights and biases
    if (rank == 0) {
        for (int i = 0; i < MNIST_LABELS; i++) {
            network->b[i] -= learning_rate * global_gradient.b_grad[i] / ((float) dataset->size);
            for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
                network->W[i][j] -= learning_rate * global_gradient.W_grad[i][j] / ((float) dataset->size);
            }
        }
    }

    // Broadcast the updated network to all processes
    MPI_Bcast(network->b, MNIST_LABELS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < MNIST_LABELS; i++) {
        MPI_Bcast(network->W[i], MNIST_IMAGE_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    return global_loss;
}