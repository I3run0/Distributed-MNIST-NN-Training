#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

#include "../include/mnist_file_ompc.h"
#include "../include/neural_network_ompc.h"

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

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
    float sum, max;

    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
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
void neural_network_hypothesis(uint8_t * image, float * b, float W[MNIST_LABELS][MNIST_IMAGE_SIZE], float activations[MNIST_LABELS])
{
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = b[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            activations[i] += W[i][j] * PIXEL_SCALE(image[j]);
        }
    }

    neural_network_softmax(activations, MNIST_LABELS);
}

/**
 * Update the gradients for this step of gradient descent using the gradient
 * contributions from a single training example (image).
 * 
 * This function returns the loss ontribution from this training example.
 */
float neural_network_gradient_update(uint8_t * image, float * b, float W[MNIST_LABELS][MNIST_IMAGE_SIZE], float * b_grad_l, float * W_grad_l, uint8_t label, int worker)
{
    float activations[MNIST_LABELS];
    float b_grad, W_grad;
    int i, j;

    // First forward propagate through the network to calculate activations
    neural_network_hypothesis(image, b, W, activations);

    for (i = 0; i < MNIST_LABELS; i++) {
        // This is the gradient for a softmax bias input
        b_grad = (i == label) ? activations[i] - 1 : activations[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            // The gradient for the neuron weight is the bias multiplied by the input weight
            W_grad = b_grad * PIXEL_SCALE(image[j]);

            // Update the weight gradient
            W_grad_l[i*MNIST_IMAGE_SIZE+j] += W_grad;
        }

        // Update the bias gradient
        b_grad_l[i] += b_grad;
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
    float *total_loss_array, total_loss, b[MNIST_LABELS], W[MNIST_LABELS][MNIST_IMAGE_SIZE];
    int nworkers = omp_get_num_devices();
    int nimages = dataset->size;
    int nchunks = nimages / nworkers;
    int i, j, h;
    float * b_grad = (float*)calloc(MNIST_LABELS, sizeof(float));
    float * W_grad = (float*)calloc(MNIST_LABELS*MNIST_IMAGE_SIZE, sizeof(float));

    // Zero initialise gradient for weights and bias vector
    memset(&gradient, 0, sizeof(neural_network_gradient_t));

    for (i = 0; i < MNIST_LABELS; i++){
        b[i] = network->b[i];
        for (j = 0; j < MNIST_IMAGE_SIZE; j++)
            W[i][j] = network->W[i][j];
    }

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0, total_loss=0; i < nworkers; i++) {
        #pragma omp target \
            depend(in: dataset->labels[i*nchunks:nchunks], dataset->images[i*nchunks*MNIST_IMAGE_SIZE:nchunks*MNIST_IMAGE_SIZE]) \
            map(to: b[0:MNIST_LABELS], W[0:MNIST_LABELS][0:MNIST_IMAGE_SIZE]) \
            map(tofrom: b_grad[0:MNIST_LABELS], W_grad[0:MNIST_LABELS*MNIST_IMAGE_SIZE]) \
            map(tofrom: total_loss) \
            device(i) nowait
        {
            #pragma omp teams distribute parallel for
            for (j = i*nchunks; j < min((i+1)*nchunks, nimages); j++) {
                total_loss += neural_network_gradient_update(dataset->images + j * MNIST_IMAGE_SIZE, b, W, b_grad, W_grad, dataset->labels[j], i);
            }
        }
    }

    #pragma omp taskwait

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] -= learning_rate * b_grad[i] / ((float) nimages);
        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] -= learning_rate * W_grad[i*MNIST_IMAGE_SIZE+j] / ((float) nimages);
        }
    }

    free(W_grad);
    free(b_grad);
    return total_loss;
}