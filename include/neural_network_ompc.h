#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include "mnist_file.h"

typedef struct neural_network_t_ {
    float b[MNIST_LABELS];
    float W[MNIST_LABELS][MNIST_IMAGE_SIZE];
} neural_network_t;

typedef struct neural_network_gradient_t_ {
    float b_grad[MNIST_LABELS];
    float W_grad[MNIST_LABELS][MNIST_IMAGE_SIZE];
} neural_network_gradient_t;

void neural_network_random_weights(neural_network_t * network);
void neural_network_hypothesis(uint8_t * image, float * b, float W[MNIST_LABELS][MNIST_IMAGE_SIZE], float activations[MNIST_LABELS]);
float neural_network_gradient_update(uint8_t * image, float * b, float W[MNIST_LABELS][MNIST_IMAGE_SIZE], float * b_grad_l, float * W_grad_l, uint8_t label, int worker);
float neural_network_training_step(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate);

#endif