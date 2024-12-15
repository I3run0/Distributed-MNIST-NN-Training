#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h> // Include OpenMP header

#include "../include/mnist_file.h"
#include "../include/neural_network.h"

#define STEPS 100

/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    for (i = 0, correct = 0; i < dataset->size; i++) {
        neural_network_hypothesis(&dataset->images[i], network, activations);

        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    return ((float) correct) / ((float) dataset->size);
}

int main(int argc, char *argv[])
{
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    float loss, accuracy;
    int i, batches;
    double start_time, end_time, iteration_time, total_time = 0.0;

    // Read the datasets from the files
    train_dataset = mnist_get_dataset(TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE);
    test_dataset = mnist_get_dataset(TEST_IMAGES_FILE, TEST_LABELS_FILE);

    neural_network_random_weights(&network);

    printf("Step\tIteration Time (s)\tAverage Loss\n");

    for (i = 0; i < STEPS; i++) {
        start_time = omp_get_wtime();

        loss = neural_network_training_step(train_dataset, &network, 0.5);

        end_time = omp_get_wtime();
        iteration_time = end_time - start_time;
        total_time += iteration_time;

        printf("%04d\t%.6f\t\t%.2f\t\n", i, iteration_time, loss / train_dataset->size);
    }

    start_time = omp_get_wtime();
    accuracy = calculate_accuracy(test_dataset, &network);
    end_time = omp_get_wtime();
    iteration_time = end_time - start_time;
    total_time += iteration_time;
    printf("\nFinal Accuracy: %.6f\n", accuracy);
    printf("Total Duration: %.6f seconds\n", total_time);
    printf("Mean Iteration Time: %.6f seconds\n", total_time / STEPS);

    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

    return 0;
}
