#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "../include/mnist_file_ompc.h"
#include "../include/neural_network_ompc.h"

#define STEPS 100

void retrieve_data_from_device(int device, mnist_dataset_t *dataset) {
    #pragma omp target exit data map(release: dataset[0:1]) \
                                    depend(in: dataset) \
                                    device(device) nowait
    #pragma omp target exit data depend(in: dataset->images, dataset->labels) \
                                    map(release: dataset->images[0:dataset->size * MNIST_IMAGE_SIZE], \
                                            dataset->labels[0:dataset->size]) \
                                    device(device) nowait
    printf("Data sent to device %d\n", device);

}
/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t *dataset, neural_network_t *network) {
    float activations[MNIST_LABELS], max_activation, b[MNIST_LABELS], W[MNIST_LABELS][MNIST_IMAGE_SIZE];
    int i, j, correct, predict;
    for (i = 0; i < MNIST_LABELS; i++) {
        b[i] = network->b[i];
        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            W[i][j] = network->W[i][j];
        }
    }
    for (i = 0, correct = 0; i < dataset->size; i++) {
        neural_network_hypothesis(dataset->images + i * MNIST_IMAGE_SIZE, b, W, activations);

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
    return ((float)correct) / ((float)dataset->size);
}

int main(int argc, char *argv[]) {
    mnist_dataset_t *train_dataset, *test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    float loss, accuracy;
    int i, batches, nworkers;
    double start_time, end_time, iteration_time, total_time = 0;

    
    // Read the datasets from the files
    train_dataset = mnist_get_dataset(TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE, MNIST_DATASET_SIZE);
    test_dataset = mnist_get_dataset(TEST_IMAGES_FILE, TEST_LABELS_FILE, 0);
    
    // Initialize weights and biases with random values
    neural_network_random_weights(&network);

    // Get the number of devices
    nworkers = omp_get_num_devices();
    printf("Number of devices: %d\n", nworkers);

    int nimages = train_dataset->size;
    int nchunks = nimages / nworkers;
    // Send data to all devices

    for (i = 0; i < nworkers; i++) {
        #pragma omp target enter data map(to: train_dataset->images[i*nchunks*MNIST_IMAGE_SIZE:nchunks*MNIST_IMAGE_SIZE], \
                                            train_dataset->labels[i*nchunks:nchunks]) \
                                            depend(out: train_dataset->images[i*nchunks*MNIST_IMAGE_SIZE:nchunks*MNIST_IMAGE_SIZE],\
                                            train_dataset->labels[i*nchunks:nchunks]) \
                                            device(i) nowait
        printf("Sending %d images to device %d\n", nchunks, i);
    }

    start_time = omp_get_wtime(); // Start timer for the whole training process
    
    printf("Step\tIteration Time (s)\tAverage Loss\n");

    for (i = 0; i < STEPS; i++) {
        start_time = omp_get_wtime(); // Start timer for this iteration

        // Run one step of gradient descent and calculate the loss
        loss = neural_network_training_step(train_dataset, &network, 0.5);

        end_time = omp_get_wtime(); // End timer for this iteration
        iteration_time = end_time - start_time; // Time for this iteration
        total_time += iteration_time; // Accumulate total time

        accuracy = calculate_accuracy(test_dataset, &network);

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

    // Retrieve data from all devices
    for (i = 0; i < nworkers; i++) {
        retrieve_data_from_device(i, train_dataset);
    }

    printf("Cleaning...\n");
    // Cleanup
    printf("Done.\n");
    return 0;
}
