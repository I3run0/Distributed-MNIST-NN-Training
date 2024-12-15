#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

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

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

int main(int argc, char *argv[])
{
    mnist_dataset_t *train_dataset, *test_dataset;
    neural_network_t network;
    float loss, accuracy;
    int i, rank, size;
    int provided;
    double start, end, total_time = 0.0;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    train_dataset = mnist_get_dataset(TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE);
    test_dataset = mnist_get_dataset(TEST_IMAGES_FILE, TEST_LABELS_FILE);
    neural_network_random_weights(&network);
  

    MPI_Bcast(&network, sizeof(neural_network_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0 )
        printf("Step\tIteration Time (s)\tAverage Loss\n");

    for (i = 0; i < STEPS; i++) {
        if (rank == 0) {
            start = omp_get_wtime();
        }

        loss = neural_network_training_step_parallel(train_dataset, &network, 0.5);
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            end = omp_get_wtime();
            double iteration_time = end - start;
            total_time += iteration_time;

            printf("%04d\t%.6f\t\t%.2f\t\n", i, iteration_time, loss / train_dataset->size);
        }

    }

    if (rank == 0) {
        start = omp_get_wtime();
        accuracy = calculate_accuracy(test_dataset, &network);
        end = omp_get_wtime();
        double iteration_time = end - start;
        total_time += iteration_time;
        printf("\nFinal Accuracy: %.6f\n", accuracy);
        printf("Total Duration: %.6f seconds\n", total_time);
        printf("Mean Iteration Time: %.6f seconds\n", total_time / STEPS);
    }

    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);
    
    MPI_Finalize();

    return 0;
}
