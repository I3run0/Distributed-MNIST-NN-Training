#ifndef MNIST_FILE_H_
#define MNIST_FILE_H_

#include <stdint.h>

#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_IMAGE_MAGIC 0x00000803

#ifndef MNIST_IMAGE_WIDTH
#define MNIST_IMAGE_WIDTH 28
#endif

#ifndef MNIST_IMAGE_HEIGHT
#define MNIST_IMAGE_HEIGHT 28
#endif

#ifndef MNIST_IMAGE_SIZE
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT
#endif

#ifndef MNIST_LABELS
#define MNIST_LABELS 10
#endif

#ifndef MNIST_DATASET_SIZE
#define MNIST_DATASET_SIZE 60000
#endif

/**
 * Downloaded from: http://yann.lecun.com/exdb/mnist/
 */
#ifndef TRAIN_IMAGES_FILE
#define TRAIN_IMAGES_FILE "../data/train-images-idx3-ubyte"
#endif

#ifndef  TRAIN_LABELS_FILE
#define TRAIN_LABELS_FILE "../data/train-labels-idx1-ubyte"
#endif

#ifndef TEST_IMAGES_FILE
#define TEST_IMAGES_FILE "../data/t10k-images-idx3-ubyte"
#endif

#ifndef TEST_LABELS_FILE
#define TEST_LABELS_FILE "../data/t10k-labels-idx1-ubyte"
#endif

typedef struct mnist_label_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_labels;
} __attribute__((packed)) mnist_label_file_header_t;

typedef struct mnist_image_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t number_of_rows;
    uint32_t number_of_columns;
} __attribute__((packed)) mnist_image_file_header_t;

typedef struct mnist_image_t_ {
    uint8_t pixels[MNIST_IMAGE_SIZE];
} __attribute__((packed)) mnist_image_t;

typedef struct mnist_dataset_t_ {
    mnist_image_t * images;
    uint8_t * labels;
    uint32_t size;
} mnist_dataset_t;

mnist_dataset_t * mnist_get_dataset(const char * image_path, const char * label_path);
void mnist_free_dataset(mnist_dataset_t * dataset);
int mnist_batch(mnist_dataset_t * dataset, mnist_dataset_t * batch, int batch_size, int batch_number);

#endif
