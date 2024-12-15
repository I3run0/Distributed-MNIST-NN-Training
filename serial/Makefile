CC = gcc
CFLAGS = -lm -fopenmp
SOURCE_FILES = mnist.c mnist_file.c neural_network.c
OUTPUT_DIR = bin

# Default target
all: $(OUTPUT_DIR) mnist-1x mnist-2x mnist-4x

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)
# mnist-1x target
mnist-1x: $(SOURCE_FILES)
	$(CC) $(SOURCE_FILES) $(CFLAGS) -DMNIST_IMAGE_WIDTH=28 -DMNIST_IMAGE_HEIGHT=28 -DTRAIN_IMAGES_FILE=\"../data/upscaled_datasets/train-images-idx3-ubyte-upscaled-1x.ubyte\" -DTEST_IMAGES_FILE=\"../data/upscaled_datasets/t10k-images-idx3-ubyte-upscaled-1x.ubyte\" -o $(OUTPUT_DIR)/mnist-1x

# mnist-2x target
mnist-2x: $(SOURCE_FILES)
	$(CC) $(SOURCE_FILES) $(CFLAGS) -DMNIST_IMAGE_WIDTH=56 -DMNIST_IMAGE_HEIGHT=56 -DTRAIN_IMAGES_FILE=\"../data/upscaled_datasets/train-images-idx3-ubyte-upscaled-2x.ubyte\" -DTEST_IMAGES_FILE=\"../data/upscaled_datasets/t10k-images-idx3-ubyte-upscaled-2x.ubyte\" -o $(OUTPUT_DIR)/mnist-2x

# mnist-4x target
mnist-4x: $(SOURCE_FILES)
	$(CC) $(SOURCE_FILES) $(CFLAGS) -DMNIST_IMAGE_WIDTH=112 -DMNIST_IMAGE_HEIGHT=112 -DTRAIN_IMAGES_FILE=\"../data/upscaled_datasets/train-images-idx3-ubyte-upscaled-4x.ubyte\" -DTEST_IMAGES_FILE=\"../data/upscaled_datasets/t10k-images-idx3-ubyte-upscaled-4x.ubyte\" -o $(OUTPUT_DIR)/mnist-4x

# Clean up compiled files
clean:
	rm -f $(OUTPUT_DIR)/mnist-1x $(OUTPUT_DIR)/mnist-2x $(OUTPUT_DIR)/mnist-4x
	rm -rf $(OUTPUT_DIR)