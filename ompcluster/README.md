# MNIST Neural Network OmpCluster Implementation

This directory contains a customized implementation of the [mnist-neural-network-plain-c](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c) project to enable distributed neural network training. The approach used is **data parallelism**, where each node processes a subset of the original dataset. 

For inter-node parallelization, **OmpCluster (OMPC)** is employed, while for intra-node parallelization, **OpenMP** is utilized. The folder includes the necessary `Makefile` and job script to compile the binaries and run experiments.

**OMPC** is compatible with both CPUs and GPUs as devices; but, the specific version used in this project does not support the use of both at the same time, and they must be compiled using different containers. The recipes for each container are available in the folder `recipes`.

All experiments were conducted in the **Sorgan Cluster**, provided by the [Laboratory of Computing Systems (LSC)](https://lsc.ic.unicamp.br/) at UNICAMP. Follow the instructions below to compile and run the code in this environment.

---

## How to Compile the Code

Ensure you have an the desired container installed, with the compatible MPI implementation installed (e.g., MPICH). To compile the code, run:

> **&#9432; INFO:**  To compile and run using GPU, use `--nv <ompc_gpu_image_path>` instead of only `<ompc_gpu_image_path>`.

```bash
apptainer exec <ompc_image_path> clang -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu mnist.c mnist_file.c neural_network.c -lm -o mnist -g 
```

To test locally, you can execute the binary inside the OMPC container:

Enter the container
```bash
apptainer shell <ompc_image_path>
```


---

## Running Experiments on the Sorgan Cluster

### 1. Install Dependencies

To generate the necessary datasets, install the following Python packages:

```bash
pip install pillow numpy
```

### 2. Generate Large Datasets

Run the dataset generation script to create datasets with increased image sizes (e.g., 56x56 and 112x112):

```bash
cd ../data && ./generate_new_datasets.sh && cd ../mpi_openmp
```

This should generate a directory called `upscaled_datasets` in the `data/` directory. Ensure this step completes successfully before proceeding.

### 3. Enter the Container

Enter the desired ompc container.

```bash
apptainer shell <ompc_image_path>
```

### 4. Compile the Code

Run the `make` command to compile the code:

```bash
make
```

If successful, a `bin/` directory will be created, containing binaries for training the neural network on datasets of various sizes.

### 5. Submit the Job

The Sorgan cluster uses the Slurm workload manager. Exit the container and submit the job by running:

> **&#9432; INFO:**  The GPU job is `job_gpu.sh`.
```bash
sbatch job.sh
```

---

## Output and Results

The results of the experiments will be saved in the `output/` folder, which includes:

- **Log Files**: Detailed logs for each run.
- **CSV File**: A summary of timing statistics for the different binaries, representing neural network training on datasets of varying sizes.