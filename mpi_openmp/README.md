# MNIST Neural Network MPI + OpenMP Implementation

This directory contains a customized implementation of the [mnist-neural-network-plain-c](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c) project to enable distributed neural network training. The approach used is **data parallelism**, where each node processes a subset of the original dataset. 

For inter-node parallelization, **Message Passing Interface (MPI)** is employed, while for intra-node parallelization, **OpenMP** is utilized. The folder includes the necessary `Makefile` and job script to compile the binaries and run experiments.

All experiments were conducted in the **Sorgan Cluster**, provided by the [Laboratory of Computing Systems (LSC)](https://lsc.ic.unicamp.br/) at UNICAMP. Follow the instructions below to compile and run the code in this environment.

---

## How to Compile the Code

Ensure you have an MPI implementation installed (e.g., MPICH). To compile the code, run:

```bash
mpicc mnist.c mnist_file.c neural_network.c -lm -fopenmp -o mnist
```

To test locally, you can execute the binary using MPI with two processes as follows:

```bash
mpirun -np 2 ./mnist
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

### 3. Load the MPICH Module

Load the required MPICH implementation on the Sorgan cluster using:

```bash
module load mpich
```

### 4. Compile the Code

Run the `make` command to compile the code:

```bash
make
```

If successful, a `bin/` directory will be created, containing binaries for training the neural network on datasets of various sizes.

### 5. Submit the Job

The Sorgan cluster uses the Slurm workload manager. Submit the job by running:

```bash
sbatch job.sh
```

---

## Output and Results

The results of the experiments will be saved in the `output/` folder, which includes:

- **Log Files**: Detailed logs for each run.
- **CSV File**: A summary of timing statistics for the different binaries, representing neural network training on datasets of varying sizes.