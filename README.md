<p align="center">
  <img src="https://github.com/mo652-24s2/projeto-final-parallel-mnist/blob/main/mnist.png">
</p>

# DISTRIBUTED MNIST NEURAL NETWORK TRAINING

This repository was developed as part of the *High-Performance Computing* course (MO652) at the Institute of Computing, UNICAMP. It explores various approaches to parallelizing a simple neural network, originally implemented in the [mnist-neural-network-plain-c](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c) repository.

The neural network is a basic two-layer model designed to solve the classic [MNIST dataset](https://yann.lecun.com/exdb/mnist/). The repository is organized into the following subdirectories:

- **`serial/`**: Contains the original implementation of the neural network without parallelization.
- **`mpi_openmp/`**: Contains a parallelized implementation using MPI and OpenMP.
- **`ompcluster/`**: Contains an implementation using the experimental OmpCluster framework for parallelization.
- **`data/`**: Contains the original MNIST dataset and scripts to generate upscaled versions of the dataset with images resized to 56x56 and 112x112. These larger datasets are used for extended experiments.
- **`include/`**: Contains header files used in the C implementations.

## Experimental Environment

The provided neural network implementations were tested on the **Sorgan Cluster**, hosted by the [Laboratory of Computing Systems (LSC)](https://lsc.ic.unicamp.br/) at UNICAMP. All performance metrics and experimental results were collected in this environment. 

It is important to note that OmpCluster is an experimental framework developed by LSC. It aims to enable cluster programming using OpenMP directives. As such, computing results obtained with OmpCluster may not represent final, optimized production-grade performance.

## Reproducing the Experiments

To reproduce the experiments on the Sorgan cluster, please follow these steps:

1. **Generate Large Datasets**: Refer to the scripts in the `data/` directory to generate the larger versions of the MNIST dataset (56x56 and 112x112).
2. **Understand Implementation-Specific Requirements**: Read the `README.md` files within each implementation folder (`serial/`, `mpi_openmp/`, and `ompcluster/`) for specific instructions on running the implementations and conducting experiments.
3. **Run on a Different Environment**: If you plan to run these implementations on a different cluster or system, additional configuration steps may be required. Be aware of potential adjustments needed for compatibility.

## References

- Original Neural Network Implementation: [mnist-neural-network-plain-c](https://github.com/AndrewCarterUK/mnist-neural-network-plain-c)
- MNIST Dataset: [Yann LeCun's MNIST Page](https://yann.lecun.com/exdb/mnist/)
- Laboratory of Computing Systems (LSC): [https://lsc.ic.unicamp.br/](https://lsc.ic.unicamp.br/)
