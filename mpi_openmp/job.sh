#!/bin/bash
#SBATCH --job-name=mpi_openmp_mnist.txt
#SBATCH --output=output/err.txt
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=ib

module purge
module load mpich/4.2.1-cuda/12.1.1-ucx

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=48

# make all

test_mnist_binary() {
    local binary_path=$1
    local bins=$(ls $1)
    local nnodes=$2
    local pe=$3
    local output=$4

    mkdir -p temp

    echo "Binary,Nodes,Final Accuracy,Total Duration,Mean Iteration Time" > $output/stats.csv

    # Loop over each binary in the specified directory
    for binary in $bins; do
        for nodes in $(seq 1 $nnodes); do        
            mpirun -np $nodes --map-by ppr:1:socket:PE=$pe ./$binary_path/$binary > temp/temp.log

            # Extract the last three metrics from the output log
            final_accuracy=$(grep "Final Accuracy" temp/temp.log | awk '{print $3}')
            total_duration=$(grep "Total Duration" temp/temp.log | awk '{print $3}')
            mean_iteration_time=$(grep "Mean Iteration Time" temp/temp.log | awk '{print $4}')

            # Append the statistics to the CSV file for the current binary
            echo "$binary,$nodes,$final_accuracy,$total_duration,$mean_iteration_time" >> $output/stats.csv
        done
    done

    rm -rf temp
}

# Call the function with the appropriate parameters
test_mnist_binary "bin" $SLURM_NTASKS $OMP_NUM_THREADS "output"
