#!/bin/bash
#SBATCH --job-name=mnist
#SBATCH --output=output/err.txt
#SBATCH --time=05:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --partition=cpu

module purge
module load gcc

cd $SLURM_SUBMIT_DIR

echo $(pwd)
# make clean
# make all

test_mnist_binary() {
    local binary_path=$1
    local bins=$(ls $1)
    local output=$2

    mkdir -p temp

    echo "Binary,Final Accuracy,Total Duration,Mean Iteration Time" > $output/stats.csv

    # Loop over each binary in the specified directory
    for binary in $bins; do
        echo "./$binary_path/$binary"

        $binary_path/$binary > temp/temp.log

        # Extract the last three metrics from the output log
        final_accuracy=$(grep "Final Accuracy" temp/temp.log | awk '{print $3}')
        total_duration=$(grep "Total Duration" temp/temp.log | awk '{print $3}')
        mean_iteration_time=$(grep "Mean Iteration Time" temp/temp.log | awk '{print $4}')

        # Append the statistics to the CSV file for the current binary
        echo "$binary,$final_accuracy,$total_duration,$mean_iteration_time" >> $output/stats.csv
    done

    rm -rf temp
}

# Call the function with the appropriate parameters
test_mnist_binary "bin" "output"