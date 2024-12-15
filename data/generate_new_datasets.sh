#!/bin/bash

declare -a datasets=("t10k-images-idx3-ubyte" "train-images-idx3-ubyte")

# Scales to upscale the images
declare -a scales=(1 2 4)

# Base size of MNIST images
base_size=28

python_script="upscale_mnist.py"

# Output directory for the upscaled datasets
output_dir="upscaled_datasets"
mkdir -p "$output_dir"

for dataset in "${datasets[@]}"; do
    for scale in "${scales[@]}"; do

        new_size=$((base_size * scale))
        
        output_file="${output_dir}/${dataset}-upscaled-${scale}x.ubyte"
        
        echo "Upscaling ${dataset} to ${new_size}x${new_size}..."
        python3 "$python_script" "$dataset" "$output_file" "$new_size" "$new_size"
        
        # Check if the script succeeded
        if [[ $? -eq 0 ]]; then
            echo "Upscaled dataset saved to ${output_file}."
        else
            echo "Failed to upscale ${dataset} to ${new_size}x${new_size}."
            exit 1
        fi
    done
done

echo "All datasets upscaled successfully!"
