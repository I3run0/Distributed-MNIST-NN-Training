import argparse
import numpy as np
from PIL import Image

def read_mnist_images(file_path):
    """Read MNIST images from the .ubyte file."""
    with open(file_path, 'rb') as f:
        # Read the header
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

def upscale_images(images, new_size):
    """Upscale MNIST images to the new size."""
    upscaled_images = []
    for image in images:
        img = Image.fromarray(image, mode='L')  # Convert to PIL Image
        upscaled_img = img.resize(new_size, Image.BICUBIC)  # Upscale
        upscaled_images.append(np.array(upscaled_img, dtype=np.uint8))
    return np.array(upscaled_images)

def save_mnist_images(file_path, images, new_size):
    """Save upscaled MNIST images in .ubyte format."""
    num_images = images.shape[0]
    with open(file_path, 'wb') as f:
        # Write the header
        f.write((2051).to_bytes(4, 'big'))  # Magic number for images
        f.write(num_images.to_bytes(4, 'big'))  # Number of images
        f.write(new_size[0].to_bytes(4, 'big'))  # New number of rows
        f.write(new_size[1].to_bytes(4, 'big'))  # New number of columns
        
        # Write the image data
        f.write(images.tobytes())

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Upscale MNIST images.")
    parser.add_argument("input_file", help="Path to the input .ubyte file.")
    parser.add_argument("output_file", help="Path to save the upscaled .ubyte file.")
    parser.add_argument("new_size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="New size (width height) for upscaling.")
    args = parser.parse_args()

    # Read original MNIST images
    images = read_mnist_images(args.input_file)
    print(f"Loaded {images.shape[0]} images of size {images.shape[1]}x{images.shape[2]}.")
    
    # Upscale images
    upscaled_images = upscale_images(images, tuple(args.new_size))
    print(f"Upscaled images to size {args.new_size[0]}x{args.new_size[1]}.")
    
    # Save upscaled images in .ubyte format
    save_mnist_images(args.output_file, upscaled_images, tuple(args.new_size))
    print(f"Upscaled images saved to {args.output_file}.")

if __name__ == "__main__":
    main()
