import os
import imageio
import argparse

# Function to create a video from .png files in the current directory
def create_video_from_pngs(input_folder, output_video_path, fps=30, format='png'):
    # Get a list of .png files in the current directory
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.'+format)]
    # Sort the files to ensure correct order
    png_files.sort()

    # Read the images into a list
    images = []
    for filename in png_files:
        filepath = os.path.join(input_folder, filename)
        images.append(imageio.imread(filepath))

    # Save the images as a video
    imageio.mimsave(output_video_path, images, fps=fps)

# Set the output video path and the frames per second (fps)


# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('-i', '--input_folder', required=True, help='Input image folder path')
parser.add_argument('-o', '--output', default='output.mp4', help='Output file path (default: output.mp4)')
parser.add_argument('-f', '--fps', type=int, default=30, help='Frames per second (default: 30)')
parser.add_argument('-d', '--format', default='png', help='Input file format (default: png)')

# Parse the arguments
args = parser.parse_args()

# Call the function to create the video
create_video_from_pngs(args.input_folder, args.output, args.fps, args.format)
