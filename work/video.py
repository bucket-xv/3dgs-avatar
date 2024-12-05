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

# use cv2 to convert the images to a video 
def create_video_from_pngs_with_cv2(input_folder, output_video_path, fps=30, format='png'):
    import cv2 
    # Get a list of .png files in the current directory
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.' + format)]
    png_files.sort()

    # Read the first image to get the dimensions
    first_image_path = os.path.join(input_folder, png_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 output
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame
    for filename in png_files:
        filepath = os.path.join(input_folder, filename)
        frame = cv2.imread(filepath)
        video.write(frame)

    # Release the video writer
    video.release()


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
create_video_from_pngs_with_cv2(args.input_folder, args.output, args.fps, args.format)
