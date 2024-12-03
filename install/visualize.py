import os
import imageio

# Function to create a video from .png files in the current directory
def create_video_from_pngs(input_folder, output_video_path, fps=30):
    # Get a list of .png files in the current directory
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
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
input_folder = '/users/xch/multimodal/3dgs-avatar/exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default/test-video/renders'
output_video_path = 'output.mp4'
fps = 5  # Adjust the frame rate as needed

# Call the function to create the video
create_video_from_pngs(input_folder,output_video_path, fps)
