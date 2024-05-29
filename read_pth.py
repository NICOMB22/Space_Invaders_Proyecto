import os
import torch

# Get the current directory
current_dir = os.path.dirname(__file__)

# Construct the path to the .pth file
file_path = os.path.join(current_dir, 'models', 'dqn_spaceinvaders_1080000_steps.zip', 'pytorch_variables.pth')

# Load the model
model = torch.load(file_path)

# Now you can use the model for inference or further training
