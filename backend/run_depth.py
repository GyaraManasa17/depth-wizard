import cv2
import torch
import numpy as np
from transformers import pipeline

# Load Depth Anything V2 (Small checkpoint - fast, good for testing)
print("Loading model...")
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
print("Model loaded.")

# Load our test image
image_path = "../data/test_hills.png"
from PIL import Image
image = Image.open(image_path)
print(f"Image loaded: {image.size}")

# Run depth estimation
print("Running inference...")
result = pipe(image)
depth = result["depth"]  # this is a PIL image showing relative depth

# Save the output
output_path = "../data/test_depth_output_hills.png"
depth.save(output_path)
print(f"Depth map saved to {output_path}")