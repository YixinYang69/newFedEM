import os
from PIL import Image

# Directory paths
input_dir = "raw_data/celeba/img_align_celeba/"
output_dir = "raw_data/celeba/img_align_celeba_comp/"

# Target dimensions
target_width = 45
target_height = 55

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each image file in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a JPEG image
    if filename.endswith(".jpg"):
        # Load the image using PIL
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        
        # Resize the image to the target dimensions
        img_resized = img.resize((target_width, target_height))
        
        # Save the resized image to the output directory
        output_path = os.path.join(output_dir, filename)
        img_resized.save(output_path)
        
        print(f"Resized and saved: {filename} -> {output_path}")

print("All images resized successfully.")
