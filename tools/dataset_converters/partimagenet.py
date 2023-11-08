# Define the mappings based on the given relationships
class_mappings = {
    0: 1,  # Quadruped Head
    1: 1,  # Quadruped Body
    2: 1,  # Quadruped Foot
    3: 1,  # Quadruped Tail
    4: 2,  # Biped Head
    5: 2,  # Biped Body
    6: 2,  # Biped Hand
    7: 2,  # Biped Foot
    8: 2,  # Biped Tail
    9: 3,  # Fish Head
    10: 3,  # Fish Body
    11: 3,  # Fish Fin
    12: 3,  # Fish Tail
    13: 4,  # Bird Head
    14: 4,  # Bird Body
    15: 4,  # Bird Wing
    16: 4,  # Bird Foot
    17: 4,  # Bird Tail
    18: 5,  # Snake Head
    19: 5,  # Snake Body
    20: 6,  # Reptile Head
    21: 6,  # Reptile Body
    22: 6,  # Reptile Foot
    23: 6,  # Reptile Tail
    24: 7,  # Car Body
    25: 7,  # Car Tier
    26: 7,  # Car Side Mirror
    27: 8,  # Bicycle Body
    28: 8,  # Bicycle Head
    29: 8,  # Bicycle Seat
    30: 8,  # Bicycle Tier
    31: 9,  # Boat Body
    32: 9,  # Boat Sail
    33: 10,  # Aeroplane Head
    34: 10,  # Aeroplane Body
    35: 10,  # Aeroplane Engine
    36: 10,  # Aeroplane Wing
    37: 10,  # Aeroplane Tail
    38: 11, # Bottle Mouth
    39: 11,  # Bottle Body
    40: 0, #background
}

# This function applies the class mapping to a given annotation
import os
from PIL import Image
import numpy as np

# Define the directory paths
input_directory = '/data1/yunfei/SpformerV1/data/PartImageNet/annotations/val'
output_directory = '/data1/yunfei/SpformerV1/data/PartImageNet/annotations_object/val'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Function to process and map image tensor based on annotations
def process_and_map_image_tensor(image_tensor, mappings):
    # Create an output tensor of the same shape as image_tensor
    mapped_tensor = np.zeros_like(image_tensor)
    if np.unique(image_tensor).max() > 40 or np.unique(image_tensor).min() <0:
        raise(ValueError)
    # Assume image_tensor contains class labels as pixel values
    for original_label, new_label in mappings.items():
        # Map the pixel values based on the class_mappings
        mapped_tensor[image_tensor == original_label] = new_label
    return mapped_tensor

# Iterate over each file in the input directory
for filename in os.listdir(input_directory):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):  # Check for image files
        # Load the image as a tensor (here we're using NumPy for example)
        image_path = os.path.join(input_directory, filename)
        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        image_tensor = np.array(image)

        # Process and map the image tensor
        new_image_tensor = process_and_map_image_tensor(image_tensor, class_mappings)

        # Convert the tensor back to an image
        new_image = Image.fromarray(new_image_tensor.astype(np.uint8))

        # Save the new image to the output directory
        new_image_path = os.path.join(output_directory, filename)
        new_image.save(new_image_path)

print("Processing complete.")

