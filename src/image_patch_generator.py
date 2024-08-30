import os
import cv2
import numpy as np
from lxml import etree

# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use environment variables for user-specific paths, with defaults relative to the script directory
INPUT_IMAGE_PATH = os.environ.get('INPUT_IMAGE_PATH', os.path.join(SCRIPT_DIR, 'Coda_della_Volpe_SE.tif'))
OUTPUT_DIRECTORY = os.environ.get('OUTPUT_DIRECTORY', os.path.join(SCRIPT_DIR, 'dataset'))
CVAT_ANNOTATIONS_PATH = os.environ.get('CVAT_ANNOTATIONS_PATH', os.path.join(SCRIPT_DIR, 'Annotations', 'second.xml'))

# Extract the original image name without the file extension
original_image_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Load the input image
image = cv2.imread(INPUT_IMAGE_PATH)

# Load the CVAT annotations
tree = etree.parse(CVAT_ANNOTATIONS_PATH)
root = tree.getroot()

# Get the image dimensions
image_height, image_width, _ = image.shape

# Set the square size
square_size = 768

# Calculate the number of rows and columns
num_rows = image_height // square_size
num_cols = image_width // square_size

# Iterate over the image in square patches
for row in range(num_rows):
    for col in range(num_cols):
        # Calculate the patch coordinates
        x_start = col * square_size
        y_start = row * square_size
        x_end = x_start + square_size
        y_end = y_start + square_size

        # Extract the square patch from the image
        patch = image[y_start:y_end, x_start:x_end]

        # Check if the patch is empty (blank) or not of the desired size
        if np.count_nonzero(patch) == 0 or patch.shape[:2] != (square_size, square_size):
            continue  # Skip empty or incorrect-sized patches

        # Save the patch image
        patch_filename = f"patch_{original_image_name}_{row}_{col}.jpg"
        patch_path = os.path.join(OUTPUT_DIRECTORY, patch_filename)
        cv2.imwrite(patch_path, patch)

        # Create the annotation XML content for the patch
        annotation_root = etree.Element("annotation")
        folder_element = etree.SubElement(annotation_root, "folder")
        folder_element.text = "dataset"
        filename_element = etree.SubElement(annotation_root, "filename")
        filename_element.text = patch_filename

        # Iterate over the CVAT annotations
        for object_elem in root.findall("object"):
            name = object_elem.find("name").text

            # Get the bounding box coordinates
            bbox = object_elem.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Check if the bounding box is within the patch
            if x_start <= xmin <= x_end and y_start <= ymin <= y_end:
                # Calculate the relative coordinates within the patch
                rel_xmin = max(xmin - x_start, 0)
                rel_ymin = max(ymin - y_start, 0)
                rel_xmax = min(xmax - x_start, square_size)
                rel_ymax = min(ymax - y_start, square_size)

                # Create the object element in the annotation XML
                object_element = etree.SubElement(annotation_root, "object")
                name_element = etree.SubElement(object_element, "name")
                name_element.text = name
                bndbox_element = etree.SubElement(object_element, "bndbox")
                xmin_element = etree.SubElement(bndbox_element, "xmin")
                xmin_element.text = str(rel_xmin)
                ymin_element = etree.SubElement(bndbox_element, "ymin")
                ymin_element.text = str(rel_ymin)
                xmax_element = etree.SubElement(bndbox_element, "xmax")
                xmax_element.text = str(rel_xmax)
                ymax_element = etree.SubElement(bndbox_element, "ymax")
                ymax_element.text = str(rel_ymax)

        # Save the annotation XML file for the patch
        annotation_filename = f"patch_{original_image_name}_{row}_{col}.xml"
        annotation_path = os.path.join(OUTPUT_DIRECTORY, annotation_filename)
        with open(annotation_path, "wb") as file:
            file.write(etree.tostring(annotation_root, pretty_print=True))

print("Processing complete. Check the output directory for results.")