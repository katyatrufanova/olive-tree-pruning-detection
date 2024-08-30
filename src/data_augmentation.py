import os
import cv2
import numpy as np
import glob
from lxml import etree
import albumentations as A
import math
import copy
import random

# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use environment variables for user-specific paths
DATASET_PATH = os.environ.get('DATASET_PATH', os.path.join(SCRIPT_DIR, 'dataset'))
OUTPUT_DIRECTORY = os.environ.get('OUTPUT_DIRECTORY', os.path.join(SCRIPT_DIR, 'augmented_dataset'))
LABEL_PRUNED = int(os.environ.get('LABEL_PRUNED', 1))

def count_images_with_trees(annotation_files, label_pruned):
    num_images_pruned = 0
    num_images_non_pruned = 0

    for annotation_file in annotation_files:
        tree = etree.parse(annotation_file)
        root = tree.getroot()

        pruned_tree_present = any(
            int(object_elem.find("name").text) == label_pruned
            for object_elem in root.findall("object")
        )

        if pruned_tree_present:
            num_images_pruned += 1
        else:
            num_images_non_pruned += 1

    return num_images_pruned, num_images_non_pruned

def update_annotation(root, bboxes):
    annotation_root = copy.deepcopy(root)

    for object_elem, bbox in zip(annotation_root.findall("object"), bboxes):
        xmin, ymin, xmax, ymax = bbox
        object_elem.find("bndbox/xmin").text = str(xmin)
        object_elem.find("bndbox/ymin").text = str(ymin)
        object_elem.find("bndbox/xmax").text = str(xmax)
        object_elem.find("bndbox/ymax").text = str(ymax)

    return annotation_root

def perform_augmentation(image, bboxes, augmentation_transform):
    bboxes = [
        (
            float(bbox.find("bndbox/xmin").text),
            float(bbox.find("bndbox/ymin").text),
            float(bbox.find("bndbox/xmax").text),
            float(bbox.find("bndbox/ymax").text),
        )
        for bbox in bboxes.findall("object")
    ]

    augmented = augmentation_transform(image=image, bboxes=bboxes, category_ids=[0] * len(bboxes))
    augmented_image = augmented["image"]
    augmented_bboxes = augmented["bboxes"]

    return augmented_image, augmented_bboxes

def count_trees_per_image(annotation_files, label_pruned):
    num_trees_per_image = []

    for annotation_file in annotation_files:
        tree = etree.parse(annotation_file)
        root = tree.getroot()

        num_pruned_trees = sum(
            int(object_elem.find("name").text) == label_pruned
            for object_elem in root.findall("object")
        )

        num_trees_per_image.append(num_pruned_trees)

    return num_trees_per_image

def augment_dataset(dataset_path, output_directory, label_pruned):
    os.makedirs(output_directory, exist_ok=True)

    image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
    annotation_files = [os.path.join(dataset_path, f"{os.path.splitext(os.path.basename(image_file))[0]}.xml")
                        for image_file in image_files]

    num_images_pruned, num_images_non_pruned = count_images_with_trees(annotation_files, label_pruned)

    num_trees_per_image = count_trees_per_image(annotation_files, label_pruned)
    avg_num_pruned_trees = np.mean(num_trees_per_image)

    desired_rotations = min(10, math.ceil(avg_num_pruned_trees))

    for image_file, annotation_file in zip(image_files, annotation_files):
        image = cv2.imread(image_file)
        image_filename = os.path.basename(image_file)
        image_filename_without_extension = os.path.splitext(image_filename)[0]

        tree = etree.parse(annotation_file)
        root = tree.getroot()

        non_pruned_tree_present = any(
            int(object_elem.find("name").text) != label_pruned
            for object_elem in root.findall("object")
        )

        if non_pruned_tree_present:
            rotation_angles = random.sample(range(1, 361), desired_rotations)

            for rotation_number, rotation_angle in enumerate(rotation_angles):
                augmentation_transform = A.Compose([
                    A.Rotate(limit=rotation_angle, p=1.0, border_mode=cv2.BORDER_CONSTANT),
                ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

                augmented_image, augmented_bbox = perform_augmentation(image, root, augmentation_transform)

                rotated_image_filename = f"{image_filename_without_extension}_rotation{rotation_number + 1}.jpg"
                rotated_annotation_filename = f"{image_filename_without_extension}_rotation{rotation_number + 1}.xml"

                rotated_image_path = os.path.join(output_directory, rotated_image_filename)
                cv2.imwrite(rotated_image_path, augmented_image)

                rotated_annotation_root = update_annotation(root, augmented_bbox)

                rotated_annotation_path = os.path.join(output_directory, rotated_annotation_filename)
                rotated_annotation_tree = etree.ElementTree(rotated_annotation_root)
                rotated_annotation_tree.write(rotated_annotation_path, pretty_print=True, encoding="utf-8")

        # Copy the original image and annotation to the augmented dataset folder
        image_path = os.path.join(output_directory, image_filename)
        cv2.imwrite(image_path, image)

        annotation_path = os.path.join(output_directory, f"{image_filename_without_extension}.xml")
        tree.write(annotation_path, pretty_print=True, encoding="utf-8")
    
if __name__ == "__main__":
    augment_dataset(DATASET_PATH, OUTPUT_DIRECTORY, LABEL_PRUNED)

    num_images_pruned_balanced, num_images_non_pruned_balanced = count_images_with_trees(
        glob.glob(os.path.join(OUTPUT_DIRECTORY, "*.xml")), LABEL_PRUNED
    )

    print("Number of images with at least one pruned tree:", num_images_pruned_balanced)
    print("Number of images with at least one non-pruned tree:", num_images_non_pruned_balanced)