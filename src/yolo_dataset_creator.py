import os
import glob
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from shutil import copyfile

def convert_annotation(xml_file, img_size, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(output_file, 'w') as f:
        for obj in root.iter('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            x_center = (xmin + xmax) / 2 / img_size
            y_center = (ymin + ymax) / 2 / img_size
            width = (xmax - xmin) / img_size
            height = (ymax - ymin) / img_size

            f.write(f"{name} {x_center} {y_center} {width} {height}\n")

def create_yolo_dataset(src_folder, dest_folder, train_ratio=0.8, val_ratio=0.2):
    os.makedirs(dest_folder, exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, 'labels', 'test'), exist_ok=True)

    img_list = glob.glob(os.path.join(src_folder, '*.jpg'))
    random.shuffle(img_list)
    train_size = int(train_ratio * len(img_list))
    val_size = int(val_ratio * train_size)
    test_size = len(img_list) - train_size

    train_list = img_list[:train_size]
    val_list = train_list[-val_size:]
    train_list = train_list[:-val_size]
    test_list = img_list[train_size:]

    def process_images(img_list, img_dest, label_dest):
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            xml_name = os.path.splitext(img_name)[0] + '.xml'
            xml_path = os.path.join(src_folder, xml_name)
            yolo_name = os.path.splitext(img_name)[0] + '.txt'
            yolo_path = os.path.join(dest_folder, 'labels', label_dest, yolo_name)

            if os.path.exists(xml_path):
                convert_annotation(xml_path, img.shape[1], yolo_path)

            dest_img_path = os.path.join(dest_folder, 'images', img_dest, img_name)
            cv2.imwrite(dest_img_path, img)

    process_images(train_list, 'train', 'train')
    process_images(val_list, 'val', 'val')
    process_images(test_list, 'test', 'test')

# Get the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use environment variables for user-specific paths, with defaults relative to the script directory
SRC_FOLDER = os.environ.get('SRC_FOLDER', os.path.join(SCRIPT_DIR, 'augmented_dataset'))
DEST_FOLDER = os.environ.get('DEST_FOLDER', os.path.join(SCRIPT_DIR, 'yolo_dataset'))
TRAIN_RATIO = float(os.environ.get('TRAIN_RATIO', '0.8'))

create_yolo_dataset(SRC_FOLDER, DEST_FOLDER, TRAIN_RATIO)

print(f"YOLO dataset creation complete. Check the output directory: {DEST_FOLDER}")