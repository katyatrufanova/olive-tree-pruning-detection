# Olive Tree Pruning Detection

This repository contains the code and resources for an olive tree pruning detection system using YOLOv7-tiny. The project aims to detect and classify pruned and unpruned olive trees in drone-captured images using object detection techniques.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training and Testing](#training-and-testing)
  - [Demo](#demo)
- [Experimental Results](#experimental-results)
- [Documentation](#documentation)
- [License](#license)

## Project Structure

```
olive-tree-pruning-detection/
│
├── data/
│ └── sample_data/
│ └── [sample images]
│
├── docs/
│ └── Report.pdf
│
├── notebooks/
│ ├── demo.ipynb
│ └── training_testing.ipynb
│
├── src/
│ ├── data_augmentation.py
│ ├── image_patch_generator.py
│ └── yolo_dataset_creator.py
│
├── README.md
└── requirements.txt
```

## Installation

1. Clone this repository:
```
git clone https://github.com/katyatrufanova/olive-tree-pruning-detection.git
cd olive-tree-pruning-detection
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Clone the YOLOv7 repository:
```
git clone https://github.com/WongKinYiu/yolov7
```

## Dataset

The project utilizes two high-resolution TIF images of olive tree fields in Apulia, Italy, captured using a drone. The dataset contains both pruned and unpruned trees, labeled using the Computer Vision Annotation Tool (CVAT). Due to the proprietary nature of the full dataset, only sample images are provided in the `data/sample_data` directory for demonstration purposes.

## Usage

### Data Preprocessing

1. Run the image patch generator script:
```
python src/image_patch_generator.py --input_dir /path/to/original/images --output_dir /path/to/output/patches
```

2. Perform data augmentation:
```
python src/data_augmentation.py --input_dir /path/to/patches --output_dir /path/to/augmented/data
```

3. Create YOLO format dataset:
```
python src/yolo_dataset_creator.py --input_dir /path/to/augmented/data --output_dir /path/to/yolo/dataset
```

### Training and Testing

Open and run the `notebooks/training_testing.ipynb` notebook:
```
jupyter notebook notebooks/training_testing.ipynb
```

### Demo

1. Download the trained weights file `best_weights.pt` from [Google Drive](https://drive.google.com/file/d/1r6f1aSViv6WjBZpvnNQ4jDSpvxXxGGnn/view?usp=sharing) and place it in the root directory of the project.

2. Open and run the `notebooks/demo.ipynb` notebook:
```
jupyter notebook notebooks/demo.ipynb
```

3. Follow the instructions in the notebook to run inference on sample images or your own images.

## Experimental Results

Three experiments were conducted to evaluate the performance of the YOLOv7-tiny model in detecting and labeling pruned and unpruned trees. The best results were achieved in Experiment 3 with the following hyperparameters:

- Epochs: 200
- Learning Rate: 0.001 (initial) to 0.0001 (final)
- Optimizer: Adam
- Weight Decay: 0

Results of Experiment 3 (best performance):
- Precision: 0.47
- Recall: 0.66
- mAP@.5: 0.527
- mAP@.5:.95: 0.349

For detailed results and comparisons, please refer to the `docs/Report.pdf` file.

## Documentation

For a comprehensive overview of the project, including detailed information on materials and methods, please refer to the `docs/Report.pdf` file.

