# Indoor-Outdoor Classifier

This project classifies images into three categories: **indoor**, **outdoor**, or **doubt** using the Places365 dataset and a pre-trained ResNet-18 model.

## Directory Structure

The repository has the following directory structure:

.
├── classification_log.txt # Logs classification results with filename, label, confidence, and scene category.
├── classify_images.py # Python script for classifying images into indoor, outdoor, or doubt.
├── install.py # Setup script for installing necessary dependencies.
├── model
│ └── pytorch_model.bin # Model weights for classification (if separate from places365 directory).
├── places365
│ ├── categories_places365.txt # Categories/labels of the Places365 dataset.
│ └── resnet18_places365.pth.tar # Pre-trained ResNet-18 model for image classification.
└── README.md # This README file.


## Requirements

Before running this project, ensure you have the following installed:

- Python 3.x
- PyTorch (>=1.7.0)
- torchvision
- PIL (Pillow)
- Other dependencies listed in the `install.py` script

You can install the required dependencies by running the provided `install.py` script.

## Installation

1. **Clone the Repository:**

    First, clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/indoor_outdoor_classifier.git
    cd indoor_outdoor_classifier
    ```

2. **Install Dependencies:**

    If you have Python 3 installed, run the following command to install the necessary libraries:

    ```bash
    python install.py
    ```

    This will install the required dependencies such as `torch`, `torchvision`, and `Pillow`.

## Dataset

This project uses the **Places365** dataset, which contains scene categories for various indoor and outdoor environments.

- **Indoor categories** include places like `kitchen`, `living_room`, `classroom`, etc.
- **Outdoor categories** include places like `beach`, `mountain`, `park`, etc.
- If the confidence score for a scene category is below a specified threshold, the image is labeled as **doubt**.

## How to Use

1. **Download Pre-trained Model and Labels:**

    The necessary model weights and labels will be downloaded automatically if they don’t already exist in the `places365` directory. If they are already present, the download will be skipped.

    The following files are required for the classification:
    - `resnet18_places365.pth.tar` (model weights)
    - `categories_places365.txt` (categories/labels)

    If these files are not present, they will be downloaded using the URLs defined in the `install.py` script.

2. **Classify Images:**

    To classify images, simply run the `classify_images.py` script:

    ```bash
    python classify_images.py
    ```

    The script will classify images in the `shard_94/images/` directory and move them into one of the three subdirectories:
    - `indoor/`
    - `outdoor/`
    - `doubt/`

    It will also log the classification details (filename, label, confidence, scene category) into the `classification_log.txt` file.

3. **Log File:**

    A log file `classification_log.txt` will be generated containing the results of each image processed. It includes:
    - **Filename**: Name of the image.
    - **Label**: Classification result (indoor, outdoor, doubt).
    - **Confidence**: The confidence score for the classification.
    - **Scene Category**: The predicted category from the Places365 dataset.

## Example Output

When running the classification script, you will see output like the following:

image_1.jpg → indoor (0.87) - Scene: bedroom
image_2.jpg → outdoor (0.91) - Scene: beach
image_3.jpg → doubt (0.05) - Scene: museum


## Notes

- The classification script uses a ResNet-18 model pre-trained on the **Places365** dataset for scene classification.
- Images are resized and normalized before being passed to the model for inference.
- The classification is based on a confidence threshold of `0.05`. Images with a lower confidence score are labeled as **doubt**.
