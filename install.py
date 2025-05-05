import os
import urllib.request
import torch
from torchvision import models

# Download model weights
os.makedirs('places365', exist_ok=True)
weights_url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
file_path = 'places365/resnet18_places365.pth.tar'

if not os.path.exists(file_path):
    print("Downloading Places365 weights...")
    urllib.request.urlretrieve(weights_url, file_path)
else:
    print("Weights already downloaded.")

# Download label file
labels_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
labels_path = 'places365/categories_places365.txt'

if not os.path.exists(labels_path):
    print("Downloading labels...")
    urllib.request.urlretrieve(labels_url, labels_path)
else:
    print("Labels already downloaded.")

