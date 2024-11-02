import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from imutils import paths

from torchvision import transforms
from torch.utils.data import Dataset
from imutils import paths

import cv2


import urllib.request 
from PIL import Image 
import numpy as np 
from patchify import patchify



class ImageDataset():
    def __init__(self, image_dir, transform=None):
        #set all image file paths in directory
        self.image_paths = list(paths.list_images(image_dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = read_image(img_path)

        if self.transform:

            image = self.transform(image)
        

        return image


def patching(image):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)
    elif isinstance(image, Image.Image):
        image = np.array(image)

    img_arr = np.asarray(image)
    patches = patchify(image, (224, 224, 3), step = 224)
    return patches





def main():

    print(torch.__version__)

    # Define transformations for resizing and converting images to tensors
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    BATCH_SIZE = 32

    # DATA_DIR = os.path.abspath('../Train-Val-DS' train))
    TRAIN_DIR = os.path.join('../Train-Val-DS/train')
    TEST_DIR = os.path.join('../Train-Val-DS/val')

    # Display an error message if the dataset directories are not found
    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory '{TRAIN_DIR}' not found.")
        return
    if not os.path.exists(TEST_DIR):
        print(f"Testing directory '{TEST_DIR}' not found.")
        return

    # Load the datasets for training and testing
    train_set = ImageDataset(image_dir=TRAIN_DIR, transform=transform)
    test_set = ImageDataset(image_dir=TEST_DIR, transform=transform)

    # Create DataLoaders for training and testing sets
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)
    test_dataloader = DataLoader(dataset=test_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)


    #####################################################################################
    # Display a random image from the training set
    image_tensor = train_set[1]  # Get the image tensor at index 1
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C)
    # Convert from RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Display the image
    cv2.imshow('Image Window', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ######################################################################################

    patching(train_set[1])
    

if __name__ == "__main__":
    main()
