import torch
import os

import torchvision.transforms as transforms
from torchvision import datasets
from imutils import paths
import random


def create_train_data(images):
    count = len(images)


def patch_embedding():
    print("PATCH EMBEDDING NOT IMPLEMNTED")

def patch_embedding():
    print("PATCH EMBEDDING NOT IMPLEMNTED")



def main():
    print(torch.__version__)

    #Resize images
    transform = transforms.Compose([
        transforms.Resize((101, 101)), 
        transforms.ToTensor()
    ])
    
    TRAIN_DIR = 'Implementation Project/Train-Val-DS/train'
    VAL_DIR = 'Implementation Project/Train-Val-DS/val'

    train_set = list(paths.list_images(TRAIN_DIR))
    val_set = list(paths.list_images(VAL_DIR))

    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(val_set)}")

    random.shuffle(train_set)
    random.shuffle(val_set)

    ROWS = 224
    COLS =224
    CHANNELS = 3



if __name__ == "__main__":
    main()
