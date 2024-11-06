# this file contains random code that is no longer used


"""
from __future__ import print_function

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from imutils import paths
import json
from torch.nn.utils.rnn import pad_sequence

#PyTorch custom dataset class
class ImageDataset(Dataset):
    def __init__(self, image_dir, class_dir, caption_dir, transform=None):

        #Set all image file paths in the directory
        self.image_paths = list(paths.list_images(image_dir))
        self.transform = transform

        #Store imge classes and captions
        with open(class_dir, 'r') as f:
            self.class_data = json.load(f)
        
        with open(caption_dir, 'r') as f:
            self.caption_data = json.load(f)

    def __load_json__(self):
        pass
          
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = read_image(img_path)
        filename = os.path.basename(img_path) #return image filename
        filename = filename.split(".")[0]
        
        #Find name of the file
        base_filename = filename.rsplit('_', 1)[0]
        #print(base_filename)

        class_attributes = []
        caption_attributes = []

        #Find corresponding captions and classes based on filename
        img_class = next((item for item in self.class_data if base_filename in item['name']), None)
        img_caption = next((item for item in self.caption_data if base_filename in item['name']), None)

        # Get the class label and attributes (if available)
        if img_caption:
            caption_attributes = img_caption.get("attributes", [])
            #print(caption_attributes)

        if img_class:
            class_attributes = img_class.get("attributes", [])
            #print(caption_attributes)

        #transform into tensor
        if self.transform:
            image = self.transform(image)

        return [image, caption_attributes, class_attributes]
    

def collate_fn (set):
    images = []
    classes = set[1]
    captions = set[2]

    class_len = 0
    caption_len = 0

    #find longest class attributes
    for x in classes:
        if len(x) > class_len:
            class_len = len(x)

    #find longest caption attributes
    for x in captions:
        if len(x) > caption_len:
            caption_len = len(x)


    for sample in set:
        images.append(sample[0])

        # Convert caption and class attributes to tensors and append
        caption_attributes.append(torch.tensor(sample[1], dtype=torch.long))  # Convert list to tensor
        class_attributes.append(torch.tensor(sample[2], dtype=torch.long))    # Convert list to tensor

    # Stack images into a single tensor of shape (batch_size, 3, 224, 224)
    images = torch.stack(images, dim=0)

    # Pad caption and class attributes to the longest sequence length in the batch
    captions = pad_sequence(captions, batch_first=True, padding_value=caption_len)
    classes = pad_sequence(classes, batch_first=True, padding_value=class_len)

    return images, captions, classes


def main():
    #Print PyTorch version
    print(torch.__version__)

    # Initialise Variables
    IMG_SIZE = 64
    ROWS = 256
    COLS = 256
    CHANNELS = 3
    BATCH_SIZE = 16

    # Data location Variables Images
    TRAIN_DIR = os.path.join('../Train-Val-DS/train')
    VAL_DIR = os.path.join('../Train-Val-DS/val')

    #Data location Variables Captions
    TRAIN_CAPT_DIR = os.path.join('../train-val-annotations/capt_train_annotations.json')
    VAL_CAPT_DIR = os.path.join('../train-val-annotations/capt_val_annotations.json')

    #Data location variables Classes
    TRAIN_CLASS_DIR = os.path.join('../train-val-annotations/class_train_annotation.json')
    VAL_CLASS_DIR = os.path.join('../train-val-annotations/class_val_annotation.json')

    # Transformations for resizing and converting images to tensors
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Initialise Data directory locations
    dataset_home = '../Train-Val-DS/'
    subdirs = ['train/', 'val/']
    for subdir in subdirs:
        # Create label subdirectories
        newdir = dataset_home + subdir
        os.makedirs(newdir, exist_ok=True)

    # Display an error if data set directory not found
    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory '{TRAIN_DIR}' not found.")
        return
    if not os.path.exists(VAL_DIR):
        print(f"Validation directory '{VAL_DIR}' not found.")
        return

    # Load the datasets for training and validation
    train_set = ImageDataset(image_dir=TRAIN_DIR, class_dir=TRAIN_CLASS_DIR, caption_dir=TRAIN_CAPT_DIR, transform=transform) 
    val_set = ImageDataset(image_dir=VAL_DIR, class_dir=VAL_CLASS_DIR, caption_dir=VAL_CAPT_DIR, transform=transform)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(dataset=train_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=0)
    val_dataloader = DataLoader(dataset=val_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=0)

    # Iterate over the dataset
   # for images, class_attributes, caption_attributes in train_set:
        # Process and display the differences
       # print(f"Class Attributes: {class_attributes}")
       # print(f"Caption Attributes: {caption_attributes}")

    # Display number of images loaded
    print(f"Number of training images: {len(train_set)}")
    print(f"Number of validation images: {len(val_set)}")

    for i, (images, caption_attributes, class_attributes) in enumerate(train_dataloader):
        print(f"Batch {i+1}:")
        print(f"Images shape: {images.shape}")
        print(f"Caption attributes shape: {caption_attributes.shape}")
        print(f"Class attributes shape: {class_attributes.shape}")
        if i >= 1:  # Only test the first few batches
            break

    for i, (images, caption_attributes_batch, class_attributes_batch) in enumerate(train_dataloader):
        if i >= 1:  # Display only the first batch
            break
        plt.figure(figsize=(10, 10))
        for j in range(min(len(images), 16)):
            # Convert the image tensor to a NumPy array and permute dimensions for display
            img = images[j].permute(1, 2, 0).numpy()
            if img.max() <= 1:  # Check if the image is normalized (e.g., [0, 1])
                img = (img * 255).astype("uint8")
            
            # Safely retrieve class and caption attributes for each image
            class_attr = class_attributes_batch[j].tolist() if j < len(class_attributes_batch) else "No class attribute"
            caption_attr = caption_attributes_batch[j].tolist() if j < len(caption_attributes_batch) else "No caption attribute"
            
            plt.subplot(4, 4, j + 1)
            plt.imshow(img)
            plt.title(f"Class Attr: {class_attr}, Caption Attr: {caption_attr}")
            plt.axis("off")
        plt.show()


    num_classes = 2
    epochs = 10 #50 #100
    data_augmentation = True #True 


if __name__ == "__main__":
    main()
    
"""