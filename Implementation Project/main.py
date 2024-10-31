import torch
import os
import random
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from imutils import paths

def main():
    print(torch.__version__)

    # Transform: Resize and Convert images to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])

    BATCH_SIZE = 32
    TRAIN_DIR = '../Train-Val-DS/train'
    VAL_DIR = '../Train-Val-DS/val'

    #displays if error occured loading data set
    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory '{TRAIN_DIR}' not found.")
    if not os.path.exists(VAL_DIR):
        print(f"Validation directory '{VAL_DIR}' not found.")




    # Create dataset and dataloader for training and validation
    train_set = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    val_set = datasets.ImageFolder(root=VAL_DIR, transform=transform)

    train_dataloader = DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    val_dataloader = DataLoader(
        dataset=val_set,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    # Print number of files in each set
    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(val_set)}")

    # Display a grid of random training images
    num_rows, num_cols = 5, 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for i in range(num_rows):
        for j in range(num_cols):
            # Choose a random index from the training dataset
            image_index = random.randrange(len(train_set))
            image, label = train_set[image_index]

            # Display the image in the subplot
            axs[i, j].imshow(image.permute(1, 2, 0))

            # Set the title of the subplot as the corresponding class name
            axs[i, j].set_title(train_set.classes[label], color="white")

            # Disable the axis for better visualization
            axs[i, j].axis(False)

    # Set the super title of the figure
    fig.suptitle(f"Random {num_rows * num_cols} images from the training dataset", fontsize=16, color="white")

    # Set the background color of the figure as black
    fig.set_facecolor(color='white')

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()
