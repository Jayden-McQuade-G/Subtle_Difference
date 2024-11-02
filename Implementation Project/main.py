import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Define a simple CNN for feature extraction
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 56 * 56, 256)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

# Define an LSTM for generating text descriptions based on features
class ImageDescriptionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ImageDescriptionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Generate a simple description of an image based on its features
def generate_description(features, lstm_model):
    dummy_input = features.unsqueeze(1).repeat(1, 10, 1)  # Sequence length of 10
    output = lstm_model(dummy_input)
    description = "Image has noticeable features with abstract patterns."  # Placeholder for actual logic
    return description

# Compare images based on their features
def compare_images(features1, features2):
    difference = torch.abs(features1 - features2).mean().item()
    if difference < 0.1:
        return "The images are almost identical."
    elif difference < 0.5:
        return "The images have noticeable differences in structure and details."
    else:
        return "The images are significantly different with major changes in patterns."

# Load and preprocess images
def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0), image

# Tkinter UI setup
def create_ui():
    def open_image1():
        filepath = filedialog.askopenfilename()
        if filepath:
            img1, pil_img1 = load_image(filepath)
            panel1.config(image=ImageTk.PhotoImage(pil_img1))
            panel1.image = ImageTk.PhotoImage(pil_img1)
            global image_tensor1
            image_tensor1 = img1

    def open_image2():
        filepath = filedialog.askopenfilename()
        if filepath:
            img2, pil_img2 = load_image(filepath)
            panel2.config(image=ImageTk.PhotoImage(pil_img2))
            panel2.image = ImageTk.PhotoImage(pil_img2)
            global image_tensor2
            image_tensor2 = img2

    def analyze_images():
        if image_tensor1 is not None and image_tensor2 is not None:
            features1 = cnn_model(image_tensor1)
            features2 = cnn_model(image_tensor2)
            desc1 = generate_description(features1, lstm_model)
            desc2 = generate_description(features2, lstm_model)
            comparison_result = compare_images(features1, features2)
            
            description1_label.config(text=f"Image 1 Description: {desc1}")
            description2_label.config(text=f"Image 2 Description: {desc2}")
            comparison_label.config(text=f"Comparison Result: {comparison_result}")

    # Initialize models
    global cnn_model, lstm_model, image_tensor1, image_tensor2
    cnn_model = SimpleCNN()
    lstm_model = ImageDescriptionLSTM(input_size=256, hidden_size=128, output_size=50)
    image_tensor1 = None
    image_tensor2 = None

    # Create the main tkinter window
    root = tk.Tk()
    root.title("Image Comparison Tool")

    # UI layout
    panel1 = tk.Label(root)
    panel1.grid(row=0, column=0, padx=10, pady=10)
    panel2 = tk.Label(root)
    panel2.grid(row=0, column=1, padx=10, pady=10)

    btn1 = tk.Button(root, text="Open Image 1", command=open_image1)
    btn1.grid(row=1, column=0, padx=10, pady=10)
    btn2 = tk.Button(root, text="Open Image 2", command=open_image2)
    btn2.grid(row=1, column=1, padx=10, pady=10)

    analyze_btn = tk.Button(root, text="Analyze Images", command=analyze_images)
    analyze_btn.grid(row=2, column=0, columnspan=2, pady=10)

    description1_label = tk.Label(root, text="Image 1 Description: ")
    description1_label.grid(row=3, column=0, columnspan=2)
    description2_label = tk.Label(root, text="Image 2 Description: ")
    description2_label.grid(row=4, column=0, columnspan=2)
    comparison_label = tk.Label(root, text="Comparison Result: ")
    comparison_label.grid(row=5, column=0, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    create_ui()
