# Vocabulary for describing subtle differences between image pairs
vocabulary = [
    "<start>", "<end>", "<pad>", "<unk>",  # Special tokens
    "color", "difference", "change", "modified", "added", "removed", "altered",
    "background", "foreground", "left", "right", "top", "bottom",
    "small", "large", "tiny", "huge", "slightly", "significantly",
    "object", "shape", "position", "texture", "pattern",
    "dark", "light", "red", "blue", "green", "yellow", "black", "white", "gray",
    "circle", "square", "triangle", "line", "dot", "stripe", "curve",
    "near", "far", "close", "distant",
    "visible", "hidden", "blurred", "sharp", "clear", "faded", "bright",
    "shadow", "highlight", "edge", "corner", "center", "middle",
    "appears", "disappears", "replaced", "shifted", "moved", "rotated",
    "texture", "surface", "pattern", "different", "similar",
    "new", "old", "background", "foreground", "overlapping",
    "part", "section", "contrast", "brightness", "sharpness", "size",
    "inside", "outside", "around", "between", "connected", "disconnected"
]

# Mapping each word to an index for use in the model
vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np



class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(CaptionGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        lstm_input = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(lstm_input)
        outputs = self.fc(hiddens)
        return outputs

# Instantiate the LSTM model
embed_size = 256
hidden_size = 512
vocab_size = len(vocabulary)
caption_model = CaptionGenerator(vocab_size, embed_size, hidden_size)

# Dummy input and target captions
dummy_features = torch.randn(1, embed_size)  # Replace with ViT output
dummy_captions = torch.tensor([[vocab_dict["<start>"], vocab_dict["object"], vocab_dict["changed"], vocab_dict["<end>"]]])
outputs = caption_model(dummy_features, dummy_captions)

# Set up the training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(vit_model.parameters()) + list(caption_model.parameters()), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for img1, img2, target_caption in dataset_loader:  # Replace with actual data loader
        patches1 = preprocess_image(img1)
        patches2 = preprocess_image(img2)

        # Phase 1: Difference Identification
        differences = vit_model(patches1 - patches2)

        # Phase 2: Caption Generation
        outputs = caption_model(differences, target_caption)
        loss = criterion(outputs.view(-1, vocab_size), target_caption.view(-1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def generate_caption(image1, image2):
    patches1 = preprocess_image(image1)
    patches2 = preprocess_image(image2)
    differences = vit_model(patches1 - patches2)

    # Generate captions using LSTM
    caption = ["<start>"]
    for i in range(20):  # Limit to 20 words
        caption_tensor = torch.tensor([[vocab_dict[word] for word in caption]], dtype=torch.long)
        outputs = caption_model(differences, caption_tensor)
        _, predicted = outputs[:, -1, :].max(1)
        word = list(vocab_dict.keys())[list(vocab_dict.values()).index(predicted.item())]
        caption.append(word)
        if word == "<end>":
            break
    return " ".join(caption[1:-1])

# Test the caption generation
image1 = Image.open("test_image1.jpg")  # Replace with test images
image2 = Image.open("test_image2.jpg")
print("Generated Caption:", generate_caption(image1, image2))
