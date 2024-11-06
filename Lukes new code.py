# ahahha_spacy.py

import os
import json
import spacy
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageTk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
import logging
from functools import partial

# -----------------------------
# 1. Initialize spaCy and Logging
# -----------------------------

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("missing_images.log"),
        logging.StreamHandler()
    ]
)

# Load spaCy's English model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading 'en_core_web_sm' model for spaCy as it was not found.")
    from spacy.cli import download

    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


def tokenize_spacy(text):
    """
    Tokenizes input text using spaCy.

    Args:
        text (str): The text to tokenize.

    Returns:
        list: List of token strings.
    """
    doc = nlp(text.lower())
    return [token.text for token in doc]


# -----------------------------
# 2. Define Collate Function Globally
# -----------------------------

def collate_fn(data, word_to_idx, max_length):
    """
    Creates mini-batch tensors from the list of tuples (before_image, after_image, caption).

    Args:
        data: list of tuples (before_image, after_image, caption).
        word_to_idx (dict): Mapping from words to indices.
        max_length (int): Maximum length for padding.

    Returns:
        before_images: Tensor of shape (batch_size, 3, 224, 224)
        after_images: Tensor of shape (batch_size, 3, 224, 224)
        captions: Tensor of shape (batch_size, max_caption_length)
    """
    before_images, after_images, captions = zip(*data)

    # Stack images
    before_images = torch.stack(before_images, 0)
    after_images = torch.stack(after_images, 0)

    # Convert captions to sequences
    sequences = [caption_to_sequence(caption, word_to_idx) for caption in captions]

    # Pad sequences
    padded_sequences, _ = pad_caption_sequences(sequences, word_to_idx, max_length=max_length)

    return before_images, after_images, padded_sequences


# -----------------------------
# 3. Dataset Class
# -----------------------------

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_dir, annotation_file, word_to_idx, transform=None):
        """
        Args:
            image_dir (str): Parent directory containing 'train' and 'val' subdirectories.
            annotation_file (str): Path to the JSON annotations file.
            word_to_idx (dict): Mapping from words to indices.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.annotations = self.load_annotations(annotation_file)
        self.word_to_idx = word_to_idx
        self.transform = transform
        self.image_pairs, self.captions = self.load_image_pairs()

    def load_annotations(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def find_image_file(self, image_set, image_name):
        """
        Finds the image file with the given name and any common image extension.

        Args:
            image_set (str): 'train' or 'val'.
            image_name (str): Name of the image without extension.

        Returns:
            str: Full path to the image file if found, else None.
        """
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        for ext in extensions:
            image_path = os.path.join(self.image_dir, image_set, image_name + ext)
            logging.debug(f"Searching for image: {image_path}")
            if os.path.exists(image_path):
                logging.debug(f"Found image: {image_path}")
                return image_path
        logging.debug(f"Image not found: {image_set}/{image_name}")
        return None

    def load_image_pairs(self):
        image_pairs = []
        captions = []

        for entry in tqdm(self.annotations, desc="Loading Image Pairs"):
            # Extract image filenames
            contents = entry.get('contents', [])
            if len(contents) < 2:
                logging.info(f"Insufficient image contents for entry: {entry.get('name', 'Unknown')}")
                continue

            # Determine if the entry is for 'train' or 'val' based on the annotation file
            image_set = 'train' if 'train' in entry.get('name', '').lower() else 'val'

            # Extract 'after' and 'before' image names without 'train/' or 'val/' prefixes
            after_img_filename = contents[0].get('name', '').replace(f"{image_set}/", '')
            before_img_filename = contents[1].get('name', '').replace(f"{image_set}/", '')

            if not after_img_filename or not before_img_filename:
                logging.info(f"Missing image filenames in entry: {entry.get('name', 'Unknown')}")
                continue

            # Find image files with any common extension
            after_img = self.find_image_file(image_set, after_img_filename)
            before_img = self.find_image_file(image_set, before_img_filename)

            if after_img and before_img:
                image_pairs.append((before_img, after_img))
                # Concatenate all 'value's from 'attributes' as the caption
                attribute_values = [attr['value'] for attr in entry.get('attributes', []) if 'value' in attr]
                concatenated_captions = ' '.join(attribute_values)
                captions.append(concatenated_captions)
            else:
                missing = []
                if not after_img:
                    missing.append(f"{image_set}/{after_img_filename}")
                if not before_img:
                    missing.append(f"{image_set}/{before_img_filename}")
                logging.info(
                    f"Missing images for entry: {entry.get('name', 'Unknown')}. Missing files: {', '.join(missing)}")

        return image_pairs, captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        before_img_path, after_img_path = self.image_pairs[idx]
        caption = self.captions[idx]

        # Load images
        before_image = Image.open(before_img_path).convert('RGB')
        after_image = Image.open(after_img_path).convert('RGB')

        if self.transform:
            before_image = self.transform(before_image)
            after_image = self.transform(after_image)

        return before_image, after_image, caption


# -----------------------------
# 4. Vocabulary Building
# -----------------------------

def build_vocabulary(captions, threshold=5):
    """
    Builds a vocabulary dictionary based on word frequency.

    Args:
        captions (list): List of caption strings.
        threshold (int): Minimum frequency for a word to be included.

    Returns:
        word_to_idx (dict): Mapping from words to unique indices.
        idx_to_word (dict): Mapping from indices to words.
        vocab_size (int): Size of the vocabulary.
    """
    tokens = []
    for caption in captions:
        tokens.extend(tokenize_spacy(caption))  # Using spaCy tokenizer

    counter = Counter(tokens)
    # Remove words that appear less than the threshold
    vocab = [word for word, count in counter.items() if count >= threshold]

    # Add special tokens
    vocab = ['<pad>', '<start>', '<end>', '<unk>'] + vocab

    # Create mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}

    return word_to_idx, idx_to_word, len(vocab)


def caption_to_sequence(caption, word_to_idx):
    """
    Converts a caption string into a list of word indices.

    Args:
        caption (str): The caption string.
        word_to_idx (dict): Mapping from words to indices.

    Returns:
        sequence (list): List of word indices.
    """
    tokens = tokenize_spacy(caption)  # Using spaCy tokenizer
    sequence = [word_to_idx.get('<start>')]
    for token in tokens:
        if token in word_to_idx:
            sequence.append(word_to_idx[token])
        else:
            sequence.append(word_to_idx.get('<unk>'))
    sequence.append(word_to_idx.get('<end>'))
    return sequence


def preprocess_captions(captions, word_to_idx):
    """
    Converts a list of captions into sequences of word indices.

    Args:
        captions (list): List of caption strings.
        word_to_idx (dict): Mapping from words to indices.

    Returns:
        sequences (list): List of lists containing word indices.
    """
    sequences = [caption_to_sequence(caption, word_to_idx) for caption in captions]
    return sequences


def pad_caption_sequences(sequences, word_to_idx, max_length=None):
    """
    Pads sequences to the same length.

    Args:
        sequences (list): List of lists containing word indices.
        word_to_idx (dict): Mapping from words to indices.
        max_length (int, optional): Maximum length for padding. If None, uses the longest sequence.

    Returns:
        padded_sequences (Tensor): Tensor of padded sequences.
        max_length (int): The maximum sequence length.
    """
    if not max_length:
        max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            seq = seq + [word_to_idx['<pad>']] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        padded_sequences.append(torch.tensor(seq, dtype=torch.long))
    padded_sequences = torch.stack(padded_sequences)
    return padded_sequences, max_length


# -----------------------------
# 5. Model Architecture
# -----------------------------

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        # Load the pretrained ResNet-50 model with updated weights parameter
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, 512)  # Keeping output size 512
        self.bn = nn.BatchNorm1d(512, momentum=0.01)

    def forward(self, images):
        """
        Forward propagation.

        Args:
            images: input images, a tensor of dimensions (batch_size, 3, image_size, image_size)

        Returns:
            features: encoded images, a tensor of dimension (batch_size, 512)
        """
        with torch.no_grad():
            features = self.resnet(images)  # (batch_size, 2048, 7, 7)
        features = self.adaptive_pool(features)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        features = self.fc(features)  # (batch_size, 512)
        features = self.bn(features)  # (batch_size, 512)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args:
            embed_size: dimension of word embeddings
            hidden_size: dimension of LSTM hidden states
            vocab_size: size of vocabulary
            num_layers: number of LSTM layers
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features, captions):
        """
        Forward propagation.

        Args:
            features: encoded images, a tensor of dimension (batch_size, embed_size)
            captions: captions, a tensor of dimension (batch_size, max_caption_length)

        Returns:
            outputs: scores for vocabulary, a tensor of dimension (batch_size, max_caption_length+1, vocab_size)
        """
        embeddings = self.embed(captions)  # (batch_size, max_caption_length, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings),
                               dim=1)  # (batch_size, max_caption_length+1, embed_size)
        embeddings = self.dropout(embeddings)
        hiddens, _ = self.lstm(embeddings)  # (batch_size, max_caption_length+1, hidden_size)
        outputs = self.linear(hiddens)  # (batch_size, max_caption_length+1, vocab_size)
        return outputs

    def sample(self, features, word_to_idx, idx_to_word, max_length=20):
        """
        Generate captions for given image features using greedy search.

        Args:
            features: encoded images, a tensor of dimension (1, embed_size)
            word_to_idx (dict): mapping from words to indices
            idx_to_word (dict): mapping from indices to words
            max_length (int): maximum length of generated caption

        Returns:
            generated_caption (str): the generated caption string
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        states = None

        for i in range(max_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (1, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs: (1, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)  # inputs: (1, embed_size)
            inputs = inputs.unsqueeze(1)  # (1, 1, embed_size)
            if idx_to_word.get(predicted.item(), '<unk>') == '<end>':
                break

        sampled_caption = []
        for word_id in sampled_ids:
            word = idx_to_word.get(word_id, '<unk>')
            if word == '<end>':
                break
            sampled_caption.append(word)
        generated_caption = ' '.join(sampled_caption)
        return generated_caption


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, before_images, after_images, captions):
        # Encode before and after images
        before_features = self.encoder(before_images)
        after_features = self.encoder(after_images)

        # Compute feature difference
        image_features = after_features - before_features  # (batch_size, embed_size)

        # Decode captions
        outputs = self.decoder(image_features, captions)
        return outputs


# -----------------------------
# 6. Training Utilities
# -----------------------------

def train(model, data_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    for i, (before_imgs, after_imgs, captions) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}/{total_epochs}")):
        before_imgs = before_imgs.to(device)
        after_imgs = after_imgs.to(device)
        captions = captions.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(before_imgs, after_imgs, captions[:, :-1])  # Exclude the last token for inputs
        targets = captions[:, 1:]  # Shifted by one for targets

        # Adjust outputs to remove the first time step
        outputs = outputs[:, 1:, :]  # (batch_size, max_length, vocab_size)

        # Ensure outputs and targets have the same number of elements
        if outputs.size(1) != targets.size(1):
            min_length = min(outputs.size(1), targets.size(1))
            outputs = outputs[:, :min_length, :]
            targets = targets[:, :min_length]

        # Compute loss
        outputs = outputs.reshape(-1, outputs.size(-1))  # (batch_size * max_length, vocab_size)
        targets = targets.reshape(-1)  # (batch_size * max_length)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch}/{total_epochs}], Average Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for before_imgs, after_imgs, captions in tqdm(data_loader, desc="Validation"):
            before_imgs = before_imgs.to(device)
            after_imgs = after_imgs.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(before_imgs, after_imgs, captions[:, :-1])
            targets = captions[:, 1:]

            # Adjust outputs to remove the first time step
            outputs = outputs[:, 1:, :]  # (batch_size, max_length, vocab_size)

            # Ensure outputs and targets have the same number of elements
            if outputs.size(1) != targets.size(1):
                min_length = min(outputs.size(1), targets.size(1))
                outputs = outputs[:, :min_length, :]
                targets = targets[:, :min_length]

            # Compute loss
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


# -----------------------------
# 7. Evaluation Utilities
# -----------------------------

def calculate_bleu_score(reference, candidate):
    """
    Calculates BLEU-4 score between reference and candidate captions.

    Args:
        reference (str): Reference caption.
        candidate (str): Generated caption.

    Returns:
        score (float): BLEU-4 score.
    """
    reference_tokens = tokenize_spacy(reference)
    candidate_tokens = tokenize_spacy(candidate)

    reference = [reference_tokens]
    candidate = candidate_tokens

    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    return score


def evaluate_model(model, encoder, data_loader, word_to_idx, idx_to_word, device, max_length):
    model.eval()
    bleu_scores = []
    with torch.no_grad():
        for before_imgs, after_imgs, captions in tqdm(data_loader, desc="Evaluating"):
            before_imgs = before_imgs.to(device)
            after_imgs = after_imgs.to(device)

            # Encode images
            before_features = encoder(before_imgs)
            after_features = encoder(after_imgs)
            image_features = after_features - before_features  # Feature difference

            # Generate captions
            generated_captions = []
            for i in range(image_features.size(0)):
                feature = image_features[i].unsqueeze(0).to(device)
                caption = model.decoder.sample(feature, word_to_idx, idx_to_word, max_length)
                generated_captions.append(caption)

            # Calculate BLEU scores
            for i in range(captions.size(0)):
                # Extract reference caption
                target_caption = captions[i].lower()

                # Get generated caption
                generated_caption = generated_captions[i]

                # Calculate BLEU-4 score
                bleu = calculate_bleu_score(target_caption, generated_caption)
                bleu_scores.append(bleu)

    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    print(f"Average BLEU-4 Score: {average_bleu:.4f}")
    return average_bleu


# -----------------------------
# 8. Caption Generation Utility
# -----------------------------

def generate_caption(model, encoder, before_image, after_image, word_to_idx, idx_to_word, device, max_length=20):
    """
    Generates a caption for a pair of images.

    Args:
        model: The Image Captioning model.
        encoder: The CNN encoder.
        before_image (PIL Image): The 'before' image.
        after_image (PIL Image): The 'after' image.
        word_to_idx (dict): Mapping from words to indices.
        idx_to_word (dict): Mapping from indices to words.
        device: Computation device.
        max_length (int): Maximum length of the generated caption.

    Returns:
        generated_caption (str): The generated caption.
    """
    model.eval()
    with torch.no_grad():
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet means
                                 std=[0.229, 0.224, 0.225])
        ])

        # Preprocess images
        before_tensor = transform(before_image).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        after_tensor = transform(after_image).unsqueeze(0).to(device)

        # Encode images
        before_features = encoder(before_tensor)
        after_features = encoder(after_tensor)
        image_features = after_features - before_features  # Feature difference

        # Generate caption
        generated_caption = model.decoder.sample(image_features, word_to_idx, idx_to_word, max_length)

    return generated_caption


# -----------------------------
# 9. GUI Utility (Optional)
# -----------------------------

def display_caption(model, encoder, image_pair, word_to_idx, idx_to_word, device, max_length):
    """
    Displays image pairs and their generated captions in a GUI.

    Args:
        model: The Image Captioning model.
        encoder: The CNN encoder.
        image_pair (tuple): Tuple containing paths to 'before' and 'after' images.
        word_to_idx (dict): Mapping from words to indices.
        idx_to_word (dict): Mapping from indices to words.
        device: Computation device.
        max_length (int): Maximum length of the generated caption.
    """
    before_img_path, after_img_path = image_pair
    before_image = Image.open(before_img_path).convert('RGB')
    after_image = Image.open(after_img_path).convert('RGB')

    # Generate caption
    generated_caption = generate_caption(model, encoder, before_image, after_image, word_to_idx, idx_to_word, device,
                                         max_length)

    # Load images for display
    img_before = before_image.resize((300, 300))
    img_after = after_image.resize((300, 300))

    # Create GUI window
    root = tk.Tk()
    root.title("Image Captioning")

    # Display Before Image
    tk_before = ImageTk.PhotoImage(img_before)
    label_before = tk.Label(root, image=tk_before)
    label_before.image = tk_before  # Keep a reference
    label_before.pack(side="left", padx=10, pady=10)
    label_before_title = tk.Label(root, text="Before")
    label_before_title.pack(side="left")

    # Display After Image
    tk_after = ImageTk.PhotoImage(img_after)
    label_after = tk.Label(root, image=tk_after)
    label_after.image = tk_after  # Keep a reference
    label_after.pack(side="left", padx=10, pady=10)
    label_after_title = tk.Label(root, text="After")
    label_after_title.pack(side="left")

    # Display Captions
    caption_text = f"Generated Caption:\n{generated_caption}"
    label_caption = tk.Label(root, text=caption_text, wraplength=600, justify="left", font=("Helvetica", 12))
    label_caption.pack(pady=10)

    root.mainloop()


# -----------------------------
# 10. Main Function
# -----------------------------

def main():
    # -----------------------------
    # Paths
    # -----------------------------
    # Set image_dir to the parent directory containing 'train' and 'val' subdirectories
    image_dir = r"C:\Users\minen\Downloads\Train-Val-DS"

    train_annotation_file = r"C:\Users\minen\Downloads\train-val-annotations\capt_train_annotations.json"
    val_annotation_file = r"C:\Users\minen\Downloads\train-val-annotations\capt_val_annotations.json"

    # -----------------------------
    # Check if Annotation Files Exist
    # -----------------------------
    if not os.path.isfile(train_annotation_file):
        print(f"Error: Training annotation file not found at '{train_annotation_file}'.")
        return
    if not os.path.isfile(val_annotation_file):
        print(f"Error: Validation annotation file not found at '{val_annotation_file}'.")
        return

    # -----------------------------
    # Define Image Transformations
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet means
                             std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # Load Annotations
    # -----------------------------
    with open(train_annotation_file, 'r') as f:
        train_annotations = json.load(f)
    with open(val_annotation_file, 'r') as f:
        val_annotations = json.load(f)

    # -----------------------------
    # Debugging: Inspect the First Few Entries
    # -----------------------------
    print("\n--- Training Annotations Sample ---")
    for i, entry in enumerate(train_annotations[:2]):  # Inspect first 2 entries
        print(f"Entry {i + 1} Keys: {list(entry.keys())}")
        print(f"Entry {i + 1} Attributes: {[attr['key'] for attr in entry.get('attributes', [])]}")
    print("\n--- Validation Annotations Sample ---")
    for i, entry in enumerate(val_annotations[:2]):  # Inspect first 2 entries
        print(f"Entry {i + 1} Keys: {list(entry.keys())}")
        print(f"Entry {i + 1} Attributes: {[attr['key'] for attr in entry.get('attributes', [])]}")

    # -----------------------------
    # Extract Captions
    # -----------------------------
    try:
        train_captions = [' '.join([attr['value'] for attr in entry.get('attributes', []) if 'value' in attr]) for entry
                          in train_annotations]
    except KeyError as e:
        print(f"Error: Missing key in training annotations - {e}")
        print("Please ensure each entry in the training annotation JSON has an 'attributes' key with 'value' fields.")
        return

    try:
        val_captions = [' '.join([attr['value'] for attr in entry.get('attributes', []) if 'value' in attr]) for entry
                        in val_annotations]
    except KeyError as e:
        print(f"Error: Missing key in validation annotations - {e}")
        print("Please ensure each entry in the validation annotation JSON has an 'attributes' key with 'value' fields.")
        return

    # -----------------------------
    # Build Vocabulary
    # -----------------------------
    word_to_idx, idx_to_word, vocab_size = build_vocabulary(train_captions, threshold=5)
    print(f"\nVocabulary Size: {vocab_size}")

    # -----------------------------
    # Preprocess Captions
    # -----------------------------
    train_sequences = preprocess_captions(train_captions, word_to_idx)
    val_sequences = preprocess_captions(val_captions, word_to_idx)

    # -----------------------------
    # Pad Sequences
    # -----------------------------
    train_padded, max_length = pad_caption_sequences(train_sequences, word_to_idx, max_length=None)
    val_padded, _ = pad_caption_sequences(val_sequences, word_to_idx, max_length=max_length)
    print(f"Maximum Sequence Length: {max_length}")

    # -----------------------------
    # Instantiate Datasets
    # -----------------------------
    train_dataset = ImageCaptioningDataset(image_dir, train_annotation_file, word_to_idx, transform=transform)
    val_dataset = ImageCaptioningDataset(image_dir, val_annotation_file, word_to_idx, transform=transform)

    # -----------------------------
    # Check if Datasets Are Empty
    # -----------------------------
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Please check your image directory and annotations.")
        return
    if len(val_dataset) == 0:
        print("Error: Validation dataset is empty. Please check your image directory and annotations.")
        return

    # -----------------------------
    # Define Collate Function with Partial
    # -----------------------------
    collate_fn_partial = partial(collate_fn, word_to_idx=word_to_idx, max_length=max_length)

    # -----------------------------
    # Create DataLoaders
    # -----------------------------
    batch_size = 16
    num_workers = 4  # Adjust based on your system

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn_partial)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn_partial)

    # -----------------------------
    # Hyperparameters and Device Configuration
    # -----------------------------
    embed_size = 512  # Updated from 256 to 512 to match EncoderCNN output
    hidden_size = 512
    num_layers = 1
    learning_rate = 1e-4
    num_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # -----------------------------
    # Instantiate Encoder and Decoder
    # -----------------------------
    encoder = EncoderCNN()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    # Move models to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # -----------------------------
    # Instantiate the Image Captioning Model
    # -----------------------------
    model = ImageCaptioningModel(encoder, decoder).to(device)

    # -----------------------------
    # Loss and Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # -----------------------------
    # Training Loop
    # -----------------------------
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # -----------------------------
    # Plot Training and Validation Loss
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # -----------------------------
    # Evaluate the Model
    # -----------------------------
    average_bleu = evaluate_model(model, encoder, val_loader, word_to_idx, idx_to_word, device, max_length)

    # -----------------------------
    # Generate a Sample Caption
    # -----------------------------
    sample_idx = 0  # Change as needed
    if sample_idx < len(val_dataset):
        before_img_path, after_img_path = val_dataset.image_pairs[sample_idx]
        before_image = Image.open(before_img_path).convert('RGB')
        after_image = Image.open(after_img_path).convert('RGB')

        generated_caption = generate_caption(model, encoder, before_image, after_image, word_to_idx, idx_to_word,
                                             device, max_length)
        print(f"\nGenerated Caption: {generated_caption}")
        print(f"Reference Caption: {' '.join(val_dataset.captions[sample_idx].split())}")

        # -----------------------------
        # Optional: Display in GUI
        # -----------------------------
        display_caption(model, encoder, val_dataset.image_pairs[sample_idx], word_to_idx, idx_to_word, device,
                        max_length)
    else:
        print("Sample index is out of range for the validation dataset.")


if __name__ == "__main__":
    main()
