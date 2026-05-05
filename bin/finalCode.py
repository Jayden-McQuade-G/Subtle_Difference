#ahahha.py

#Import Library Modules
import os
import json
import spacy
import torch
import torch.nn as nn
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
from torch.cuda import amp


### Initialize spaCy and Logging ###
#----------------------------------#

#Define Logging structure to track code progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("missing_images.log"),
        logging.StreamHandler()
    ]
)

#Load English Vocabulary from spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading 'en_core_web_sm' model for spaCy as it was not found.")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

#Use spaCy library to tokenise input text
#input: target "text"
#output: list of tokenised text
def tokenize_spacy(text):
    doc = nlp(text.lower())
    return [token.text for token in doc]


## Define Collate Function ##
#---------------------------#

#Formats images and captions to correctly sized tensors
#Input: List of data (before_image, after_image, caption), word to index dictionary, maximum length for padding
#Output: correctly formatted Tensors, before_image, after_image, captions, list of string captions
def collate_fn(data, word_to_idx, max_length):
    before_images, after_images, captions = zip(*data)

    #Stack images
    before_images = torch.stack(before_images, 0)
    after_images = torch.stack(after_images, 0)

    #add all captions to a list
    caption_list = [caption_to_sequence(caption, word_to_idx) for caption in captions]

    #add padding = make all captions same length
    padded_caption_list, _ = pad_caption_sequences(caption_list, word_to_idx, max_length=max_length)

    return before_images, after_images, padded_caption_list, captions  # Return captions strings as well


## Dataset Class ##
#-----------------#
#Define custom dataset class, with variables and methods used for loading and proccessing of images
class ImageCaptioningDataset(Dataset):

    #Initialise datase - runs when that database class is instantiated
    #Input: image parent path, path to annotations, word to index map, transform to apply
    def __init__(self, image_dir, annotation_file, word_to_idx, transform=None):
        self.image_dir = image_dir
        self.annotations = self.load_annotations(annotation_file)
        self.word_to_idx = word_to_idx
        self.transform = transform
        self.image_pairs, self.captions = self.load_image_pairs()

    #load annoations json file to a list of annotations
    #input: annotations path
    #output: list of annotations
    def load_annotations(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    #Locates a specific file using an image name
    #input: path of target image folder, target image name
    #output: (if found)target image path or  (if not found)none
    def find_image_file(self, image_set, image_name):
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] #target any extension
        for ext in extensions:
            image_path = os.path.join(self.image_dir, image_set, image_name + ext)
            logging.debug(f"Searching for image: {image_path}")
            if os.path.exists(image_path):
                logging.debug(f"Found image: {image_path}")
                return image_path
        logging.debug(f"Image not found: {image_set}/{image_name}")
        return None

    #Used to load images and captions and match the captions to the target file names
    def load_image_pairs(self):
        image_pairs = [] #store before and after image
        captions = [] #store captions for the pair

        #loop over each file caption in the json list and display progress bar=(TQDM)
        for entry in tqdm(self.annotations, desc="Loading Image Pairs"):
            #add file names in contents section to a list  
            contents = entry.get('contents', []) #/////REVIEW/////

            #if an image doesnt have a corresponding before or after image ignore the image
            if len(contents) < 2:
                logging.info(f"Insufficient image contents for entry: {entry.get('name', 'Unknown')}")
                continue

            #checks if the target set is train or target and stores
            image_set = 'train' if 'train' in entry.get('name', '').lower() else 'val'

            #Get file name for image pair
            after_img_filename = contents[0].get('name', '').replace(f"{image_set}/", '')
            before_img_filename = contents[1].get('name', '').replace(f"{image_set}/", '')

            if not after_img_filename or not before_img_filename:
                logging.info(f"Missing image filenames in entry: {entry.get('name', 'Unknown')}")
                continue

            #finds and stores the before and after images matching the above before/after name
            after_img = self.find_image_file(image_set, after_img_filename)
            before_img = self.find_image_file(image_set, before_img_filename)

            #checks if all data founds and if so adds returns the images + captions
            if after_img and before_img:
                image_pairs.append((before_img, after_img))
                #Extracts each individual caption and adds them to a string ////Review////
                attribute_values = [attr['value'] for attr in entry.get('attributes', []) if 'value' in attr]
                concatenated_captions = ' '.join(attribute_values)
                captions.append(concatenated_captions)
            else: #store missing images/poisoned data
                missing = []
                if not after_img:
                    missing.append(f"{image_set}/{after_img_filename}")
                if not before_img:
                    missing.append(f"{image_set}/{before_img_filename}")
                logging.info(f"Missing images for entry: {entry.get('name', 'Unknown')}. Missing files: {', '.join(missing)}")

        return image_pairs, captions

    #Return total data points in the set
    def __len__(self):
        return len(self.captions)

    #retrieves a data point from the set at specofoed index
    def __getitem__(self, idx):
        before_img_path, after_img_path = self.image_pairs[idx]
        caption = self.captions[idx]

        # Load images
        before_image = Image.open(before_img_path).convert('RGB')
        after_image = Image.open(after_img_path).convert('RGB')

        if self.transform:
            before_image = self.transform(before_image)
            after_image = self.transform(after_image)

        return before_image, after_image, caption  # Return caption string


## Create Vocabulary ##
#--------------------#
#input: list of captions, min frequency reqired to add word to vocab
#output: wordindex map, index word map, size of vocabulary  ////Word index concept ///REVIEW/////
def build_vocabulary(captions, threshold=1):
    tokens = []
    for caption in captions:
        tokens.extend(tokenize_spacy(caption))  # Using spaCy tokenizer
    counter = Counter(tokens)

    #Dont include words less then min frequency
    vocab = [word for word, count in counter.items() if count >= threshold]

    #extra terms
    vocab = ['<pad>', '<start>', '<end>', '<unk>'] + vocab

    #map words to index vice versa
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}

    return word_to_idx, idx_to_word, len(vocab)

#make caption into list of word indecies ///Review///
#input: caption, word idex
#output: list of word index
def caption_to_sequence(caption, word_to_idx):
    tokens = tokenize_spacy(caption)  # Using spaCy tokenizer
    sequence = [word_to_idx.get('<start>')]
    for token in tokens:
        if token in word_to_idx:
            sequence.append(word_to_idx[token])
        else:
            sequence.append(word_to_idx.get('<unk>'))
    sequence.append(word_to_idx.get('<end>'))
    return sequence

#list of caption to word index ///REVIEW///
#input: caption, word index
#output: list of word index
def preprocess_captions(captions, word_to_idx):
    sequences = [caption_to_sequence(caption, word_to_idx) for caption in captions]
    return sequences

#make captions have equal lengths via padding
#input: word index list, word index, max padd length
#output: padded caption, max length
def pad_caption_sequences(sequences, word_to_idx, max_length=None):
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

### CNN STRUCTURE class ####
#Encoder
#-------------------#
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size

        #define Architectre
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # (batch, 64, 112, 112)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (batch, 64, 56, 56)

        #CNN layers
        self.layer1 = self._make_layer(64, 128, blocks=2)   # (batch, 128, 56, 56)
        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2)  # (batch, 256, 28, 28)
        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2)  # (batch, 512, 14, 14)

        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  #Adaptive pooling layer
        self.fc = nn.Linear(512, 512)                      #fully Connected layer   
        self.bn = nn.BatchNorm1d(512, momentum=0.01)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, images):
        x = self.conv1(images)        # (batch, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)           # (batch, 64, 56, 56)

        x = self.layer1(x)            # (batch, 128, 56, 56)
        x = self.layer2(x)            # (batch, 256, 28, 28)
        x = self.layer3(x)            # (batch, 512, 14, 14)

        x = self.adaptive_pool(x)     # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)     # (batch, 512)
        x = self.fc(x)                # (batch, 512)
        x = self.bn(x)                # (batch, 512)
        return x

## LTSM class
#Decorder RNN class (LSTM Model)
#input: dimension of word embedd, dimension of ltsm hidden state, size of vocab, num of ltsm layer
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    #forward propogation
    #input: feature vector, captions
    #output: vocab scores 
    def forward(self, features, captions):
        embeddings = self.embed(captions)  # (batch_size, max_caption_length, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # (batch_size, max_caption_length+1, embed_size)
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
            outputs = self.linear(hiddens.squeeze(1))    # outputs: (1, vocab_size)
            _, predicted = outputs.max(1)                # predicted: (1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)               # inputs: (1, embed_size)
            inputs = inputs.unsqueeze(1)                 # (1, 1, embed_size)
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

def train(model, data_loader, criterion, optimizer, device, epoch, total_epochs, scaler):
    model.train()
    total_loss = 0
    for i, (before_imgs, after_imgs, captions, captions_strings) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}/{total_epochs}")):
        before_imgs = before_imgs.to(device)
        after_imgs = after_imgs.to(device)
        captions = captions.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
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
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

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
        for before_imgs, after_imgs, captions, captions_strings in tqdm(data_loader, desc="Validation"):
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
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

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


def evaluate_model(model, encoder, data_loader, val_captions, word_to_idx, idx_to_word, device, max_length):
    model.eval()
    bleu_scores = []
    caption_iter = iter(val_captions)  # Create an iterator over the original captions
    with torch.no_grad():
        for before_imgs, after_imgs, captions, captions_strings in tqdm(data_loader, desc="Evaluating"):
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
            for i in range(image_features.size(0)):
                try:
                    target_caption = next(caption_iter).lower()
                except StopIteration:
                    target_caption = ''
                generated_caption = generated_captions[i]
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
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),      # Data augmentation
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
    word_to_idx, idx_to_word, vocab_size = build_vocabulary(train_captions, threshold=1)
    print(f"\nVocabulary Size: {vocab_size}")

    # -----------------------------
    # Inspect Vocabulary Coverage
    # -----------------------------
    key_words = ['pineapple', 'rough', 'wider', 'light', 'yellow', 'thorns', 'smoother', 'darker']
    missing_words = [word for word in key_words if word not in word_to_idx]
    print(f"Missing Key Words in Vocabulary: {missing_words}")

    # If any key words are missing, add them manually
    for word in missing_words:
        word_to_idx[word] = len(word_to_idx)
        idx_to_word[len(idx_to_word)] = word
    vocab_size = len(word_to_idx)
    print(f"Updated Vocabulary Size after adding missing words: {vocab_size}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        collate_fn=collate_fn_partial
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important for evaluation
        num_workers=num_workers,
        pin_memory=True,  # Speeds up transfer to GPU
        collate_fn=collate_fn_partial
    )

    # -----------------------------
    # Hyperparameters and Device Configuration
    # -----------------------------
    embed_size = 512  # Updated from 256 to 512 to match EncoderCNN output
    hidden_size = 512
    num_layers = 1
    learning_rate = 1e-4
    num_epochs = 5  # Reduced from 20 to 5 as per your request

    # Enable cuDNN benchmarking for optimized performance
    torch.backends.cudnn.benchmark = True

    # Set device
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

    # Initialize GradScaler for mixed precision
    scaler = amp.GradScaler()

    # -----------------------------
    # Training Loop
    # -----------------------------
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, scaler)
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
    average_bleu = evaluate_model(model, encoder, val_loader, val_dataset.captions, word_to_idx, idx_to_word, device, max_length)

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
        print(f"Reference Caption: {val_dataset.captions[sample_idx].lower()}")

        # -----------------------------
        # Optional: Display in GUI
        # -----------------------------
        display_caption(model, encoder, val_dataset.image_pairs[sample_idx], word_to_idx, idx_to_word, device,
                        max_length)
    else:
        print("Sample index is out of range for the validation dataset.")

if __name__ == "__main__":
    main()
