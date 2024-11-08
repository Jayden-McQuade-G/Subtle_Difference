
#main.py

#Import Library Modules
import os
import json
import spacy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from collections import Counter
from tqdm import tqdm

import logging
from functools import partial
from torch.cuda import amp

import tkinter as tk

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    

### Initialize spaCy and Logging ###
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
    print("'en_core_web_sm' not found.... Downloading")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

#Use spaCy library to tokenise input text
#input: target "text"
#output: list of tokenised text
def tokenize_text(target_txt):
    var = nlp(target_txt.lower())
    return [token.text for token in var]



## Dataset Class ##
#Define custom dataset class, with variables and methods used for loading and proccessing of images
class ImageCaptioningDataset(Dataset):

    #Initialise datase - runs when that database class is instantiated
    #Input: image parent path, path to annotations, word to index map, transform to apply
    def __init__(self, img_folder_path, annotation_file, word_index, transform=None):

        self.img_folder_path = img_folder_path
        self.annotations = self.load_annotations(annotation_file)
        self.word_index = word_index
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
            image_path = os.path.join(self.img_folder_path, image_set, image_name + ext)
            print(f"Searching for image: {image_path}")
            if os.path.exists(image_path):
                print(f"Image Found: {image_path}")
                return image_path
            
        print(f"Image not found: {image_set}/{image_name}")
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
    def __getitem__(self, index):

        before_img_path, after_img_path = self.image_pairs[index]
        caption = self.captions[index]

        #load images
        before_image = Image.open(before_img_path).convert('RGB')
        after_image = Image.open(after_img_path).convert('RGB')

        if self.transform:
            before_image = self.transform(before_image)
            after_image = self.transform(after_image)

        return before_image, after_image, caption  # Return caption string

#Collate function
#Formats images and captions to correctly sized tensors
#Input: List of data (before_image, after_image, caption), word to index dictionary, maximum length for padding
#Output: correctly formatted Tensors, before_image, after_image, captions, list of string captions
def collate_fn(data, word_index, max_length):
    before_images, after_images, captions = zip(*data)

    #Stack images
    before_images = torch.stack(before_images, 0)
    after_images = torch.stack(after_images, 0)
    #add all captions to a list
    caption_list = [caption_to_sequence(caption, word_index) for caption in captions]

    #add padding = make all captions same length
    padded_captions, _ = pad_caption_sequences(caption_list, word_index, max_length=max_length)

    return before_images, after_images, padded_captions, captions


## Create Vocabulary ##
#input: list of captions, min frequency reqired to add word to vocab
#output: wordindex map, index word map, size of vocabulary  ////Word index concept ///REVIEW/////
def build_vocabulary(captions, threshold=1):
    tokens = []
    for caption in captions:
        tokens.extend(tokenize_text(caption))
    counter = Counter(tokens)

    #Dont include words less then min frequency
    vocab = [word for word, count in counter.items() if count >= threshold]

    #extra terms
    vocab = ['<pad>', '<start>', '<end>', '<unk>'] + vocab

    #map words to index vice versa
    word_index = {word: index for index, word in enumerate(vocab)}
    index_word = {index: word for index, word in enumerate(vocab)}

    return word_index, index_word, len(vocab)

#make caption into list of word indecies ///Review///
#input: caption, word idex
#output: list of word index
def caption_to_sequence(caption, word_index):
    tokens = tokenize_text(caption)
    sequence = [word_index.get('<start>')]
    for token in tokens:
        if token in word_index:
            sequence.append(word_index[token])
        else:
            sequence.append(word_index.get('<unk>'))
    sequence.append(word_index.get('<end>'))
    return sequence

#list of caption to word index ///REVIEW///
#input: caption, word index
#output: list of word index
def preprocess_captions(captions, word_index):
    caption_list = [caption_to_sequence(caption, word_index) for caption in captions]

    return caption_list

#make captions have equal lengths via padding
#input: word index list, word index, max padd length
#output: padded caption, max length
def pad_caption_sequences(sequences, word_index, max_length=None):
    if not max_length:
        max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            seq = seq + [word_index['<pad>']] * (max_length - len(seq))
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        #CNN layers
        self.layer1 = self._make_layer(64, 128, blocks=2)   
        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2) 
        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2)  

        
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
        x = self.conv1(images)        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)           

        x = self.layer1(x)            
        x = self.layer2(x)            
        x = self.layer3(x)            

        x = self.adaptive_pool(x)    
        x = x.view(x.size(0), -1)     
        x = self.fc(x)                
        x = self.bn(x)  

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
        embeddings = self.embed(captions)  
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  
        embeddings = self.dropout(embeddings)
        hiddens, _ = self.lstm(embeddings)  
        outputs = self.linear(hiddens) 

        return outputs


    #generate captions for image features with greedy search 
    #input: feature vector, index word, max len
    #output: generated caption
    def sample(self, features, word_index, index_word, max_length=20):
        sampled_ids = []
        inputs = features.unsqueeze(1)  
        states = None

        for i in range(max_length):
            hiddens, states = self.lstm(inputs, states)  
            outputs = self.linear(hiddens.squeeze(1))    
            _, predicted = outputs.max(1)                
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)               
            inputs = inputs.unsqueeze(1)                 
            if index_word.get(predicted.item(), '<unk>') == '<end>':
                break

        sampled_caption = []
        for word_id in sampled_ids:
            word = index_word.get(word_id, '<unk>')
            if word == '<end>':
                break
            sampled_caption.append(word)
        generated_caption = ' '.join(sampled_caption)

        return generated_caption

#custom imgae captioning class for difference captioning
class ImageCaptioningModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, before_images, after_images, captions):

        #Create feature vertexes of each image
        before_features = self.encoder(before_images)
        after_features = self.encoder(after_images)

        #Calculate degree of difference by subtracting images
        image_features = after_features - before_features  # (batch_size, embed_size)

        #create sequence of captions detailing the difference between two images
        outputs = self.decoder(image_features, captions)

        return outputs


## Training Utilities
#Contains code for training and validating the model

#Training 
def train(model, data_loader, criterion, optimizer, proccess_unit, epoch, total_epochs, scaler):
    #initialise model training, set to train state
    model.train()
    total_loss = 0 #used to track loss accross batches may help us improve certain portions ///Review///
    for i, (before_imgs, after_imgs, captions, captions_strings) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}/{total_epochs}")):
        before_imgs = before_imgs.to(proccess_unit)
        after_imgs = after_imgs.to(proccess_unit)
        captions = captions.to(proccess_unit)

        #Refreshes gradients, resolve pytorch gradient accumulation.
        optimizer.zero_grad()

        #This part may improve performance///Review///
        with torch.cuda.amp.autocast(proccess_unit=='cuda'):
            outputs = model(before_imgs, after_imgs, captions[:, :-1])
            targets = captions[:, 1:]

            #Adjust outputs to remove the first time step
            outputs = outputs[:, 1:, :]

            #Adjust outputs and targets to the same length
            if outputs.size(1) != targets.size(1):
                min_length = min(outputs.size(1), targets.size(1))
                outputs = outputs[:, :min_length, :]
                targets = targets[:, :min_length]

            #calculates loss with criterion
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

        #backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()


        total_loss += loss.item() #append loss to total

        if (i + 1) % 100 == 0:
            print(f"Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")

    #calculate average loss and display to console
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch}/{total_epochs}], Average Loss: {avg_loss:.4f}")

    return avg_loss

#Validation code
def validate(model, data_loader, criterion, proccess_unit):
    #initiate model validation
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for before_imgs, after_imgs, captions, captions_strings in tqdm(data_loader, desc="Validation"):
            before_imgs = before_imgs.to(proccess_unit)
            after_imgs = after_imgs.to(proccess_unit)
            captions = captions.to(proccess_unit)

            #Forward pass
            outputs = model(before_imgs, after_imgs, captions[:, :-1])
            targets = captions[:, 1:]

            outputs = outputs[:, 1:, :]  

            #Adjust outputs and targets to the same length
            if outputs.size(1) != targets.size(1):
                min_length = min(outputs.size(1), targets.size(1))
                outputs = outputs[:, :min_length, :]
                targets = targets[:, :min_length]

            #calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            total_loss += loss.item()

    #calc average loss display to console
    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss - {avg_loss:.4f}")

    return avg_loss


## Evaluate code ## ///Review///
#---------------------------#

#calculate bleu score between reference an generated captions
#input: reference captions and gnerated captions
#outputs score of caption quality
def compute_bleu(ref_caption, gen_caption):

    ref_tokens = tokenize_text(ref_caption)
    gen_tokens = tokenize_text(gen_caption)

    ref_caption = ref_tokens
    gen_caption = gen_tokens

    smoothie = SmoothingFunction().method4
    score = sentence_bleu(ref_caption, gen_caption, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return score

#Consensus-based Image Description Evaluation
#evaluate the CIDEr score on the model
#input: reference caption, generated caption
#output: cider score
def compute_cider(ref_list, gen_list):
    #put in dictionary format
    gen_dict = {}
    ref_dict = {}

    for x in  range(len(gen_list)):
        gen_dict[x] = [{'image_id': x, 'caption': gen_list[x]}]
        ref_dict[x] = [{'image_id': x, 'caption': ref_list[x]}]

      #tokenise dictionaries with PTB
    ptb_tokenizer = PTBTokenizer()
    gen_dict = ptb_tokenizer.tokenize(gen_dict)
    ref_dict = ptb_tokenizer.tokenize(ref_dict)

    #use pycocoevalcap library to calculate Cider score
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(ref_dict, gen_dict)

    return score

#evaluate model calls all evaluation methods, preproccesses inputs and provides outputs
def evaluate_model(model, encoder, data_loader, val_captions, word_index, index_word, proccess_unit, max_length):
    model.eval()
    bleu_scores = []
    cider_scores = []

    for index in range(len(val_captions)):
        val_captions[index] = val_captions[index].lower()

    with torch.no_grad():
        for before_imgs, after_imgs, captions, captions_strings in tqdm(data_loader, desc="Evaluating"):
            before_imgs = before_imgs.to(proccess_unit)
            after_imgs = after_imgs.to(proccess_unit)

            #image encoding
            before_features = encoder(before_imgs)
            after_features = encoder(after_imgs)
            image_features = after_features - before_features  # Feature difference

            #generate captions
            gen_captions = []
            for i in range(image_features.size(0)):
                feature = image_features[i].unsqueeze(0).to(proccess_unit)
                caption = model.decoder.sample(feature, word_index, index_word, max_length)
                gen_captions.append(caption.lower())

            #Calculate BLEU scoresres
            for i in range(image_features.size(0)):
                try:
                    ref_caption = next(iter(val_captions))
                except StopIteration:
                    ref_caption = ''
                gen_caption = gen_captions[i]
                #bleu
                bleu_score = compute_bleu(ref_caption, gen_caption)
                bleu_scores.append(bleu_score)

            #calculate CIDEr score
            cider_score = compute_cider(val_captions, gen_captions)
            cider_scores.append(cider_score)

    #calculate score averages
    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    average_cider = sum(cider_scores) / len(cider_scores) if cider_scores else 0

    print(f"bleu score average: {average_bleu:.4f}")
    print(f"CIDEr score average: {average_cider:.4f}")

    return average_bleu, average_cider


#generate captions function
# -----------------------------

#generates captions for an image pair, define differences
#input: captioning model, before_image, after_image, wordindex map, indexword map , proccessing unit, captionlength
#output the generated caption
def generate_caption(model, encoder, before_image, after_image, word_index, index_word, proccess_unit, max_length=20):

    model.eval()  #initialise model validation
    with torch.no_grad():

        #define image transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        #proccess images via transform, change dimensions + structure
        before_tensor = transform(before_image).unsqueeze(0).to(proccess_unit)
        after_tensor = transform(after_image).unsqueeze(0).to(proccess_unit)

        #Encoding feature vertex for use
        before_features = encoder(before_tensor)
        after_features = encoder(after_tensor)
        image_features = after_features - before_features 

        #decode features to produce captions
        generated_caption = model.decoder.sample(image_features, word_index, index_word, max_length)

    return generated_caption


#Graphical user interface //review implementation of Nics Code//
# -----------------------------
#input: image caption model, CNN encoder, list(before_img, after_img), word index map, index word map, proccess_unit, max length of caption
def displayGUI(model, encoder, image_pair, word_index, index_word, proccess_unit, max_length):

    before_img_path, after_img_path = image_pair
    before_image = Image.open(before_img_path).convert('RGB')
    after_image = Image.open(after_img_path).convert('RGB')

    #generate caption
    generated_caption = generate_caption(model, encoder, before_image, after_image, word_index, index_word, proccess_unit,
                                         max_length)

    #load before and after images
    img_before = before_image.resize((244, 244))
    img_after = after_image.resize((244, 244))

    #create tkinter window
    root = tk.Tk()
    root.title("Conditional Image Difference Captioning Task")

    #before image widget
    tk_before = ImageTk.PhotoImage(img_before)
    label_before = tk.Label(root, image=tk_before)
    label_before.image = tk_before 
    label_before.pack(side="left", padx=10, pady=10)
    label_before_title = tk.Label(root, text="Before")
    label_before_title.pack(side="left")

    #after image widget
    tk_after = ImageTk.PhotoImage(img_after)
    label_after = tk.Label(root, image=tk_after)
    label_after.image = tk_after
    label_after.pack(side="left", padx=10, pady=10)
    label_after_title = tk.Label(root, text="After")
    label_after_title.pack(side="left")

    #captions widget
    caption_text = f"Generated Caption:\n{generated_caption}"
    label_caption = tk.Label(root, text=caption_text, wraplength=600, justify="left", font=("Helvetica", 12))
    label_caption.pack(pady=10)

    root.mainloop()


# main function code
#----------------------
def main():

    #set variables
    #path to dataset files, images, train annotations and val annotations
    img_folder_path = '../Train-Val-DS'
    train_annotation_path = '../train-val-annotations/capt_train_annotations.json'
    val_annotation_path = '../train-val-annotations/capt_val_annotations.json'
    
    batch_size = 16
    num_workers = 4  #leveraging Multithreading - Default = 4 

   #confirm annotation files
    if not os.path.isfile(train_annotation_path):
        print(f"Error: Training annotations file not found'{train_annotation_path}'.")
        return
    if not os.path.isfile(val_annotation_path):
        print(f"Error: Validation annotations file not found '{val_annotation_path}'.")
        return

    #define transform images to correct formatt
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),      
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #load train and val annotations into lists via json.load
    with open(train_annotation_path, 'r') as f:
        train_annotations = json.load(f)
    with open(val_annotation_path, 'r') as f:
        val_annotations = json.load(f)

    
    #Debug code view first few annotations
    print("\nview first train annotation (debug):")
    print(f"Train #1 Keys: {list(train_annotations[0].keys())}")
    print(f"Train #1 Attributes: {[attr['key'] for attr in train_annotations[0].get('attributes', [])]}")

    print("\nView first Val annotation (debug):")
    print(f"Validate #1 Keys: {list(val_annotations[0].keys())}")
    print(f"Validate #1 Attributes: {[attr['key'] for attr in val_annotations[0].get('attributes', [])]}")
    

    #proccess train captions, to formatted list for proccessing
    try:
        train_captions = [' '.join([attr['value'] for attr in entry.get('attributes', []) if 'value' in attr]) for entry
                          in train_annotations]
    except KeyError as e:
        print(f"Error: key missing in training annotations: {e}")
        print("Each entry in the training annotation JSON must have 'attributes' key with 'value' fields.")
        return

    #process validation captions to formatted list for proccessing
    try:
        val_captions = [' '.join([attr['value'] for attr in entry.get('attributes', []) if 'value' in attr]) for entry
                        in val_annotations]
    except KeyError as e:
        print(f"Error: Key missing in validation annotations: {e}")
        print("Please ensure each entry in the validation annotation JSON has an 'attributes' key with 'value' fields.")
        return

    #create vocabulary
    word_index, index_word, vocab_size = build_vocabulary(train_captions, threshold=1)
    print(f"\nVocabulary Size: {vocab_size}")

    #preproccess captions
    train_sequences = preprocess_captions(train_captions, word_index)
    val_sequences = preprocess_captions(val_captions, word_index)

    #padding captions lists to ensure they are the same length
    train_padded, max_length = pad_caption_sequences(train_sequences, word_index, max_length=None)
    val_padded, _ = pad_caption_sequences(val_sequences, word_index, max_length=max_length)
    print(f"Max Caption Sequence Length: {max_length}")

    #instantiate image datasets
    train_dataset = ImageCaptioningDataset(img_folder_path, train_annotation_path, word_index, transform=transform)
    val_dataset = ImageCaptioningDataset(img_folder_path, val_annotation_path, word_index, transform=transform)

    #debug check if datasets loaded correctly ///review///
    if len(train_dataset) == 0:
        print("Error loading Training set.")
        return
    if len(val_dataset) == 0:
        print("Error loading Validating set.")
        return

    #partial collate function, removes collate wrapper to allow us to use collate and prefill wordindex map and max length
    collate_fn_partial = partial(collate_fn, word_index=word_index, max_length=max_length)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, #shuffle set (True helps reduce over fitting)
        num_workers=num_workers,
        pin_memory=True,  #improves memory transfer between GPU and CPU
        collate_fn=collate_fn_partial
    )
    val_loader = DataLoader( 
        val_dataset,
        batch_size=batch_size,
        shuffle=False, #shuffle set (False maintains consistency accross validation)
        num_workers=num_workers,
        pin_memory=True,  #improves memory transfer between GPU and CPU
        collate_fn=collate_fn_partial
    )
    
    #Model configuration
    embed_size = 512 
    hidden_size = 512
    num_layers = 1
    learning_rate = 1e-4
    num_epochs = 2 ### Default to 5

    #cuDNN benchmarking optimises performance for ML tasks - Targets NVIDIA GPU (LUKE)
    torch.backends.cudnn.benchmark = True

    #confirms GPU availability and sets proccess_unit to CPU or GPU designed to endure GPU functionality is properly implemented
    proccess_unit = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {proccess_unit}")

    #define encoder and decoder variables
    encoder = EncoderCNN()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    #sets models to use target proccessing unit
    encoder = encoder.to(proccess_unit)
    decoder = decoder.to(proccess_unit)

    #image captioning model
    model = ImageCaptioningModel(encoder, decoder).to(proccess_unit)

    #Loss and Optimisation
    criterion = nn.CrossEntropyLoss(ignore_index=word_index['<pad>'])
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    #GrandScalar to increase performance ///Review///
    scaler = amp.GradScaler()

    #Training loop based on defined number of Epochs
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, proccess_unit, epoch, num_epochs, scaler)
        val_loss = validate(model, val_loader, criterion, proccess_unit)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    #plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #evaluate model performanace
    average_bleu, average_cider = evaluate_model(model, encoder, val_loader, val_dataset.captions, word_index, index_word, proccess_unit, max_length)

    #Debug create sample captions ///review///
    index = 0
    if index < len(val_dataset):

        before_img_path, after_img_path = val_dataset.image_pairs[index]
        before_image = Image.open(before_img_path).convert('RGB')
        after_image = Image.open(after_img_path).convert('RGB')

        generated_caption = generate_caption(model, encoder, before_image, after_image, word_index, index_word,
                                             proccess_unit, max_length)
        print(f"\nGenerated Caption: {generated_caption}")
        print(f"Reference Caption: {val_dataset.captions[index].lower()}")

        #Call Gui to display images and captions --- ERRORS WITH GUI MUST FIX
        displayGUI(model, encoder, val_dataset.image_pairs[index], word_index, index_word, proccess_unit,
                        max_length)
    else:
        print("Sample index is out of range for the validation dataset.")

if __name__ == "__main__":
    main()

