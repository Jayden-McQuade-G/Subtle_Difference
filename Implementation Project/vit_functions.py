import math
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from imutils import paths


from torch.utils.data import Dataset
from imutils import paths


#Define Dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Load all image paths
        self.image_paths = list(paths.list_images(image_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read the image
        img_path = self.image_paths[idx]
        image = read_image(img_path)

        # Ensure the image is in [C, H, W] format
        if image.dim() == 2:  # If the image is grayscale
            image = image.unsqueeze(0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        return image
    
#Convert Images into Patches
class PatchEmbeddings():
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels  = config["num_channels"]
        self.hidden_size = config ["hidden_size"]

        #Calc number of patches 
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
#Combine patch embedding with class token and position embeddings
class Embeddings():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)

        #learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1,1, config["hidden_size"]))

        #create Position embeddings for CLS token and Patch embeddings
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.ddropout == nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size,_, _ = x.size()
        # Expand the [CLS] Token to the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # concat the CLS token to beginning of input sequence
        # results in sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

#takes a series of embeddings, computs query, key and value vectors for each embedding
#Single Attention Head
class AttentionHead():
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self,hidden_size = hidden_size
        self.attention_head_size = hidden_size
        
        #Create query, key and value projection layers 
        self.query = nn.Linear(hidden_size, attention_head_size, bias = bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias = bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias = bias)

        self.dropout == nn.Dropout(dropout)

    #project input into query, key and value, same input ti generate vakues, self attention
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        #Calc attention Scoores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size )
        attention_probs = nn.functional.softmax(attention_probs)

        #calc attention output 
        attention_output = torch.matmul(attention_probs, value)
        return(attention_output, attention_probs)
    
class MultiHeadAttention():
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        #the attention head size = hidden size / number if attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_headsself.all_head_size
        self.all_head_size = self.num_attention_heads  * self.attention_head_size

        #decide to use bias or not
        self.qkv_bias = config["qkv_bias"]

        #create list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        #create linear layer to project the attention output back to the hiddensize, most the time all_head_size == hidden_size
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout == nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        #Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]

        #concat the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        #Project the concat attention output back to hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        #return the attention output and the attention probabilities 
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


    class MultiLayerPerceptron():
        pass

    class Block():
        pass

    class Encoder():
        pass

    class ViTForClassfication():
        pass



