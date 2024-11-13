import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from collections import Counter
import nltk
nltk.download('punkt')  # Ensure NLTK's tokenizer is available


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>",
                     3: "<unk>"}  # Index-to-string mapping
        # String-to-index mapping
        self.stoi = {v: k for k, v in self.itos.items()}
        self.index = 4  # Start indexing new words from 4

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in nltk.tokenize.word_tokenize(sentence.lower()):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = self.index
                    self.itos[self.index] = word
                    self.index += 1

    def numericalize(self, text):
        tokenized_text = nltk.tokenize.word_tokenize(text.lower())
        return [self.stoi.get(word, self.stoi["<unk>"]) for word in tokenized_text]


class Flickr8KDataset(Dataset):
    def __init__(self, img_dir, captions_file, vocab, transform=None, max_length=20):
        """
        Args:
            img_dir (string): Directory with all the images.
            captions_file (string): Path to the captions CSV file.
            vocab (Vocabulary): The Vocabulary object.
            transform (callable, optional): Optional transform to be applied on an image.
            max_length (int): Maximum length of captions.
        """
        self.img_dir = img_dir
        self.captions_file = captions_file
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length
        self.img_captions = self.load_captions()

    def load_captions(self):
        captions = []
        with open(self.captions_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                img_name = parts[0]
                caption = parts[1]
                captions.append((img_name, caption))
        return captions

    def __len__(self):
        return len(self.img_captions)

    def __getitem__(self, idx):
        img_name, caption = self.img_captions[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Convert caption to tokenized, padded format
        numerical_caption = [self.vocab.stoi["<start>"]] + \
            self.vocab.numericalize(caption) + \
                            [self.vocab.stoi["<end>"]]

        # Pad caption
        if len(numerical_caption) < self.max_length:
            numerical_caption += [self.vocab.stoi["<pad>"]
                                  ] * (self.max_length - len(numerical_caption))
        else:
            numerical_caption = numerical_caption[:self.max_length]

        return image, torch.tensor(numerical_caption)


# Parameters
img_dir = '/home/ducanh/neural-network-and-deeplearning/dataset/Images'
captions_file = '/home/ducanh/neural-network-and-deeplearning/dataset/captions.txt'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize Vocabulary and build from captions
vocab = Vocabulary(freq_threshold=5)

# Load all captions to build the vocabulary
with open(captions_file, 'r') as file:
    all_captions = [line.strip().split(',')[1] for line in file]

vocab.build_vocabulary(all_captions)

# Initialize Dataset and DataLoader
dataset = Flickr8KDataset(
    img_dir=img_dir, captions_file=captions_file, vocab=vocab, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Check example
for images, captions in dataloader:
    print(images.shape)  # Should be [32, 3, 224, 224]
    print(captions.shape)  # Should be [32, max_length]
    break
