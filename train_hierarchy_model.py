import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torch.nn as nn
import os
from tqdm import tqdm
import copy
import argparse
import string
from PIL import Image

from hierarchy_model import Encoder, HierarchicalDecoder, EncoderDecoder
import utils  # Assuming you have a utils.py file with necessary utility functions

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class Flickr8kDataset(Dataset):
    """Flickr8k dataset."""

    def __init__(self, img_dir, split_file, ann_file, vocab_file, transform=None, max_seq_length=30):
        self.img_dir = img_dir
        self.split_file = split_file
        self.ann_file = ann_file
        self.vocab_file = vocab_file
        self.transform = transform
        self.max_seq_length = max_seq_length

        # Load vocabulary
        with open(self.vocab_file, 'r') as f:
            vocab = f.read().splitlines()
        vocab = ['<pad>', '<start>', '<end>'] + vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.pad_id = self.word2idx['<pad>']
        self.sos_id = self.word2idx['<start>']
        self.eos_id = self.word2idx['<end>']

        self.image_file_names, self.captions, self.tokenized_captions = self.tokenizer()


    def tokenizer(self):
        # Load split image filenames
        with open(self.split_file, 'r') as f:
            image_filenames = f.read().splitlines()

        # Load annotations
        captions_dict = {}
        with open(self.ann_file, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) < 2:
                    continue
                filename, caption = tokens[0], tokens[1]
                filename = filename.split('#')[0]
                if filename in image_filenames:
                    if filename not in captions_dict:
                        captions_dict[filename] = []
                    captions_dict[filename].append(caption)

        image_file_names = []
        captions = []
        tokenized_captions = []
        for filename, caps in captions_dict.items():
            for cap in caps:
                image_file_names.append(filename)
                cap_tokens = self.clean_caption(cap)
                # Truncate the caption tokens
                cap_tokens = cap_tokens[: self.max_seq_length - 2]
                captions.append(cap_tokens)
                cap_token_ids = [self.word2idx.get(word, self.word2idx['<pad>']) for word in cap_tokens]
                cap_token_ids = [self.sos_id] + cap_token_ids + [self.eos_id]
                tokenized_captions.append(torch.tensor(cap_token_ids))
        return image_file_names, captions, tokenized_captions

    def clean_caption(self, caption):
        # Lowercase and remove punctuation
        caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = caption.split()
        return tokens

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        image_name = self.image_file_names[idx]
        cap_tokens = self.tokenized_captions[idx]

        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'caption': cap_tokens,
            'pad_id': self.pad_id,
            'sos_id': self.sos_id,
            'eos_id': self.eos_id,
            'max_seq_length': self.max_seq_length
        }
        return sample

def collate_fn(batch):
    """
    Custom collate function to pad sequences and handle batching.
    """
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    pad_id = batch[0]['pad_id']
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_id)
    return images, captions

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        images, captions = batch
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        # Shift captions for teacher forcing
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        outputs = model(images, inputs)
        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_samples += captions.size(0)

    average_loss = total_loss / total_samples
    return average_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images, captions = batch
            images = images.to(device)
            captions = captions.to(device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_samples += captions.size(0)

    average_loss = total_loss / total_samples
    return average_loss

def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets and dataloaders
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = Flickr8kDataset(
        img_dir=args.img_dir,
        split_file=args.train_dir,
        ann_file=args.ann_dir,
        vocab_file=args.vocab_file,
        transform=transform,
        max_seq_length=args.max_seq_length,

    )

    val_dataset = Flickr8kDataset(
        img_dir=args.img_dir,
        split_file=args.val_dir,
        ann_file=args.ann_dir,
        vocab_file=args.vocab_file,
        transform=transform,
        max_seq_length=args.max_seq_length,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Build the model
    encoder = Encoder(
        model_type=args.encoder_type,
        fine_tune=bool(args.fine_tune),
        embed_size=args.embed_size,  # Pass the embed_size
    )

    decoder = HierarchicalDecoder(
        vocab_size=train_dataset.vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        pad_id=train_dataset.pad_id,
        sos_id=train_dataset.sos_id,
        eos_id=train_dataset.eos_id,
    )

    model = EncoderDecoder(encoder, decoder)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )

    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                },
                os.path.join(args.save_dir, f"{args.experiment_name}_best_model.pth"),
            )

        # Adjust learning rate if needed
        # utils.adjust_learning_rate(optimizer, epoch, args.learning_rate_decay)

    print(f"Training complete. Best validation loss: {best_loss:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Script for Hierarchical Transformer Model')
    parser.add_argument('--img-dir', type=str, default='./dataset/Flickr8k_Dataset/')
    parser.add_argument('--ann-dir', type=str, default='./dataset/Flickr8k_text/Flickr8k.token.txt')
    parser.add_argument('--train-dir', type=str, default='./dataset/Flickr8k_text/Flickr_8k.trainImages.txt')
    parser.add_argument('--val-dir', type=str, default='./dataset/Flickr8k_text/Flickr_8k.devImages.txt')
    parser.add_argument('--vocab-file', type=str, default='./vocab.txt')
    parser.add_argument('--encoder-type', type=str, default='resnet50')
    parser.add_argument('--fine-tune', type=int, choices=[0, 1], default=0)
    parser.add_argument('--embed-size', type=int, default=512)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-seq-length', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--save-dir', type=str, default='./model_saves/')
    parser.add_argument('--experiment-name', type=str, default='hierarchical_transformer_experiment')
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_model(args)
