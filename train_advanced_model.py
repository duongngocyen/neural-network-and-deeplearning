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
import nltk
from nltk.translate.bleu_score import corpus_bleu
import eval  # Assuming you have an eval.py file with necessary functions
import utils  # Assuming you have a utils.py file with necessary utility functions

from hierarchy_model import Encoder, HierarchicalDecoder, EncoderDecoder

nltk.download('punkt')
nltk.download('wordnet')

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Flickr8kDataset(Dataset):
    """Flickr8k dataset."""

    def __init__(self, img_dir, split_dir, ann_dir, vocab_file, transform=None, max_seq_length=30):
        """
        Args:
            img_dir (string): Directory with all the images.
            split_dir (string): Directory with all the file names which belong to a certain split (train/dev/test).
            ann_dir (string): File containing all captions.
            vocab_file (string): File containing the vocabulary.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_seq_length (int): Maximum sequence length.
        """
        self.img_dir = img_dir
        self.split_dir = split_dir
        self.ann_dir = ann_dir
        self.vocab_file = vocab_file
        self.transform = transform
        self.max_seq_length = max_seq_length

        self.image_file_names, self.captions, self.tokenized_captions = self.tokenizer()
        self.vocab_size = len(self.word2idx)
        self.pad_id = self.word2idx['<pad>']
        self.sos_id = self.word2idx['<start>']
        self.eos_id = self.word2idx['<end>']

    def tokenizer(self):
        # Load vocabulary
        with open(self.vocab_file, 'r') as f:
            vocab = f.read().splitlines()
        # Add special tokens
        vocab = ['<pad>', '<start>', '<end>'] + vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Load split image filenames
        with open(self.split_dir, 'r') as f:
            image_filenames = f.read().splitlines()

        # Load annotations
        captions_dict = {}
        with open(self.ann_dir, 'r') as f:
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

        self.captions_dict = captions_dict  # Store for later use

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

        # Get all reference captions for this image
        references = self.captions_dict[image_name]
        ref_tokenized = []
        for cap in references:
            cap_tokens_ref = self.clean_caption(cap)
            cap_tokens_ref = cap_tokens_ref[: self.max_seq_length - 2]
            cap_token_ids_ref = [self.word2idx.get(word, self.word2idx['<pad>']) for word in cap_tokens_ref]
            cap_token_ids_ref = [self.sos_id] + cap_token_ids_ref + [self.eos_id]
            ref_tokenized.append(torch.tensor(cap_token_ids_ref))

        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'caption': cap_tokens,
            'references': ref_tokenized,
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
    references = [item['references'] for item in batch]
    pad_id = batch[0]['pad_id']
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_id)
    # References need to be padded individually
    max_ref_len = max(len(ref) for refs in references for ref in refs)
    padded_references = []
    for refs in references:
        padded_refs = pad_sequence(refs, batch_first=True, padding_value=pad_id)
        padded_references.append(padded_refs)
    return images, captions, padded_references

def adjust_learning_rate(optimizer, n_iter, warmup_steps, embed_size):
    """
    Adjust the learning rate according to the model size and number of iterations.
    """
    lr = (embed_size ** -0.5) * min(n_iter ** -0.5, n_iter * (warmup_steps ** -1.5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def train_one_epoch(model, dataloader, optimizer, device, n_iter, args):
    model.train()
    total_loss = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(ignore_index=model.decoder.pad_id, reduction='sum')

    for batch in tqdm(dataloader, desc=f"Epoch {n_iter}"):
        images, captions, _ = batch
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

        if args.grad_clip is not None:
            clip_gradient(optimizer, args.grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_samples += captions.size(0)

        # Adjust learning rate
        adjust_learning_rate(optimizer, n_iter, args.warmup_steps, args.embed_size)
        n_iter += 1

    average_loss = total_loss / total_samples
    return average_loss, n_iter

def evaluate_model(model, dataloader, device, args):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, captions, references_batch = batch
            images = images.to(device)
            captions = captions.to(device)

            for i in range(images.size(0)):
                image = images[i]
                generated_caption = generate_caption(model, image, model.decoder.idx2word, device, max_length=args.max_seq_length)
                hypotheses.append(generated_caption)

                # Get the reference captions
                refs = []
                for ref in references_batch[i]:
                    ref_tokens = [model.decoder.idx2word[idx.item()] for idx in ref if idx.item() not in [model.decoder.pad_id, model.decoder.sos_id, model.decoder.eos_id]]
                    refs.append(' '.join(ref_tokens))
                references.append(refs)

    # Compute BLEU score
    bleu4 = corpus_bleu([[ref.split() for ref in refs] for refs in references], [hyp.split() for hyp in hypotheses])

    # You can integrate additional metrics here using your eval.py and utils.py modules

    return bleu4

def generate_caption(model, image, idx2word, device, max_length=20):
    model.eval()
    with torch.no_grad():
        # Encode the image
        encoder_outputs = model.encoder(image.to(device).unsqueeze(0))

        # Initialize the caption with <start> token
        caption = [model.decoder.sos_id]
        caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)

        for _ in range(max_length):
            # Get the embeddings
            embeddings = model.decoder.embedding(caption_tensor)
            embeddings = model.decoder.positional_encoding(embeddings)
            embeddings = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)

            # Prepare masks
            tgt_mask = model.decoder.generate_square_subsequent_mask(embeddings.size(0)).to(device)
            tgt_key_padding_mask = None  # Assuming no padding in the input

            # Decode through all stages
            if model.decoder.decoder_type == 'transformer':
                scene_outputs = model.decoder.scene_decoder(
                    tgt=embeddings,
                    memory=encoder_outputs.permute(1, 0, 2),
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                object_outputs = model.decoder.object_decoder(
                    tgt=scene_outputs,
                    memory=encoder_outputs.permute(1, 0, 2),
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                attribute_outputs = model.decoder.attribute_decoder(
                    tgt=object_outputs,
                    memory=encoder_outputs.permute(1, 0, 2),
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                output = model.decoder.output_layer(attribute_outputs[-1])  # (batch_size, vocab_size)
            else:
                # RNN Decoder
                outputs, _ = model.decoder.lstm(embeddings.permute(1, 0, 2))  # (batch_size, seq_len, hidden_size)
                outputs = model.decoder.attention(outputs)  # (batch_size, seq_len, embed_size)
                output = model.decoder.output_layer(outputs[:, -1, :])  # (batch_size, vocab_size)

            predicted_id = output.argmax(1).item()

            # Add the predicted word to the caption
            caption.append(predicted_id)
            caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)

            # Break if <end> token is generated
            if predicted_id == model.decoder.eos_id:
                break

    # Convert token ids to words
    words = [idx2word[idx] for idx in caption[1:-1]]  # Exclude <start> and <end>
    sentence = ' '.join(words)
    return sentence

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
        split_dir=args.train_dir,
        ann_dir=args.ann_dir,
        vocab_file=args.vocab_file,
        transform=transform,
        max_seq_length=args.max_seq_length,
    )

    val_dataset = Flickr8kDataset(
        img_dir=args.img_dir,
        split_dir=args.val_dir,
        ann_dir=args.ann_dir,
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
        batch_size=args.batch_size_val,
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
        hidden_size=args.decoder_hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        pad_id=train_dataset.pad_id,
        sos_id=train_dataset.sos_id,
        eos_id=train_dataset.eos_id,
        idx2word=train_dataset.idx2word,  # Pass idx2word
        decoder_type=args.decoder_type,
    )

    model = EncoderDecoder(encoder, decoder)
    model = model.to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )

    # Training loop
    best_bleu4 = 0.0
    poor_iters = 0
    n_iter = 1
    epoch = 1

    if args.use_checkpoint:
        checkpoint = torch.load(os.path.join(args.save_dir, args.checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        n_iter = checkpoint['n_iter']
        best_bleu4 = checkpoint['best_bleu4']

    while epoch <= args.num_epochs:
        print(f"Epoch {epoch}/{args.num_epochs}")
        train_loss, n_iter = train_one_epoch(
            model, train_loader, optimizer, device, n_iter, args
        )

        bleu4 = evaluate_model(
            model, val_loader, device, args
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation BLEU-4: {bleu4:.4f}")

        # Save the best model based on BLEU-4 score
        if bleu4 > best_bleu4:
            poor_iters = 0
            best_bleu4 = bleu4
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'n_iter': n_iter,
                    'best_bleu4': best_bleu4,
                },
                os.path.join(args.save_dir, f"{args.experiment_name}_best_model.pth"),
            )
        else:
            poor_iters += 1

        if poor_iters > args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

        epoch += 1

    print(f"Training complete. Best BLEU-4 score: {best_bleu4:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Script for Hierarchical Transformer Model')
    parser.add_argument('--img-dir', type=str, default='./dataset/Flickr8k_Dataset/')
    parser.add_argument('--ann-dir', type=str, default='./dataset/Flickr8k_text/Flickr8k.token.txt')
    parser.add_argument('--train-dir', type=str, default='./dataset/Flickr8k_text/Flickr_8k.trainImages.txt')
    parser.add_argument('--val-dir', type=str, default='./dataset/Flickr8k_text/Flickr_8k.devImages.txt')
    parser.add_argument('--test-dir', type=str, default='./dataset/Flickr8k_text/Flickr_8k.testImages.txt')
    parser.add_argument('--vocab-file', type=str, default='./vocab.txt')
    parser.add_argument('--encoder-type', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet101'])
    parser.add_argument('--fine-tune', type=int, choices=[0, 1], default=0)
    parser.add_argument('--embed-size', type=int, default=512)
    parser.add_argument('--decoder-hidden-size', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-seq-length', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--batch-size-val', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--save-dir', type=str, default='./model_saves/')
    parser.add_argument('--experiment-name', type=str, default='hierarchical_transformer_experiment')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--decoder-type', choices=['transformer', 'rnn'], default='transformer')
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--use-checkpoint', type=int, default=0)
    parser.add_argument('--checkpoint-name', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_model(args)