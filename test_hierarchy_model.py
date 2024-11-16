import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torch.nn as nn
import os
from tqdm import tqdm
import argparse
import string
from PIL import Image
import nltk
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

from hierarchy_model import Encoder, HierarchicalDecoder, EncoderDecoder

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')

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
    padded_references = []
    for refs in references:
        padded_refs = pad_sequence(refs, batch_first=True, padding_value=pad_id)
        padded_references.append(padded_refs)
    return images, captions, padded_references

def generate_caption(model, image, idx2word, device, max_length=20):
    model.eval()
    with torch.no_grad():
        # Encode the image
        encoder_outputs = model.encoder(image.unsqueeze(0).to(device))

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

def validate(model, dataloader, criterion, device, args, idx2word):
    model.eval()
    total_loss = 0
    total_samples = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images, captions, references_batch = batch
            images = images.to(device)
            captions = captions.to(device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)

            loss = criterion(outputs, targets_flat)
            total_loss += loss.item()
            total_samples += captions.size(0)

            # Generate captions for evaluation
            for i in range(images.size(0)):
                image = images[i]
                generated_caption = generate_caption(model, image, idx2word, device, max_length=args.max_seq_length)
                hypotheses.append(generated_caption)

                # Get the reference captions
                refs = []
                for ref in references_batch[i]:
                    ref_tokens = [idx2word[idx.item()] for idx in ref if idx.item() not in [model.decoder.pad_id, model.decoder.sos_id, model.decoder.eos_id]]
                    refs.append(' '.join(ref_tokens))
                references.append(refs)

    average_loss = total_loss / total_samples

    # Prepare data for evaluation metrics
    refs_dict = {idx: refs for idx, refs in enumerate(references)}
    hyps_dict = {idx: [hypotheses[idx]] for idx in range(len(hypotheses))}

    # Compute BLEU scores using NLTK
    bleu1 = corpus_bleu([[ref.split() for ref in refs] for refs in references], [hyp.split() for hyp in hypotheses], weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu([[ref.split() for ref in refs] for refs in references], [hyp.split() for hyp in hypotheses], weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu([[ref.split() for ref in refs] for refs in references], [hyp.split() for hyp in hypotheses], weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu([[ref.split() for ref in refs] for refs in references], [hyp.split() for hyp in hypotheses])

    # Compute CIDEr score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(refs_dict, hyps_dict)

    # Compute ROUGE score
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(refs_dict, hyps_dict)

    # Compute METEOR score
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(refs_dict, hyps_dict)

    # Log the metrics
    print(f"Validation Loss: {average_loss:.4f}")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"CIDEr: {cider_score:.4f}")
    print(f"ROUGE_L: {rouge_score:.4f}")
    print(f"METEOR: {meteor_score:.4f}")

    return average_loss, bleu4, cider_score, rouge_score, meteor_score

def test_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader for testing
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Prepare test dataset and dataloader
    test_dataset = Flickr8kDataset(
        img_dir=args.img_dir,
        split_file=args.test_dir,
        ann_file=args.ann_dir,
        vocab_file=args.vocab_file,
        transform=transform,
        max_seq_length=args.max_seq_length,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Build the model
    encoder = Encoder(
        model_type=args.encoder_type,
        fine_tune=bool(args.fine_tune),
        embed_size=args.embed_size,
    )

    decoder = HierarchicalDecoder(
        vocab_size=test_dataset.vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        pad_id=test_dataset.pad_id,
        sos_id=test_dataset.sos_id,
        eos_id=test_dataset.eos_id,
        # Removed idx2word to fix the error
    )

    model = EncoderDecoder(encoder, decoder)
    model = model.to(device)

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=test_dataset.pad_id)

    # Evaluate on the test set
    test_loss, test_bleu4, test_cider, test_rouge, test_meteor = validate(
        model, test_loader, criterion, device, args, idx2word=test_dataset.idx2word
    )

    print("\nTest Set Evaluation:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test BLEU-4: {test_bleu4:.4f}")
    print(f"Test CIDEr: {test_cider:.4f}")
    print(f"Test ROUGE_L: {test_rouge:.4f}")
    print(f"Test METEOR: {test_meteor:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing Script for Hierarchical Transformer Model')
    parser.add_argument('--img-dir', type=str, default='./dataset/Flickr8k_Dataset/')
    parser.add_argument('--ann-dir', type=str, default='./dataset/Flickr8k_text/Flickr8k.token.txt')
    parser.add_argument('--test-dir', type=str, default='./dataset/Flickr8k_text/Flickr_8k.testImages.txt')
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
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--checkpoint-name', type=str, required=True)
    args = parser.parse_args()

    test_model(args)