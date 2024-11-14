import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer
from PIL import Image
import argparse
import logging
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dataset Class
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, caption_file, split_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.captions = self.load_captions(caption_file, split_file)

    def load_captions(self, caption_file, split_file):
        with open(split_file, "r") as f:
            image_filenames = set(f.read().strip().split("\n"))

        captions = {}
        with open(caption_file, "r") as f:
            for line in f:
                img_name, caption = line.split("\t")
                img_name = img_name.split("#")[0]
                if img_name in image_filenames:
                    if img_name not in captions:
                        captions[img_name] = []
                    captions[img_name].append(caption.strip())

        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = list(self.captions.keys())[idx]
        caption = self.captions[img_name][0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "caption": caption}

# Encoder with ResNet18
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, hidden_size)

    def forward(self, images):
        features = self.resnet(images)  # Spatial features
        pooled_features = self.adaptive_pool(features)
        global_features = pooled_features.view(pooled_features.size(0), -1)  # Global features
        return self.fc(global_features), features

# Hierarchical Attention Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_heads, num_layers, max_seq_len):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, captions, encoder_output, mask=None):
        seq_len = captions.size(1)
        # Adjust positional encoding dynamically
        if seq_len > self.positional_encoding.size(1):
            # Extend the positional encoding dynamically
            extra_positions = torch.zeros(1, seq_len - self.positional_encoding.size(1), self.embedding.embedding_dim).to(self.positional_encoding.device)
            self.positional_encoding = torch.cat([self.positional_encoding, extra_positions], dim=1)

        embedded_captions = self.embedding(captions) + self.positional_encoding[:, :seq_len, :]

        
        embedded_captions = self.embedding(captions) + self.positional_encoding[:, :seq_len, :]
        encoder_output = encoder_output.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        embedded_captions = embedded_captions.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        output = self.transformer_decoder(embedded_captions, encoder_output, tgt_mask=mask)
        output = output.permute(1, 0, 2)  # Back to (batch_size, seq_len, embed_size)
        return self.fc(output)

    def generate(self, global_features, max_seq_len, start_token_id, end_token_id):
        device = global_features.device
        batch_size = global_features.size(0)
        generated_captions = torch.full((batch_size, max_seq_len), start_token_id, dtype=torch.long).to(device)

        for t in range(1, max_seq_len):
            mask = torch.triu(torch.ones((t, t), dtype=torch.bool)).to(device)
            outputs = self.forward(generated_captions[:, :t], global_features.unsqueeze(1).repeat(1, t, 1), mask)
            predicted_tokens = torch.argmax(outputs[:, -1, :], dim=-1)
            generated_captions[:, t] = predicted_tokens
            if (predicted_tokens == end_token_id).all():
                break

        return generated_captions

# Full Hierarchical Attention Model
class HierarchicalAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_size, encoder_hidden_size, decoder_hidden_size, num_layers, num_heads, max_seq_len):
        super(HierarchicalAttentionModel, self).__init__()
        self.encoder = Encoder(encoder_hidden_size)
        self.decoder = TransformerDecoder(vocab_size, embed_size, decoder_hidden_size, num_heads, num_layers, max_seq_len)

    def forward(self, images, captions):
        encoder_features, _ = self.encoder(images)
        outputs = self.decoder(captions, encoder_features)
        return outputs

# Scoring Functions
def compute_metrics(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(references, hypotheses)
        if isinstance(score, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

# Training Function
def train(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = len(tokenizer.vocab)

    model = HierarchicalAttentionModel(
        vocab_size, args.embed_size, args.encoder_hidden_size, args.decoder_hidden_size, args.num_layers, args.num_heads, args.max_seq_len
    )
    model.to(args.device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Flickr8kDataset(args.img_dir, args.caption_file, args.train_split, transform)
    val_dataset = Flickr8kDataset(args.img_dir, args.caption_file, args.val_split, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        logger.info(f"Epoch {epoch + 1}/{args.epochs} started.")

        for batch in train_dataloader:
            images = batch["image"].to(args.device)
            captions = tokenizer(batch["caption"], padding=True, return_tensors="pt").input_ids.to(args.device)

            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} completed. Average Loss: {avg_loss}")

        model.eval()
        references, hypotheses = [], []
        eval_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch["image"].to(args.device)
                captions = tokenizer(batch["caption"], padding=True, return_tensors="pt").input_ids.to(args.device)

                references.extend(batch["caption"])
                outputs = model(images, captions[:, :-1])
                eval_loss += criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1)).item()
                generated_outputs = model(images, captions=None)
                hypotheses.extend(tokenizer.batch_decode(generated_outputs, skip_special_tokens=True))

        avg_eval_loss = eval_loss / len(val_dataloader)
        logger.info(f"Validation Loss: {avg_eval_loss}")

        metrics = compute_metrics({i: [r] for i, r in enumerate(references)}, {i: [h] for i, h in enumerate(hypotheses)})
        logger.info(f"Validation Metrics: {metrics}")

# Main Script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--caption-file", type=str, required=True, help="Path to the caption file")
    parser.add_argument("--train-split", type=str, required=True, help="Path to the training split file")
    parser.add_argument("--val-split", type=str, required=True, help="Path to the validation split file")
    parser.add_argument("--embed-size", type=int, default=256, help="Embedding size")
    parser.add_argument("--encoder-hidden-size", type=int, default=512, help="Hidden size for the encoder")
    parser.add_argument("--decoder-hidden-size", type=int, default=512, help="Hidden size for the decoder")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max-seq-len", type=int, default=20, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    args = parser.parse_args()

    train(args)
