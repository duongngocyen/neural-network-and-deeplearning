import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer
from PIL import Image
import argparse
import logging
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
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
        features = self.resnet(images)
        pooled_features = self.adaptive_pool(features)
        global_features = pooled_features.view(pooled_features.size(0), -1)
        return self.fc(global_features), features

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions, global_features):
        if captions is not None:
            embedded_captions = self.embedding(captions)
            inputs = torch.cat([embedded_captions, global_features.unsqueeze(1).repeat(1, embedded_captions.size(1), 1)], dim=-1)
            outputs, _ = self.lstm(inputs)
            outputs = self.fc(outputs)
            return outputs
        else:
            batch_size = global_features.size(0)
            device = global_features.device
            max_seq_len = 20
            generated_captions = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
            inputs = torch.zeros((batch_size, 1, global_features.size(1) + self.embedding.embedding_dim)).to(device)

            hidden = None
            for t in range(max_seq_len):
                if t == 0:
                    word_embeddings = self.embedding(torch.zeros(batch_size, 1, dtype=torch.long).to(device))
                else:
                    word_embeddings = self.embedding(generated_captions[:, t-1].unsqueeze(1))

                inputs[:, :, :self.embedding.embedding_dim] = word_embeddings
                inputs[:, :, self.embedding.embedding_dim:] = global_features.unsqueeze(1)
                output, hidden = self.lstm(inputs, hidden)
                token_scores = self.fc(output.squeeze(1))
                predicted_tokens = torch.argmax(token_scores, dim=-1)
                generated_captions[:, t] = predicted_tokens

            return generated_captions

# Full Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(vocab_size, hidden_size, embed_size)

    def forward(self, images, captions):
        global_features, _ = self.encoder(images)
        outputs = self.decoder(captions, global_features)
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

    model = ImageCaptioningModel(vocab_size, args.embed_size, args.hidden_size)
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

        # Validation
        model.eval()
        references, hypotheses = [], []
        eval_loss = 0  # Initialize evaluation loss
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch["image"].to(args.device)
                captions = tokenizer(batch["caption"], padding=True, return_tensors="pt").input_ids.to(args.device)

                references.extend(batch["caption"])

                # Compute outputs for evaluation loss
                outputs = model(images, captions[:, :-1])
                eval_loss += criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1)).item()

                # Generate captions
                generated_outputs = model(images, captions=None)
                hypotheses.extend(tokenizer.batch_decode(generated_outputs, skip_special_tokens=True))

        avg_eval_loss = eval_loss / len(val_dataloader)
        logger.info(f"Validation Loss: {avg_eval_loss}")

        # Compute evaluation metrics
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
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden size for LSTM")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    args = parser.parse_args()

    train(args)