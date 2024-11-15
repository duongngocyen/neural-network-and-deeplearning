import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from sklearn.model_selection import train_test_split
import random

# Import evaluation metrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import tqdm


class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, processor):
        self.img_dir = img_dir
        self.processor = processor

        # Read the captions file and create a list of (image, caption) pairs
        self.data = []
        with open(captions_file, 'r') as f:
            next(f)  # Skip the header line if it exists
            for line in f:
                line = line.strip()
                if line == '':
                    continue  # Skip empty lines
                # Split only on the first comma
                img_name, caption = line.split(',', 1)
                self.data.append((img_name.strip(), caption.strip()))

        # Group captions by image
        self.image_captions = {}
        for img_name, caption in self.data:
            if img_name in self.image_captions:
                self.image_captions[img_name].append(caption)
            else:
                self.image_captions[img_name] = [caption]

        # Get the list of unique images
        self.images = list(self.image_captions.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        captions = self.image_captions[img_name]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Randomly select a caption
        caption = random.choice(captions)

        # Preprocess the image and caption
        encoding = self.processor(
            images=image, text=caption, return_tensors="pt", padding='max_length', truncation=True)

        # Remove batch dimension
        for k in encoding.keys():
            encoding[k] = encoding[k].squeeze()

        # Set labels (input_ids) for training
        encoding['labels'] = encoding['input_ids'].clone()

        # For evaluation
        encoding['caption'] = caption
        encoding['image'] = image

        return encoding


class SubsetByImages(Dataset):
    def __init__(self, dataset, image_list):
        self.dataset = dataset
        self.image_indices = [dataset.images.index(img) for img in image_list]

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        return self.dataset[self.image_indices[idx]]


def collate_fn(batch):
    # Collate function to handle batches of varying sizes
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    captions = [item['caption'] for item in batch]
    images = [item['image'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels,
        'captions': captions,
        'images': images
    }


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


def main():
    # Initialize the processor and model
    processor = BlipProcessor.from_pretrained(
        'Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained(
        'Salesforce/blip-image-captioning-base')

    # Create the full dataset
    img_dir = 'dataset/Flickr8k_images'  # Replace with your images directory
    # Replace with your captions file path
    captions_file = 'dataset/captions.txt'

    full_dataset = Flickr8kDataset(
        img_dir=img_dir,
        captions_file=captions_file,
        processor=processor
    )

    # Split dataset into training and validation sets (ensure images are not shared)
    train_imgs, val_imgs = train_test_split(
        full_dataset.images, test_size=0.2, random_state=42)

    # Create train and validation datasets
    train_dataset = SubsetByImages(full_dataset, train_imgs)
    val_dataset = SubsetByImages(full_dataset, val_imgs)
    print("Length of training dataset: ", len(train_dataset))
    print("Length of validation dataset: ", len(val_dataset))
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, collate_fn=collate_fn)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm.tqdm(train_loader):
            # Move tensors to the configured device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        model.eval()
        total_val_loss = 0
        references = {}
        hypotheses = {}
        img_id = 0  # Unique image ID for evaluation

        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                captions = batch['captions']

                # Forward pass
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                # Generate captions
                generated_ids = model.generate(
                    pixel_values=pixel_values, max_length=50)
                generated_captions = processor.batch_decode(
                    generated_ids, skip_special_tokens=True)

                for ref, hyp in zip(captions, generated_captions):
                    references[img_id] = [ref]   # Ground truth caption(s)
                    hypotheses[img_id] = [hyp]
                    img_id += 1

        avg_val_loss = total_val_loss / len(val_loader)

        # Compute evaluation metrics
        # print("Reference: ", references)
        # print("Hypothesis: ", hypotheses)
        metrics = compute_metrics(references, hypotheses)
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Metrics: {metrics_str}")

    # Save the fine-tuned model
    model.save_pretrained('blip-finetuned-flickr8k')
    processor.save_pretrained('blip-finetuned-flickr8k')


if __name__ == '__main__':
    main()
