# Import necessary libraries for handling datasets, deep learning, image processing, and evaluation
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from sklearn.model_selection import train_test_split
import random

# Import evaluation metrics for evaluating model performance
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import tqdm

# Define a custom dataset class for Flickr8k
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, processor):
        self.img_dir = img_dir  # Directory containing image files
        self.processor = processor  # Processor for preprocessing images and captions

        # Read the captions file and create a list of (image, caption) pairs
        self.data = []
        with open(captions_file, 'r') as f:
            next(f)  # Skip the header line if it exists
            for line in f:
                line = line.strip()
                if line == '':
                    continue  # Skip empty lines
                # Split each line into image name and caption
                img_name, caption = line.split(',', 1)
                self.data.append((img_name.strip(), caption.strip()))

        # Group captions by image for efficient access
        self.image_captions = {}
        for img_name, caption in self.data:
            if img_name in self.image_captions:
                self.image_captions[img_name].append(caption)
            else:
                self.image_captions[img_name] = [caption]

        # Get the list of unique image names
        self.images = list(self.image_captions.keys())

    def __len__(self):
        # Return the total number of unique images
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve the image and its captions using the index
        img_name = self.images[idx]
        captions = self.image_captions[img_name]

        # Construct the full image path and open the image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Randomly select one caption for the image
        caption = random.choice(captions)

        # Preprocess the image and caption for model input
        encoding = self.processor(
            images=image, text=caption, return_tensors="pt", padding='max_length', truncation=True)

        # Remove the batch dimension for simplicity
        for k in encoding.keys():
            encoding[k] = encoding[k].squeeze()

        # Duplicate input IDs as labels for training purposes
        encoding['labels'] = encoding['input_ids'].clone()

        # Include raw caption and image in the output for evaluation
        encoding['caption'] = caption
        encoding['image'] = image

        return encoding

# Define a subset class to filter datasets by a specific set of images
class SubsetByImages(Dataset):
    def __init__(self, dataset, image_list):
        self.dataset = dataset  # Original dataset
        # Find indices corresponding to the specified image list
        self.image_indices = [dataset.images.index(img) for img in image_list]

    def __len__(self):
        # Return the number of images in the subset
        return len(self.image_indices)

    def __getitem__(self, idx):
        # Retrieve the data for the specified index
        return self.dataset[self.image_indices[idx]]

# Custom collate function to handle batches with varying data
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    # Captions and images are kept as lists for evaluation purposes
    captions = [item['caption'] for item in batch]
    images = [item['image'] for item in batch]

    # Return a dictionary of batched data
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels,
        'captions': captions,
        'images': images
    }

# Function to compute evaluation metrics for generated captions
def compute_metrics(references, hypotheses):
    # Define the metrics to use for evaluation
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    # Compute scores for each metric
    for scorer, method in scorers:
        score, _ = scorer.compute_score(references, hypotheses)
        if isinstance(score, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

# Main function to set up data, train, validate, and save the model
def main():
    # Initialize the processor and pre-trained model
    processor = BlipProcessor.from_pretrained(
        'Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained(
        'Salesforce/blip-image-captioning-base')

    # Specify dataset paths
    img_dir = 'dataset/Flickr8k_images'  # Path to image directory
    captions_file = 'dataset/captions.txt'  # Path to captions file

    # Create the full dataset using the custom Flickr8kDataset class
    full_dataset = Flickr8kDataset(
        img_dir=img_dir,
        captions_file=captions_file,
        processor=processor
    )

    # Split the dataset into training and validation sets, ensuring no image overlap
    train_imgs, val_imgs = train_test_split(
        full_dataset.images, test_size=0.2, random_state=42)

    # Create subset datasets for training and validation
    train_dataset = SubsetByImages(full_dataset, train_imgs)
    val_dataset = SubsetByImages(full_dataset, val_imgs)
    print("Length of training dataset: ", len(train_dataset))
    print("Length of validation dataset: ", len(val_dataset))

    # Set up data loaders with batching and shuffling
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, collate_fn=collate_fn)

    # Define optimizer and learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Set the number of training epochs and device (CPU/GPU)
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        # Iterate through training batches
        for batch in tqdm.tqdm(train_loader):
            # Move data to the configured device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # Perform forward pass to compute loss
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels)
            loss = outputs.loss

            # Perform backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        references = {}  # Store ground truth captions
        hypotheses = {}  # Store generated captions
        img_id = 0  # Image ID counter for evaluation

        with torch.no_grad():
            # Iterate through validation batches
            for batch in tqdm.tqdm(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                captions = batch['captions']

                # Perform forward pass to compute validation loss
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                # Generate captions for evaluation
                generated_ids = model.generate(
                    pixel_values=pixel_values, max_length=50)
                generated_captions = processor.batch_decode(
                    generated_ids, skip_special_tokens=True)

                # Map generated captions to ground truth captions
                for ref, hyp in zip(captions, generated_captions):
                    references[img_id] = [ref]   # Store ground truth
                    hypotheses[img_id] = [hyp]  # Store predictions
                    img_id += 1

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)

        # Compute evaluation metrics for the generated captions
        metrics = compute_metrics(references, hypotheses)
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        # Print training and validation metrics for the epoch
        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Metrics: {metrics_str}")

    # Save the fine-tuned model and processor
    model.save_pretrained('./advanced1/blip-finetuned-flickr8k')
    processor.save_pretrained('./advanced1/blip-finetuned-flickr8k')


# Entry point of the script
if __name__ == '__main__':
    main()