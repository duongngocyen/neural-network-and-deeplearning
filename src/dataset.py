import os
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, tokenizer=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.root_dir, f"{annotation['image_id']:012d}.jpg")
        caption = annotation["caption"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        if self.tokenizer:
            caption = self.tokenizer(caption, padding="max_length", truncation=True, return_tensors="pt")

        return image, caption

def get_coco_loader(root_dir, annotation_file, tokenizer, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = COCODataset(root_dir, annotation_file, transform=transform, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)