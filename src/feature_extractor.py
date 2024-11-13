import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import logging

logging.basicConfig(filename='../log/feature_extraction.log', level=logging.INFO)

class FeatureExtractor:
    def __init__(self, model_name="resnet50"):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model = getattr(models, model_name)(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval()

    def extract_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
        return features.flatten()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract image features using a CNN.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save extracted features.")
    args = parser.parse_args()

    feature_extractor = FeatureExtractor()
    features = {}

    for img_file in os.listdir(args.image_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(args.image_dir, img_file)
            features[img_file] = feature_extractor.extract_features(img_path)

    torch.save(features, args.output_file)
    logging.info(f"Extracted features saved to {args.output_file}.")