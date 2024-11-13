import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.feature_extractor import FeatureExtractor
from src.dataset import get_coco_loader
from src.rnn_model import RNNCaptioning
from src.transformer_model import EncoderDecoderCaptioning

def train(model, dataloader, optimizer, criterion, device, writer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (features, captions) in enumerate(dataloader):
            features = features.to(device)
            captions = captions["input_ids"].squeeze(1).to(device)

            # Forward pass
            outputs = model(features, captions)
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Log epoch loss
        writer.add_scalar("Loss/train", total_loss / len(dataloader), epoch)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train image captioning model using RNN or Transformer.")
    parser.add_argument("--model", type=str, choices=["rnn", "transformer"], required=True, help="Model type.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COCO dataset features.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to COCO annotations file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--feature_dim", type=int, default=2048, help="Feature dimension size from CNN.")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size for model.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size.")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader
    tokenizer = None  # You can initialize a tokenizer if needed
    dataloader = get_coco_loader(args.data_dir, args.annotation_file, tokenizer, args.batch_size)

    # Initialize Model
    if args.model == "rnn":
        print("Training RNN-based captioning model...")
        model = RNNCaptioning(
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=args.vocab_size,
        ).to(device)
    elif args.model == "transformer":
        print("Training Transformer-based captioning model...")
        model = EncoderDecoderCaptioning(
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            vocab_size=args.vocab_size,
        ).to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding index

    # TensorBoard
    writer = SummaryWriter(log_dir="../log/tensorboard/")

    # Train
    train(model, dataloader, optimizer, criterion, device, writer, epochs=args.epochs)

    # Save Model
    model_path = f"../model/{args.model}_captioning.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training completed. Model saved to {model_path}.")