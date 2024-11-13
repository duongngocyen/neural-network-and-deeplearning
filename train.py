import torch
import torchvision.transforms as transforms
from dataset import Flickr8KDataset, Vocabulary
from model import ImageCaptioningModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from parser import create_parser


def train_image_captioning_model(model, train_dataloader, val_dataloader, vocab_size, num_epochs=10, learning_rate=0.001, save_path="best_image_captioning_model.pth"):
    """
    Trains the image captioning model and saves the model with the lowest validation loss.

    Args:
        model: The image captioning model (CNN encoder + RNN decoder).
        train_dataloader: DataLoader for the training set.
        val_dataloader: DataLoader for the validation set.
        vocab_size: Size of the vocabulary.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        save_path: Path to save the best model.
    """
    # Define loss and optimizer
    # Ignore padding token in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")  # Track the best validation loss

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, captions in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            # Exclude last token for decoder input
            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, vocab_size),
                             captions[:, 1:].reshape(-1))  # Shift target by 1

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.to(device)
                captions = captions.to(device)

                outputs = model(images, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, vocab_size),
                                 captions[:, 1:].reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(
                f"Saved Best Model with Validation Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    vocab = Vocabulary(freq_threshold=5)
    with open('path/to/flickr8k/captions.txt', 'r') as file:
        all_captions = [line.strip().split(',')[1] for line in file]
    vocab.build_vocabulary(all_captions)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = Flickr8KDataset(img_dir=args.img_dir, captions_file=args.caption_file,
                              vocab=vocab, transform=transform, max_length=args.max_length)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = ImageCaptioningModel(
        args.emb_size, args.hidden_size, len(vocab))

    train_image_captioning_model(
        model, train_dataloader, val_dataloader, len(vocab), args.no_epochs, args.lr)
