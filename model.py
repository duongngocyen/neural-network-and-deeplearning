import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use a pretrained ResNet model, remove the fully connected layer
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last FC layer
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        features = self.fc(features)  # (batch_size, embed_size)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        # (batch_size, caption_length, embed_size)
        embeddings = self.embed(captions)
        # Append features at the start
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # (batch_size, caption_length, hidden_size)
        hiddens, _ = self.lstm(embeddings)
        # (batch_size, caption_length, vocab_size)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None, max_len=20):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for _ in range(max_len):
            # (batch_size, 1, hidden_size)
            hiddens, states = self.lstm(inputs, states)
            # (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # (batch_size, max_len)
        return sampled_ids


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(
            embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


# Example usage:
embed_size = 256  # Embedding size for image and word embeddings
hidden_size = 512  # Hidden state size of LSTM
# Vocabulary size (this should be the size of your dataset's vocabulary)
vocab_size = 5000

model = ImageCaptioningModel(embed_size, hidden_size, vocab_size)

# Example input: batch of images and captions
images = torch.randn(16, 3, 224, 224)  # Batch of 16 images of size 3x224x224
# Batch of 16 captions with length 20
captions = torch.randint(0, vocab_size, (16, 20))

outputs = model(images, captions)
print(outputs.shape)  # Should output: (16, 20, vocab_size)
