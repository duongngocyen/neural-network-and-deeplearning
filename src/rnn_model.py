import torch
import torch.nn as nn

class RNNCaptioning(nn.Module):
    def __init__(self, feature_dim, hidden_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        # Encode image features
        features = self.fc(features).unsqueeze(1)  # Add time dimension

        # Combine features with captions
        embeddings, _ = self.lstm(features)
        outputs = self.fc_out(embeddings)

        return outputs