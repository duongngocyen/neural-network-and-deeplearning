import torch
import torch.nn as nn
import torchvision.models as models
import warnings
class Encoder(nn.Module):
    """
    Encoder: CNN-based encoder to extract image features.
    """
    def __init__(self, model_type='resnet50', encoded_image_size=14, fine_tune=False, embed_size=512):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        model = getattr(models, model_type)
        resnet = model(weights='IMAGENET1K_V1')

        # Remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling to fix the feature size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        # Add a linear layer to project features to embed_size
        self.feature_proj = nn.Linear(2048, embed_size)

        self.fine_tune(fine_tune)

    def forward(self, images):
        """
        Forward propagation.
        """
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.flatten(2).permute(0, 2, 1)  # Shape: (batch_size, num_patches, 2048)
        out = self.feature_proj(out)  # Project to (batch_size, num_patches, embed_size)
        return out  # Now has shape (batch_size, num_patches, embed_size)

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for certain layers.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        if fine_tune:
            # Fine-tune the last two layers
            for c in list(self.resnet.children())[-2:]:
                for p in c.parameters():
                    p.requires_grad = True


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.exp(
                torch.arange(0, d_model, 2).float()
                * -(torch.log(torch.tensor(10000.0)) / d_model)
            )
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class HierarchicalDecoder(nn.Module):
    """
    Hierarchical Decoder with three stages:
    1. Scene Description
    2. Object Description
    3. Attribute and Action Description
    """
    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        num_layers,
        num_heads,
        dropout,
        max_seq_length,
        pad_id,
        sos_id,
        eos_id,
    ):
        super(HierarchicalDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(embed_size, dropout, max_seq_length)

        # Stage 1: Scene Description Decoder
        self.scene_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dropout=dropout,
        )
        self.scene_decoder = nn.TransformerDecoder(
            self.scene_decoder_layer, num_layers=num_layers
        )

        # Stage 2: Object Description Decoder
        self.object_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dropout=dropout,
        )
        self.object_decoder = nn.TransformerDecoder(
            self.object_decoder_layer, num_layers=num_layers
        )

        # Stage 3: Attribute and Action Description Decoder
        self.attribute_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dropout=dropout,
        )
        self.attribute_decoder = nn.TransformerDecoder(
            self.attribute_decoder_layer, num_layers=num_layers
        )

        # Final output linear layer
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, encoder_outputs, targets):
        """
        Forward pass through the hierarchical decoder.
        """
        # Prepare embeddings and positional encodings
        embeddings = self.embedding(targets)  # (batch_size, seq_len, embed_size)
        embeddings = self.positional_encoding(embeddings)

        # Transpose for transformer: (seq_len, batch_size, embed_size)
        embeddings = embeddings.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(embeddings.size(0)).to(embeddings.device)
        tgt_key_padding_mask = targets.eq(self.pad_id)

        # Stage 1: Scene Description
        scene_outputs = self.scene_decoder(
            tgt=embeddings,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Stage 2: Object Description
        object_outputs = self.object_decoder(
            tgt=scene_outputs,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Stage 3: Attribute and Action Description
        attribute_outputs = self.attribute_decoder(
            tgt=object_outputs,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Compute output logits
        outputs = self.output_layer(attribute_outputs)  # (seq_len, batch_size, vocab_size)

        # Transpose back to (batch_size, seq_len, vocab_size)
        outputs = outputs.permute(1, 0, 2)

        return outputs  # Shape: (batch_size, seq_len, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. Mask out subsequent positions."""
        mask = torch.triu(torch.ones(sz, sz) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture with hierarchical decoder.
    """
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        # Encode images
        encoder_outputs = self.encoder(images)

        # Decode captions
        outputs = self.decoder(encoder_outputs, captions)

        return outputs