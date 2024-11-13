from transformers import ViTModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn

class TransformerCaptioning(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.caption_generator = GPT2LMHeadModel.from_pretrained("gpt2")

        # Resize tokenizer embedding to handle additional tokens
        self.caption_generator.resize_token_embeddings(len(self.tokenizer))

    def forward(self, images, captions=None):
        # Extract features from images
        features = self.feature_extractor(images)["last_hidden_state"]

        # Prepare inputs for the caption generator
        inputs = self.tokenizer(captions, return_tensors="pt", padding="max_length", truncation=True)
        outputs = self.caption_generator(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return outputs.logits