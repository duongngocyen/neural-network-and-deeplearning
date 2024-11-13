import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description="ImageCaptioning Model")
    parser.add_argument(
        "--emd_size", type=int, default=256, help="Embedding size"
    )
    parser.add_argument(
        "--no_epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=5000, help="Vocabulary size"
    )
    parser.add_argument(
        "--max_length", type=int, default=20, help="Max sequence length of caption"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden layer size"
    )
    parser.add_argument(
        "--img_dir", type=str, default="/home/ducanh/neural-network-and-deeplearning/dataset/Images", help="Image directory"
    )
    parser.add_argument(
        "--caption_file", type=str, default="/home/ducanh/neural-network-and-deeplearning/dataset/captions.txt", help="Caption file"
    )
    return parser
