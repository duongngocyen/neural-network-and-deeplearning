# Neural Network and Deep Learning Project: Flickr8k Dataset

This project implements an image captioning system that integrates CNN-based feature extraction with RNN, Transformer, and Hierarchical Transformer-based caption generation. The Flickr8k dataset is used for training and evaluation.

---

## 1. Setup

To set up the project, first install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Download Flickr8k Dataset

Download the Flickr8k dataset manually from [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398) and place the extracted files in the `dataset` folder. The folder structure should look like this:

```
dataset/
├── Flickr8k_Dataset/         # Images
├── Flickr8k_text/            # Text annotations
│   ├── Flickr8k.token.txt    # Captions
│   ├── Flickr_8k.trainImages.txt # Train split
│   ├── Flickr_8k.devImages.txt   # Validation split
│   └── Flickr_8k.testImages.txt  # Test split
```

---

## 3. Train and Evaluate Models

This project includes training scripts for various models. You can run the respective scripts to train and evaluate them:

### Train the RNN Model

To benchmark the RNN-based captioning model:

```bash
sh benchmark_rnn.sh
```

### Train the Transformer Model

To benchmark the Transformer-based captioning model:

```bash
sh benchmark_transformer.sh
```

### Train the BLIP Model

To train the BLIP model:

```bash
sh train_blip.sh
```

### Train the Hierarchical Transformer Model

To train the Hierarchical Transformer model:

```bash
sh benchmark_hierarchy_model.sh

```

---

## 4. Evaluate Models

After training, you can evaluate each model using their respective scripts. The evaluation results will include metrics such as BLEU, CIDEr, ROUGE, and METEOR.

---

## 5. Project Structure

```
<root>
├── dataset/                     # Dataset folder
│   ├── Flickr8k_Dataset/        # Flickr8k images
│   ├── Flickr8k_text/           # Text annotations and splits
│   └── vocab.txt                # Vocabulary file
├── log/                         # Logs for training/testing
├── model/                       # Save trained models
├── script/
│   ├── benchmark_rnn.sh         # Script to train and benchmark RNN
│   ├── benchmark_transformer.sh # Script to train and benchmark Transformer
│   ├── train_blip.sh            # Script to train BLIP
│   ├── hierarchical_model.sh    # Script to train Hierarchical Transformer
├── src/
│   ├── dataset.py               # Dataset loader for Flickr8k
│   ├── feature_extractor.py     # CNN for feature extraction
│   ├── rnn_model.py             # RNN-based captioning model
│   ├── transformer_model.py     # Transformer-based captioning model
│   ├── hierarchical_model.py    # Hierarchical Transformer model
│   ├── train.py                 # Training logic
│   ├── evaluate.py              # Evaluation script
│   └── utils.py                 # Utility functions
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

---

## 6. Dependencies

The project requires the following Python packages, listed in `requirements.txt`:

- `torch`
- `torchvision`
- `transformers`
- `nltk`
- `pycocoevalcap`

To install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Acknowledgments

This project uses the Flickr8k dataset and leverages pretrained CNNs, RNNs, Transformers, and hierarchical models for image captioning. It is inspired by advances in computer vision and natural language processing.
```