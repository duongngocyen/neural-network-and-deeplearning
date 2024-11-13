# Neural Network and Deep Learning Project

This project implements an image captioning system that integrates CNN-based feature extraction with RNN and Transformer-based caption generation. The MS COCO dataset is used for training and evaluation.

---

## 1. Setup

To set up the project, first install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Download COCO Dataset

To download the MS COCO dataset, run the following command:

```bash
bash script/download_coco.sh
```

The dataset will be stored in the `data/coco` folder with the following structure:

```
data/coco/
├── train2017/          # Training images
├── val2017/            # Validation images
└── annotations/        # JSON annotation files
```

---

## 3. Extract Features Using CNN

To extract features from the images using a pretrained CNN (e.g., ResNet50), run:

```bash
python src/feature_extractor.py --image_dir data/coco/train2017 --output_file data/coco/features_train.pt
```

The extracted features will be saved in the `data/coco/features_train.pt` file. 

---

## 4. Train Models

### Train the RNN Model

To train the RNN-based captioning model, run:

```bash
python src/train.py --model rnn --data_dir data/coco/features_train.pt --annotation_file data/coco/annotations/captions_train2017.json --epochs 10
```

The trained model will be saved in the `model` folder.

### Train the Transformer Model

To train the Transformer-based captioning model, run:

```bash
python src/train.py --model transformer --data_dir data/coco/features_train.pt --annotation_file data/coco/annotations/captions_train2017.json --epochs 10
```

The trained model will also be saved in the `model` folder.

---

## 5. Evaluate Models

To evaluate a trained model, use the `evaluate.py` script. You can compute metrics such as BLEU and METEOR:

```bash
python src/evaluate.py --model_path model/transformer_captioning.pth --data_dir data/coco/features_val.pt --annotation_file data/coco/annotations/captions_val2017.json
```

---

## 6. Project Structure

```
<root>
├── data/
│   ├── coco/                      # MS COCO dataset
│       ├── train2017/             # Training images
│       ├── val2017/               # Validation images
│       └── annotations/           # Annotations (JSON)
├── log/                           # Logs for training/testing
├── model/                         # Save trained models
├── script/
│   ├── download_coco.sh           # Script to download the COCO dataset
├── src/
│   ├── dataset.py                 # Dataset loader for COCO
│   ├── feature_extractor.py       # CNN for feature extraction
│   ├── rnn_model.py               # RNN-based captioning model
│   ├── transformer_model.py       # Transformer-based captioning model
│   ├── train.py                   # Training logic
│   ├── evaluate.py                # Evaluation script
│   └── utils.py                   # Utility functions
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 7. Dependencies

The project requires the following Python packages, listed in `requirements.txt`:

- `torch`
- `torchvision`
- `transformers`
- `nltk`
- `tensorboard`

To install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 9. Acknowledgments

This project uses the MS COCO dataset and leverages pretrained CNNs and Transformers for feature extraction and text generation. The work is inspired by recent advances in computer vision and natural language processing.