#!/bin/bash
python src/train.py \
  --data_dir data/coco/train2017 \
  --annotation_file data/coco/annotations/captions_train2017.json \
  --batch_size 32 \
  --epochs 10