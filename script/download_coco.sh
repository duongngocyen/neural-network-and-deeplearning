#!/bin/bash

# Download MS COCO dataset
echo "Downloading MS COCO dataset..."
mkdir -p data/coco

# Train and validation images
wget http://images.cocodataset.org/zips/train2017.zip -P data/coco/
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco/

echo "Extracting files..."
unzip data/coco/train2017.zip -d data/coco/
unzip data/coco/val2017.zip -d data/coco/
unzip data/coco/annotations_trainval2017.zip -d data/coco/

echo "Dataset download and extraction complete."