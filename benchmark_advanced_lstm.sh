#!/bin/bash

python advanced_captioning_lstm.py \
    --img-dir ./dataset/Flickr8k_Dataset/ \
    --caption-file ./dataset/Flickr8k_text/Flickr8k.token.txt \
    --train-split ./dataset/Flickr8k_text/Flickr_8k.trainImages.txt \
    --val-split ./dataset/Flickr8k_text/Flickr_8k.devImages.txt \
    --embed-size 256 \
    --hidden-size 512 \
    --batch-size 16 \
    --lr 5e-4 \
    --epochs 50 > advanced_lstm.log 2>&1