python multimodal.py \
    --img-dir ./dataset/Flickr8k_Dataset/ \
    --caption-file ./dataset/Flickr8k_text/Flickr8k.token.txt \
    --train-split ./dataset/Flickr8k_text/Flickr_8k.trainImages.txt \
    --val-split ./dataset/Flickr8k_text/Flickr_8k.devImages.txt \
    --batch-size 16 \
    --lr 5e-5 \
    --epochs 10 > multimodal.log 2>&1