python advanced_captioning_transformer.py \
    --img-dir ./dataset/Flickr8k_Dataset/ \
    --caption-file ./dataset/Flickr8k_text/Flickr8k.token.txt \
    --train-split ./dataset/Flickr8k_text/Flickr_8k.trainImages.txt \
    --val-split ./dataset/Flickr8k_text/Flickr_8k.devImages.txt \
    --embed-size 256 \
    --encoder-hidden-size 512 \
    --decoder-hidden-size 512 \
    --num-layers 4 \
    --num-heads 8 \
    --max-seq-len 20 \
    --batch-size 16 \
    --lr 5e-4 \
    --epochs 50 \
    --device cuda > advanced_transformer.log 2>&1
