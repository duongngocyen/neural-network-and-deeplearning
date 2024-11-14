#!/bin/bash

# Directory for experiment logs
LOG_DIR="./experiment_transformer_logs/layers_heads_no_finetune"
mkdir -p $LOG_DIR

run_experiment() {
  CMD=$1
  EXP_NAME=$2

  echo "Running: $EXP_NAME"
  $CMD > "$LOG_DIR/${EXP_NAME}.log" 2>&1
  echo "Experiment $EXP_NAME completed. Logs saved to $LOG_DIR/${EXP_NAME}.log"
}

echo "Starting: Effect of varying number of transformer layers and heads (No Fine-Tuning)"

for NUM_HEADS in 1 2 3; do
  for NUM_LAYERS in 3 5 7; do
    CMD="python train.py --encoder-type resnet18 --decoder-type transformer --num-heads $NUM_HEADS --num-tf-layers $NUM_LAYERS --fine-tune 0 --experiment-name resnet18_bs64_ft0_l${NUM_LAYERS}_h${NUM_HEADS}"
    run_experiment "$CMD" "resnet18_bs64_ft0_l${NUM_LAYERS}_h${NUM_HEADS}"
  done
done

echo "All no-fine-tuning layer/head experiments completed."