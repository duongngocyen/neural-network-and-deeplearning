#!/bin/bash

# Create a directory for experiment logs
LOG_DIR="./experiment_transformer_logs"
mkdir -p $LOG_DIR

run_experiment() {
  CMD=$1
  EXP_NAME=$2

  echo "Running: $EXP_NAME"
  $CMD > "$LOG_DIR/${EXP_NAME}.log" 2>&1
  echo "Experiment $EXP_NAME completed. Logs saved to $LOG_DIR/${EXP_NAME}.log"
}

# 1. Effect of larger CNN models on caption quality
echo "Starting: Effect of larger CNN models on caption quality"

for ENCODER in resnet18 resnet50 resnet101; do
  CMD="python train.py --encoder-type $ENCODER --decoder-type transformer --num-heads 1 --num-tf-layers 3 --experiment-name ${ENCODER}_bs64_ft0_l3_h1"
  run_experiment "$CMD" "${ENCODER}_bs64_ft0_l3_h1"
done

# 2. Effect of finetuning on caption quality
echo "Starting: Effect of finetuning on caption quality"

for ENCODER in resnet18 resnet50 resnet101; do
  CMD="python train.py --encoder-type $ENCODER --decoder-type transformer --num-heads 1 --num-tf-layers 3 --fine-tune 1 --experiment-name ${ENCODER}_bs64_ft1_l3_h1"
  run_experiment "$CMD" "${ENCODER}_bs64_ft1_l3_h1"
done

# 3. Effect of varying number of transformer layers and heads
echo "Starting: Effect of varying number of transformer layers and heads"

for NUM_HEADS in 1 2 3; do
  for NUM_LAYERS in 3 5 7; do
    CMD="python train.py --encoder-type resnet18 --decoder-type transformer --num-heads $NUM_HEADS --num-tf-layers $NUM_LAYERS --experiment-name resnet18_bs64_ft0_l${NUM_LAYERS}_h${NUM_HEADS}"
    run_experiment "$CMD" "resnet18_bs64_ft0_l${NUM_LAYERS}_h${NUM_HEADS}"
  done
done

echo "All experiments completed."