#!/bin/bash

# Create a directory for experiment logs
LOG_DIR="./experiment_rnn_logs"
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
  CMD="python train.py --encoder-type $ENCODER --experiment-name ${ENCODER}_h512_bs64_ft0"
  run_experiment "$CMD" "${ENCODER}_h512_bs64_ft0"
done

# 2. Effect of finetuning on caption quality
echo "Starting: Effect of finetuning on caption quality"

CMD="python train.py --encoder-type resnet18 --experiment-name resnet18_h512_bs64_ft1 --fine-tune 1"
run_experiment "$CMD" "resnet18_h512_bs64_ft1"

CMD="python train.py --encoder-type resnet50 --experiment-name resnet50_h512_bs32_ft1 --fine-tune 1 --batch-size 32"
run_experiment "$CMD" "resnet50_h512_bs32_ft1"

CMD="python train.py --encoder-type resnet101 --experiment-name resnet101_h512_bs32_ft1 --fine-tune 1 --batch-size 32"
run_experiment "$CMD" "resnet101_h512_bs32_ft1"

# 3. Effect of varying LSTM units
echo "Starting: Effect of varying LSTM units"

# Using ResNet18
echo "Running with ResNet18"
for HIDDEN_SIZE in 256 512 1024; do
  CMD="python train.py --decoder-hidden-size $HIDDEN_SIZE --encoder-type resnet18 --experiment-name resnet18_h${HIDDEN_SIZE}_bs64_ft0"
  run_experiment "$CMD" "resnet18_h${HIDDEN_SIZE}_bs64_ft0"
done

# Using ResNet50
echo "Running with ResNet50"
for HIDDEN_SIZE in 256 512; do
  CMD="python train.py --decoder-hidden-size $HIDDEN_SIZE --encoder-type resnet50 --experiment-name resnet50_h${HIDDEN_SIZE}_bs64_ft0"
  run_experiment "$CMD" "resnet50_h${HIDDEN_SIZE}_bs64_ft0"
done
CMD="python train.py --decoder-hidden-size 1024 --encoder-type resnet50 --experiment-name resnet50_h1024_bs32_ft0 --batch-size 32"
run_experiment "$CMD" "resnet50_h1024_bs32_ft0"

# Using ResNet101
echo "Running with ResNet101"
for HIDDEN_SIZE in 256 512; do
  CMD="python train.py --decoder-hidden-size $HIDDEN_SIZE --encoder-type resnet101 --experiment-name resnet101_h${HIDDEN_SIZE}_bs64_ft0"
  run_experiment "$CMD" "resnet101_h${HIDDEN_SIZE}_bs64_ft0"
done
CMD="python train.py --decoder-hidden-size 1024 --encoder-type resnet101 --experiment-name resnet101_h1024_bs32_ft0 --batch-size 32"
run_experiment "$CMD" "resnet101_h1024_bs32_ft0"

echo "All experiments completed."