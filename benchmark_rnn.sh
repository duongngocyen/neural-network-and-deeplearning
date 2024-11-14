#!/bin/bash

# Directory for experiment logs
LOG_DIR="./experiment_rnn_logs/prioritized"
mkdir -p $LOG_DIR

run_experiment() {
  CMD=$1
  EXP_NAME=$2

  echo "Running: $EXP_NAME"
  $CMD > "$LOG_DIR/${EXP_NAME}.log" 2>&1
  echo "Experiment $EXP_NAME completed. Logs saved to $LOG_DIR/${EXP_NAME}.log"
}

# Priority 1: Baseline experiments with varying LSTM hidden units (no fine-tuning)
echo "Priority 1: Baseline experiments (RNN, varying hidden units, no fine-tuning)"

for ENCODER in resnet18 resnet50 resnet101; do
  for HIDDEN_SIZE in 256; do
    CMD="python train.py --decoder-hidden-size $HIDDEN_SIZE --encoder-type $ENCODER --experiment-name ${ENCODER}_h${HIDDEN_SIZE}_bs64_ft0 --fine-tune 0 --batch-size 64"
    run_experiment "$CMD" "${ENCODER}_h${HIDDEN_SIZE}_bs64_ft0"
  done
done

for ENCODER in resnet18; do
  for HIDDEN_SIZE in 512 1024; do
    CMD="python train.py --decoder-hidden-size $HIDDEN_SIZE --encoder-type $ENCODER --experiment-name ${ENCODER}_h${HIDDEN_SIZE}_bs64_ft0 --fine-tune 0 --batch-size 64"
    run_experiment "$CMD" "${ENCODER}_h${HIDDEN_SIZE}_bs64_ft0"
  done
done

# Priority 2: Fine-tuning with varying LSTM hidden units
echo "Priority 2: Fine-tuning experiments (RNN, varying hidden units)"

for ENCODER in resnet18; do
  for HIDDEN_SIZE in 256; do
    CMD="python train.py --decoder-hidden-size $HIDDEN_SIZE --encoder-type $ENCODER --experiment-name ${ENCODER}_h${HIDDEN_SIZE}_bs64_ft1 --fine-tune 1 --batch-size 64"
    run_experiment "$CMD" "${ENCODER}_h${HIDDEN_SIZE}_bs64_ft1"
  done
done

echo "Prioritized RNN experiments completed."