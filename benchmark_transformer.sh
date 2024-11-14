#!/bin/bash

# Directory for experiment logs
LOG_DIR="./experiment_transformer_logs/prioritized"
mkdir -p $LOG_DIR

run_experiment() {
  CMD=$1
  EXP_NAME=$2

  echo "Running: $EXP_NAME"
  $CMD > "$LOG_DIR/${EXP_NAME}.log" 2>&1
  echo "Experiment $EXP_NAME completed. Logs saved to $LOG_DIR/${EXP_NAME}.log"
}

# Priority 1: Baseline experiment with no fine-tuning and smallest configurations
echo "Priority 1: Baseline experiments"

CMD="python train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 3 --fine-tune 0 --experiment-name resnet18_bs64_ft0_l3_h1"
run_experiment "$CMD" "resnet18_bs64_ft0_l3_h1"

CMD="python train.py --encoder-type resnet50 --decoder-type transformer --num-heads 1 --num-tf-layers 3 --fine-tune 0 --experiment-name resnet50_bs64_ft0_l3_h1"
run_experiment "$CMD" "resnet50_bs64_ft0_l3_h1"

CMD="python train.py --encoder-type resnet101 --decoder-type transformer --num-heads 1 --num-tf-layers 3 --fine-tune 0 --experiment-name resnet101_bs64_ft0_l3_h1"
run_experiment "$CMD" "resnet101_bs64_ft0_l3_h1"

# Priority 2: Effect of varying Transformer layers and heads (no fine-tuning)
echo "Priority 2: Effect of varying layers and heads (no fine-tuning)"

for NUM_HEADS in 1 2 3; do
  for NUM_LAYERS in 3 5; do
    CMD="python train.py --encoder-type resnet18 --decoder-type transformer --num-heads $NUM_HEADS --num-tf-layers $NUM_LAYERS --fine-tune 0 --experiment-name resnet18_bs64_ft0_l${NUM_LAYERS}_h${NUM_HEADS}"
    run_experiment "$CMD" "resnet18_bs64_ft0_l${NUM_LAYERS}_h${NUM_HEADS}"
  done
done

# Priority 3: Fine-tuning for best baseline model
echo "Priority 3: Fine-tuning best baseline (resnet50, 3 layers, 1 head)"

CMD="python train.py --encoder-type resnet50 --decoder-type transformer --num-heads 1 --num-tf-layers 3 --fine-tune 1 --experiment-name resnet50_bs64_ft1_l3_h1"
run_experiment "$CMD" "resnet50_bs64_ft1_l3_h1"

echo "Prioritized experiments completed."