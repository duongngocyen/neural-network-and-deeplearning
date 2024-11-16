#!/bin/bash

# Directory for experiment logs
LOG_DIR="./experiment_transformer_logs/hierarchy"
mkdir -p $LOG_DIR

run_experiment() {
  CMD=$1
  EXP_NAME=$2

  echo "Running: $EXP_NAME"
  $CMD > "$LOG_DIR/${EXP_NAME}.log" 2>&1
  echo "Experiment $EXP_NAME completed. Logs saved to $LOG_DIR/${EXP_NAME}.log"
}

# Example experiments
CMD="python train_hierarchy_model.py --encoder-type resnet50 --fine-tune 0 --batch-size 64 --num-epochs 50 --experiment-name resnet50_ft0_hierarchy"
run_experiment "$CMD" "resnet50_ft0_hierarchy"

python test_hierarchy_model.py --checkpoint-name model_saves/resnet50_ft1_hierarchy_best_model.pth

echo "Hierarchical transformer experiments completed."