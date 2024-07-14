#!/bin/bash
# TRAIN_FLAG = $1
# CHECKPOINT_PATH = $2
# DATA_PATH = $3 # data/FB15K
# structural_path = $4
# text_path = $5

if [ $# -eq 0 ]
then
  echo "Please specify test or train."
  exit 0
fi

if [ $1 == "train" ]
then
nohup python -u train.py \
  --data_path $3 \
  --data_size 1
elif [ $1 == "test" ]
then
nohup python -u evaluate.py \
    --data_path $3 \
else
  echo 'Please specify test or train.'
fi