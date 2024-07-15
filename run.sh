#!/bin/bash
# TRAIN_FLAG = $1
# CHECKPOINT_PATH = $2
# DATA_PATH = $3 # data/FB15K
# HF_TOKEN = $4

if [ $# -eq 0 ]
then
  echo "Please specify test or train."
  exit 0
fi

if [ $1 == "train" ]
then
nohup python -u train.py \
  --checkpoint_path $2 \
  --dataset_directory $3 \
  --repo_token $4 \
  --data_size 1.0 \
  --model_id meta-llama/Llama-2-7b-hf \
  --entities_filename entity2text.txt \
  --descriptions_filename entity2textlong.txt \
  --padding 50 \
  --batch_size 32 \
  --patience 3 \
  --learning_rate  5e-5 \
  --decay 0.25 \
  --epochs 5 \
  --verbose True \
  --attention_dropout 0.0
elif [ $1 == "test" ]
then
nohup python -u evaluate.py \
  --checkpoint_path $2 \
  --dataset_directory $3 \
  --repo_token $4 \
  --data_size 1.0 \
  --model_id meta-llama/Llama-2-7b-hf \
  --entities_filename entity2text.txt \
  --descriptions_filename entity2textlong.txt \
  --padding 50 \
  --batch_size 32 \
  --verbose True \
  --attention_dropout 0.0
else
  echo 'Please specify test or train.'
fi