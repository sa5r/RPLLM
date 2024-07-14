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
  --args.checkpoint_path $2 \
  --dataset_directory $3 \
  --args.repo_token $4 \
  --data_size 1.0 \
  --model_id meta-llama/Llama-2-7b-hf \
  --args.entities_filename entity2text.txt \
  --args.descriptions_filename entity2textlong.txt \
  --args.padding 50 \
  --args.batch_size 32 \
  --args.patience 3 \
  --args.learning_rate  5e-5 \
  --args.decay 0.25 \
  --args.epochs 5 \
  --args.verbose True \
  --args.attention_dropout 0.0
elif [ $1 == "test" ]
then
nohup python -u train.py \
  --dataset_directory $3 \
  --data_size 1 \
  --model_id meta-llama/Llama-2-7b-hf \
  --args.entities_filename entity2text.txt \
  --args.descriptions_filename entity2textlong.txt \
  --args.padding 50 \
  --args.batch_size 32 \
  --args.patience 3 \
  --args.repo_token $4 \
  --args.checkpoint_path $2 \
  --args.learning_rate  5e-5 \
  --args.decay 0.25 \
  --args.epochs 5 \
  --args.verbose True \
  --args.attention_dropout 0.0 \
else
  echo 'Please specify test or train.'
fi