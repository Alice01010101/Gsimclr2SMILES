#!/bin/bash 

DATASET=USPTO_STEREO
MODEL=g2s_series_rel
TASK=reaction_prediction               #retrosynthesis                 
REPR_START=smiles
REPR_END=smiles
N_WORKERS=4

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

python my_preprocess.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --task="$TASK" \
  --representation_start=$REPR_START \
  --representation_end=$REPR_END \
  --train_src="./my_data/$DATASET/src-train.txt" \
  --train_tgt="./my_data/$DATASET/tgt-train.txt" \
  --val_src="./my_data/$DATASET/src-val.txt" \
  --val_tgt="./my_data/$DATASET/tgt-val.txt" \
  --test_src="./my_data/$DATASET/src-test.txt" \
  --test_tgt="./my_data/$DATASET/tgt-test.txt" \
  --log_file="$PREFIX.preprocess.log" \
  --preprocess_output_path="./my_preprocessed/$PREFIX/" \
  --seed=42 \
  --max_src_len=1024 \
  --max_tgt_len=1024 \
  --num_workers="$N_WORKERS"
