#!/bin/bash

#LOAD_FROM="./checkpoints/Old_USPTO_STEREO_g2s_series_rel_smiles_smiles.1/STEREO_simclr_tau100_epoch90.pt"
LOAD_FROM=""
MODEL=g2s_series_rel
TASK=reaction_prediction                #retrosynthesis       
DATASET=Old_USPTO_STEREO
MPN_TYPE=dgat #dgat
MAX_REL_POS=4
ACCUM_COUNT=4
ENC_PE=none
ENC_H=128 #256 为D-MPNN的维度，与下面encoder的维度要对齐
BATCH_SIZE=256 #4096
ENC_EMB_SCALE=sqrt
MAX_STEP=400000 #500000
ENC_LAYER=4
BATCH_TYPE=tokens
REL_BUCKETS=11 #10

EXP_NO=1
REL_POS=emb_only
ATTN_LAYER=0
LR=0.0001      #0.0001
DROPOUT=0.3

REPR_START=smiles
REPR_END=smiles

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

#添加一行EMB_SIZE，与上面ENC_H对齐
EMB_SIZE=128
#把emb_size从256改成512，1024
FILTER_SIZE=256

CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 --master_port 9999 my_train.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --task="$TASK" \
  --representation_end=$REPR_END \
  --load_from="$LOAD_FROM" \
  --train_bin="./my_preprocessed/$PREFIX/train_0.npz" \
  --valid_bin="./my_preprocessed/$PREFIX/val_0.npz" \
  --log_file="$PREFIX.train.$EXP_NO.log" \
  --vocab_file="./my_preprocessed/$PREFIX/vocab_$REPR_END.txt" \
  --save_dir="./checkpoints/$PREFIX.$EXP_NO" \
  --embed_size=256 \
  --mpn_type="$MPN_TYPE" \
  --encoder_num_layers="$ENC_LAYER" \
  --encoder_hidden_size="$ENC_H" \
  --encoder_norm="$ENC_NORM" \
  --encoder_skip_connection="$ENC_SC" \
  --encoder_positional_encoding="$ENC_PE" \
  --encoder_emb_scale="$ENC_EMB_SCALE" \
  --attn_enc_num_layers="$ATTN_LAYER" \
  --attn_enc_hidden_size="$EMB_SIZE" \
  --attn_enc_heads=8 \
  --attn_enc_filter_size="$FILTER_SIZE" \
  --rel_pos="$REL_POS" \
  --rel_pos_buckets="$REL_BUCKETS" \
  --decoder_num_layers=6 \
  --decoder_hidden_size=256 \
  --decoder_attn_heads=8 \
  --decoder_filter_size="$FILTER_SIZE" \
  --dropout="$DROPOUT" \
  --attn_dropout="$DROPOUT" \
  --max_relative_positions="$MAX_REL_POS" \
  --seed=42 \
  --epoch=100 \
  --max_steps="$MAX_STEP" \
  --warmup_steps=100 \
  --lr="$LR" \
  --weight_decay=0.0 \
  --clip_norm=20.0 \
  --batch_type="$BATCH_TYPE" \
  --train_batch_size="$BATCH_SIZE" \
  --valid_batch_size="$BATCH_SIZE" \
  --predict_batch_size="$BATCH_SIZE" \
  --accumulation_count="$ACCUM_COUNT" \
  --num_workers=0 \
  --beam_size=5 \
  --predict_min_len=1 \
  --predict_max_len=512 \
  --log_iter=100 \
  --eval_iter=2000 \
  --save_iter=5000 \
  --margin=4.0

