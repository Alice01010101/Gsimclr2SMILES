import argparse
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from my_models.ShareEncoder_MT import ShareEncoder_MT
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from my_utils import parsing
from my_utils.data_utils import load_vocab, G2SDataset
from my_utils.train_utils import get_lr, grad_norm, NoamLR, param_count, param_norm, set_seed, setup_logger
from my_models.contrastive_loss import calculate_loss, simclr_loss_vectorized, others_simclr_loss
import random

def get_train_parser():
    parser = argparse.ArgumentParser("train")
    parsing.add_common_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser

def main(args):
    parsing.log_args(args)
    # initialization ----------------- vocab
    if not os.path.exists(args.vocab_file):
        raise ValueError(f"Vocab file {args.vocab_file} not found!")
    vocab = load_vocab(args.vocab_file)
    vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

    # initialization ----------------- model
    os.makedirs(args.save_dir, exist_ok=True)

    #############################################
    local_rank=args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device=local_rank
    #############################################

    if args.model == "g2s_series_rel":
        model_class=ShareEncoder_MT
        dataset_class=G2SDataset
        assert args.compute_graph_distance
    else:
        raise ValueError(f"Model {args.model} not supported!")
    
    model=model_class(args,vocab).to(local_rank)

    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

    #############TODO############################################
    if args.load_from:
        state=torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")
    #############################################################

    
    if torch.cuda.device_count()>1:
        model=DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    logging.info(model)
    logging.info(f"Number of total parameters = {param_count(model.module)}")

    # initialization ----------------- optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )


    scheduler=optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=args.epoch,
        eta_min=0,
        last_epoch=-1
    )

    # initialization ----------------- data
    src_train_dataset = dataset_class(args,is_reac=True,file=args.train_bin)
    tgt_train_dataset = dataset_class(args,is_reac=False,file=args.train_bin)

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    o_start = time.time()
    total_steps=0
    accum=0
    accs=[]

    logging.info("Start training")
    for epoch in range(1, args.epoch+1):
        model.train()
        model.zero_grad() #当model中的参数和optimizer中的参数不相同时，两者的zero_grad()不等价

        src_train_dataset.shuffle_in_bucket(bucket_size=1000)
        tgt_train_dataset.shuffle_in_bucket(bucket_size=1000)
        src_train_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.train_batch_size
        )
        tgt_train_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.train_batch_size
        )
        src_train_sampler=torch.utils.data.distributed.DistributedSampler(src_train_dataset)
        tgt_train_sampler=torch.utils.data.distributed.DistributedSampler(tgt_train_dataset)

        src_train_loader=DataLoader(
            dataset=src_train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            sampler=src_train_sampler,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=False,
        )
        tgt_train_loader=DataLoader(
            dataset=tgt_train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            sampler=tgt_train_sampler,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=False,
        )

        src_train_loader.sampler.set_epoch(epoch)
        tgt_train_loader.sampler.set_epoch(epoch)

        total_loss,total_num=0,0
        for src_batch,tgt_batch in zip(src_train_loader,tgt_train_loader):
            src_batch.to(device)
            tgt_batch.to(device)
            
            #Enable autocasting for the forward pass(model+loss)
            with torch.cuda.amp.autocast(enabled=args.enable_amp):

                loss1,loss2,acc=model(src_batch,tgt_batch)
                loss = loss1 + loss2

                """
                ##############################################################
                #方法一：使用simclr的loss
                #loss=simclr_loss_vectorized(src_emb,tgt_emb,tau=100)
                loss_1=others_simclr_loss(src_emb,tgt_emb,100)
                ##############################################################

                ##############################################################
                #方法二：使用MolR的loss
                #loss=calculate_loss(src_emb,tgt_emb,args)
                ##############################################################
                """
            scaler.scale(loss).backward()
            total_loss += loss.item()*args.train_batch_size 
            total_num += args.train_batch_size
            total_steps += 1
            accum +=1
            accs.append(acc.item()*100)

            if accum == args.accumulation_count:
                #Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer) #相当于zero_grad()?
                #Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                nn.utils.clip_grad_norm_(model.parameters(),args.clip_norm)
                #Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                scaler.step(optimizer)
                #Update the scale for next iteration.
                scaler.update()
                #scheduler.step()
                model.zero_grad()
                accum=0
                
            if total_steps%100==0:
                logging.info('Step {} Loss: {:.4f}, lr: {},acc: {:.4f}'.format(total_steps,total_loss/total_num,get_lr(optimizer),np.mean(accs)))
        
        #当前epoch结束
        logging.info('Train Epoch: [{}/{}] Loss: {:.4f},lr {:.6f}'.format(epoch, args.epoch, total_loss / total_num,get_lr(optimizer)))
        #前两个epoch作为warmup_step(修改，发现lr有点大)，之后使用CosineAnnealingLR对学习率进行调整
        #if epoch>=2:
        scheduler.step()
        #每5个epoch保存一次
        if epoch%5==0:
            if dist.get_rank() ==0:
                logging.info('Model Saving at epoch {}'.format(epoch))
                state={
                    "args":args,
                    "state_dict":model.module.state_dict()
                }
                torch.save(state,os.path.join(args.save_dir,f"End_to_End_epoch{epoch}.pt"))

if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    torch.set_printoptions(profile="full")
    main(args)

        
                    

