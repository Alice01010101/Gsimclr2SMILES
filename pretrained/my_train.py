import argparse
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from my_models.Gsimclr import Gsimclr
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from my_utils import parsing
from my_utils.data_utils import load_vocab, G2SDataset
from my_utils.train_utils import get_lr, grad_norm, NoamLR, param_count, param_norm, set_seed, setup_logger
from my_models.contrastive_loss import calculate_loss, simclr_loss_vectorized, others_simclr_loss,local_global_loss_
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

    #更改
    #############################################
    local_rank=args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device=local_rank

    torch.autograd.set_detect_anomaly(True)
    #############################################
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device=torch.device("cpu")

    if args.model == "g2s_series_rel":
        model_class=Gsimclr
        dataset_class=G2SDataset
        #assert args.compute_graph_distance
    else:
        raise ValueError(f"Model {args.model} not supported!")
    
    model=model_class(args).to(device)
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

    if args.load_from and dist.get_rank()==0:
        state=torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

    if torch.cuda.device_count()>1:
        model=DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logging.info(model)
    logging.info(f"Number of parameters = {param_count(model)}")

    # initialization ----------------- optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    ###########################################################################
    """
    scheduler = NoamLR(
        optimizer,
        model_size=args.decoder_hidden_size,
        warmup_steps=args.warmup_steps
    )
    """
    ############################################################################
    scheduler=optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=args.epoch,
        eta_min=0,
        last_epoch=-1
    )
    ############################################################################

    # initialization ----------------- data
    src_train_dataset = dataset_class(args,is_reac=True,file=args.train_bin)
    tgt_train_dataset = dataset_class(args,is_reac=False,file=args.train_bin)
    src_valid_dataset = dataset_class(args,is_reac=True,file=args.valid_bin)
    tgt_valid_dataset = dataset_class(args,is_reac=False,file=args.valid_bin)

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    o_start = time.time()
    total_steps=0
    accum=0

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
            num_workers=6,
            sampler=src_train_sampler,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=False,
        )
        tgt_train_loader=DataLoader(
            dataset=tgt_train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=6,
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
                src_global_emb = model(src_batch) #[b,h]
                tgt_global_emb = model(tgt_batch) #[b,h]
                """
                src_emb,src_global_emb,mlen1 = model(src_batch) #[b,h]
                tgt_emb,tgt_global_emb,mlen2 = model(tgt_batch) #[b,h]
                
                loss1=local_global_loss_(src_emb,src_global_emb,mlen1)
                loss2=local_global_loss_(tgt_emb,tgt_global_emb,mlen2)
                """
                loss=simclr_loss_vectorized(src_global_emb,tgt_global_emb,50)
                ##############################################################
                #方法一：使用simclr的loss
                #loss=simclr_loss_vectorized(src_emb,tgt_emb,tau=100)
                #loss=others_simclr_loss(src_emb,tgt_emb,100)
                ##############################################################

                ##############################################################
                #方法二：使用MolR的loss
                #loss=calculate_loss(src_emb,tgt_emb,args)
                ##############################################################

            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            total_loss += loss.item()*args.train_batch_size 
            total_num += args.train_batch_size
            total_steps += 1
            accum +=1

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
                logging.info('Step {} Loss: {:.4f}, lr {}'.format(total_steps,total_loss/total_num,get_lr(optimizer)))

        scheduler.step()

        #当前epoch结束
        #使用molr中的valid方法,引入指标mrr
        model.eval()
        src_valid_dataset.shuffle_in_bucket(bucket_size=1000)
        tgt_valid_dataset.shuffle_in_bucket(bucket_size=1000)

        src_valid_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.valid_batch_size
        )
        tgt_valid_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.valid_batch_size
        )
        src_valid_sampler=torch.utils.data.distributed.DistributedSampler(src_valid_dataset)
        tgt_valid_sampler=torch.utils.data.distributed.DistributedSampler(tgt_valid_dataset)

        src_valid_loader=DataLoader(
            dataset=src_valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=6,
            sampler=src_valid_sampler,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=False,
        )
        tgt_valid_loader=DataLoader(
            dataset=tgt_valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=6,
            sampler=tgt_valid_sampler,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=False,
        )

        src_valid_loader.sampler.set_epoch(epoch)
        tgt_valid_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            all_product_embeddings=[]

            for product_batch in tgt_valid_loader:
                product_batch.to(device)
                product_batch_global_embeddings=model(product_batch)
                all_product_embeddings.append(product_batch_global_embeddings)
            all_product_embeddings = torch.cat(all_product_embeddings,dim=0)
            #rank
            all_rankings = []

            i=0
            for reaction_batch in src_valid_loader:
                reaction_batch.to(device)
                reaction_batch_global_embeddings=model(reaction_batch)
                ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.valid_batch_size, 30000)), dim=1)
                i += args.valid_batch_size
                if torch.cuda.is_available():
                    ground_truth = ground_truth.to(device)
                cdist = torch.cdist(reaction_batch_global_embeddings,all_product_embeddings,p=2)
                sorted_indices = torch.argsort(cdist, dim=1)
                rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
                #rankings = ((sorted_indices == ground_truth).nonzero(*,bool as_tuple)[:, 1] + 1).tolist()
                all_rankings.extend(rankings)
            
            # calculate metrics
            all_rankings = np.array(all_rankings)
            mrr = float(np.mean(1 / all_rankings))
            mr = float(np.mean(all_rankings))
            h1 = float(np.mean(all_rankings <= 1))
            h3 = float(np.mean(all_rankings <= 3))
            h5 = float(np.mean(all_rankings <= 5))
            h10 = float(np.mean(all_rankings <= 10))
        logging.info('Train Epoch: [{}/{}] Loss: {:.4f},lr {:.6f},mrr: {:.4f}  mr: {:.4f}  h1: {:.4f}  h3: {:.4f}  h5: {:.4f}  h10: {:.4f}'.format(epoch, args.epoch, total_loss / total_num,get_lr(optimizer),mrr, mr, h1, h3, h5, h10))
        #前两个epoch作为warmup_step(修改，发现lr有点大)，之后使用CosineAnnealingLR对学习率进行调整

        #每5个epoch保存一次
        if epoch%5==0:
            if dist.get_rank()==0:
                logging.info('Model Saving at epoch {}'.format(epoch))
                state={
                    "args":args,
                    "state_dict":model.module.state_dict()
                }
                torch.save(state,os.path.join(args.save_dir,f"STEREO_add_valid_MLP_epoch{epoch}.pt"))

if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    torch.set_printoptions(profile="full")
    main(args)

        
                    

