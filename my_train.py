import argparse
import logging
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from my_models.Gsimclr import Gsimclr
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from my_utils import parsing
from my_utils.data_utils import load_vocab, G2SDataset
from my_utils.train_utils import get_lr, grad_norm, NoamLR, param_count, param_norm, set_seed, setup_logger
from my_models.contrastive_loss import calculate_loss, simclr_loss_vectorized

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device=torch.device("cpu")

    if args.model == "g2s_series_rel":
        model_class=Gsimclr
        dataset_class=G2SDataset
        assert args.compute_graph_distance
    else:
        raise ValueError(f"Model {args.model} not supported!")
    
    model=model_class(args)
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

    if args.load_from:
        state=torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

    ######################################################################
    #修改，使用多gpu进行训练
    #if torch.cuda.device_count()>1:
    #    model=nn.DataParallel(model,device_ids=[0,1])
    ######################################################################
    model.to(device)
    model.train()

    logging.info(model)
    logging.info(f"Number of parameters = {param_count(model)}")

    # initialization ----------------- data
    src_train_dataset = dataset_class(args,is_reac=True,file=args.train_bin)
    tgt_train_dataset = dataset_class(args,is_reac=False,file=args.train_bin)
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
    #print('len(src_train_dataset)',len(src_train_dataset)) ##0
    #print('args.epoch',args.epoch)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=args.epoch,
        eta_min=0,
        last_epoch=-1
    )
    ############################################################################


    #src_valid_dataset = dataset_class(args, is_reac=True,file=args.valid_bin)
    #tgt_valid_dataset = dataset_class(args,is_reac=False,file=args.valid_bin)

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    o_start = time.time()
    total_steps=0
    accum=0

    logging.info("Start training")
    for epoch in range(args.epoch):
        epoch += 1
        model.zero_grad() #当model中的参数和optimizer中的参数不相同时，两者的zero_grad()不等价

        ###########################################################
        """
        train_dataset=zip(src_train_dataset,tgt_train_dataset)
        src_train_loader,tgt_train_loader=DataLoader(
            dataset=train_dataset,
            batch_size=1, 
            shuffle=True,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )
        """
        ############################################################
        ##########################################################################################
        
        ########################################################
        #提取出sort()和shuffle_in_bucket()函数，生成的data_indices作为公共值传入src和tgt
        #目的--shuffle
        data_size=src_train_dataset.data_size #数据集总长度
        src_lens=src_train_dataset.src_lens
        bucket_size=1000
        data_indices=np.argsort(src_lens)
        for i in range(0, data_size, bucket_size):
            np.random.shuffle(data_indices[i:i + bucket_size])
        #将data_indices传回
        src_train_dataset.data_indices=data_indices
        tgt_train_dataset.data_indices=data_indices
        ########################################################

        src_train_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.train_batch_size
        )
        tgt_train_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.train_batch_size
        )

        src_train_loader=DataLoader(
            dataset=src_train_dataset,
            batch_size=1, #???????
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )
        tgt_train_loader=DataLoader(
            dataset=tgt_train_dataset,
            batch_size=1, #?????????
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )
        
        ###############################################################################

        total_loss,total_num=0,0
        for src_batch,tgt_batch in zip(src_train_loader,tgt_train_loader):

            src_batch.to(device)
            tgt_batch.to(device)
                
            #Enable autocasting for the forward pass(model+loss)
            with torch.cuda.amp.autocast(enabled=args.enable_amp):
                src_emb = model(src_batch) #[b,h]
                tgt_emb = model(tgt_batch) #[b,h]

                #print('src_emb.size()',src_emb.size())
                #print('tgt_emb.size()',tgt_emb.size())
                ##############################################################
                #方法一：使用simclr的loss
                #loss=simclr_loss_vectorized(src_emb,tgt_emb,tau=10)
                ##############################################################

                ##############################################################
                #方法二：使用MolR的loss
                loss=calculate_loss(src_emb,tgt_emb,args)
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

        #当前epoch结束
        logging.info('Train Epoch: [{}/{}] Loss: {:.4f},lr {:.6f}'.format(epoch, args.epoch, total_loss / total_num,get_lr(optimizer)))
        
        #前两个epoch作为warmup_step(修改，发现lr有点大)，之后使用CosineAnnealingLR对学习率进行调整
        #if epoch>=2:
        scheduler.step()
        
        #每5个epoch保存一次
        if epoch%5==0:
            logging.info('Model Saving at epoch {}'.format(epoch))
            state={
                "args":args,
                "state_dict":model.state_dict()
            }
            torch.save(state,os.path.join(args.save_dir,f"STEREO_dataset_epoch{epoch}.pt"))

if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    torch.set_printoptions(profile="full")
    main(args)

        
                    

