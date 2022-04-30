from numpy import argsort
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from my_utils.chem_utils import ATOM_FDIM,BOND_FDIM
from my_utils.data_utils import G2SBatch
from my_models.graphfeat import GraphFeatEncoder 
from my_models.attention_xl import AttnEncoderXL
from my_models.contrastive_loss import calculate_loss, simclr_loss_vectorized, others_simclr_loss
from my_utils.data_utils import G2SDataset
from onmt.decoders import TransformerDecoder
from onmt.modules.embeddings import Embeddings
from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
from typing import Any, Dict
from torch_geometric.nn import Set2Set
class ShareEncoder_MT(nn.Module): 
    def __init__(self,args,vocab:Dict[str,int],feature_dim=128):
        super(ShareEncoder_MT,self).__init__()
        self.args=args
        self.vocab=vocab
        self.vocab_size=len(self.vocab)

        while args.enable_amp and not self.vocab_size % 8 == 0:
            self.vocab_size += 1
        
        # encoder
        self.encoder = GraphFeatEncoder(
            args,
            n_atom_feat=sum(ATOM_FDIM),
            n_bond_feat=BOND_FDIM
        )
        
        if args.attn_enc_num_layers > 0:
            self.attention_encoder = AttnEncoderXL(args)
        else:
            self.attention_encoder = None
        
        self.decoder_embeddings = Embeddings(
            word_vec_size=args.embed_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            dropout=args.dropout
        )

        self.decoder = TransformerDecoder(
            num_layers=args.decoder_num_layers,
            d_model=args.decoder_hidden_size,
            heads=args.decoder_attn_heads,
            d_ff=args.decoder_filter_size,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=args.dropout,
            attention_dropout=args.attn_dropout,
            embeddings=self.decoder_embeddings,
            max_relative_positions=args.max_relative_positions,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=-3,
            alignment_heads=0
        )

        if not args.attn_enc_hidden_size == args.decoder_hidden_size:
            self.bridge_layer = nn.Linear(args.attn_enc_hidden_size, args.decoder_hidden_size, bias=True)

        self.output_layer = nn.Linear(args.decoder_hidden_size, self.vocab_size, bias=True)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab["_PAD"],
            reduction="mean"
        )

        # projection head 
        self.head2 = nn.Sequential(
                               nn.Linear(256,256,bias=True),
                               nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True),
                               nn.Linear(256, feature_dim, bias=True))

        self.constantpad1d = nn.ConstantPad1d((1, 0), self.vocab["_SOS"])


    def sharencoder(self,reaction_batch:G2SBatch):
        
        hatom,_ = self.encoder(reaction_batch)

        #print('hatom.size()',hatom.size()) #512/632不等
        atom_scope=reaction_batch.atom_scope
        memory_lengths=[scope[-1][0]+scope[-1][1]-scope[0][0] for scope in atom_scope]

        assert 1+sum(memory_lengths) == hatom.size(0), \
            f"Memory lengths calculation error, encoder output: {hatom.size(0)},memory_lengths:{memory_lengths}"

        memory_bank=torch.split(hatom,[1]+memory_lengths,dim=0)
        padded_memory_bank=[]
        max_length=max(memory_lengths)

        for length,h in zip(memory_lengths,memory_bank[1:]):
            m=nn.ZeroPad2d((0,0,0,max_length-length))
            padded_memory_bank.append(m(h))
        
        padded_memory_bank=torch.stack(padded_memory_bank,dim=1) #[max_t,b,h]
        memory_lengths = torch.tensor(memory_lengths,
                                      dtype=torch.long,
                                      device=padded_memory_bank.device)
        
        """
        if self.attention_encoder is not None:
            padded_memory_bank = self.attention_encoder(
                padded_memory_bank,
                memory_lengths,
                reaction_batch.distances
            )
        """

        self.decoder.state["src"] = np.zeros(max_length)

        return padded_memory_bank,memory_lengths

    def decoderT(self,src_batch:G2SBatch):
        padded_memory_bank, memory_lengths = self.sharencoder(src_batch)
        # adapted from onmt.models
        dec_in = src_batch.tgt_token_ids[:, :-1]                       # pop last, insert SOS for decoder input
        # m = nn.ConstantPad1d((1, 0), self.vocab["_SOS"])
        # dec_in = m(dec_in)
        dec_in = self.constantpad1d(dec_in)
        dec_in = dec_in.transpose(0, 1).unsqueeze(-1)                       # [b, max_tgt_t] => [max_tgt_t, b, 1]

        dec_outs, _ = self.decoder(
            tgt=dec_in,
            memory_bank=padded_memory_bank,
            memory_lengths=memory_lengths
        )

        dec_outs = self.output_layer(dec_outs)                                  # [t, b, h] => [t, b, v]
        dec_outs = dec_outs.permute(1, 2, 0)                                    # [t, b, v] => [b, v, t]

        loss = self.criterion(
            input=dec_outs,
            target=src_batch.tgt_token_ids
        )
        predictions = torch.argmax(dec_outs, dim=1)                             # [b, t]
        mask = (src_batch.tgt_token_ids != self.vocab["_PAD"]).long()
        accs = (predictions == src_batch.tgt_token_ids).float()
        accs = accs * mask
        acc = accs.sum() / mask.sum()
        
        return loss,acc

    def simclrL(self,src_batch:G2SBatch,tgt_batch:G2SBatch):
        src_emb,_=self.sharencoder(src_batch)
        tgt_emb,_=self.sharencoder(tgt_batch)
        src_1=torch.sum((src_emb.transpose(0,1)),dim=1)
        tgt_1=torch.sum((tgt_emb.transpose(0,1)),dim=1)
        src_2=self.head2(src_1)
        tgt_2=self.head2(tgt_1)

        loss=others_simclr_loss(src_2,tgt_2,tau=100)
        return loss

    def forward(self,src_batch:G2SBatch,tgt_batch:G2SBatch):
        
        loss1,acc = self.decoderT(src_batch)
        loss2 = self.simclrL(src_batch,tgt_batch)
        return loss1,loss2,acc

