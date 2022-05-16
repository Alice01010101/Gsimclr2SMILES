from numpy import argsort
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from my_utils.chem_utils import ATOM_FDIM,BOND_FDIM
from my_utils.data_utils import G2SBatch
from my_models.graphfeat import GraphFeatEncoder 
from my_models.attention_xl import AttnEncoderXL

from torch_geometric.nn import Set2Set

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fcs(x) + self.linear_shortcut(x)

class Gsimclr(nn.Module):
    def __init__(self,args,feature_dim=64):
        super(Gsimclr,self).__init__()
        self.args=args

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
        
        #used for producing a global-embedding
        #dim从512->256
        #self.set2set=Set2Set(256,processing_steps=3) 

        # projection head 
        self.head=MLP(128,feature_dim)
        
        self.g = nn.Sequential(
                               nn.Linear(128,128,bias=True),
                               nn.BatchNorm1d(128),
                               nn.ReLU(),
                               nn.Linear(128, feature_dim, bias=True))
        
        """
        self.g = nn.Sequential(
            nn.Linear(128,feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.PReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.PReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(128, feature_dim)
        """

    def f(self,reaction_batch:G2SBatch):

        #print('reaction_batch.size',reaction_batch.size) #b
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
        
        
        if self.attention_encoder is not None:
            
            padded_memory_bank = self.attention_encoder(
                padded_memory_bank,
                memory_lengths,
                reaction_batch.distances
            )
            
        return padded_memory_bank,memory_lengths

    def forward(self, x):
        """
        local_emb,memory_lengths= self.f(x) #[max_t,b,h]
        local_emb=local_emb.transpose(0,1) #[b,max_t,h]
        global_emb=torch.sum(local_emb,dim=1)
        b=local_emb.shape[0]
        max_t=local_emb.shape[1]
        hid=local_emb.shape[2]
        local_emb=local_emb.reshape(-1,hid) #[b*max_t,h]

        local_enc=self.head(local_emb)
        global_enc=self.head(global_emb)
    
        local_enc=self.g(local_emb)
        global_enc=self.g(global_emb)

        #local_enc=local_enc.transpose(1,2) #[b,max_t, h] 
        local_enc=local_enc.reshape(b,max_t,-1)
        mlen=len(memory_lengths)
        #return local_enc,global_enc,mlen
        """
        local_emb,_= self.f(x) #[max_t,b,h]
        local_emb=local_emb.transpose(0,1) #[b,max_t,h]
        global_emb=torch.sum(local_emb,dim=1) #[b,h]
        global_enc=self.head(global_emb)
        return global_enc