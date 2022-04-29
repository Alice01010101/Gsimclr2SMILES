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
class Gsimclr(nn.Module):
    def __init__(self,args,feature_dim=128):
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
        self.g = nn.Sequential(
                               nn.Linear(256,256,bias=True),
                               nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True),
                               nn.Linear(256, feature_dim, bias=True))


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
        x_1,memory_lengths= self.f(x) #[max_t,b,h]
        ################################################################
        #方法一：使用torch.mean()
        #x_2=torch.mean((x_1.transpose(0,1)),dim=1)       #->[b,max_t,h]->[b,h]
        #方法二：使用torch.sum()
        x_2=torch.sum((x_1.transpose(0,1)),dim=1)
        """
        #方法三：使用Set2Set()
        max_t=max(memory_lengths) #之前已经使用长度补齐的操作了
        batch_index=[idx for idx,_ in enumerate(memory_lengths) for i in range(max_t)]
        batch_index=torch.tensor(batch_index).cuda()
        x_1_1=x_1.transpose(0,1)
        #print('x_1_1.size()',x_1_1.size())
        x_1_1=x_1_1.reshape(-1,256)
        x_2=self.set2set(x_1_1,batch_index)
        #print('x_2.size()',x_2.size())
        """
        ################################################################
        #print('x_2.size()',x_2.size())
        x_3=self.g(x_2)
        #print('x_3.size()',x_3.size())

        return x_3
