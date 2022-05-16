import torch
import numpy as np
import math
import torch.nn.functional as F
 
def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    norm_dot_product=np.dot(z_i,z_j)/(torch.linalg.norm(z_i)*torch.linalg.norm(z_j))
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        sum_k=0
        sum_k_N=0

        for i in range(2*N):
            z_i=out[i]
            if i !=k+N:
                sum_k +=torch.exp(sim(z_k,z_i)/tau)
            if i !=k:
                sum_k_N +=torch.exp(sim(z_k_N,z_i)/tau)

        loss_k=-np.log(torch.exp(sim(z_k,z_k_N)/tau)/sum_k)
        loss_k_N=-np.log(torch.exp(sim(z_k_N,z_k)/tau)/sum_k_N)
        total_loss +=loss_k+loss_k_N
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None

    left_norm=out_left/torch.linalg.norm(out_left,dim=1,keepdim=True)
    right_norm=out_right/torch.linalg.norm(out_right,dim=1,keepdim=True)
    out=torch.mm(left_norm,right_norm.T) #(N,N)
    pos_pairs=torch.diag(out).view(-1,1) #(N,1)

    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None

    out_norm=out/torch.linalg.norm(out,dim=1,keepdim=True)
    sim_matrix=torch.mm(out_norm,out_norm.T)

    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    exponential = torch.exp(sim_matrix)/tau
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    denom = torch.sum(exponential,dim=1,keepdim=True)

    sim_pairs=sim_positive_pairs(out_left,out_right) #(N,1)
    sim_pairs=torch.cat([sim_pairs,sim_pairs],dim=0)

    numerator = None
    numerator=torch.exp(sim_pairs/tau).to(device)

    loss = torch.mean(-torch.log(numerator/denom))
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def calculate_loss(reactant_embeddings, product_embeddings, args,device='cuda'):

    #求两个向量的L2-norm
    dist = torch.cdist(reactant_embeddings, product_embeddings, p=2)

    #print('dist.size()',dist.size()) [b,b]
    batch_size=len(dist)
    #取对角线上的值
    pos = torch.diag(dist)
    mask = torch.eye(batch_size)
    if torch.cuda.is_available():
        mask = mask.to(device)
    neg = (1 - mask) * dist + mask * args.margin
    neg = torch.relu(args.margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / batch_size / (batch_size - 1)

    return loss



def others_simclr_loss(x, x_aug,tau):
    
    T = tau
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss

def get_positive_expectation(p_samples, average=True):
    """Computes the positive part of a JS Divergence.
    Args:
        p_samples: Positive samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Ep = log_2 - F.softplus(- p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, average=True):
    """Computes the negative part of a JS Divergence.
    Args:
        q_samples: Negative samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Eq = F.softplus(-q_samples) + q_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq

def local_global_loss_(l_enc,g_enc,mlen):
    """
    print('mlen',mlen)
    print('num_graphs',num_graphs)
    print('num_nodes',num_nodes)
    mlen 128
    num_graphs 128
    num_nodes 5504
    """
    max_t=l_enc.shape[0]
    hid=l_enc.shape[2]
    l_enc=l_enc.reshape(-1,hid)

    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    

    device = g_enc.device

    pos_mask = torch.zeros((num_nodes,num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes,num_graphs)).to(device)

    total_len=0
    for i in range(mlen):
        for j in range(max_t):
            pos_mask[total_len+j][i]=1.
            neg_mask[total_len+j][i]=0.
        total_len +=max_t
    
    res = torch.mm(l_enc,g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos
