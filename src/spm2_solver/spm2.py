import torch
import numpy as np
from multiprocessing import Pool
from torch import Tensor
from IPython.core.debugger import Tracer


def spm2(s: Tensor, os: Tensor, wm1: torch.Tensor, rho, n1: Tensor=None, n2: Tensor=None, nproc: int=1) -> Tensor:
    
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))
        
    
    if len(os.shape) == 2:
        os = os.unsqueeze(0)

    elif len(os.shape) == 3:
        pass
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))

    
    device = s.device
    
    
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach()
    os_mat= os.cpu().detach()
    
    
    
    wm1 = wm1.cpu().detach()
    
    if n1 is not None:
        n1 = n1.cpu().detach()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().detach()
    else:
        n2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_spm2_kernel, zip(perm_mat, os_mat, wm1,rho, n1, n2))
            perm_mat = np.stack(mapresult.get())
            
    else:
        perm_mat = np.stack([_spm2_kernel(perm_mat[b], os_mat[b], wm1, rho, n1[b], n2[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)
    wm1 = wm1.to(device)
    
    
    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat



def _spm2_kernel(s: torch.Tensor, os: torch.Tensor,  wm1: torch.Tensor, rho, n1=None, n2=None):
       
        
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    
    n1d = n1
    n2d = n2
        
    rho = rho
    
   
  
    s_mat = s
    s_sliced = s[:n1, :n2]
    os_sliced = os[:n1, :n2]
    
    
    os_sliced = torch.clamp(os_sliced, min=0.0)
    wm1 = torch.clamp(wm1, min=0.0)
        
    alpha_6,_=torch.max(os_sliced,dim=1)
    beta_6,_=torch.max(os_sliced,dim=0)
    
 
        
    w1 = (torch.sigmoid(wm1*alpha_6)-0.5)*2
    w2 = (torch.sigmoid(wm1*beta_6)-0.5)*2
    
    w1 = w1.reshape(-1,1)
    w2 = w2.reshape(1, -1)
    
    w1 = w1.expand(n1,n2)
    w2 = w2.expand(n1,n2)
    
    c_base = ((w1+w2)*rho).numpy()
    n = max(n1,n2)
    perm_mat = 0
    
    
    s_sliced = s_sliced.numpy()
    perm_mat_t1= stable_matching(s_sliced, c_base)
    s_size_mat_row_difference = s.shape[0] - n1
    s_size_mat_column_difference = s.shape[1] - n2
    perm_mat = np.pad(perm_mat_t1,((0,s_size_mat_row_difference),(0,s_size_mat_column_difference)),'constant')
    
    
 
   
    return perm_mat




def create_candidate_mat(similarity_mat, threshold):
    
    setA_maxSimilarities = similarity_mat.max(axis=1)

    setA_candidate_mat = (
        (similarity_mat == setA_maxSimilarities[:, np.newaxis]) & (similarity_mat >= 1 - threshold)
    )
 
   
    setB_maxSimilarities = similarity_mat.max(axis=0)
   
    setB_candidate_mat = (
        (similarity_mat.T == setB_maxSimilarities[:, np.newaxis]).T & (similarity_mat >= 1 - threshold)
        
    ).T
    
    
    return setA_candidate_mat, setB_candidate_mat

def stable_matching(similarity_mat, threshold):
    n_setA, n_setB = similarity_mat.shape
    
    setA_candidate_mat, setB_candidate_mat = create_candidate_mat(similarity_mat,threshold)
    
    setB_matches = [-1] * n_setB
    matching_mat = np.zeros((n_setA, n_setB), dtype=int)

    
    unmatched_A = list(range(n_setA))
    setA_proposals = np.zeros(n_setA, dtype=int)
    
    while unmatched_A:
        node_A = unmatched_A.pop(0)
        
        node_A_candidates = np.where(setA_candidate_mat[node_A])[0]
        
        
        
        while setA_proposals[node_A] < len(node_A_candidates):
            node_B = node_A_candidates[setA_proposals[node_A]]
            setA_proposals[node_A] += 1
            
            
            if not setB_candidate_mat[node_B, node_A]:
                continue
            
            if setB_matches[node_B] == -1:
                
                setB_matches[node_B] = node_A
                matching_mat[node_A, node_B] = 1
                break
    
    return matching_mat