import torch
import numpy as np
from multiprocessing import Pool
from torch import Tensor
from IPython.core.debugger import Tracer


def spm1(s: Tensor, os: Tensor, wm1: torch.Tensor, rho, n1: Tensor=None, n2: Tensor=None, nproc: int=1) -> Tensor:
    
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
            mapresult = pool.starmap_async(_spm1_kernel, zip(perm_mat, os_mat, wm1,rho, n1, n2))
            perm_mat = np.stack(mapresult.get())
            
    else:
        perm_mat = np.stack([_spm1_kernel(perm_mat[b], os_mat[b], wm1, rho, n1[b], n2[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)
    wm1 = wm1.to(device)
    
    
    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat

def _spm1_kernel(s: torch.Tensor, os: torch.Tensor,  wm1: torch.Tensor, rho, n1=None, n2=None):
       
        
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    
    
    s_mat = s
    s_sliced = s[:n1, :n2]
    os_sliced = os[:n1, :n2]
    os_sliced = torch.clamp(os_sliced, min=0.0)
  
        
    alpha_6,_=torch.max(os_sliced,dim=1)
    beta_6,_=torch.max(os_sliced,dim=0)
    
        
    w1 = (torch.sigmoid(wm1*alpha_6)-0.5)*2
    w2 = (torch.sigmoid(wm1*beta_6)-0.5)*2
    
    
    w1 = w1.reshape(-1,1)
    w2 = w2.reshape(1, -1)
    
    w1 = w1.expand(n1,n2)
    w2 = w2.expand(n1,n2)
    
    c_base = ((w1+w2)*rho).numpy()
     
    s_mat = s
    s_sliced = s[:n1, :n2]
    
    perm_mat = 0
    
    s_sliced = s_sliced.numpy()
        
    candidate_matrixt = s_sliced
        
    perm_mat_t1= stable_matching(s_sliced,c_base)
        
    s_size_mat_row_difference = s.shape[0] - n1
    s_size_mat_column_difference = s.shape[1] - n2
    perm_mat = np.pad(perm_mat_t1,((0,s_size_mat_row_difference),(0,s_size_mat_column_difference)),'constant')
    
    
    return perm_mat



def create_preferences(similarity_mat,threshold):
    n_setA, n_setB = similarity_mat.shape
    
    non_candidates = similarity_mat < 1 - threshold
    
    setA_similarities = similarity_mat.copy()
    setA_similarities[non_candidates] = -1  
    setA_preferences = (-setA_similarities).argsort(axis=1)
    
   
    setB_choices = np.full((n_setB, n_setA), n_setA)  
    
    setB_sorted_indices = (-similarity_mat.T).argsort(axis=1)
    
    
    valid_candidates = ~non_candidates.T  
    
    for node_B in range(n_setB):
       
        valid_candidate_indices = valid_candidates[node_B]
        valid_candidates_sorted = setB_sorted_indices[node_B][valid_candidate_indices[setB_sorted_indices[node_B]]]
        
        setB_choices[node_B, valid_candidates_sorted] = np.arange(len(valid_candidates_sorted))
    
    candidates_totals_setA = (~non_candidates).sum(axis=1)
    
    return setA_preferences, setB_choices, candidates_totals_setA 

def stable_matching(similarity_mat,threshold):
   
    setA_preferences, setB_choices, candidates_totals_setA = create_preferences(similarity_mat,threshold)
    n_setA, n_setB = similarity_mat.shape
    
    setB_matches = [-1] * n_setB
    setA_proposals = np.zeros(n_setA, dtype=int)
    unmatched_A = list(range(n_setA))
    
    matching_mat = np.zeros((n_setA, n_setB), dtype=int)
    
    while unmatched_A:
        node_A = unmatched_A.pop(0)
        while setA_proposals[node_A] < candidates_totals_setA[node_A]:
            node_B = setA_preferences[node_A][setA_proposals[node_A]]
            setA_proposals[node_A] += 1
            
            if setB_choices[node_B, node_A] == n_setA:  
                continue
                
            if setB_matches[node_B] == -1:
                setB_matches[node_B] = node_A
                matching_mat[node_A][node_B] = 1  
                break
            else:
                current_match = setB_matches[node_B]
                if setB_choices[node_B, node_A] < setB_choices[node_B, current_match]:
                    matching_mat[current_match][node_B] = 0  
                    matching_mat[node_A][node_B] = 1  
                    setB_matches[node_B] = node_A
                    unmatched_A.append(current_match)
                    break
    
    return matching_mat
