import json
from pathlib import Path
from utils import compute_scores
import numpy as np

def get_gt_matrix(folder_name):
    json_file_path = Path(folder_name) / 'similarity_map.json'
    with open(json_file_path) as jsn:
        sim_dict = json.load(jsn)
    size = len(sim_dict.keys())
    mat = np.zeros([size,size])
    for i in range(size):
        for j in range(i+1,size):
            mat[i][j] = sim_dict[str(i)][str(j)]
      
    return mat

def get_sim_matrix(emb):
    sim = np.zeros([emb.shape[0],emb.shape[0]])
    # with open('score.txt','w+') as sc:
    for i in range(emb.shape[0]):
        for j in range(i+1, emb.shape[0]):
            score = compute_scores(emb[i],emb[j])
            
            # sc.write(f"{score} {i} {j}")
            # sc.write('\n')
            sim[i][j] = score
    
    return sim

def get_error_matrix(sim,gt):
    assert np.isfinite(sim).any()
    assert np.isfinite(gt).any()
    return abs(gt-sim)