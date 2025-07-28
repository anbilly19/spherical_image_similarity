import json
from pathlib import Path
from utils import compute_scores
import numpy as np
from utils import save_sim_json
from plots import save_sim_image

def get_gt_matrix(folder_name):
    json_file_path = Path(folder_name) / 'similarity_map.json'
    with open(json_file_path) as jsn:
        sim_dict = json.load(jsn)
    size = len(sim_dict.keys())
    mat = np.zeros([size,size])
    for i in range(size):
        for j in range(i+1,size):
            try:
                mat[i][j] = sim_dict[str(i)][str(j)]
            except KeyError:
                mat[i][j] = 0.0
      
    return mat

def get_sim_matrix(emb,save:bool,folder_name,model_name):
    sim = np.zeros([emb.shape[0],emb.shape[0]])
    # with open('score.txt','w+') as sc:
    for i in range(emb.shape[0]):
        for j in range(i+1, emb.shape[0]):
            score = compute_scores(emb[i],emb[j])
            
            # sc.write(f"{score} {i} {j}")
            # sc.write('\n')
            sim[i][j] = score
    if save:
        save_sim_image(np.copy(sim),folder_name,model_name)
        save_sim_json(sim,folder_name,model_name)
    return sim

def get_error_matrix(sim,gt):
    assert np.isfinite(sim).any()
    assert np.isfinite(gt).any()
    # error_mat = abs(gt-sim)
    error_mat = 0.5*(sim - gt) + 0.5 
    error_mat[np.diag_indices_from(error_mat)] = -1
    return error_mat