from torch.nn.functional import cosine_similarity

import json
def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = cosine_similarity(emb_one, emb_two, dim = 0)
    return scores.numpy()

def save_sim_json(sim,data_name,model_name):
    n = len(sim)
    sim_dict = {
        str(i): {str(j): sim[i][j] for j in range(i+1,n)}
        for i in range(n)
    }
    with open(f"sim_son/{data_name}/{model_name}.json","w+") as js:
        json.dump(sim_dict,js)
    