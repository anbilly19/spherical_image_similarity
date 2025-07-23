from torch.nn.functional import cosine_similarity

def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = cosine_similarity(emb_one, emb_two, dim = 0)
    return scores.numpy()