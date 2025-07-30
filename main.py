from plots import save_error_image
from models import load_model, extract_embeddings
from dataset_loader import load_dataset
from matrix_maker import get_gt_matrix, get_sim_matrix, get_error_matrix
from jsonargparse import ArgumentParser
from typing import List

parser = ArgumentParser()
parser.add_argument("--data_name", default="tokyo", help="dataset folder")
parser.add_argument("--model_list", type=List[str])
parser.add_argument("--save", type=bool, default=False, help="To save the similarity image and json map")
parser.add_argument("--local", type=bool, default=True, help="To access model locally, downloads if not present")
parser.add_argument("--ckpt_path",default=None, type=str, help="local path to model archive file")
parser.add_argument("--config",default = "config.yaml", action="config") # overrides defaults
cfg = parser.parse_args()
if __name__ == "__main__":
    for model_name in cfg.model_list:
        
        model, transform = load_model(model_name,cfg.local,cfg.ckpt_path)
        dataloader = load_dataset(cfg.data_name,transform)
        gt_matrix = get_gt_matrix(cfg.data_name)
        embeddings_tensor = extract_embeddings(model,dataloader)
        sim_matrix = get_sim_matrix(embeddings_tensor,cfg.save,cfg.data_name,model_name)
        error_matrix = get_error_matrix(sim_matrix,gt_matrix)
        save_error_image(error_matrix,cfg.data_name,model_name)