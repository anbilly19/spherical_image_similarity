from plots import save_error_image
from models import load_model, extract_embeddings
from dataset_loader import load_dataset
from matrix_maker import get_gt_matrix, get_sim_matrix, get_error_matrix

model_name = 'vit_pe_spatial_large_patch14_448.fb'
folder_name = 'harmony'
if __name__ == "__main__":
    model, transform = load_model(model_name)
    dataloader = load_dataset(folder_name,transform)
    gt_matrix = get_gt_matrix(folder_name)
    embeddings_tensor = extract_embeddings(model,dataloader)
    sim_matrix = get_sim_matrix(embeddings_tensor)
    error_matrix = get_error_matrix(sim_matrix,gt_matrix)
    save_error_image(error_matrix,folder_name,model_name)