from plots import save_error_image
from models import load_model, extract_embeddings
from dataset_loader import load_dataset
from matrix_maker import get_gt_matrix, get_sim_matrix, get_error_matrix

model_list = [
    'vit_pe_spatial_large_patch14_448.fb', 
                  'convnextv2_base.fcmae_ft_in22k_in1k_384',
                  'vit_base_patch14_dinov2.lvd142m',
                  'vit_base_patch16_rope_mixed_ape_224.naver_in1k',
                  'vit_base_r50_s16_224.orig_in21k',
                  'vit_pe_core_large_patch14_336.fb',
                  'vit_pe_spatial_large_patch14_448.fb',
                  'hf-hub:MahmoodLab/uni'
                  ]
folder_name = 'shapespark'
if __name__ == "__main__":
    for model_name in model_list:
        
        model, transform = load_model(model_name)
        dataloader = load_dataset(folder_name,transform)
        gt_matrix = get_gt_matrix(folder_name)
        embeddings_tensor = extract_embeddings(model,dataloader)
        sim_matrix = get_sim_matrix(embeddings_tensor)
        error_matrix = get_error_matrix(sim_matrix,gt_matrix)
        save_error_image(error_matrix,folder_name,model_name)