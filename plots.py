from PIL import Image
import numpy as np
def save_error_image(error_matrix, data_name, model_name):
    pixel_matrix= error_matrix * 255
    error_image = Image.fromarray(pixel_matrix).convert("L") # convert for grayscale
    error_image.save(f'errors/{data_name}/{model_name.split("/")[-1]}_gray.png')

def save_sim_image(sim_matrix,data_name,model_name):
    sim_matrix[sim_matrix < 0] = 0 #handles negatives
    pixel_matrix = sim_matrix * 255
    pixel_matrix[np.diag_indices_from(pixel_matrix)] = 127
    sim_image = Image.fromarray(pixel_matrix).convert("L") # convert for grayscale
    sim_image.save(f'sim_image/{data_name}/{model_name.split("/")[-1]}.png')