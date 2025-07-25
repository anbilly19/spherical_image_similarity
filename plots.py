from PIL import Image

def save_error_image(error_matrix, data_name, model_name):
    pixel_matrix= error_matrix * 255
    error_image = Image.fromarray(pixel_matrix).convert("L") # convert for grayscale
    error_image.save(f'errors/{data_name}/{model_name.split("/")[-1]}_gray.png')