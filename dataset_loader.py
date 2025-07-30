from torchvision.datasets import ImageFolder
from pathlib import Path
from torch.utils.data import DataLoader

def load_dataset(folder_name,transform):
    image_folder_name = Path(folder_name)
    dataset = ImageFolder(image_folder_name, transform)
    loader = DataLoader(dataset, shuffle=False, batch_size = 16)
    return loader
        