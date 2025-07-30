import timm
import torch
from urllib.request import urlretrieve
import os
def load_model(model_name,local,ckpt_path):
    if local:
        ckpt_path = ckpt_path if ckpt_path else f"ckpt/{model_name}/model.safetensors"
        if not os.path.isfile(ckpt_path):
            os.makedirs(os.path.dirname(ckpt_path), exist_ok = True)
            url = (f'https://huggingface.co/timm/{model_name}/resolve/main/model.safetensors')
            _,_ = urlretrieve(url, ckpt_path)
        
        model = timm.create_model(
            model_name,
            pretrained=True,
            pretrained_cfg_overlay=dict(file=ckpt_path),
            num_classes=0,  # remove classifier nn.Linear
        )
    else:
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )

    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    print(f"{model_name}: {transform=}")
    return model,transform

def extract_embeddings(model,dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    emb_list = []
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            output = model.forward_features(img)  # Shape: (batch_size, 768, 12, 12)
            embeddings = model.forward_head(output, pre_logits=True)  # Shape: (batch_size, emb_dim)
            emb_list.append(embeddings.cpu())  # Move to CPU and collect batch

    # Concatenate all batches along the first dimension
    emb_tensor = torch.cat(emb_list, dim=0)  # Final shape: (total_samples, emb_dim)
    return emb_tensor