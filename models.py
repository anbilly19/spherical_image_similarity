import timm
import torch
import numpy as np
def load_model(model_name):
    model = timm.create_model(
        'vit_base_patch14_dinov2.lvd142m',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )

    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    return model,transform

def extract_embeddings(model,dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    emb_list = []
    with torch.no_grad():
        for img,_ in dataloader:
            output = model.forward_features(img.to(device))
            # output is unpooled, a (1, 768, 12, 12) shaped tensor

            embeddings = model.forward_head(output, pre_logits=True).cpu()
            emb_list.append(embeddings)
    emb_tensor = torch.from_numpy(np.array(emb_list)).squeeze(1)
    return emb_tensor