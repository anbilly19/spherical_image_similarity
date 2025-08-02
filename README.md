Configure model_name from timm and folder_name of any dataset

### Usage
Python >=3.10
Entry point: ```python main.py```

Arguments can be supplied via config.yaml or in the run command.
Any command line argument supplied after the config file will override that value.
For e.g.
if in config.yaml 
```local:True```
User runs the following command: ```python main.py --config config.yaml --local False```
Code will run with ```local=False```

if ```ckpt_path``` is provided model will be loaded from that file. It can any of (.safetensors | .bin). [Reference on timm module usage](https://github.com/huggingface/pytorch-image-models/discussions/1826#discussioncomment-599645)


Models tested:
1. [convnextv2_base.fcmae_ft_in22k_in1k_384](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384)
2. [vit_base_patch16_rope_mixed_ape_224.naver_in1k](https://huggingface.co/timm/vit_base_patch16_rope_mixed_ape_224.naver_in1k)
3. [vit_pe_core_large_patch14_336.fb](https://huggingface.co/timm/vit_pe_core_large_patch14_336.fb)
4. [vit_base_r50_s16_224.orig_in21k](https://huggingface.co/timm/vit_base_r50_s16_224.orig_in21k)
5. [vit_pe_spatial_large_patch14_448.fb](https://huggingface.co/timm/vit_pe_spatial_large_patch14_448.fb)
6. [vit_base_patch14_dinov2.lvd142m](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m)

### Expected outputs
errors/<data_name>: contains error matrix images for different models.
```if save=True```
    sim_son/<data_name>: contains similarity json in the format of similarity_map for different models.
    sim_image/<data_name>: contains similarity matrix images for different models.  
