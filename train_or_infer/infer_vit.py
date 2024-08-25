#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append(".")
sys.path.append("..")

from PIL import Image
import torch
from torchvision import datasets, transforms

from models.vit import VIT
from configs import config_vit as cfg_vit
from einops import rearrange, repeat

def infer(
    model_path = "checkpoints/model.pt.0.11",
    image_path = "data/dogs-vs-cats/test/100.jpg"
):
    model = VIT(
        image_size = cfg_vit.image_size,
        patch_size = cfg_vit.patch_size,
        num_classes = cfg_vit.num_classes,
        dim = cfg_vit.dim,
        depth = cfg_vit.depth,
        heads = cfg_vit.heads,
        mlp_dim = cfg_vit.mlp_dim,
        pool = cfg_vit.pool,
        channels = cfg_vit.channels,
        dim_head = cfg_vit.dim_head,
        dropout = cfg_vit.dropout,
        emb_dropout = cfg_vit.emb_dropout
    )

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(checkpoint)
    model.eval()
    img = img = Image.open(image_path)
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    img_transformed = test_transforms(img)
    img_transformed = rearrange(img_transformed, 'c h w -> 1 c h w')

    label = model(img_transformed)
    print(label)
    label = label.argmax(dim = 1)
    print(label)

if __name__ == "__main__":
    infer()