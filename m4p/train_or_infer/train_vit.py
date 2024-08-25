#!/usr/bin/env python
# coding=utf-8

import sys
sys.path.append(".")
sys.path.append("..")

import torch
from models.vit import VIT
from configs import config_vit as cfg_vit

def main():
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

    image = torch.randn(1, 3, 256, 256)
    preds = model(image)
    print(preds)


if __name__ == "__main__":
    main()