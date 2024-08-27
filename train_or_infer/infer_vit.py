#!/usr/bin/env python
# coding=utf-8

import torch

from PIL import Image
from einops import rearrange
from torchvision import transforms

from m4p.models.vit import VIT, VitConfig, ViT

from absl import flags
from absl import app

@gin.configurable
def infer_vit(
    model_path: str = "checkpoints/model.pt.0",
    image_path: str = "data/dogs-vs-cats/test/1.jpg"
):
    model = ViT(VitConfig())

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
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


def main(_argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    infer_vit(VitConfig())


if __name__ == "__main__":
    flags.DEFINE_multi_string('gin_file', None, 'path')
    flags.DEFINE_multi_string('gin_param', None, 'newline')
    FLAGS = flags.FLAGS
    app.run(main)