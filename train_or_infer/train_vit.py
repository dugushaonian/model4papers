#!/usr/bin/env python
# coding=utf-8

import gin
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from absl import flags
from absl import app

from m4p.models.vit import VIT, VitConfig, ViT
from m4p.dataprocess import dogs_vs_cats

@gin.configurable
def train_vit(
    cfg_vit: VitConfig,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1,
    train_dataset_dir: str = "data/dogs-vs-cats/train/",
    test_dataset_dir: str = "data/dogs-vs-cats/test/",
    amp: bool = False,
    mixed_precision_type: str = "fp16"
) -> None:
    accelerator = Accelerator(
        split_batches = True,
        mixed_precision=mixed_precision_type if amp else 'no'
    )
    device = accelerator.device

    # dataset
    train_loader, valid_loader, test_loader = dogs_vs_cats.dogs_vs_cats_dataloader(train_dataset_dir, test_dataset_dir, batch_size)
    dataloader, valid_loader, test_loader = accelerator.prepare(train_loader, valid_loader, test_loader)

    # model
    model = ViT(cfg_vit)

    # optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr = learning_rate,
        weight_decay=weight_decay
    )

    # scheduler
    scheduler = LinearLR(optimizer)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    CEL = nn.CrossEntropyLoss()

    with tqdm(initial=0, total=epochs * len(train_loader), disable=not Accelerator.is_main_process) as pbar:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            epoch_accuracy = 0
            optimizer.zero_grad()
            train_len = 1
            for i, (data, label) in enumerate(dataloader):
                data = data.to(device)
                label = label.to(device)
                # print(data.shape)
                with accelerator.autocast():
                    pred = model(data)
                    loss = CEL(pred, label)
                    acc = (pred.argmax(dim=1) == label).float().mean()
                    epoch_loss += loss
                    epoch_accuracy += acc
                    pbar.set_description(f'loss: {loss:.4f}, acc: {acc:.4f}')
                    pbar.update(1)
                accelerator.backward(loss)
                train_len += 1
            epoch_loss = epoch_loss / train_len
            epoch_accuracy = epoch_accuracy / train_len

            epoch_val_loss = 0
            epoch_val_acc = 0
            epoch_val_len = 1
            with torch.no_grad():
                for data, label in valid_loader:
                    epoch_val_len += 1
                    data = data.to(device)
                    label = label.to(device)
                    val_pred = model(data)
                    val_loss = CEL(val_pred, label)
                    acc = (val_pred.argmax(dim = 1) == label).float().mean()
                    epoch_val_acc += acc
                    epoch_val_loss += val_loss
            epoch_val_loss /= epoch_val_len
            epoch_val_acc /= epoch_val_len
            pbar.set_description(f'epoch_loss: {epoch_loss:.4f}, epoch_accuracy: {epoch_accuracy:.4f}, epoch_val_loss: {epoch_val_loss:.4f}, epoch_val_acc: {epoch_val_acc:.4f}')

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            accelerator.wait_for_everyone()
            pbar.update(1)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, f"checkpoints/model.pt.{epoch}"
            )
    

def main(_argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_vit(VitConfig())


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        'gin_file', None, 'path'
    )
    flags.DEFINE_multi_string(
            'gin_param', None, 'newline'
        )
    FLAGS = flags.FLAGS
    app.run(main)
