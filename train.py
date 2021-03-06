#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：fangpf
@Date    ：2021/12/8 14:56 
'''
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision.models import resnet50

from config import get_config
from dataset.river_dataset import RiverDataset
from logger import create_logger
from model.build import build_model
from utils.utils import load_pretrained_weights, build_optimizer, build_scheduler

import tensorboardX as tb

parser = argparse.ArgumentParser(description="river pollution level classify")
parser.add_argument("--b", type=int, default=16, help="train batch size")
parser.add_argument("--e", type=int, default=20, help="train epochs")
parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")

args = parser.parse_args()

BATCH_SIZE = args.b
EPOCHS = args.e

NUM_CLASSES = 4


def train(config):
    dataset = RiverDataset('data/train_data', 'data/train_data/train_label.csv', 448)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    '''
    model = resnet50(num_classes=NUM_CLASSES)
    load_pretrained_weights(model, 'weights/resnet50-19c8e357.pth')
    '''
    model = build_model(config)
    load_pretrained_weights(model, 'weights/swin_tiny_patch4_window7_224.pth')

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # print(model)

    model = model.cuda()
    optimizer = build_optimizer(model, 'adam', lr=0.0005)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=EPOCHS)
    loss_func = nn.CrossEntropyLoss()
    model.train()

    writer = tb.SummaryWriter()

    for epoch in range(EPOCHS):
        for index, data in enumerate(train_loader):
            im, label = data
            im = im.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()
            out = model(im)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                num_epoch = epoch * len(train_loader) + index
                writer.add_scalar('loss', loss, num_epoch)

            if index % 20 == 0:
                logger.info(
                    'Epoch: [{}/{}] [{}/{}] loss = {:.4f}'.format(epoch + 1, EPOCHS, index + 1, len(train_loader),
                                                                  loss))

        scheduler.step()

        if (epoch + 1) % 10 == 0 and epoch + 1 != EPOCHS:
            name = 'weights/swin_transform_{}.pth'.format(epoch + 1)
            torch.save(model.state_dict(), name)

    torch.save(model.state_dict(), 'weights/swin_transform_last.pth')
    writer.close()


if __name__ == '__main__':
    config = get_config(args)
    train(config)
