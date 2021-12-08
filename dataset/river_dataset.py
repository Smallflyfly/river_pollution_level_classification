#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：fangpf
@Date    ：2021/12/8 15:18 
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class RiverDataset(Dataset):
    def __init__(self, data_root, csv_file, image_size):
        super(RiverDataset, self).__init__()
        self.data_root = data_root
        self.train_images = []
        self.train_labels = []
        self.image_size = image_size

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.Tensor(),
            transforms.Resize((448, 448)),
            transforms.Normalize([0.33708435, 0.42723662, 0.41629601], [0.2618102, 0.31948383, 0.33079577])
        ])

        self.init_data()

    def init_data(self):
        a = 1
