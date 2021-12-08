#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：fangpf
@Date    ：2021/12/8 15:18 
'''
import os

import pandas
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class RiverDataset(Dataset):
    def __init__(self, data_root, csv_file, image_size):
        super(RiverDataset, self).__init__()
        self.data_root = data_root
        self.csv_file = csv_file
        self.train_images = []
        self.train_labels = []
        self.image_size = image_size
        self.image_label_map = {}

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize([0.51231626, 0.54201973, 0.41985212], [0.23131444, 0.22577731, 0.24543156])
        ])

        self.read_csv()

        self.init_data()

    def read_csv(self):
        dataframe = pandas.read_csv(self.csv_file)
        images = dataframe['image_name']
        labels = dataframe['label']
        for image, label in zip(images, labels):
            label = int(label)
            self.image_label_map[image] = label

    def init_data(self):
        images = os.listdir(os.path.join(self.data_root, 'train_image'))
        for image in images:
            label = self.image_label_map[image]
            image = os.path.join(os.path.join(self.data_root, 'train_image'), image)
            self.train_images.append(image)
            self.train_labels.append(label)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        image = self.train_images[index]
        label = self.train_labels[index]
        im = Image.open(image)
        im = self.transforms(im)
        return im, label