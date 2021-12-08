#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2021/12/8 15:22
"""
import os

import cv2
import numpy as np


IMAGE_ROOT = './data/train_data/train_image'


def cal():
    mean, std = None, None
    images = os.listdir(IMAGE_ROOT)
    for image in images:
        im = cv2.imread(os.path.join(IMAGE_ROOT, image))
        im = im[:, :, ::-1] / 255.
        if mean is None and std is None:
            mean, std = cv2.meanStdDev(im)
        else:
            mean_, std_ = cv2.meanStdDev(im)
            mean_stack = np.stack((mean, mean_), axis=0)
            std_stack = np.stack((std, std_), axis=0)
            mean = np.mean(mean_stack, axis=0)
            std = np.mean(std_stack, axis=0)
    return mean.reshape((1, 3))[0], std.reshape((1, 3))[0]


if __name__ == '__main__':
    mean, std = cal()
    print(mean, std)
    # [0.51231626 0.54201973 0.41985212] [0.23131444 0.22577731 0.24543156]
