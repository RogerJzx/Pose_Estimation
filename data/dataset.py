import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T

from torch.utils import data
import random
import cv2
def change_bg(im, mask, bg):
    bg = cv2.resize(bg, (im.shape[1], im.shape[0]))

    im_mask = cv2.bitwise_and(im, mask)
    return im_mask + cv2.bitwise_and(bg, cv2.bitwise_not(mask))
def load_data(imgpath, shape, bg_path):
    # print(imgpath)
    label_path = imgpath.replace("pcg", 'labels').replace("mask", 'labels').replace('jpg', 'txt')
    mask_path = imgpath.replace("pcg", 'mask')#.replace('/00', '/').replace('jpg', 'png')
    # print(imgpath, cv2.imread(imgpath))
    im = cv2.imread(imgpath)[..., ::-1]
    mask = cv2.imread(mask_path)[..., ::-1]
    bg = cv2.imread(bg_path)[..., ::-1]

    # print(label_path, mask_path, imgpath)
    with open(label_path) as f:
        label = f.readline()

    # print(label)
    # raw point xyz
    label = [float(i) for i in label.split()]
    img_bg = change_bg(im, mask, bg)
    # img_bg = im

    im_resize = cv2.resize(img_bg, shape)


    sx, sy = float(shape[0] * 1. / img_bg.shape[0]), float(shape[1] * 1. / img_bg.shape[1])
    label.extend([0] * (1050 - len(label)))

    for j in range(50 - 1):

        if label[21 * j + 1] == 0:
            break
        for i in range(9):
            xx = int(label[21 * j + i * 2 + 1] * im.shape[0] * sx)
            yy = int(label[21 * j + i * 2 + 2] * im.shape[1] * sy)
            label[21 * j + i * 2 + 1] = xx * 1. / shape[0]
            label[21 * j + i * 2 + 2] = yy * 1. / shape[1]

    return im_resize, np.array(label)


def load_data1(imgpath, shape, bg_path):
    label_path = imgpath.replace("JPEGImages", 'labels').replace('jpg', 'txt')
    mask_path = imgpath.replace("JPEGImages", 'mask').replace('/00', '/').replace('jpg', 'png')
    #print(imgpath, cv2.imread(imgpath))
    im = cv2.imread(imgpath)[..., ::-1]
    mask = cv2.imread(mask_path)[..., ::-1]
    bg = cv2.imread(bg_path)[..., ::-1]

    with open(label_path) as f:
        label = f.readline()

    # raw point xyz
    label = [float(i) for i in label.split()]
    img_bg = change_bg(im, mask, bg)

    im_resize = cv2.resize(img_bg, shape)

    sx, sy = float(shape[0] * 1. / img_bg.shape[0]), float(shape[1] * 1. / img_bg.shape[1])
    label.extend([0] * (1050 - len(label)))

    for j in range(50 - 1):

        if label[21 * j + 1] == 0:
            break
        for i in range(9):
            xx = int(label[21 * j + i * 2 + 1] * im.shape[0] * sx)
            yy = int(label[21 * j + i * 2 + 2] * im.shape[1] * sy)
            label[21 * j + i * 2 + 1] = xx * 1. / shape[0]
            label[21 * j + i * 2 + 2] = yy * 1. / shape[1]

    return im_resize, np.array(label)


class listDataset(data.Dataset):
    def __init__(self, root_path, shape=None, shuffle=True, transform=None,
                 target_transform=None, train=False, bg_file_names=None):
        self.prefix = '/'.join(root_path.split('/')[:-3])
        with open(root_path, 'r') as f:
            self.lines = f.readlines()
        if shuffle:
            random.shuffle(self.lines)
        self.nSample = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.shape = shape
        self.bg_files_names = bg_file_names
        self.train = train

    def __getitem__(self, index):

        imgpath = self.prefix + '/' + self.lines[index].rstrip()

        random_bg_index = random.randint(0, len(self.bg_files_names) - 1)

        bg_path = self.bg_files_names[random_bg_index]

        img, label = load_data(imgpath, self.shape, bg_path)
        #print(imgpath)
        if not self.train:

            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
        label = torch.from_numpy(label)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

    def __len__(self):
        return self.nSample

