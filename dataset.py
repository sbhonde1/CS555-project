import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import csv
import os
from torch.nn import functional as F
from collections import deque
from albumentations.pytorch.functional import img_to_tensor
from PIL import Image
SIZE_S = 608
SIZE_R = 419
MODE_TRAIN = 'train'
MODE_TEST = 'test'
MODE_VALID = 'valid'


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_real_box(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            float_left = float(row[0])
            float_top = float(row[1])
            float_right = float(row[2])
            float_bot = float(row[3])
            int_left = int(float_left)  # X0
            int_right = int(float_right)  # X1
            int_top = int(float_top)  # Y0
            int_bot = int(float_bot)  # Y1
            break
    return [float_left, float_top, float_right, float_bot]


def load_synthetic_box(path):
    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            x = float(row[1])
            y = float(row[2])
            w = float(row[3])
            h = float(row[4])
            float_left = (x - w / 2.) * SIZE_S
            float_right = (x + w / 2.) * SIZE_S
            float_top = (y - h / 2.) * SIZE_S
            float_bot = (y + h / 2.) * SIZE_S
            int_left = int(float_left)          # X0
            int_right = int(float_right)        # X1
            int_top = int(float_top)            # Y0
            int_bot = int(float_bot)            # Y1
            break
    return [float_left, float_top, float_right, float_bot]


class MugDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images = list(sorted(os.listdir(os.path.join(self.root, self.mode, 'images'))))
        self.labels = list(sorted(os.listdir(os.path.join(self.root, self.mode, 'labels'))))

    def __len__(self):
        return len(self.images)

    # def __getitem__(self, idx):
    #     img_path = os.path.join(self.root, self.mode, 'images', self.images[idx])
    #     box_path = os.path.join(self.root, self.mode, 'labels', self.labels[idx])
    #     image = load_image(img_path)
    #     box = None
    #     if self.transform is not None:
    #         image = self.transform(image)
    #     if self.mode == MODE_TRAIN or self.mode == MODE_TEST:
    #         box = load_synthetic_box(box_path)
    #         box = torch.as_tensor(box, dtype=torch.float32)
    #     if self.mode == MODE_VALID:
    #         box = load_real_box(box_path)
    #         box = torch.as_tensor(box, dtype=torch.float32)
    #     return image, box

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.mode, 'images', self.images[idx])
        box_path = os.path.join(self.root, self.mode, 'labels', self.labels[idx])
        image = Image.open(img_path).convert("RGB")
        box = None
        if self.transform is not None:
            image = self.transform(image)
        if self.mode == MODE_TRAIN or self.mode == MODE_TEST:
            box = load_synthetic_box(box_path)
        if self.mode == MODE_VALID:
            box = load_real_box(box_path)
        boxes = [box]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((2,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((2,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target["boxes"] = boxes
        target["image_id"] = image_id
        target["area"] = area
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        return image, target
