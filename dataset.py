import os
import random
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.transforms import *
import cv2
#apt install libgl1-mesa-glx

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

TEN_CLASS_PREFIX_LABEL = {
    "n01443537" : 0, #goldfish, Carassius auratus
    "n01484850" : 1, #great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
    "n01514859" : 2, #cock
    "n01582220" : 3, #magpie
    "n01614925" : 4, #bald eagle, American eagle, Haliaeetus leucocephalus
    "n01629819" : 5, #European fire salamander, Salamandra salamandra
    "n01664065" : 6, #loggerhead, loggerhead turtle, Caretta caretta
    "n01755581" : 7, #diamondback, diamondback rattlesnake, Crotalus adamanteus
    "n02099601" : 8, #gol-d-en retriever
    "n02129165" : 9, #lion, king of beasts, Panthera leo
}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class TenClassBaseDataset(data.Dataset):
    def __init__(self, data_dir, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, num_classes=2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.image_paths = []
        self.labels = []
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            ToTensor(),
            ToPILImage(),
            Resize(resize),
            RandomVerticalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.setup()
        self.calc_statistics()

    def setup(self):
        file_list = os.listdir(self.data_dir)

        for i, file in enumerate(file_list):
            _file_name, ext = os.path.splitext(file)
            if _file_name[0] == ".":  # "." 로 시작하는 파일 무시
                continue

            pwd_filename, _ = _file_name.split("_")
            file_class = pwd_filename.split("/")[-1]

            self.labels.append(TEN_CLASS_PREFIX_LABEL[file_class])

            self.image_paths.append(os.path.join(self.data_dir, file))

            #print(self.image_paths[-1], self.labels[-1])

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #print(image)

        label = self.get_label(index)

        image_transform = self.transform(image)
        return image_transform, label

    def __len__(self):
        return len(self.image_paths)

    def get_label(self, index):
        return self.labels[index]

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set


class TestDataset(data.Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            ToTensor(),
            ToPILImage(),
            Resize(resize, Image.BILINEAR),
            RandomVerticalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


def ten_class_dataset(resize, data_dir, val_ratio):
    return TenClassBaseDataset(val_ratio=val_ratio, resize=resize, data_dir=data_dir, num_classes=10)