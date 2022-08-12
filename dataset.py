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

DOG_PRIFIXS = [
    "n02105641", "n02105855", "n02106030", "n02106166", "n02106382",
    "n02106550", "n02106662", "n02107142", "n02107312", "n02107574",
    "n02107683", "n02107908", "n02108000", "n02108089", "n02108422",
    "n02108551", "n02108915", "n02109047", "n02109525", "n02109961",
]

DOG_PRIFIXS_LABEL = {
    "n02105641" : 0, #Old English sheepdog, bobtail
    "n02105855" : 1, #Shetland sheepdog, Shetland sheep dog, Shetland
    "n02106030" : 2, #collie
    "n02106166" : 3, #Border collie
    "n02106382" : 4, #Bouvier des Flandres, Bouviers des Flandres
    "n02106550" : 5, #Rottweiler
    "n02106662" : 6, #German shepherd, German shepherd dog, German police dog, alsatian
    "n02107142" : 7, #Doberman, Doberman pinscher
    "n02107312" : 8, #miniature pinscher
    "n02107574" : 9, #Greater Swiss Mountain dog
    "n02107683" : 10, #Bernese mountain dog
    "n02107908" : 11, #Appenzeller
    "n02108000" : 12, #EntleBucher
    "n02108089" : 13, #boxer
    "n02108422" : 14, #bull mastiff
    "n02108551" : 15, #Tibetan mastiff
    "n02108915" : 16, #French bulldog
    "n02109047" : 17, #Great Dane
    "n02109525" : 18, #Saint Bernard, St Bernard
    "n02109961" : 19, #Eskimo dog, husky
}

BIRD_PRIFIXS = [
    "n01514668", "n01514859", "n01518878", "n01530575", "n01531178",
    "n01532829", "n01534433", "n01537544", "n01558993", "n01560419",
    "n01580077", "n01582220", "n01592084", "n01601694", "n01608432",
    "n01614925", "n01616318", "n01622779", "n01819313", "n01847000",
]

BIRD_PRIFIXS_LABEL = {
    "n01514668" : 0, #cock
    "n01514859" : 1, #hen
    "n01518878" : 2, #ostrich, Struthio camelus
    "n01530575" : 3, #brambling, Fringilla montifringilla
    "n01531178" : 4, #goldfinch, Carduelis carduelis
    "n01532829" : 5, #house finch, linnet, Carpodacus mexicanus
    "n01534433" : 6, #junco, snowbird
    "n01537544" : 7, #indigo bunting, indigo finch, indigo bird, Passerina cyanea
    "n01558993" : 8, #robin, American robin, Turdus migratorius
    "n01560419" : 9, #bulbul
    "n01580077" : 10, #jay
    "n01582220" : 11, #magpie
    "n01592084" : 12, #chickadee
    "n01601694" : 13, #water ouzel, dipper
    "n01608432" : 14, #kite
    "n01614925" : 15, #bald eagle, American eagle, Haliaeetus leucocephalus
    "n01616318" : 16, #vulture
    "n01622779" : 17, #great grey owl, great gray owl, Strix nebulosa
    "n01819313" : 18, #sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
    "n01847000" : 19, #drake
}

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

class DogBirdBaseDataset(data.Dataset):
    def __init__(self, data_dir, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2, num_classes=2, dataset_select=0): #dataset_sel, 0 == dogbird, 1 == dog, 2 == bird, 3 == 10class
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.image_paths = []
        self.labels = []
        self.num_classes = num_classes
        self.dataset_select = dataset_select
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

            if self.dataset_select == 0:  # dogbird dataset
                if file_class in BIRD_PRIFIXS :
                    self.labels.append(0) #0 == bird
                if file_class in DOG_PRIFIXS :
                    self.labels.append(1) #1 == dog

            elif self.dataset_select == 1:  # dog dataset
                self.labels.append(DOG_PRIFIXS_LABEL[file_class])

            elif self.dataset_select == 2:  # bird dataset
                self.labels.append(BIRD_PRIFIXS_LABEL[file_class])

            elif self.dataset_select == 3:  # 10class dataset
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


def DogBaseDataset(resize, data_dir, val_ratio):
    return DogBirdBaseDataset(val_ratio=val_ratio, resize=resize, data_dir=data_dir, num_classes=20, dataset_select=1)

def BirdBaseDataset(resize, data_dir, val_ratio):
    return DogBirdBaseDataset(val_ratio=val_ratio, resize=resize, data_dir=data_dir, num_classes=20, dataset_select=2)

def TenClassBaseDataset(resize, data_dir, val_ratio):
    return DogBirdBaseDataset(val_ratio=val_ratio, resize=resize, data_dir=data_dir, num_classes=10, dataset_select=3)
