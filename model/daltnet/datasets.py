import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class ImageDataset_sRGB(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode
        self.size = 512
        file_better = open(os.path.join(root,'better.txt'),'r')
        better_files = sorted(file_better.readlines())
        self.better_files = list()
        for i in range(len(better_files)):
            self.better_files.append(better_files[i].replace("\n",""))

        file = open(os.path.join(root,'train_del_bad1.txt'),'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root,"A","train_del_bad",set1_input_files[i][:-1]))
            self.set1_expert_files.append(os.path.join(root,"B","train_del_bad",set1_input_files[i][:-1]))

        # file = open(os.path.join(root,'train_label.txt'),'r')
        # set2_input_files = sorted(file.readlines())
        # self.set2_input_files = list()
        # self.set2_expert_files = list()
        # for i in range(len(set2_input_files)):
        #     self.set2_input_files.append(os.path.join(root,"input","JPG/480p",set2_input_files[i][:-1] + ".jpg"))
        #     self.set2_expert_files.append(os.path.join(root,"expertC","JPG/480p",set2_input_files[i][:-1] + ".jpg"))

        file = open(os.path.join(root,'test.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root,"A","test",test_input_files[i][:-1]) + ".jpg")
            self.test_expert_files.append(os.path.join(root,"B","test",test_input_files[i][:-1]) + ".jpg")

    def __getitem__(self, index):
        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])

        if img_name in self.better_files:
            img_label = 1
        else:
            img_label = 0

        if self.mode == "train":
            ratio_H = np.random.uniform(0.6,1.0)
            ratio_W = np.random.uniform(0.6,1.0)
            W,H = img_input._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(crop_h, crop_w))
            img_input = TF.crop(img_input, i, j, h, w)
            img_exptC = TF.crop(img_exptC, i, j, h, w) 
            #img_input = TF.resized_crop(img_input, i, j, h, w, (320,320))  /17  
            #img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (320,320))  /batch_size 32

            if np.random.random() > 0.5:
                img_input = TF.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)

            img_input = TF.resize(img_input, (self.size,self.size), interpolation=Image.NEAREST)
            img_exptC = TF.resize(img_exptC, (self.size,self.size), interpolation=Image.NEAREST)

        img_input = TF.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)

        return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name, "img_label": img_label}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

