import torch
from torch.utils.data import Dataset
from classes import *
from PIL import Image
import numpy as np

class OnePieceDataset(Dataset):
    items=[]
    def __init__(self,w,h):
        self.w=w
        self.h=h
        with open("./hw2/annotations.txt", "r") as f:
            for line in f.readlines():
                self.items.append(line.split(" "))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item=self.items[idx]

        #one hot encoding of the class of the item
        label=torch.zeros(len(classes))
        label[int(item[-1].strip())]=1
        
        #load the image as pil image
        image=Image.open(" ".join(item[0:-1])).convert('RGB')
        image = image.resize((self.w, self.h)) 

        #convert it to a tensor
        image=torch.tensor(np.array(image))
        
        return image, label