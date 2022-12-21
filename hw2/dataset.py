import torch
from torch import Dataset
from classes import *

class OnePieceDataset(Dataset):
    self.items=[]
    def __init__(self):
        with open("./hw2/annotations.txt", "r") as f:
            for line in f.readlines().split("\n"):
                self.items.append(line.split(" "))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item=self.items[idx]

        #one hot encoding of the class of the item
        label=torch.zeros(len(classes))
        label[int(item[1])]=1
        
        #load the image and return it as a tensor
        image=torch.load(item[0])
        
        return image, label