import torch
from dataset import OnePieceDataset
from torch.utils.data import DataLoader

#declare parameters
num_epochs=10
batch_size=32

#read the dataset
dataset=OnePieceDataset()

#split in train and test set
split=[int(0.8*len(dataset)),int(0.2*len(dataset))+1]
train,test = torch.utils.data.random_split(dataset,split)

#dataloader
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)



print("finished")