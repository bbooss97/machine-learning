import torch
from dataset import OnePieceDataset
from torch.utils.data import DataLoader
from torch import nn
from nn import Mlp

#declare parameters
num_epochs=10
batch_size=32
lr=0.001

#read the dataset
dataset=OnePieceDataset(400,400)

#split in train and test set
split=[int(0.8*len(dataset)),int(0.2*len(dataset))+1]
train,test = torch.utils.data.random_split(dataset,split)

#dataloader
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

#define the model
model=Mlp()

#define loss and the optimizer
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)


for epoch in range(num_epochs):
    #train
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):
        #reshape the images
        # images=images.reshape()
        
        #forward pass
        outputs=model(images)
        
        #calculate the loss
        l=loss(outputs,labels)
        
        #backpropagation
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        #print the loss
        print("epoch: {}/{}, step: {}/{}, loss: {}".format(epoch+1,num_epochs,i+1,len(train_dataloader),l.item()))
        
    #test
    model.eval()
    with torch.no_grad():
        correct=0
        total=0
        for images, labels in test_dataloader:
            #reshape the images
            images=images.reshape(-1,28*28)
            
            #forward pass

print("finished")