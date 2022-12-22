import torch
from dataset import OnePieceDataset
from torch.utils.data import DataLoader
from torch import nn
from nn import Mlp

#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#declare parameters
num_epochs=2
batch_size=32
lr=0.001
w,h=400,400

#read the dataset
dataset=OnePieceDataset(w,h)

#split in train and test set
split=[int(0.8*len(dataset)),int(0.2*len(dataset))+1]
train,test = torch.utils.data.random_split(dataset,split)

#dataloader
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True , drop_last=True)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True , drop_last=True)

#define the model
model=Mlp(w,h)
model.to(device)

#define loss and the optimizer
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)


for epoch in range(num_epochs):

    #train
    model.train()

    for i, (images, labels) in enumerate(train_dataloader):

        #move the data to the device
        images=images.to(device)
        labels=labels.to(device)

        #reshape the images
        images=images.reshape(batch_size,-1)
        
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

    # Initialize variables to store metrics
    loss = 0.0
    accuracy = 0.0

    # Loop over the data in the test set
    with torch.no_grad():
        for images, labels in test_dataloader:

            # Move the data to the device
            images = images.to(device)
            labels = labels.to(device)

            # Reshape the images
            images=images.reshape(batch_size,-1)

            # Forward pass: compute predictions and loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Compute running metrics
            loss += loss.item()
            accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()

    # Compute average metrics
    avg_loss = loss / len(test_dataloader)
    avg_accuracy = accuracy / len(test_dataloader)

    # Print the metrics
    print(f'Test loss: {avg_loss:.4f}')
    print(f'Test accuracy: {avg_accuracy:.4f}')

print("finished")