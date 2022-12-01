import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import random
import warnings
import torch
from torch.utils.data import Dataset, DataLoader

#default tensor type
# torch.set_default_tensor_type(torch.FloatTensor)


#make everything reproducible
random.seed(50)
np.random.seed(50)


dataset=pd.read_csv("./hw1/train_set.tsv", sep='\t', header=0)
dataset_numpy=dataset.to_numpy().astype(np.float32)
x=dataset_numpy[:,:-2]
y=dataset_numpy[:,-2:]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(x)
x=scaler.transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=45)
x_train_tensor=torch.from_numpy(x_train)
y_train_tensor=torch.from_numpy(y_train)
x_test_tensor=torch.from_numpy(x_test)
y_test_tensor=torch.from_numpy(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(f"the classes and the number of occurrences in the train set are {np.unique(y_train[:,0], return_counts=True)}")
print(f"the classes and the number of occurrences in the test set are {np.unique(y_test[:,0], return_counts=True)}")

class data(Dataset):
    def __init__(self, x, y):
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_data=data(x_train, y_train)
test_data=data(x_test, y_test)

x_train=torch.from_numpy(x_train).to('cuda')
y_train=torch.from_numpy(y_train).to('cuda')
x_test=torch.from_numpy(x_test).to('cuda')
y_test=torch.from_numpy(y_test).to('cuda')

train_loader=DataLoader(train_data, batch_size=32, shuffle=True)
test_loader=DataLoader(test_data, batch_size=len(test_data), shuffle=True)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(35, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc21 = nn.Linear(50, 50)
        self.fc22 = nn.Linear(50, 50)
        self.fc23 = nn.Linear(50, 50)
        self.fc231 = nn.Linear(50, 50)
        self.fc232 = nn.Linear(50, 50)
        self.fc233 = nn.Linear(50, 50)
        self.fc234 = nn.Linear(50, 50)
        self.fc235 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 5)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.m(x)
        x = torch.relu(self.fc21(x))
        x = torch.relu(self.fc22(x))
        x = self.m(x)
        x = torch.relu(self.fc23(x))
        x = torch.relu(self.fc231(x))
        x = self.m(x)
        x = torch.relu(self.fc232(x))
        x = torch.relu(self.fc233(x))
        x = torch.relu(self.fc234(x))
        x = torch.sigmoid(self.fc235(x))
        x =  F.softmax(self.fc3(x), dim=1)
        return x

net = Net()
net.cuda()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
massimo=0
for epoch in range(10000):  # loop over the dataset multiple times
    running_loss = 0.0
    print(epoch)
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.to('cuda')
        labels=labels.to('cuda')
        labels=labels[:,0]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        #one hot encoding
        labels=labels.long()
        cmp=torch.zeros(labels.shape[0],5)
        for i in range(labels.shape[0]):
            cmp[i,labels[i]]=1
        cmp=cmp.to('cuda')


        loss = criterion(outputs, cmp)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    with torch.no_grad():
        predicted=net(x_test)
        predicted=torch.argmax(predicted, dim=1)
        x_test=x_test.cpu()
        y_test=y_test.cpu()
        predicted=predicted.cpu()
        f1_micro=f1_score(y_test[:,0], predicted, average='micro')
        f1_macro=f1_score(y_test[:,0], predicted, average='macro')
        if f1_micro>massimo:
            massimo=f1_micro

        print(f"massimo f1 score {massimo} ,     current {f1_micro}" )
        x_test=x_test.to('cuda')
        y_test=y_test.to('cuda')

print('Finished Training')
x_train=x_train.to('cpu')
y_train=y_train.to('cpu')
x_test=x_test.to('cpu')
y_test=y_test.to('cpu')
net=net.to('cpu')
with torch.no_grad():
    predicted=net(torch.tensor(x_train))
    predicted=torch.argmax(predicted, dim=1)
    print("train")
    print(f"the accuracy is {accuracy_score(y_train[:,0], predicted)}")
    print(f"the f1 score is {f1_score(y_train[:,0], predicted, average='macro')}")
    print(f"the precision score is {precision_score(y_train[:,0], predicted, average='macro')}")
    print(f"the recall score is {recall_score(y_train[:,0], predicted, average='macro')}")
    print(f"the confusion matrix is \n{confusion_matrix(y_train[:,0], predicted)}")






    predicted=net(torch.tensor(x_test))
    predicted=torch.argmax(predicted, dim=1)
    print("test")
    print(f"the accuracy is {accuracy_score(y_test[:,0], predicted)}")
    print(f"the f1 score is {f1_score(y_test[:,0], predicted, average='macro')}")
    print(f"the precision score is {precision_score(y_test[:,0], predicted, average='macro')}")
    print(f"the recall score is {recall_score(y_test[:,0], predicted, average='macro')}")
    print(f"the confusion matrix is \n{confusion_matrix(y_test[:,0], predicted)}")


print(massimo)






