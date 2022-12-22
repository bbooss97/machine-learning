#read the data folder and create a file with the annotations
import os 
from classes import *

with open("./hw2/annotations.txt", "w") as f:
    for i,n in enumerate(classes):
        for file in os.listdir("./hw2/Data/Data/"+n):
            f.write("./hw2/Data/Data/"+n+"/"+file+" "+str(i)+"\n")
    print(n+" done")