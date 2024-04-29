import os
import torch
from torch import nn
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, nFeatures, nHiddenLayers, hiddenLayerWidth, nClasses):
    super().__init__()
    self.layers = nn.Sequential()
    self.nFeatures = nFeatures
    self.nHiddenLayers = nHiddenLayers
    self.hiddenLayerWidth = hiddenLayerWidth
    self.nClasses = nClasses
    
    self.layers.add_module("1st Layer", nn.Linear(self.nFeatures, self.hiddenLayerWidth))
    self.layers.add_module("1st Layer Act", nn.ReLU())
    for i in range(self.nHiddenLayers):
        self.layers.add_module(f"{i}e Layer", nn.Linear(self.hiddenLayerWidth, self.hiddenLayerWidth))
        self.layers.add_module(f"{i}e Layer Act", nn.ReLU())
    self.layers.add_module("Output Layer", nn.Linear(self.hiddenLayerWidth, self.nClasses))

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


def get_FFSD_data(filename: str):
    df = pd.read_csv(filename)
    df["Target"] = df["Target"].str[1:].astype(int)
    df["Source"] = df["Source"].str[1:].astype(int)
    df["Location"] = df["Location"].str[1:].astype(int)
    df["Type"] = df["Type"].str[2:].astype(int)
    df_1 = df.loc[df["Labels"] == 1]
    df_0 = df.loc[df["Labels"] == 0]
    data_1 = df_1.to_numpy(dtype="float32")
    data_0 = df_0.to_numpy(dtype="float32")[:len(data_1)]
    data = np.concatenate((data_0, data_1))
    np.random.shuffle(data)
    
    inputs = torch.from_numpy(data[:,:-1])
    labels = data[:,-1].reshape((data[:,-1].shape[0], 1))
    trainLabels = []
    for value in labels:
        if value == 1:
            trainLabels.append([1,0])
        else:
            trainLabels.append([0,1])    
    labels = torch.from_numpy(np.asarray(trainLabels, dtype="float32"))
    
    return inputs, labels

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

if __name__ == "__main__":
    
    inputs, labels = get_FFSD_data(r"antifraud\data\S-FFSD.csv")
    
    trainInputs, testInputs, trainLabels, testLabels = train_test_split(inputs, labels)
    
    lr = 0.001
    batchSize = 5
    
    mlp = MLP(nFeatures=6, nHiddenLayers=5, hiddenLayerWidth=50, nClasses=2)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr)
    losses = []
    accs = []
    for epoch in range(0, 10):
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
        correct = 0
        nSamples = 0
        for input, label in zip(batch(trainInputs, batchSize), batch(trainLabels, batchSize)):
            optimizer.zero_grad()
            output = mlp(input)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

    testOutputs = mlp(testInputs)
    acc = accuracy(testOutputs.argmax(dim=1), testLabels.argmax(dim=1))
    print("TEST ACC: ", acc)