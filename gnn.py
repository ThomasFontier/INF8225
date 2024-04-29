import os
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GraphConv, MessagePassing
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import pandas as pd
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import KarateClub
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.utils import to_networkx
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def process_data():
    data = pd.read_csv(r'S-FFSD.csv')
    data = data[data['Labels'] != 2]
    print(data.head())

    X = data.drop(['Labels'], axis=1)
    y = data['Labels']

    over = RandomOverSampler(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=1.0)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    encoder = LabelEncoder()
    combined_nodes = pd.concat([X_resampled['Source'], X_resampled['Target']])
    encoder.fit(combined_nodes)

    X_resampled['Source'] = encoder.transform(X_resampled['Source'])
    X_resampled['Target'] = encoder.transform(X_resampled['Target'])

    features = pd.get_dummies(X_resampled[['Source', 'Target', 'Location', 'Type']])
    
    N = encoder.classes_.size 
    adjacency_matrix = np.zeros((N, N))
    weighted_adjacency_matrix = np.zeros((N, N))

    for _, row in X_resampled.iterrows():
        adjacency_matrix[row['Source'], row['Target']] = 1
        weighted_adjacency_matrix[row['Source'], row['Target']] += row['Amount']

    X_train, X_test, y_train, y_test = train_test_split(range(len(y_resampled)), np.eye(2)[y_resampled], test_size=0.2, random_state=48, stratify=y_resampled)
    
    return [adjacency_matrix, weighted_adjacency_matrix], features, X_train, y_train, X_test, y_test

# https://github.com/OMS1996/pytorch-geometric/blob/main/3_Graph_Classification.ipynb

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hiddenDim):
        super().__init__()
        self.gcn1 = GCNConv(num_features, hiddenDim)
        self.gcn2 = GCNConv(hiddenDim, hiddenDim)
        self.gcn3 = GCNConv(hiddenDim, hiddenDim)
        self.out = torch.nn.Linear(hiddenDim, num_classes)
        
    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index).relu()
        h = self.gcn2(h, edge_index).relu()
        h = self.gcn3(h, edge_index).relu()
        z = self.out(h)
        return h, z

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hiddenDim):
        super().__init__()
        self.gat = GATConv(num_features, hiddenDim)
        self.out = torch.nn.Linear(hiddenDim, num_classes)
        
    def forward(self, x, edge_index):
        h = self.gat(x, edge_index).relu()
        z = self.out(h)
        return h, z

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class GNN_COMP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hiddenDim):
        super().__init__()
        self.gcn1 = GCNConv(num_features, hiddenDim)
        self.gat1 = GATConv(hiddenDim, hiddenDim)
        self.gcn2 = GCNConv(hiddenDim, hiddenDim)
        self.gat2 = GATConv(hiddenDim, hiddenDim)
        self.out = torch.nn.Linear(hiddenDim, num_classes)
        
    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index).relu()
        h = self.gat1(h, edge_index).relu()
        h = self.gcn2(h, edge_index).relu()
        h = self.gat2(h, edge_index).relu()
        z = self.out(h)
        return h, z

def load_DBLP():
    df = pd.read_csv(r'dblp.csv')
    df = df.dropna(axis=1)
    print(df)
    
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

if __name__ == "__main__":
    rownetworks, features, X_train, y_train, X_test, y_test = process_data()
    print("Adjacency Matrices:", rownetworks[0].shape)
    print("Weighted adjacency Matrices:", rownetworks[1])
    print("Features:", features)
    print("Train Indices:", X_train)
     
    features = features.to_numpy()
    x = []
    for val in features:
        x.append(val)
    
    edge_index = torch.Tensor(rownetworks[0]).nonzero().t().contiguous()
    x = torch.Tensor(x)
    y = torch.Tensor(y_train)
    colors = []
    for label in y:
        if label[0] == 1:
            colors.append(0.1)
        else:
            colors.append(0.9)
    
    model = GNN_COMP(x.shape[1], 2, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    nEpochs = 1000

    for epoch in range(nEpochs):
        optimizer.zero_grad()
        h, z = model(x, edge_index)
        
        loss = criterion(z, y)
        acc = accuracy(z.argmax(dim=1), y.argmax(dim=1))
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.2f} | Train Acc: {acc*100:.2f}%')
    
    h, z = model(x, edge_index)
    loss = criterion(z, y)
    acc = accuracy(z.argmax(dim=1), y.argmax(dim=1))
    print(f'Test Loss: {loss:.2f} | Test Acc: {acc*100:.2f}%')