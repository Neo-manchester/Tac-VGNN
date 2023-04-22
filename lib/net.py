import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_max, scatter_mean


###  define GNN architecture  ###

class GNN_Net(torch.nn.Module):
    def __init__(self):
        super(GNN_Net, self).__init__()
        self.conv1 = GCNConv(3, 16)  ## input, Data(x, y, area)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = GCNConv(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = GCNConv(32, 48)
        self.bn3 = nn.BatchNorm1d(48)
        
        self.conv4 = GCNConv(48, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.conv5 = GCNConv(64, 96)
        self.bn5 = nn.BatchNorm1d(96)
        
        self.linear1 = torch.nn.Linear(96, 128)
        self.bn11 = nn.BatchNorm1d(128)
        
        self.linear2 = torch.nn.Linear(128, 96)
        self.bn12 = nn.BatchNorm1d(96)
        
        self.linear3 = torch.nn.Linear(96, 2) ## output, (pose_2--Y, pose_6--Theta_roll)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.leaky_relu(x)
        
        # x, _ = scatter_max(x, data.batch, dim=0)
        x = scatter_mean(x, data.batch, dim=0)
        
        x = self.linear1(x)
        x = self.bn11(x)
        x = F.leaky_relu(x)
        
        x = self.linear2(x)
        x = self.bn12(x)
        x = F.leaky_relu(x)
        
        x = self.linear3(x)

        return x