import torch 
from torch import Tensor 
from torch_geometric.nn import GCNConv 

from init import *
# 2-layer GCN
class GCN(torch.nn.Module): 
    def __init__( 
        self, 
        num_node_features: int, 
        num_classes: int, 
        hidden_dim: int = 16, 
        dropout_rate: float = 0.5, 
    ) -> None: 
        super().__init__() 
        self.dropout1 = torch.nn.Dropout(dropout_rate) 
        self.conv1 = GCNConv(num_node_features, hidden_dim) 
        self.relu = torch.nn.ReLU(inplace=True) 
        self.dropout2 = torch.nn.Dropout(dropout_rate) 
        self.conv2 = GCNConv(hidden_dim, num_classes) 
 
    def forward(self, x: Tensor, edge_index: Tensor) -> torch.Tensor: 
        x = self.dropout1(x) 
        x = self.conv1(x, edge_index) 
        x = self.relu(x) 
        x = self.dropout2(x) 
        x = self.conv2(x, edge_index) 
        return x

# 8-layer GCN
class GCN_experimental(torch.nn.Module): 
    def __init__( 
        self, 
        num_node_features: int, 
        num_classes: int,
        hidden_dim: int = 16, 
        dropout_rate: float = 0.5, 
    ) -> None: 
        super().__init__() 
        self.dropout1 = torch.nn.Dropout(dropout_rate) 
        self.conv1 = GCNConv(num_node_features, 1024) 
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = GCNConv(1024, 512)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = GCNConv(512, 256)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = GCNConv(256, 128)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.conv5 = GCNConv(128, 64)
        self.relu5 = torch.nn.ReLU(inplace=True)
        self.conv6 = GCNConv(64, 32)
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.conv7 = GCNConv(32, hidden_dim)
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.dropout2 = torch.nn.Dropout(dropout_rate) 
        self.conv8 = GCNConv(hidden_dim, num_classes) 
 
    def forward(self, x: Tensor, edge_index: Tensor) -> torch.Tensor: 
        x = self.dropout1(x) 
        x = self.conv1(x, edge_index) 
        x = self.relu1(x) 
        x = self.conv2(x, edge_index) 
        x = self.relu2(x)
        x = self.conv3(x, edge_index) 
        x = self.relu3(x)
        x = self.conv4(x, edge_index) 
        x = self.relu4(x)
        x = self.conv5(x, edge_index) 
        x = self.relu5(x)
        x = self.conv6(x, edge_index) 
        x = self.relu6(x)
        x = self.conv7(x, edge_index) 
        x = self.relu7(x)
        x = self.dropout2(x)
        x = self.conv8(x, edge_index)
        return x 

# 5-layer GCN with up&down sampling
class GCN_novel(torch.nn.Module): 
    def __init__( 
        self, 
        num_node_features: int, 
        num_classes: int,
        hidden_dim: int = 16, 
        dropout_rate: float = 0.5, 
    ) -> None: 
        super().__init__() 
        self.dropout1 = torch.nn.Dropout(dropout_rate) 
        self.conv1 = GCNConv(num_node_features, 64) 
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = GCNConv(64, 256)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = GCNConv(256, 32)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = GCNConv(32, hidden_dim)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.dropout2 = torch.nn.Dropout(dropout_rate) 
        self.conv5 = GCNConv(hidden_dim, num_classes) 
 
    def forward(self, x: Tensor, edge_index: Tensor) -> torch.Tensor: 
        x = self.dropout1(x) 
        x = self.conv1(x, edge_index) 
        x = self.relu1(x) 
        x = self.conv2(x, edge_index) 
        x = self.relu2(x)
        x = self.conv3(x, edge_index) 
        x = self.relu3(x)
        x = self.conv4(x, edge_index) 
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.conv5(x, edge_index)
        return x 

# 5-layer GCN
class GCN_novelStd(torch.nn.Module): 
    def __init__( 
        self, 
        num_node_features: int, 
        num_classes: int,
        hidden_dim: int = 16, 
        dropout_rate: float = 0.5, 
    ) -> None: 
        super().__init__() 
        self.dropout1 = torch.nn.Dropout(dropout_rate) 
        self.conv1 = GCNConv(num_node_features, 256) 
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = GCNConv(256, 64)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = GCNConv(64, 32)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = GCNConv(32, hidden_dim)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.dropout2 = torch.nn.Dropout(dropout_rate) 
        self.conv5 = GCNConv(hidden_dim, num_classes) 
 
    def forward(self, x: Tensor, edge_index: Tensor) -> torch.Tensor: 
        x = self.dropout1(x) 
        x = self.conv1(x, edge_index) 
        x = self.relu1(x) 
        x = self.conv2(x, edge_index) 
        x = self.relu2(x)
        x = self.conv3(x, edge_index) 
        x = self.relu3(x)
        x = self.conv4(x, edge_index) 
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.conv5(x, edge_index)
        return x 