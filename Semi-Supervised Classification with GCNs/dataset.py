import torch_geometric.transforms as T  
from torch_geometric.datasets import Planetoid 

# import dataset with normalization
dir = ""
dataset = Planetoid(root = dir, name = "Cora", transform = T.NormalizeFeatures()) 
nodes          = dataset.data.num_nodes 
edges          = dataset.data.num_edges // 2 
training_set   = dataset[0].train_mask.sum() 
validation_set = dataset[0].val_mask.sum() 
testing_set    = dataset[0].test_mask.sum() 
exclusion_set  = nodes - training_set - validation_set - testing_set

# output dataset spec
print(f"Dataset: {dataset.name}") 
print(f"Nodes: {nodes} (train={training_set}, val={validation_set}, test={testing_set}, other={exclusion_set})") 
print(f"Edges: {edges}") 
print(f"Node features: {dataset.num_node_features}") 
print(f"Classes: {dataset.num_classes}") 
print(f"Dataset size: {dataset.len()}")