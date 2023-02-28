import torch 

from init import *
from dataset import dataset
from training import train
from visualize import plot
from model import GCN, GCN_experimental, GCN_novel, GCN_novelStd

SEED = 80
MAX_EPOCHS = 400 
LEARNING_RATE = 0.01 
WEIGHT_DECAY = 5e-4
EARLY_STOPPING = 0

torch.manual_seed(SEED) 
device = torch.device("cpu") 

print("Graph Convolutional Network (GCN):") 
GCNmodel = GCN(dataset.num_node_features, dataset.num_classes)
#GCNmodel = GCN_experimental(dataset.num_node_features, dataset.num_classes)
#GCNmodel = GCN_novel(dataset.num_node_features, dataset.num_classes)
#GCNmodel = GCN_novelStd(dataset.num_node_features, dataset.num_classes)
print(GCNmodel)

model = GCNmodel.to(device) 
data = dataset[0].to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
loss_fn = torch.nn.CrossEntropyLoss() 
history = train(model, data, optimizer, loss_fn, max_epochs = MAX_EPOCHS, early_stopping = EARLY_STOPPING)
plot(history, "2-layer GCN")