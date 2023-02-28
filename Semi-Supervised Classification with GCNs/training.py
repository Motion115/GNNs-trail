from typing import Tuple  
import numpy as np 
import torch 
from torch_geometric.data import Data 
from torch_geometric.utils import accuracy 

from init import *

def stepTrain(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn
) -> Tuple[float, float]: 

    model.train() 
    optimizer.zero_grad() 
    # 训练结点
    mask = data.train_mask
    # 全连接层输出
    logits = model(data.x, data.edge_index)[mask]
    # 选择最大值
    preds = logits.argmax(dim = 1) 
    # 监督值
    y = data.y[mask]
    # 计算损失 
    loss = loss_fn(logits, y) 
 
    acc = accuracy(preds, y) 
    loss.backward() 
    optimizer.step() 
    return loss.item(), acc 
 

def stepEval(
    model: torch.nn.Module,
    data: Data,
    loss_fn: LossFn,
    stage: Stage
) -> Tuple[float, float]:

    model.eval() 
    mask = getattr(data, f"{stage}_mask") 
    logits = model(data.x, data.edge_index)[mask] 
    preds = logits.argmax(dim = 1) 
    y = data.y[mask] 
    loss = loss_fn(logits, y)
    acc = accuracy(preds, y)
    return loss.item(), acc


def train( 
    model: torch.nn.Module, 
    data: Data, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: torch.nn.CrossEntropyLoss(), 
    max_epochs: int = 400, 
    early_stopping: int = 10, 
    print_interval: int = 10,   # 输出训练的间隔
    print_log: bool = True    # 是否输出日志信息
) -> HistoryDict:
    # 字典用于绘图
    history = {"training_loss": [], "validation_loss": [], "training_accuracy": [], "validation_accuracy": []} 
    for epoch in range(max_epochs): 
        training_loss, training_accuracy = stepTrain(model, data, optimizer, loss_fn) 
        validation_loss, validation_accuracy = stepEval(model, data, loss_fn, "val") 
        history["training_loss"      ].append(training_loss) 
        history["training_accuracy"  ].append(training_accuracy) 
        history["validation_loss"    ].append(validation_loss) 
        history["validation_accuracy"].append(validation_accuracy) 
        
        if early_stopping != 0:
            if epoch > early_stopping and validation_loss > np.mean(history["validation_loss"][-(early_stopping + 1) : -1]): 
                if print_log: 
                    print("\nEarly stopping...")  
                break 
 
        if print_log and epoch % print_interval == 0: 
            print(f"\nEpoch: {epoch}") 
            print(f"Train loss: {training_loss:.3f} | Train acc: {training_accuracy:.3f}") 
            print(f"  Val loss: {validation_loss:.3f} |   Val acc: {validation_accuracy:.3f}") 
 
    test_loss, test_acc = stepEval(model, data, loss_fn, "test") 
    if print_log == True: 
        print(f"\nEpoch: {epoch}") 
        print(f"Train loss: {training_loss:.3f} | Train acc: {training_accuracy:.3f}") 
        print(f"  Val loss: {validation_loss:.3f} |   Val acc: {validation_accuracy:.3f}") 
        print(f" Test loss: {test_loss:.3f} |  Test acc: {test_acc:.3f}")  
    return history