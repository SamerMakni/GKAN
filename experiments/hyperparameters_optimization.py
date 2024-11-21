import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gkan import *

import torch
import random

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import *

import numpy as np
import optuna


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'

experiment = 'GKAN (b-splines)'
print(f'running experiments for {experiment}')

seed = 133
epochs = 200
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

transform = T.Compose([T.NormalizeFeatures()])

dataset = Planetoid('./temp', 'Cora', transform=transform)[0]
dataset = dataset.to(device)

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    predictions, ground_truth = output[data.train_mask], data.y[data.train_mask]
    loss = F.nll_loss(predictions, ground_truth)
    accuracy = int((predictions.argmax(dim=-1) == ground_truth).sum()) / int(data.train_mask.sum())
    loss.backward()
    optimizer.step()
    return accuracy, loss.item()

def evaluate(model, data):
    model.eval()
    output = model(data)
    return output.argmax(dim=-1)

def objective(trial):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seed = trial.suggest_int("seed", 0, 10000)
    set_random_seeds(seed)

    hidden_size = trial.suggest_int("hidden_size", 8, 256)
    weight_decay = trial.suggest_float("weight_decay", 0, 0.0005)
    grid_size = trial.suggest_int("grid_size", 3, 20)
    spline_order = trial.suggest_int("spline_order", 1, 12)
    dropout_rate = trial.suggest_float("dropout_rate", 0.4, 1)

    model = GKAN(
        dataset=dataset,
        hidden_dim=hidden_size,
        num_layers=1, 
        grid_size=grid_size,
        kan_layer_type='KAN', 
        order=spline_order,
        dropout_rate=dropout_rate
    ).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

    patience = 10
    best_val_acc = 0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(epochs):
        train_accuracy, train_loss = train(model, optimizer, dataset)
        predictions = evaluate(model, dataset)
        
        val_acc = torch.eq(predictions[dataset.val_mask], dataset.y[dataset.val_mask]).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_epoch = epoch
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_model_state)
    trial.set_user_attr("best_epoch", best_epoch)
    return best_val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

print("Best trial:")
trial = study.best_trial

print("  - Value: ", trial.value)
print("  - Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("  - Epoch: ", trial.user_attrs["best_epoch"])

for i, t in enumerate(study.trials):
    print(f"Trial {i+1} - Best Epoch: {t.user_attrs.get('best_epoch', 'N/A')} - Value: {t.value}")
