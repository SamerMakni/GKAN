import torch, argparse
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from baselines import GCN
from tqdm import tqdm

def load_planetoid_dataset(name:str):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='data', name=name, transform=NormalizeFeatures())
    else:
        raise ValueError('Dataset not supported')
    return dataset

def run_node_classification(model:torch.nn.Module, dataset, epochs:int, learning_rate:float, l2_reg:float, patience:int, device:str, verbose:bool):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    # Training loop
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    # Testing loop
    def test():
        model.eval()
        logits, accs = model(data), []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        val_loss = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
        return accs, val_loss

    # Load data
    data = dataset[0].to(device)

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Train the model
    for epoch in range(epochs):
        loss = train()
        (train_acc, val_acc, test_acc), val_loss = test()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            if verbose:
                print("Early stopping triggered.")
            early_stop = True
            break
        if verbose and epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    if early_stop:
        model.load_state_dict(best_model_state)
        if verbose:
            print("Model loaded with best validation loss state.")

    # Final test
    train_acc, val_acc, test_acc = test()[0]
    if verbose:
        print(f'Final Test Acc: {test_acc:.4f}')
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--l2_reg', type=float, default=5e-4)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')
    dataset = load_planetoid_dataset(args.dataset)
    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, dropout_rate=args.dropout).to(device)
    results = []
    for _ in tqdm(range(args.num_trials)):
        test_acc = run_node_classification(model, dataset, args.epochs, args.learning_rate, args.l2_reg, args.early_stopping, device, args.verbose)
        results.append(test_acc)
    
    print (f'Average Test Accuracy: {torch.tensor(results).mean()}')
    if len(results) > 1:
        print(f'Standard Deviation: {torch.tensor(results).std()}')