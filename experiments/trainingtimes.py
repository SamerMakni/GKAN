import argparse
import time
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch.nn import Linear
import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gkan import GKAN

trials = 5

def generate_sbm_graph(n, n_classes):
    sizes = [n // n_classes] * n_classes
    p_in = 0.1
    p_out = 0.01
    p_matrix = [
        [p_in if i == j else p_out for j in range(n_classes)] for i in range(n_classes)
    ]
    G = nx.stochastic_block_model(sizes, p_matrix)
    return G

def convert_to_pyg_data(G, device):
    pyg_graph = from_networkx(G)
    pyg_graph.x = torch.randn((pyg_graph.num_nodes, 16)).to(device)
    pyg_graph.y = torch.randint(0, 5, (pyg_graph.num_nodes,)).to(device)
    pyg_graph.train_mask = torch.zeros(pyg_graph.num_nodes, dtype=torch.bool).to(device)
    pyg_graph.train_mask[:pyg_graph.num_nodes // 2] = True  # Simple train/test split
    pyg_graph.val_mask = ~pyg_graph.train_mask
    pyg_graph.edge_index = pyg_graph.edge_index.to(device)
    return pyg_graph

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv = GCNConv(16, 16)
        self.fc = Linear(16, 5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv = GATConv(16, 16, heads=8, dropout=0.6)
        self.fc = Linear(16 * 8, 5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def measure_time(model, data, epochs, device):
    model = model.to(device)
    data = data.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data) if hasattr(model, 'forward') else model()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'

    parser = argparse.ArgumentParser(
        description="GCN, GAT and GKAN Training Time Measurement"
    )
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1000, 2500, 5000, 10000, 15000], help="List of graph sizes"
    )
    parser.add_argument(
        "--export_plot",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Export plot or not",
    )
    parser.add_argument(
        "--export_results",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Export results to CSV or not",
    )

    args = parser.parse_args()
    n_classes = args.n_classes
    epochs = args.epochs
    sizes = args.sizes
    export_plot = args.export_plot
    export_results = args.export_results

    results = []
    if export_results == "yes":
        with open("training_times.csv", "w", newline="") as csvfile:
            fieldnames = ["size", "gcn_time", "gat_time", "gkan_time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for size in sizes:
                print(f"Generating graph with {size} nodes...")
                G = generate_sbm_graph(size, n_classes)
                data = convert_to_pyg_data(G, device)
                data.train_mask = torch.randperm(data.num_nodes)[: data.num_nodes // 2].to(device)
    
                gcn_time_avg, gat_time_avg, gkan_time_avg = 0, 0, 0
                for trial in range(trials):
                    gcn_model = GCN().to(device)
                    gcn_time = measure_time(gcn_model, data, epochs, device)
                    gcn_time_avg += gcn_time
                
                    gat_model = GAT().to(device)
                    gat_time = measure_time(gat_model, data, epochs, device)
                    gat_time_avg += gat_time
                
                    gkan_model = GKAN(dataset=data, hidden_dim=16, num_layers=1, kan_layer_type='KAN', aggregation_method=1, use_bias=False).to(device)
                    gkan_time = measure_time(gkan_model, data, epochs, device)
                    gkan_time_avg += gkan_time
                
                # Calculate average times
                gcn_time_avg /= trials
                gat_time_avg /= trials
                gkan_time_avg /= trials
                
                results.append((size, gcn_time_avg, gat_time_avg, gkan_time_avg))
                print(f"Size: {size}, GCN Time: {gcn_time_avg:.4f}s, GAT Time: {gat_time_avg:.4f}s, GKAN Time: {gkan_time_avg:.4f}s")
                
                writer.writerow({"size": size, "gcn_time": gcn_time_avg, "gat_time": gat_time_avg, "gkan_time": gkan_time_avg})

    else:
        for size in sizes:
            print(f"Generating graph with {size} nodes...")
            G = generate_sbm_graph(size, n_classes)
            data = convert_to_pyg_data(G, device)
            data.train_mask = torch.randperm(data.num_nodes)[: data.num_nodes // 2].to(device)

            gcn_time_avg, gat_time_avg, gkan_time_avg = 0
            for trial in trials:
                gcn_model = GCN().to(device)
                gcn_time = measure_time(gcn_model, data, epochs, device)
                gcn_time_avg = gcn_time_avg + gcn_time
                
                gat_model = GAT().to(device)
                gat_time = measure_time(gat_model, data, epochs, device)
                gat_time_avg = gat_time_avg + gat_time

                gkan_model = GKAN(dataset=data, hidden_dim=16, num_layers=1, kan_layer_type='KAN', aggregation_method=1, use_bias=False).to(device)
                gkan_time = measure_time(gkan_model, data, epochs, device)
                gkan_time_avg = gkan_time_avg + gkan_time

            results.append((size, gcn_time, gat_time, gkan_time))
            print(f"Size: {size}, GCN Time: {(gcn_time/5):.4f}s, GAT Time: {(gat_time/5):.4f}s, GKAN Time: {(gkan_time/5):.4f}s")

    sizes, gcn_times, gat_times, gkan_times = zip(*results)
    plt.plot(sizes, gcn_times, label="GCN")
    plt.plot(sizes, gat_times, label="GAT")
    plt.plot(sizes, gkan_times, label="GKAN")
    plt.xlabel("Graph Size")
    plt.ylabel("Training Time (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    if export_plot == "yes":
        plt.savefig("gnn_training_times.png")
    plt.show()
