from kan_layers import KANLayer, fourierKANLayer, chebKANLayer, fastKANLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

class GKAN(torch.nn.Module):
    def __init__(self, dataset, hidden_dim, num_layers, grid_size=5, output_dim=None, kan_layer_type='KAN', aggregation_method=1, use_bias=False, order=3, kan_out_dim=3, dropout_rate=0.5):
        super().__init__()
        self.dataset = dataset
        self.num_layers = num_layers
        self.aggregation_method = aggregation_method
        
        input_dim = dataset.x.shape[1]
        if not output_dim:
            output_dim = len(torch.unique(dataset.y))

        self.linear_input = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.kan_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        for i in range(num_layers):
            if kan_layer_type == 'KAN':
                self.kan_layers.append(KANLayer(
                    in_dim=hidden_dim, 
                    out_dim=hidden_dim,
                    num=grid_size, 
                    k=order, 
                    noise_scale=0.1, 
                    scale_base=1.0, 
                    scale_sp=1.0, 
                    base_fun=torch.nn.SiLU(), 
                    grid_eps=0.02, 
                    grid_range=[-1, 1], 
                    sp_trainable=True, 
                    sb_trainable=True, 
                    device='cuda'
                ))
            elif kan_layer_type == 'chebKAN':
                self.kan_layers.append(chebKANLayer(
                    input_dim=hidden_dim, 
                    output_dim=hidden_dim, 
                    degree=order
                ))

            elif kan_layer_type == 'fastKAN':
                self.kan_layers.append(fastKANLayer(
                    layers_hidden=hidden_dim, 
                    grid_min=-2.0,  
                    grid_max=2.0,   
                    num_grids=grid_size,   
                    use_base_update=True, 
                    base_activation=torch.nn.functional.silu, 
                    spline_weight_init_scale=0.1,
                ))
            elif kan_layer_type == 'fourierKAN':
                self.kan_layers.append(fourierKANLayer(hidden_dim, hidden_dim, grid_size, addbias=use_bias))
            else:
                raise ValueError(f"Unsupported KAN layer type: {kan_layer_type}")
        
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, data):
        data = self.dataset
        input_features = data.x
        edge_index = data.edge_index
        adjacency_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(edge_index.device), (data.num_nodes, data.num_nodes))
    
        input_features = self.linear_input(input_features)
        input_features = self.dropout(input_features)
        
        input_features = torch.sparse.mm(adjacency_matrix, input_features)
    
        for layer in self.kan_layers[:self.num_layers - 1]:
            input_features = layer(input_features)
            
            if isinstance(input_features, tuple):
                input_features = input_features[0]
            
            input_features = torch.sparse.mm(adjacency_matrix, input_features)
            
            input_features = self.dropout(input_features)
        
        input_features = self.output_layer(input_features)
        
        return F.log_softmax(input_features, dim=-1)

    # an alternative forward 
    # def forward(self, data):
    #     data = self.dataset
    #     input_features = data.x
    #     edge_index = data.edge_index
    #     adjacency_matrix = torch.sparse_coo_tensor(
    #         edge_index,
    #         torch.ones(edge_index.shape[1]).to(edge_index.device),
    #         (data.num_nodes, data.num_nodes)
    #     )
        
    #     input_features = self.linear_input(input_features)
        
    #     for i, layer in enumerate(self.layers[:self.num_layers-1]):
    #         input_features = layer(spmm(adjacency_matrix, input_features))
    #         input_features = self.dropout(input_features) 
        
    #     input_features = self.layers[-1](input_features)
        
    #     return F.log_softmax(input_features, dim=-1)
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
