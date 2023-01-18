import argparse

import numpy as np
import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops

from train_eval import *
from datasets import *

import warnings
#(deep cora)lr:0.005;wd:0.006;dropout:0.8;Acc:83.7±0.6
#(cora)lr:0.005;wd:0.022;dropout:0.85;Acc:84.0±0.7
#(CiteSeer)lr:0.005;wd:0.015;dropout:0.5;Acc:71.1±0.9
#(PubMed)lr:0.02;wd:0.028;dropout:0.83;Acc:79.9±0.7
#(wisconsin)lr:0.016;wd:0.002;dropout:0.34;Acc:92.9±3.1
#(texas)lr:0.016;wd:0.002;dropout:0.34;Acc:91.8±3.6
#(cornell)lr:0.008;wd:0.003;dropout:0.6;Acc:88.6±4.7
#(actor)lr:0.016;wd:0.002;dropout:0.45;Acc:39.9±1.3
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.016)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)

args = parser.parse_args()

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        self.alpha = 0.1

        # PPR
        TEMP = self.alpha * (1 - self.alpha) ** np.arange(args.K + 1)
        TEMP[-1] = (1 - self.alpha) ** args.K
        # # NPPR
        # TEMP = (alpha) ** np.arange(K + 1)
        # TEMP = TEMP / np.sum(np.abs(TEMP))
        # # Random
        # bound = np.sqrt(3 / (K + 1))
        # TEMP = np.random.uniform(-bound, bound, K + 1)
        # TEMP = TEMP / np.sum(np.abs(TEMP))
        self.temp = Parameter(torch.tensor(TEMP))
        
    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)

        hidden = x * self.temp[0]
        preds = []
        preds.append(x)
        for k in range(self.K):
            gamma = self.temp[k + 1]
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + gamma * x
            preds.append(hidden)
            print("gamma:", gamma)

        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K
        self.prop.reset_parameters()
    
    
class Net(torch.nn.Module):
    def __init__(self, dataset, dropout):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop = Prop(dataset.num_classes, args.K)
        self.droprate = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        # self.prop.reset_parameters()

    def forward(self, data):
        #self.droprate
        x, edge_index = data.x, data.edge_index
        # edge_dropout = 0.005
        # edge_index, _ = dropout_adj(edge_index, p=edge_dropout, force_undirected=False)
        x = F.dropout(x, p=self.droprate, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.lin2(x)
        #x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)

warnings.filterwarnings("ignore", category=UserWarning)
    
if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits if args.random_splits else None
    print("Data:", dataset[0])
    # for i in np.linspace(0.05, 0.8, 16):
    #     for j in np.linspace(0.005, 0.02, 4):
    #         for k in np.linspace(0.0005, 0.002, 4):
    #             print("dropout:{},learning_rate:{},weight_decay:{}".format(i, j, k))
    #             run(dataset, Net(dataset, i), args.runs, args.epochs, j, k, args.early_stopping, permute_masks, lcc=False)
    run(dataset, Net(dataset, args.dropout), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks, lcc=False)


elif args.dataset == "wisconsin" or args.dataset == "cornell" or args.dataset == "texas":
    dataset = get_WebKB_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits_1
    print(f'Dataset:{args.dataset}')
    print("Dataset:", dataset[0])
    # for i in np.linspace(0.05, 0.8, 16):
    #         for k in np.linspace(0.0005, 0.002, 4):
    #             print("dropout:{},weight_decay:{}".format(i, k))
    #             run_heter(dataset, Net(dataset, i), args.runs, args.epochs, args.lr, k, args.early_stopping, permute_masks, lcc=True)
    run_heter(dataset, Net(dataset, args.dropout), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks, lcc=True)


elif args.dataset == "actor":
    dataset = get_Actor_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits_1
    print(f'Dataset:{args.dataset}')
    print("Dataset:", dataset[0])
    run_heter(dataset, Net(dataset,args.dropout), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,permute_masks, lcc=True)


