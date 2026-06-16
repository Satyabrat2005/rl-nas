import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class MixedOp(nn.Module):
    def __init__(self, C, op_names):
        super().__init__()
        self.op_names = op_names
        ops_dict = {
            'conv3x3': nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ),
            'conv5x5': nn.Sequential(
                nn.Conv2d(C, C, 5, padding=2, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ),
            'sep_conv3x3': nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ),
            'sep_conv5x5': nn.Sequential(
                nn.Conv2d(C, C, 5, padding=2, groups=C, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ),
            'max_pool3x3': nn.MaxPool2d(3, stride=1, padding=1),
            'avg_pool3x3': nn.AvgPool2d(3, stride=1, padding=1),
            'skip': nn.Identity()
        }
        self.ops = nn.ModuleDict({name: ops_dict[name] for name in op_names})
        self.num_ops = len(op_names)

    def forward(self, x, weights):
        # weights: (num_ops,) raw logits
        probs = F.softmax(weights, dim=0)
        out = sum(p * self.ops[name](x) for p, name in zip(probs, self.op_names))
        return out

class Cell(nn.Module):
    def __init__(self, C, reduction=False, n_nodes=6, op_names=None):
        super().__init__()
        self.C = C
        self.n_nodes = n_nodes
        self.reduction = reduction
        if op_names is None:
            op_names = ['conv3x3', 'conv5x5', 'sep_conv3x3', 'sep_conv5x5',
                        'max_pool3x3', 'avg_pool3x3', 'skip']
        self.op_names = op_names
        self.num_ops = len(op_names)

        self.edges = nn.ModuleDict()
        for node in range(2, n_nodes):
            for edge_idx in range(2):
                key = f"node{node}_edge{edge_idx}"
                self.edges[key] = MixedOp(C, op_names)

    def forward(self, s0, s1, edge_weights=None, edge_sources=None):
        B, C, H, W = s0.shape
        assert C == self.C, f"Channel mismatch: {C} vs {self.C}"
        assert s1.shape == (B, C, H, W), "s0 and s1 must have same shape"

        node_outputs = [s0, s1]
        if edge_sources is None:
            # Default: edge0 from node-2, edge1 from node-1 (clamped)
            edge_sources = {}
            for node in range(2, self.n_nodes):
                edge_sources[(node, 0)] = max(0, node-2)
                edge_sources[(node, 1)] = max(0, node-1)

        for node in range(2, self.n_nodes):
            edge_outs = []
            for edge_idx in range(2):
                src = edge_sources.get((node, edge_idx), 0)
                inp = node_outputs[src]
                key = f"node{node}_edge{edge_idx}"
                if edge_weights is not None and key in edge_weights:
                    w = edge_weights[key]
                else:
                    w = torch.zeros(self.num_ops, device=inp.device)
                edge_out = self.edges[key](inp, w)
                edge_outs.append(edge_out)
            node_out = edge_outs[0] + edge_outs[1]
            node_outputs.append(node_out)

        # Concatenate last two nodes
        out = torch.cat([node_outputs[-2], node_outputs[-1]], dim=1)
        return out

# For convenience
OPS_NAMES = ['conv3x3', 'conv5x5', 'sep_conv3x3', 'sep_conv5x5',
             'max_pool3x3', 'avg_pool3x3', 'skip']
