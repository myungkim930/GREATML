"""
YATE gnn model with defined layers.

"""

# Python
import math

# PyTorch
import torch
import torch.nn as nn

# PyTorch Geometric
from torch_geometric.typing import Adj
from torch_geometric.utils import softmax
from torch_scatter import scatter

# From YATE
from graphlet_construction import to_undirected, to_directed


#########################
## Necessary functions ##
#########################

## YATE - Attention calculation
def yate_att_calc(edge_index: Adj, query: Adj, key: Adj):

    num_emb = query.size()[1]

    attention = torch.sum(
        torch.mul(query[edge_index[0, :], :], key), dim=1
    ) / math.sqrt(num_emb)
    attention = softmax(attention, edge_index[0, :])

    return attention


## YATE - output calculation with attention (message passing)
def yate_att_output(edge_index: Adj, attention: Adj, value: Adj):

    output = scatter(
        torch.mul(attention, value.t()).t(), edge_index[0, :], dim=0, reduce="sum"
    )

    return output


## YATE - output calculation with multi-head (message passing)
def yate_multihead(
    edge_index: Adj,
    query: Adj,
    key: Adj,
    value: Adj,
    num_heads: int = 1,
    concat=True,
):

    if concat:

        H, C = num_heads, query.size()[1] // num_heads

        for i in range(H):

            A = yate_att_calc(
                edge_index, query[:, i * C : (i + 1) * C], key[:, i * C : (i + 1) * C]
            )
            O = yate_att_output(edge_index, A, value[:, i * C : (i + 1) * C])
            try:
                output = torch.cat((output, O), dim=1)
                attention = torch.cat((attention, A), dim=0)
            except:
                output = O
                attention = A

    else:

        H, C = num_heads, query.size()[1]

        for i in range(H):

            A = yate_att_calc(
                edge_index, query[:, i * C : (i + 1) * C], key[:, i * C : (i + 1) * C]
            )

            try:
                attention = torch.cat((attention, A), dim=0)
            except:
                attention = A

        attention = attention / H
        output = yate_att_output(edge_index, attention, value)

    return output, attention


#################
## YATE Blocks ##
#################

## YATE - Attention Layer
class YATE_Attention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 1,
        concat: bool = True,
        read_out: bool = False,
    ):
        super(YATE_Attention, self).__init__()

        if concat:
            assert output_dim % num_heads == 0
            self.lin_query = nn.Linear(input_dim, output_dim)
            self.lin_key = nn.Linear(input_dim, output_dim)
            self.lin_value = nn.Linear(input_dim, output_dim)
        else:
            self.lin_query = nn.Linear(input_dim, num_heads * output_dim)
            self.lin_key = nn.Linear(input_dim, num_heads * output_dim)
            self.lin_value = nn.Linear(input_dim, num_heads * output_dim)

        if read_out == False:
            self.lin_edge = nn.Linear(input_dim, output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat = concat
        self.readout = read_out

        self.reset_parameters()

    def reset_parameters(self):

        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        if self.readout == False:
            self.lin_edge.reset_parameters()

    def forward(
        self,
        x: Adj,
        edge_index: Adj,
        edge_attr: Adj,
        return_attention=False,
    ):

        Z = torch.mul(edge_attr, x[edge_index[1, :]])

        query = self.lin_query(x)
        key = self.lin_key(Z)
        value = self.lin_value(Z)

        output, attention = yate_multihead(
            edge_index=edge_index,
            query=query,
            key=key,
            value=value,
            num_heads=self.num_heads,
            concat=self.concat,
        )

        if self.readout == False:
            edge_attr = self.lin_edge(Z)

        if return_attention:
            return output, edge_attr, attention
        else:
            return output, edge_attr


## YATE - single encoding block
class YATE_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ff_dim: int,
        num_heads: int = 1,
        concat: bool = True,
        dropout=0.1,
        read_out: bool = False,
    ):

        super().__init__()

        # Graph Attention Layer
        self.g_attn = YATE_Attention(
            input_dim, input_dim, num_heads, concat, read_out=read_out
        )

        # Two-layer MLP + Layers to apply in between the main layers for x and edges
        self.linear_net_x = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, input_dim),
        )

        self.read_out = read_out
        if self.read_out == False:
            self.linear_net_e = nn.Sequential(
                nn.Linear(input_dim, ff_dim),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(ff_dim, input_dim),
            )
            self.norm1_e = nn.LayerNorm(input_dim)
            self.norm2_e = nn.LayerNorm(input_dim)

        self.norm1_x = nn.LayerNorm(input_dim)
        self.norm2_x = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: Adj,
        edge_index: Adj,
        edge_attr: Adj,
    ):

        # Attention part
        attn_out_x, attn_out_e = self.g_attn(x, edge_index, edge_attr)

        x = x + attn_out_x
        x = self.norm1_x(x)

        # MLP part
        linear_out_x = self.linear_net_x(x)
        x = x + linear_out_x
        x = self.norm2_x(x)

        if self.read_out == False:
            edge_attr = edge_attr + attn_out_e
            edge_attr = self.norm1_e(edge_attr)

            linear_out_e = self.linear_net_e(edge_attr)
            edge_attr = edge_attr + linear_out_e
            edge_attr = self.norm2_e(edge_attr)

        return x, edge_attr


#########################
## YATE Encoding Block ##
#########################


## YATE - encoding block with several layers and the final classification layer
class YATE_Encode(nn.Module):
    def __init__(self, input_dim_x, input_dim_e, hidden_dim, num_layers, **block_args):

        super(YATE_Encode, self).__init__()

        if input_dim_x != hidden_dim:
            self.initial_x = nn.Sequential(
                nn.Linear(input_dim_x, hidden_dim),
                nn.Dropout(),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.initial_x = nn.Identity()

        self.initial_e = nn.Sequential(
            nn.Linear(input_dim_e, hidden_dim),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        self.layers = nn.ModuleList(
            [
                YATE_Block(input_dim=hidden_dim, **block_args)
                for _ in range(num_layers - 1)
            ]
        )

        self.readout_layer = YATE_Block(
            input_dim=hidden_dim, read_out=True, **block_args
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input, return_edgeinfo=False, return_attention=False):

        # Define the appropriate vars
        x, edge_index, edge_attr, edge_type, head_idx = (
            input.x,
            input.edge_index,
            input.edge_attr,
            input.edge_type,
            input.head_idx,
        )

        # Initial layer for the node/edge features
        x = self.initial_x(x)
        edge_attr = self.initial_e(edge_attr)

        # change edge_index and edge_feat to undirected
        edge_index_ud, edge_type_ud, edge_attr_ud = to_undirected(
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
        )

        if return_attention:
            attention_maps = []
            for l in self.layers:
                _, _, attention = l.g_attn(
                    x, edge_index_ud, edge_attr_ud, return_attention
                )
                attention_maps.append(attention)

        for l in self.layers:
            x, edge_attr_ud = l(x, edge_index_ud, edge_attr_ud)

        x, _ = self.readout_layer(x, edge_index_ud, edge_attr_ud)

        # Extract representations of central entities
        x = x[head_idx, :]

        # Apply a final classifier
        x = self.classifier(x)

        if return_edgeinfo and return_attention:
            edge_index, edge_type, edge_attr = to_directed(
                edge_index, edge_type, edge_index_ud, edge_type_ud, edge_attr_ud
            )
            if return_attention:
                return x, edge_attr, edge_type, attention_maps
            else:
                return x, edge_attr, edge_type
        else:
            if return_attention:
                return x, attention_maps
            elif return_attention == False:
                return x
