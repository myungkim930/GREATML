"""
Graphlet construction & augmentation (pos/neg) frameworks

"""

# Python
import numpy as np
import math
from typing import List, Union

# Pytorch
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Graphlet
from .gc_utils import k_hop_subgraph, subgraph, feature_extract_lm, add_self_loops

#####################################
## Graphlet Construction Framework ##
#####################################

# Graphlet class to construct a graphlet of a given entity
class Graphlet:
    def __init__(self, main_data, num_hops: int = 1, flow: str = "target_to_source"):

        super(Graphlet, self).__init__()

        self.main_data = main_data
        self.edge_index = main_data.edge_index
        self.edge_type = main_data.edge_type
        self.x_model = main_data.x_model
        self.ent2idx = main_data.ent2idx
        self.rel2idx = main_data.rel2idx

        self.num_hops = num_hops
        self.flow = flow

    def make_graphlet(
        self,
        cen_ent: int,
    ):

        edge_index, edge_type, mapping = k_hop_subgraph(
            edge_index=self.edge_index,
            node_idx=cen_ent,
            num_hops=self.num_hops,
            edge_type=self.edge_type,
            flow=self.flow,
        )

        edge_index, edge_type = add_self_loops(
            edge_index=edge_index, edge_type=edge_type
        )

        x, edge_feat = feature_extract_lm(
            main_data=self.main_data, node_idx=mapping[0, :], edge_type=edge_type
        )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_feat,
            y=1,
            g_idx=cen_ent,
            mapping=torch.transpose(mapping, 0, 1),
        )

        return data

    def set_augment_param(self, **kwargs):

        self.max_nodes = kwargs.get("max_nodes")
        self.n_pos = kwargs.get("n_pos")
        self.per_pos = kwargs.get("per_pos")
        self.n_neg = kwargs.get("n_neg")
        self.per_neg = kwargs.get("per_neg")

    def augment(self, cen_ent):

        data = self.make_graphlet(cen_ent)

        if data.num_nodes > self.max_nodes:
            g_temp = gen_pos(1, self.max_nodes / data.num_nodes, data)
            data = g_temp[0]

        g_p = gen_pos(self.n_pos, self.per_pos, data)
        g_p.insert(0, data)

        g_n = []

        neg_set = np.setdiff1d(
            range(self.main_data.edge_index.max().item() + 1), data.mapping[:, 0]
        )

        for i in range(len(g_p)):

            g_n += gen_neg(
                self.n_neg, self.per_neg, g_p[i], self.main_data, neg_set=neg_set
            )

        g_p = g_p + g_n

        batch = next(iter(DataLoader(g_p, batch_size=len(g_p))))
        batch = Data(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_type=batch.edge_type,
            y=batch.y,
            g_idx=batch.g_idx,
            mapping=batch.mapping,
            head_idx=batch.ptr[: batch.y.size()[0]],
        )

        return batch

    def make_batch(
        self,
        idx_cen: Union[int, List[int], Tensor],
        aug: bool = True,
        max_nodes: int = 100,
        n_pos: int = 10,
        per_pos: float = 0.8,
        n_neg: int = 10,
        per_neg: float = 0.05,
    ):

        self.set_augment_param(
            max_nodes=max_nodes,
            n_pos=n_pos,
            per_pos=per_pos,
            n_neg=n_neg,
            per_neg=per_neg,
        )

        if isinstance(idx_cen, Tensor):
            idx_cen = idx_cen.tolist()
        elif isinstance(idx_cen, int):
            idx_cen = [idx_cen]

        data = []
        start_idx = 0

        for g_idx in range(len(idx_cen)):

            if aug == True:
                data_total_temp = self.augment(cen_ent=idx_cen[g_idx])
                data_total_temp.head_idx = data_total_temp.head_idx + start_idx
                start_idx += data_total_temp.num_nodes

            else:
                data_total_temp = self.make_graphlet(cen_ent=idx_cen[g_idx])

            data.append(data_total_temp)

        d_batch = next(iter(DataLoader(data, batch_size=len(idx_cen))))

        if hasattr(d_batch, "head_idx") == False:
            d_batch.head_idx = d_batch.ptr[0 : len(idx_cen)]

        return d_batch


## Function to create positives
def gen_pos(n_pos, per_pos, data):

    g_pos = []
    check_dup = []
    nb_mark = 0
    n_neighbor = data.num_nodes - 1

    if math.comb(n_neighbor, int(n_neighbor * per_pos)) < n_pos:
        crit_pos = math.comb(n_neighbor, int(n_neighbor * per_pos))
    else:
        crit_pos = n_pos

    while nb_mark < crit_pos:

        idx_pos = list(
            np.random.choice(
                data.mapping[1:, 1],
                replace=False,
                size=int(n_neighbor * per_pos),
            )
        )

        if set(idx_pos) in check_dup:
            continue
        else:
            check_dup.append(set(idx_pos))

        idx_pos.insert(0, 0)

        if len(idx_pos) == 1:
            break

        edge_index, edge_mask, mask_p = subgraph(idx_pos, data.edge_index)

        x_pos = data.x[mask_p[0, :], :]
        edge_feat_pos = data.edge_attr[edge_mask]
        edge_type_pos = data.edge_type[edge_mask]

        mapping_pos = torch.vstack((data.mapping[mask_p[0, :], 0], mask_p[1, :]))

        pos_data = Data(
            x=x_pos,
            edge_index=edge_index,
            edge_attr=edge_feat_pos,
            edge_type=edge_type_pos,
            y=torch.tensor([1.0]),
            g_idx=data.mapping[0, 0],
            mapping=torch.transpose(mapping_pos, 0, 1),
        )

        g_pos.append(pos_data)
        nb_mark += 1

    return g_pos


## Function to create negatives
def gen_neg(n_neg, per_neg, data, main_data, neg_set=None):

    g_neg = []
    n_neighbor = data.num_nodes - 1
    nb_mark = 0

    if neg_set is None:
        neg_set = np.setdiff1d(main_data.edge_index[1, :].unique(), data.mapping[:, 0])

    idx_neg = np.random.choice(
        neg_set, (n_neg, math.ceil(n_neighbor * per_neg)), replace=False
    )

    while nb_mark < n_neg:

        idx_replace = np.random.choice(
            range(1, n_neighbor + 1),
            size=math.ceil(n_neighbor * per_neg),
            replace=False,
        )

        mapping_neg = data.mapping.clone()
        mapping_neg[idx_replace, 0] = torch.tensor(idx_neg[nb_mark, :])

        x_neg_replace = feature_extract_lm(
            main_data=main_data, node_idx=idx_neg[nb_mark, :]
        )
        x_neg = data.x.clone()
        x_neg[idx_replace, :] = x_neg_replace

        neg_data = Data(
            x=x_neg,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_type=data.edge_type,
            y=torch.tensor([0.0]),
            g_idx=data.mapping[0, 0],
            mapping=mapping_neg,
        )

        g_neg.append(neg_data)
        nb_mark += 1

    return g_neg


############################
## Augmentation Framework ##
############################

# ## Graph augmentation class
# class Augment:
#     def __init__(
#         self,
#         max_nodes: int = 100,
#         n_pos: int = 10,
#         per_pos: float = 0.8,
#         n_neg: int = 10,
#         per_neg: float = 0.05,
#     ):

#         super(Augment, self).__init__()

#         self.max_nodes = max_nodes
#         self.n_pos = n_pos
#         self.per_pos = per_pos
#         self.n_neg = n_neg
#         self.per_neg = per_neg

#     def generate(self, data, main_data):

#         if data.num_nodes > self.max_nodes:
#             g_temp = gen_pos(1, 100 / data.num_nodes, data)
#             data = g_temp[0]

#         g_p = gen_pos(self.n_pos, self.per_pos, data)
#         g_p.insert(0, data)

#         g_n = []

#         neg_set = np.setdiff1d(
#             range(main_data.edge_index.max().item() + 1), data.mapping[:, 0]
#         )

#         for i in range(len(g_p)):

#             g_n += gen_neg(self.n_neg, self.per_neg, g_p[i], main_data, neg_set=neg_set)

#         g_p = g_p + g_n

#         batch = next(iter(DataLoader(g_p, batch_size=len(g_p))))
#         batch = Data(
#             x=batch.x,
#             edge_index=batch.edge_index,
#             edge_attr=batch.edge_attr,
#             edge_type=batch.edge_type,
#             y=batch.y,
#             g_idx=batch.g_idx,
#             mapping=batch.mapping,
#             head_idx=batch.ptr[: batch.y.size()[0]],
#         )

#         return batch
