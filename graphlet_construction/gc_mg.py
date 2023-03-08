# Python
import numpy as np
import math
from typing import List, Union

# Pytorch
import torch
from torch import Tensor
from torch_geometric.data import Data

# from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# Graphlet
from .gc_utils import (
    k_hop_subgraph,
    subgraph,
    feature_extract_lm,
    add_self_loops,
    to_undirected,
)

# Graphlet class to construct a graphlet of a given entity
class Graphlet:
    def __init__(self, main_data, num_hops: int = 1, max_nodes: int = 100):

        super(Graphlet, self).__init__()

        self.main_data = main_data
        self.edge_index = main_data.edge_index
        self.edge_type = main_data.edge_type
        self.x_model = main_data.x_model
        self.ent2idx = main_data.ent2idx
        self.rel2idx = main_data.rel2idx

        self.num_hops = num_hops
        self.max_nodes = max_nodes

    def make_batch(
        self,
        cen_idx: Union[int, List[int], Tensor],
        per_perturb: float = 0.3,
        n_perturb_mask: int = 1,
        n_perturb_replace: int = 1,
    ):

        if isinstance(cen_idx, Tensor):
            cen_idx = cen_idx.tolist()
        if isinstance(cen_idx, int):
            cen_idx = [cen_idx]

        # Obtain the of entities and edge_types in the batch
        head_ = self.edge_index[0, :]
        tail_ = self.edge_index[1, :]
        node_mask = head_.new_empty(self.edge_index.max().item() + 1, dtype=torch.bool)
        node_mask.fill_(False)
        reduce_mask = head_.new_empty(head_.size(0), dtype=torch.bool)
        reduce_mask.fill_(False)

        subset = cen_idx

        for _ in range(self.num_hops):
            node_mask[subset] = True
            reduce_mask = torch.index_select(node_mask, 0, head_)
            subset = tail_[reduce_mask]

        self.edge_index_reduced = self.edge_index[:, reduce_mask]
        self.edge_type_reduced = self.edge_type[reduce_mask]

        # Obtain the list of data with original and perturbed graphs
        data_total = []
        self.neg_edge_set = self.edge_type_reduced.unique()
        self.neg_node_set = self.edge_index_reduced.unique()
        perturb_methods = ["mask_edge", "mask_node", "neg_edge", "neg_node"]

        for g_idx in range(len(cen_idx)):
            data_original_ = self._make_graphlet(node_idx=cen_idx[g_idx])
            data_original_.idx_perturb = torch.tensor([-1])
            data_perturb_mask = [
                self._perturb_graphlet(
                    data=data_original_,
                    method=perturb_methods[np.random.randint(0, 2)],
                    per_perturb=per_perturb,
                )
                for _ in range(n_perturb_mask)
            ]
            data_perturb_replace = [
                self._perturb_graphlet(
                    data=data_original_,
                    method=perturb_methods[np.random.randint(2, 4)],
                    per_perturb=per_perturb,
                )
                for _ in range(n_perturb_replace)
            ]
            data_ = data_perturb_mask + data_perturb_replace
            data_.insert(0, data_original_)
            data_total = data_total + data_

        # Form the batch with obtained graphlets
        makebatch = Batch()
        data_batch_temp = makebatch.from_data_list(
            data_total, follow_batch=["edge_index", "idx_perturb"]
        )
        for i in range(data_batch_temp.idx_perturb.size(0)):
            if data_batch_temp.idx_perturb[i] < 0:
                continue
            else:
                data_batch_temp.idx_perturb[i] = (
                    data_batch_temp.idx_perturb[i]
                    + data_batch_temp.edge_index_ptr[
                        data_batch_temp.idx_perturb_batch[i]
                    ]
                )

        edge_index, edge_type, edge_attr, idx_perturb = to_undirected(
            data_batch_temp.edge_index,
            data_batch_temp.edge_type,
            data_batch_temp.edge_attr,
            data_batch_temp.idx_perturb[data_batch_temp.idx_perturb > -1],
        )

        data_batch = Data(
            x=data_batch_temp.x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            g_idx=data_batch_temp.g_idx,
            y=data_batch_temp.y,
            idx_perturb=idx_perturb,
            head_idx=data_batch_temp.ptr[:-1],
        )

        return data_batch

    def _make_graphlet(
        self,
        node_idx: Union[int, List[int], Tensor],
        exclude_center: bool = True,
    ):

        if isinstance(node_idx, Tensor):
            node_idx = int(node_idx)
        elif isinstance(node_idx, List):
            node_idx = node_idx[0]

        edge_index, edge_type, mapping = k_hop_subgraph(
            edge_index=self.edge_index_reduced,
            node_idx=node_idx,
            num_hops=self.num_hops,
            edge_type=self.edge_type_reduced,
        )

        if self.max_nodes is not None:
            if mapping.size(1) > self.max_nodes:
                idx_keep = (mapping[1, :] > 0) & (
                    mapping[1, :] < edge_index.max().item()
                )
                idx_keep[
                    idx_keep.nonzero().squeeze()[
                        torch.randperm(idx_keep.nonzero().squeeze().size(0))[
                            0 : self.max_nodes - 2
                        ]
                    ]
                ] = False
                idx_keep = ~idx_keep
                idx_keep = idx_keep.nonzero().squeeze()
                edge_index, edge_mask, mask_ = subgraph(idx_keep, edge_index)
                edge_type = edge_type[edge_mask]
                mapping = torch.vstack((mapping[0, mask_[0, :]], mask_[1, :]))

        edge_index, edge_type = add_self_loops(
            edge_index=edge_index,
            edge_type=edge_type,
            exclude_center=exclude_center,
        )

        x, edge_feat = feature_extract_lm(
            main_data=self.main_data,
            node_idx=mapping[0, :],
            edge_type=edge_type,
            exclude_center=exclude_center,
        )

        data_out = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_feat,
            g_idx=node_idx,
            y=torch.tensor([1.0]),
            mapping=torch.transpose(mapping, 0, 1),
        )

        return data_out

    def _perturb_graphlet(
        self,
        data,
        method: str,
        per_perturb: float,
    ):

        # Obtain indexes for perturbation
        idx_perturb_ = (data.edge_index[0, :] == 0).nonzero().view(-1)
        n_candidate = idx_perturb_.size(0)
        n_perturb = math.ceil(per_perturb * n_candidate)
        idx_perturb_ = idx_perturb_[torch.randperm(n_candidate)[0:n_perturb]]
        data_perturb = data.clone()
        data_perturb.idx_perturb = idx_perturb_.clone()

        # Obtain Data class for the perturbed graphlet
        if method == "mask_edge":
            data_perturb.edge_attr[idx_perturb_, :] = torch.ones(
                n_perturb, data_perturb.edge_attr.size(1)
            )
            data_perturb.y = torch.tensor([1.0])
        elif method == "mask_node":
            data_perturb.x[
                data_perturb.edge_index[1, idx_perturb_], :
            ] = -9e15 * torch.ones(n_perturb, data_perturb.x.size(1))
            data_perturb.y = torch.tensor([1.0])
        elif method == "neg_edge":
            neg_edge_ = torch.ones(len(self.rel2idx))
            neg_edge_[data_perturb.edge_type[idx_perturb_]] = 0
            neg_edge_type = torch.multinomial(neg_edge_, n_perturb, replacement=True)
            data_perturb.edge_attr[idx_perturb_, :] = feature_extract_lm(
                self.main_data, edge_type=neg_edge_type
            )
            data_perturb.y = torch.tensor([0.0])
        elif method == "neg_node":
            neg_node_ = torch.bincount(
                self.neg_node_set,
            )
            neg_node_[neg_node_ > 0] = 1
            neg_node_[data_perturb.mapping[:, 0]] = 0
            neg_node_ = neg_node_.type(torch.float)
            neg_node_idx = torch.multinomial(neg_node_, n_perturb, replacement=True)
            data_perturb.x[
                data_perturb.edge_index[1, idx_perturb_], :
            ] = feature_extract_lm(
                self.main_data, node_idx=neg_node_idx, exclude_center=False
            )
            data_perturb.y = torch.tensor([0.0])
        else:
            print("error")

        return data_perturb
