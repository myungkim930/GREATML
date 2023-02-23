"""
Functions that can be utilized in graphlet construction

"""

# Python
import re
import numpy as np
from typing import List, Tuple, Union, Optional

# Pytorch
import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import index_to_mask

## K-hop Subgraph Extraction
def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Adj,
    edge_type: Union[int, List[int], Tensor],
    flow: str = "target_to_source",
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.
    """

    num_nodes = edge_index.max().item() + 1

    assert flow in ["source_to_target", "target_to_source"]
    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, _ = torch.cat(subsets).unique(return_inverse=True)

    node_mask.fill_(False)
    node_mask[subset] = True

    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]

    edge_type = edge_type[edge_mask]

    mapping = torch.reshape(torch.tensor((node_idx, 0)), (2, 1))
    mapping_temp = torch.vstack(
        (subset[subset != node_idx], torch.arange(1, subset.size()[0]))
    )
    mapping = torch.hstack((mapping, mapping_temp))

    edge_index_new = edge_index.clone()
    for i in range(mapping.size(1)):
        edge_index_new[edge_index == mapping[0, i]] = mapping[1, i]

    return edge_index_new, edge_type, mapping


## Subgraph with assigned nodes
def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Union[Tensor, List[int]],
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.
    """

    device = edge_index.device
    num_nodes = edge_index.max().item() + 1

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    subset = index_to_mask(subset, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    mapping = torch.vstack((edge_index.unique(), torch.argsort(edge_index.unique())))

    edge_list_new = edge_index.clone()
    for i in range(mapping.size()[1]):
        edge_list_new[edge_index == mapping[0, i]] = mapping[1, i]

    return edge_list_new, edge_mask, mapping


## Add self-loop function
def add_self_loops(
    edge_index: Adj,
    edge_type=None,
) -> Tuple[Tensor, Tensor]:
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index` or
    to the central node. Edgetype of self-loops will be added with '0'
    """

    N = edge_index.max().item() + 1

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    if edge_type is not None:
        edge_type = torch.cat(
            [edge_type, torch.zeros(N, dtype=torch.long, device=edge_index.device)],
            dim=0,
        )
        return edge_index, edge_type
    else:
        return edge_index


## Remove duplicate function
def remove_duplicates(edge_index: Adj, edge_type: Adj = None, edge_attr: Adj = None):

    nnz = edge_index.size(1)
    num_nodes = edge_index.max().item() + 1

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[0]
    idx[1:].mul_(num_nodes).add_(edge_index[1])

    if edge_type is not None:
        idx[1:].add_(edge_type * (10 ** (len(str(num_nodes)) + 1)))

    idx[1:], perm = torch.sort(
        idx[1:],
    )

    mask = idx[1:] > idx[:-1]

    edge_index = edge_index[:, perm]
    edge_index = edge_index[:, mask]

    if edge_type is not None:
        edge_type, edge_attr = edge_type[perm], edge_attr[perm, :]
        edge_type, edge_attr = edge_type[mask], edge_attr[mask, :]
        return edge_index, edge_type, edge_attr
    else:
        return edge_index


## To undirected function
def to_undirected(edge_index: Adj, edge_type: Adj = None, edge_attr: Adj = None):

    row = torch.cat([edge_index[0, :], edge_index[1, :]])
    col = torch.cat([edge_index[1, :], edge_index[0, :]])

    edge_index = torch.stack([row, col], dim=0)

    if edge_type is not None:
        edge_type = torch.cat([edge_type, edge_type])
        edge_attr = torch.vstack((edge_attr, edge_attr))
        edge_index, edge_type, edge_attr = remove_duplicates(
            edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr
        )
        return edge_index, edge_type, edge_attr
    else:
        edge_index = remove_duplicates(edge_index=edge_index)
        return edge_index


## To the original(directed) with
def to_directed(
    edge_index: Adj,
    edge_type: Adj,
    edge_index_mod: Adj,
    edge_type_mod: Adj,
    edge_attr_mod: Adj,
):

    num_nodes = edge_index.max().item() + 1
    idx = edge_index[0].clone()
    idx.mul_(num_nodes).add_(edge_index[1]).add_(
        edge_type * (10 ** (len(str(num_nodes)) + 1))
    )

    idx_mod = edge_index_mod[0].clone()
    idx_mod.mul_(num_nodes).add_(edge_index_mod[1]).add_(
        edge_type_mod * (10 ** (len(str(num_nodes)) + 1))
    )

    map = ~(idx_mod[:, None] != idx).all(dim=1)
    edge_index = edge_index_mod[:, map]
    edge_type = edge_type_mod[map]
    edge_attr = edge_attr_mod[map, :]

    return edge_index, edge_type, edge_attr


##############

## Word-Feature Extraction (Subject to change depending on the naming of entities/relations)
def feature_extract_lm(
    main_data,
    node_idx: Optional[Union[int, List[int], Tensor]] = None,
    edge_type: Optional[Union[int, List[int], Tensor]] = None,
):
    r"""Extracts node/edge features from language model."""

    if node_idx is not None:
        if isinstance(node_idx, int):
            node_idx = [node_idx]
        elif isinstance(node_idx, Tensor) and node_idx.size()[0] == 1:
            node_idx = [node_idx.tolist()]

        gent_names = main_data.ent2idx[node_idx]

        x = [
            main_data.x_model.get_sentence_vector(
                x.replace("_", " ")
                .replace("<", "")
                .replace("\n", "")
                .replace(">", "")
                .lower()
            )
            for x in gent_names
        ]

        x = np.array(x)
        x = torch.tensor(
            x,
        )

    if edge_type is not None:
        if isinstance(edge_type, int):
            edge_type = [edge_type]
        elif isinstance(edge_type, Tensor) and edge_type.size()[0] == 1:
            edge_type = edge_type.tolist()

        grel_names = main_data.rel2idx[edge_type]

        edge_feat = [
            main_data.x_model.get_sentence_vector(
                re.sub(r"\B([A-Z])", r" \1", x)
                .replace("_", " ")
                .replace("\n", "")
                .replace("<", "")
                .replace(">", "")
                .lower()
            )
            for x in grel_names
        ]

        edge_feat = np.array(edge_feat)
        edge_feat = torch.tensor(
            edge_feat,
        )

    if (node_idx is not None) and (edge_type is None):
        return x
    elif (node_idx is None) and (edge_type is not None):
        return edge_feat
    elif (node_idx is not None) and (edge_type is not None):
        return x, edge_feat
