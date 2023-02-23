from .gc_makeg import Graphlet

# from .gc_makebatch import make_batch
from .gc_utils import (
    k_hop_subgraph,
    feature_extract_lm,
    subgraph,
    add_self_loops,
    to_undirected,
    remove_duplicates,
    to_directed,
)

__all__ = [
    "Graphlet",
    "k_hop_subgraph",
    "feature_extract_lm",
    "subgraph",
    "add_self_loops",
    "to_undirected",
    "remove_duplicates",
    "to_directed",
]

classes = __all__
