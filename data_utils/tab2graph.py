"""
Class for transforming tables (DFs) into graphs
From the loaded dataframe, user can decide
what to set as main entities and targets

"""

# Python
import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Union, Optional


###############
## Tab2graph ##
###############


class Tab2graph:
    def __init__(self, data_name: str):

        super(Tab2graph, self).__init__()

        data_dir = os.getcwd() + "/data/raw/" + data_name + ".parquet"

        self.data_name = data_name
        self.data = pd.read_parquet(data_dir)
        self.data = self.data.replace('\n','', regex=True)
        self.col_names = list(self.data.columns)

    def make_graph(
        self,
        main_ent_col: str,
        target_col: Optional[Union[str, List[str]]] = None,
        save: bool = False,
    ):

        self.data = self.data.drop_duplicates(subset=[main_ent_col])

        rel_names = self.col_names.copy()
        rel_names.remove(main_ent_col)
        if target_col is not None:
            rel_names.remove(target_col)
            target = np.array(self.data[target_col])

        # triplet with pd.melt
        triplet = pd.melt(self.data, id_vars=main_ent_col, value_vars=rel_names)
        triplet = triplet.dropna()

        # Map entities relations and targets
        main_ent = triplet[main_ent_col]
        t_ent = triplet["value"]
        ent = pd.concat([main_ent, t_ent])
        ent = ent.unique()
        main_ent_ptr = len(main_ent.unique())
        # ent = np.hstack((h_ent.unique(), t_ent.unique()))
        # ent = np.unique(ent)
        ent2idx = np.dstack((ent, np.arange(len(ent))))
        ent2idx = np.reshape(ent2idx, (len(ent), 2))

        rel_names.insert(0, "hasName")
        rel2idx = np.dstack((np.array(rel_names), np.arange(len(rel_names))))
        rel2idx = np.reshape(rel2idx, (len(rel_names), 2))

        edge_index = triplet[[main_ent_col, "value"]]
        edge_index = edge_index.replace(list(ent2idx[:, 0]), list(ent2idx[:, 1]))
        edge_index = np.array(edge_index)
        edge_index = edge_index.T

        edge_type = triplet["variable"]
        edge_type = edge_type.replace(list(rel2idx[:, 0]), list(rel2idx[:, 1]))
        edge_type = np.array(edge_type)
        edge_type = edge_type.T
        edge_type = edge_type.astype(np.int64)

        # Change to torch tensor if possible
        try:
            import torch

            edge_index = torch.tensor(edge_index)
            edge_type = torch.tensor(edge_type)
        except ModuleNotFoundError:
            pass

        if save:

            data = {
                "edge_index": edge_index,
                "edge_type": edge_type,
                "ent2idx": ent2idx,
                "rel2idx": rel2idx,
                "target": target,
                "main_ent_ptr": main_ent_ptr,
            }
            save_dir = os.getcwd() + "/data/preprocessed/" + self.data_name + ".pickle"

            with open(save_dir, "wb") as pickle_file:
                pickle.dump(data, pickle_file)
