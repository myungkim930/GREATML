"""
Class for loading and setting data
"""

# Python
import pickle
import fasttext
import os
import torch
import numpy as np
import pandas as pd

## Load data and place it as object
class Load_data:
    def __init__(
        self,
        data_name: str,
        x_model: str = "fasttext",
    ):

        super(Load_data, self).__init__()

        data_dir = os.getcwd() + "/data/preprocessed/" + data_name + ".pickle"

        with open(data_dir, "rb") as pickle_file:
            data = pickle.load(pickle_file)

        self.__dict__ = data

        if x_model == "fasttext":
            self.x_model = fasttext.load_model(
                "/storage/store3/work/mkim/gitlab/cc.en.300.bin"
            )
        if isinstance(data["ent2idx"], list):
            self.ent2idx = np.array(data["ent2idx"])
        else:
            self.ent2idx = data["ent2idx"]
        if isinstance(data["rel2idx"], list):
            self.rel2idx = np.array(data["rel2idx"])
        else:
            self.rel2idx = data["rel2idx"]

        if len(np.shape(self.ent2idx)) > 1:
            self.ent2idx = self.ent2idx[:, 0]
            self.rel2idx = self.rel2idx[:, 0]

    def reduce(self, num_rel: int = 10):

        head = np.array(self.edge_index[0, :])
        count = np.bincount(head)
        idx = pd.DataFrame({"idx": np.where(count > num_rel - 1)[0]})
        mask_map = pd.concat([idx, pd.DataFrame(np.zeros((len(idx), 1)) == 0)], axis=1)
        mask_map = np.array(mask_map)

        mask = pd.DataFrame(head)[0].map(dict(mask_map))
        mask = np.array(mask.fillna(False))

        self.edge_index = self.edge_index[:, mask]
        self.edge_type = self.edge_type[mask]

        mask_h = pd.DataFrame(self.headidx2type.T)[0].map(dict(mask_map))
        mask_h = np.array(mask_h.fillna(False))
        self.headidx2type = self.headidx2type[:, mask_h]
