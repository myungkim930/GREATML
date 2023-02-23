import os
import pandas as pd
import numpy as np
import torch
import pickle

# Need to make changes with numberical values. For now, keep numerical=False.
# Need to fix for 2022 version


def make_data_yago(data_name: str, numerical: bool = False, save: bool = False):

    data_folder_dir = os.getcwd() + "/data/raw/" + data_name + "/"

    triplet_cat_dir = data_folder_dir + "yagoFacts.tsv"
    triplet_cat = pd.read_csv(
        triplet_cat_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    )
    triplet_cat = triplet_cat.dropna()

    head_cat = triplet_cat["h"]
    tail_cat = triplet_cat["t"]
    rel_cat = triplet_cat["r"]

    if numerical:
        triplet_num_dir = data_folder_dir + "yagoLiteralFacts.tsv"
        triplet_num = pd.read_csv(
            triplet_num_dir,
            sep="\t",
            header=0,
            usecols=[1, 2, 3],
            names=["h", "r", "t"],
        )
        head_num = triplet_num["h"]
        tail_num = triplet_num["t"]
        rel_num = triplet_num["r"]

        head, tail, rel = head_cat, tail_cat, rel_cat

    else:
        head, tail, rel = head_cat, tail_cat, rel_cat

    # Extracting edge_index
    head_name = pd.DataFrame(head.unique())
    tail_name = pd.DataFrame(tail.unique())
    ent_name = pd.concat([head_name, tail_name])[0]
    ent_name = pd.DataFrame(ent_name.unique())
    ent2idx = pd.concat(
        [ent_name, pd.DataFrame({"idx": np.arange(len(ent_name))})], axis=1
    )
    ent2idx = np.array(ent2idx)

    head = head.map(dict(ent2idx))
    tail = tail.map(dict(ent2idx))

    edge_index = np.stack([head, tail])
    edge_index = torch.tensor(edge_index)

    # Extracting edge_type
    rel_name = pd.DataFrame(rel.unique())
    rel_name.index = rel_name.index + 1
    rel_name.loc[0] = "<hasName>"
    rel_name = rel_name.sort_index()

    rel2idx = pd.concat(
        [rel_name, pd.DataFrame({"idx": np.arange(len(rel_name))})], axis=1
    )
    rel2idx = np.array(rel2idx)
    rel = np.array(rel.map(dict(rel2idx)))

    edge_type = torch.tensor(rel)

    # Extracting types for head
    taxo_dir = data_folder_dir + "yagoSimpleTaxonomy.tsv"
    taxo = pd.read_csv(
        taxo_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    )

    taxo = taxo[taxo["r"] == "rdfs:subClassOf"]
    taxo = taxo[taxo["t"] != "owl:Thing"]

    type_name = pd.DataFrame(taxo["t"].unique())
    type2idx = pd.concat(
        [type_name, pd.DataFrame({"idx": np.arange(len(type_name))})], axis=1
    )
    type2idx = np.array(type2idx)

    type_map = pd.concat([taxo["h"], taxo["t"].map(dict(type2idx))], axis=1)
    type_map = np.array(type_map)

    types_dir = data_folder_dir + "yagoSimpleTypes.tsv"
    types = pd.read_csv(
        types_dir, sep="\t", header=0, usecols=[1, 2, 3], names=["h", "r", "t"]
    )

    types = types.drop_duplicates(subset=["h"])
    types = types.dropna()
    ent2type = pd.concat([types["h"], types["t"].map(dict(type_map))], axis=1)
    ent2type = np.array(ent2type)

    headidx2type = np.stack(
        [head_name[0].map(dict(ent2idx)), head_name[0].map(dict(ent2type))]
    )
    headidx2type = torch.tensor(headidx2type)

    data = {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "ent2idx": ent2idx,
        "rel2idx": rel2idx,
        "headidx2type": headidx2type,
        "type2idx": type2idx,
    }

    if save:
        save_dir = os.getcwd() + "/data/preprocessed/" + data_name + ".pickle"

        with open(save_dir, "wb") as pickle_file:
            pickle.dump(data, pickle_file)

    # return data
