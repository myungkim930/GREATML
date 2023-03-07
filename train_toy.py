### Trainer for toy examples

import os
import torch
import datetime

import graphlet_construction as gc

from model import YATE_Encode
from data_utils import Load_data
from loss import infonce_loss

## Trainer class
class Trainer:
    def __init__(
        self,
        exp_setting: dict,
    ) -> None:
        self.__dict__ = exp_setting
        self.model = self.model.to(self.device)
        self.log = []

    def _run_batch(self, data):
        self.optimizer.zero_grad()
        output_x, output_edge_attr = self.model(data)
        loss_node = infonce_loss(output_x, data)
        target = data.edge_type[data.idx_perturb]
        loss_edge = self.criterion(output_edge_attr, target)
        loss = loss_node + loss_edge
        self.loss_node = round(loss_node.item(), 4)
        self.loss_edge = round(loss_edge.item(), 4)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        print(f"[GPU{self.device.index}] Epoch {epoch} | Batchsize: {self.n_batch}")
        self.model.train()
        batch_number = 1
        self.idx_extract.reset()
        while self.idx_extract.cont_flag:
            idx = self.idx_extract.sample(n_batch=self.n_batch)
            if self.idx_extract.cont_flag == False:
                break
            data = self.graphlet.make_batch(idx, **self.graphlet_setting)
            data = data.to(self.device)
            self._run_batch(data)
            print(
                f"[GPU{self.device.index}] Epoch {epoch} | Batch_No.: {batch_number}/{self.n_iter} | Loss(n/e): {self.loss_node}/{self.loss_edge}"
            )
            self.log.append(
                f"[GPU{self.device.index}] Epoch {epoch} | Batch_No.: {batch_number}/{self.n_iter} | Loss(n/e): {self.loss_node}/{self.loss_edge}"
            )
            batch_number += 1

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = self.save_dir + f"/checkpoint_ep{epoch}.pt"
        torch.save(ckp, PATH)
        PATH_LOG = self.save_dir + f"/log_train.txt"
        with open(PATH_LOG, "w") as output:
            for row in self.log:
                output.write(str(row) + "\n")
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self):
        for epoch in range(self.n_epoch):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(
    data_name: str,
    gpu_device: int,
    num_rel: int,
    num_hops: int,
    per_perturb: float,
    n_perturb_mask: int,
    n_perturb_neg: int,
    max_nodes: int,
    n_batch: int = 32,
    n_epoch: int = 50,
    save_every: int = 1,
):

    # create dictionary that set experiment settings
    exp_setting = dict()
    exp_setting["n_batch"] = n_batch
    exp_setting["n_epoch"] = n_epoch
    exp_setting["save_every"] = save_every

    # set device
    device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
    exp_setting["device"] = device

    # exp_setting["graphlet_setting"] = [num_pos, per_pos, num_neg, per_neg, max_nodes]
    exp_setting["graphlet_setting"] = dict(
        {
            "per_perturb": per_perturb,
            "n_perturb_mask": n_perturb_mask,
            "n_perturb_neg": n_perturb_neg,
            "max_nodes": max_nodes,
        }
    )

    # load data
    main_data = Load_data(data_name=data_name)
    if num_rel is not None:
        main_data.reduce(num_rel=num_rel)

    # set graph_construction framework
    graphlet = gc.Graphlet(main_data, num_hops=num_hops)
    exp_setting["graphlet"] = graphlet

    # set train for batch
    idx_extract = idx_extractor(main_data, max_nodes=max_nodes)
    exp_setting["idx_extract"] = idx_extract

    # (hypothetical) iterations per epoch
    exp_setting["n_iter"] = (main_data.edge_index[0, :].unique().size(0) // n_batch) + 1

    # load your model
    model = YATE_Encode(
        input_dim_x=300,
        input_dim_e=300,
        hidden_dim=300,
        edge_class_dim=len(main_data.rel2idx),
        num_layers=10,
        ff_dim=300,
        num_heads=2,
    )
    exp_setting["model"] = model

    # experiment settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    exp_setting["optimizer"] = optimizer
    exp_setting["criterion"] = criterion

    now = datetime.datetime.now()
    save_dir = os.getcwd() + "/data/saved_model/" + now.strftime("%Y-%m-%d|%H:%M:%S")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exp_setting["save_dir"] = save_dir

    return exp_setting


## Index sampler according to the coverage of edge_index
class idx_extractor:
    def __init__(self, main_data, max_nodes: int):
        self.main_data = main_data
        self.max_nodes = max_nodes

    def reset(self):
        self.count_head = torch.ceil(
            torch.bincount(self.main_data.edge_index[0, :]) / self.max_nodes
        )
        self.cont_flag = True

    def sample(self, n_batch: int):
        n_train = n_batch
        if n_batch > self.count_head[self.count_head > 0].size(0):
            n_train = self.count_head[self.count_head > 0].size(0)
        if n_train < n_batch / 2:
            self.cont_flag = False
        else:
            idx_sample = torch.multinomial(self.count_head, n_train)
            self.count_head[idx_sample] -= 1
            return idx_sample


##############
def main():
    os.chdir("/storage/store3/work/mkim/gitlab/GREATML")
    exp_setting = load_train_objs(
        data_name="yago3",
        gpu_device=1,
        num_hops=1,
        per_perturb=0.3,
        n_perturb=1,
        max_nodes=100,
        n_batch=256,
        n_epoch=40,
        save_every=1,
    )
    trainer = Trainer(exp_setting)
    trainer.train()


if __name__ == "__main__":
    main()

##################
