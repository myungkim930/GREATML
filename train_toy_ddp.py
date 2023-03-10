### Trainer for toy examples

import os
import torch
import datetime
import math

import graphlet_construction as gc

from time import time
from model import YATE_Encode
from data_utils import Load_data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from loss import create_target_node, Infonce_loss

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


##############
## DDP setup
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


## Trainer class
class Trainer:
    def __init__(
        self,
        exp_setting: dict,
        gpu_id: int,
    ) -> None:
        self.__dict__ = exp_setting
        self.gpu_id = gpu_id
        self.model = self.model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.criterion_node = torch.nn.L1Loss()
        # Other losses: torch.nn. BCEWithLogitsLoss BCELoss L1Loss / Infonce_loss
        self.weights = self.weights.to(gpu_id)
        self.criterion_edge = torch.nn.CrossEntropyLoss(self.weights)
        self.log = []

    def _run_step(self, step, idx):
        start_time = time()
        self.optimizer.zero_grad()
        data = self.graphlet.make_batch(idx, **self.graphlet_setting)
        data = data.to(self.gpu_id)
        output_x, output_edge_attr = self.model(data)

        # loss on nodes
        target_node = create_target_node(data=data)
        # target_node.to(self.gpu_id)
        loss_node = self.criterion_node(output_x, target_node)

        # loss on edges
        target_edge = data.edge_type[data.idx_perturb]
        loss_edge = self.criterion_edge(output_edge_attr, target_edge)

        loss = loss_node + loss_edge
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        end_time = time()
        duration = round(end_time - start_time, 4)

        loss_node = round(loss_node.detach().item(), 4)
        loss_edge = round(loss_edge.detach().item(), 4)

        self.log.append(
            f"Step {step} | Loss(n/e): {loss_node}/{loss_edge} | Duration: {duration}"
        )

        print(
            f"[GPU{self.gpu_id}] Step {step} | Loss(n/e): {loss_node}/{loss_edge} | Duration: {duration}"
        )

        del (
            loss,
            loss_node,
            loss_edge,
            output_x,
            output_edge_attr,
            target_node,
            target_edge,
            data,
        )

    def _set_perturb(self):
        self.graphlet_setting["per_perturb"] += self.perturb_window[
            "per_perturb_increase"
        ]
        self.perturb_window["step_perturb_change"] += self.perturb_window["step_diff"]
        self.perturb_window["step_diff"] *= self.perturb_window["change_step_diff"]

    def _save_checkpoint(self, step):
        ckp = self.model.module.state_dict()
        PATH = self.save_dir + f"/ckpt_step{step}.pt"
        torch.save(ckp, PATH)
        PATH_LOG = self.save_dir + f"/log_train.txt"
        with open(PATH_LOG, "w") as output:
            for row in self.log:
                output.write(str(row) + "\n")
        print(f"Step {step-1} | Training checkpoint saved at {self.save_dir}")

    def train(self):
        self.idx_extract.reset()
        self.model.train()
        step = 0
        idx = self.idx_extract.sample(n_batch=self.n_batch * torch.cuda.device_count())
        train_idx = DataLoader(
            idx,
            batch_size=self.n_batch,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(idx),
        )
        # self.train_data.sampler.set_epoch(epoch)
        while step < self.n_steps:
            for idx in train_idx:
                self._run_step(step, idx)
            step += 1
            if step > self.perturb_window["step_perturb_change"] - 1:
                self._set_perturb()
            if self.gpu_id == 0 and step % self.save_every == 0:
                self._save_checkpoint(step)


def load_train_objs(
    data_name: str,
    # gpu_device: int,
    num_hops: int,
    per_perturb: float,
    n_perturb_mask: int,
    n_perturb_replace: int,
    max_nodes: int,
    n_batch: int,
    n_steps: int,
    save_every: int,
    num_rel=None,
):

    # create dictionary that set experiment settings
    exp_setting = dict()
    exp_setting["n_batch"] = n_batch
    exp_setting["n_steps"] = n_steps
    exp_setting["save_every"] = save_every

    # set device
    # device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
    # exp_setting["device"] = device

    # exp_setting["graphlet_setting"] = [num_pos, per_pos, num_neg, per_neg, max_nodes]
    exp_setting["graphlet_setting"] = dict(
        {
            "per_perturb": per_perturb,
            "n_perturb_mask": n_perturb_mask,
            "n_perturb_replace": n_perturb_replace,
        }
    )

    exp_setting["perturb_window"] = dict(
        {
            "step_perturb_change": 100,
            "step_diff": 200,
            "change_step_diff": 2,
            "per_perturb_increase": 0.2,
        }
    )

    # load data
    main_data = Load_data(data_name=data_name)
    if num_rel is not None:
        main_data.reduce(num_rel=num_rel)

    # set graph_construction framework
    graphlet = gc.Graphlet(main_data, num_hops=num_hops, max_nodes=max_nodes)
    exp_setting["graphlet"] = graphlet

    # set train for batch
    idx_extract = Index_extractor(main_data, max_nodes=max_nodes)
    exp_setting["idx_extract"] = idx_extract

    # load your model
    model = YATE_Encode(
        input_dim_x=300,
        input_dim_e=300,
        hidden_dim=300,
        edge_class_dim=len(main_data.rel2idx),
        num_layers=6,
        ff_dim=300,
        num_heads=2,
    )
    exp_setting["model"] = model
    # experiment settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    exp_setting["optimizer"] = optimizer

    weights = torch.bincount(main_data.edge_type)
    weights[0] = main_data.edge_index[0, :].unique().size(0)
    weights = weights / torch.sum(weights)
    exp_setting["weights"] = weights

    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=100, T_mult=2, eta_max=1e-3, T_up=10, gamma=0.8
    )
    exp_setting["scheduler"] = scheduler

    now = datetime.datetime.now()
    save_dir = (
        os.getcwd()
        + "/data/saved_model/"
        + now.strftime(f"%d%m%Y_BS{n_batch}_NH{num_hops}")
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exp_setting["save_dir"] = save_dir

    return exp_setting


## Index sampler according to the coverage of edge_index
class Index_extractor:
    def __init__(self, main_data, max_nodes: int):
        self.main_data = main_data
        self.max_nodes = max_nodes

    def reset(self):
        self.count_head = torch.ceil(
            torch.bincount(self.main_data.edge_index[0, :]) / self.max_nodes
        )

    def sample(self, n_batch: int):
        if n_batch > self.count_head[self.count_head > 0].size(0):
            self.idx_extract.reset()
        idx_sample = torch.multinomial(self.count_head, n_batch)
        self.count_head[idx_sample] -= 1
        return idx_sample


## scheduler for the learning rate
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


##############
def main(rank: int, world_size: int):
    os.chdir("/storage/store3/work/mkim/gitlab/GREATML")
    exp_setting = load_train_objs(
        data_name="yago3",
        # gpu_device=1,
        num_hops=1,
        per_perturb=0.2,
        n_perturb_mask=1,
        n_perturb_replace=0,
        max_nodes=100,
        n_batch=16,
        n_steps=1000,
        save_every=100,
    )
    ddp_setup(rank, world_size)
    trainer = Trainer(exp_setting, rank)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)

##################


# join=True
