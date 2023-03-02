import torch


def _similarity(input: torch.tensor):
    norm = torch.norm(input, p=2, dim=1)
    input = input / norm.unsqueeze(1)
    similarity = torch.mm(input, input.t())

    return similarity


def _pos_mask(graph_idx: torch.tensor):
    pos_mask = torch.mul(torch.unsqueeze(graph_idx, dim=1), 1 / graph_idx) - torch.eye(
        graph_idx.size(0)
    )
    pos_mask = torch.round(pos_mask, decimals=2)
    pos_mask = pos_mask == 1

    return pos_mask


def infonce_loss(input: torch.tensor, data, tau: float):

    graph_idx = data.g_idx
    cos_sim = _similarity(input) / tau
    pos_mask = _pos_mask(graph_idx)
    num_pos_ = int(sum(pos_mask)[0])

    loss = (
        -cos_sim[pos_mask].reshape((graph_idx.size(0), num_pos_))
        + torch.logsumexp(cos_sim, dim=-1).repeat((num_pos_, 1)).t()
    )
    loss = loss.mean()

    return loss
