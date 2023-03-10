import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor


class Infonce_loss(_Loss):
    def __init__(self) -> None:
        super(Infonce_loss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _infonce(input, target)


def _infonce(input: torch.tensor, target: Tensor):
    pos_mask = (target - torch.eye(target.size(0), device=input.device)) > 0
    num_pos_ = int(sum(pos_mask)[0])
    loss = (
        -input[pos_mask].reshape((target.size(0), num_pos_))
        + torch.logsumexp(
            input[torch.eye(input.size(0)) == 0].reshape(
                input.size(0), input.size(0) - 1
            ),
            dim=-1,
        )
        .repeat((num_pos_, 1))
        .t()
    )
    loss = loss.mean()

    return loss
