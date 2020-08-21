from typing import List

import torch
from torch import Tensor


def accuracy(output: Tensor, target: Tensor, ks=(1,)) -> List[Tensor]:
    """
    Args:
        output: Tensor of shape :math:`(N, C)`
        target: Tensor of shape :math:`(N,)`

    Returns:
        Top-k accuracy for each k in ``ks``.

    .. reference::
        Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(ks)
        batch_size = target.size(0)

        # pred: (N, maxk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        # pred: (maxk, N)
        pred = pred.t()
        # correct: (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
