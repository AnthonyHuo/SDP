import torch
from torch import autograd, nn as nn


class GetMask(autograd.Function):
    @staticmethod
    def forward(ctx, scores):  # binarization

        expert_pred = torch.argmax(scores, dim=1)  # [bs]
        expert_pred_one_hot = torch.zeros_like(scores).scatter_(1, expert_pred.unsqueeze(-1), 1)

        return expert_pred, expert_pred_one_hot

    @staticmethod
    def backward(ctx, g1, g2):
        return g2


def get_device(x):
    gpu_idx = x.get_device()
    return f"cuda:{gpu_idx}" if gpu_idx >= 0 else "cpu"


class MoEBase(nn.Module):
    def __init__(self):
        super(MoEBase, self).__init__()
        self.scores = None
        self.router = None

    def set_score(self, scores):
        self.scores = scores
        for module in self.modules():
            if hasattr(module, 'scores'):
                module.scores = self.scores


class MoEConv(nn.Conv2d, MoEBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 n_expert=8):
        super(MoEConv, self).__init__(in_channels, out_channels * n_expert, kernel_size, stride, padding, dilation,
            groups, bias, )
        self.in_channels = in_channels
        self.out_channels = out_channels * n_expert
        self.expert_width = out_channels

        self.n_expert = n_expert
        assert self.n_expert >= 1
        self.layer_selection = torch.zeros([n_expert, self.out_channels])
        for cluster_id in range(n_expert):
            start = cluster_id * self.expert_width
            end = (cluster_id + 1) * self.expert_width
            idx = torch.arange(start, end)
            self.layer_selection[cluster_id][idx] = 1
        self.scores = None

    def forward(self, x):
        if self.n_expert > 1:
            if self.scores is None:
                self.scores = self.router(x)
            expert_selection, expert_selection_one_hot = GetMask.apply(self.scores)
            mask = torch.matmul(expert_selection_one_hot, self.layer_selection.to(x))  # [bs, self.out_channels]
            out = super(MoEConv, self).forward(x)
            out = out * mask.unsqueeze(-1).unsqueeze(-1)
            index = torch.where(mask.view(-1) > 0)[0]
            shape = out.shape
            out_selected = out.view(shape[0] * shape[1], shape[2], shape[3])[index].view(shape[0], -1, shape[2],
                                                                                         shape[3])
        else:
            out_selected = super(MoEConv, self).forward(x)
        self.scores = None
        return out_selected