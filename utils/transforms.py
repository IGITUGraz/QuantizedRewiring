import torch
from torch import nn
from torch.nn import functional as F


class ThresholdEncode(nn.Module):

    def __init__(self, low, high, size):
        super(ThresholdEncode, self).__init__()
        self.low = low
        self.high = high
        self.size = size
        self.th = torch.linspace(low, high, size + 2)[1:-1]

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x)
        th = torch.linspace(self.low, self.high, self.size + 2)[1:-1]
        spike_times = torch.zeros((x.shape[0], 2 * self.size))
        x_prev = x[:-1]
        x_next = x[1:]
        for j in range(self.size):
            t = th[j]
            idx_up = torch.nonzero((x_prev <= t) & (x_next > t)).squeeze()
            idx_down = torch.nonzero((x_prev > t) & (x_next <= t)).squeeze()
            spike_times[idx_up, 2 * j] = 1.
            spike_times[idx_down, 2 * j + 1] = 1.
        return spike_times


class AddPrompt(object):

    def __init__(self, duration: int):
        super().__init__()
        self.duration = duration

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        sample = F.pad(sample, (0, 1), 'constant', 0.)
        sample[-self.duration:, -1] = 1
        return sample
