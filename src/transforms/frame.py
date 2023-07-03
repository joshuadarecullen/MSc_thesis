import torch
from torch import Tensor
from torch.nn import Module

"""
convert tensor into sequence of successive possibly overlapping frames
given an nD tensor of shape (num_samples, ...), we construct a (n+1)D
tensor of shape (num_frames, window_length, ...), where each frame starts
hop_length points after the preceeding one. we do this by striding across
data in memory, duplicates are not made. we do not apply any zero padding
so any incomplete frames at the end of the sequence are discarded. doing
a backward pass attempts to unframe the data. it cannot know the original
number of samples in this scope, but we account for the change in size
according to the difference between the window length and the hop length

implementation notes:
- 11.2022 - fixed to work for nD data, rather than just 2D data
          - 'sample_rate' varies depending on the input, e.g.
            spectrogram sample rate is determined by the hop size
            i.e. the number of hops in the spectrogram per second
            whereas for a time series, the sample rate is just the
            number of samples per second
"""

class Frame(Module):
    def __init__(
        self,
        window_length: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length

    def forward(self, data: Tensor) -> Tensor:
        # extract dimension to frame and all remaining
        num_channels, num_samples, *other_dims = data.shape
        # calculate the number of frames according to the window,
        # ensure we leave off any at the end when the size of the frames doesn't match
        # TODO: add an option to include incomplete frames by pulling out from columns from prev frame
        num_frames = 1 + int(round((num_samples - self.window_length) // self.hop_length))
        # the size switches the old dimension, num_samples, to two new dimensions,
        # num_frames and window_length, making an nD tensor into an nD+1 tensor
        size = (1, num_frames, num_channels, self.window_length, *other_dims)
        # extract existing strides from old signal
        strides = data.stride()
        # to get to the next frame, we need to step by the existing stride times the hop length
        # we keep the remaining strides as these don't change
        stride = (1, strides[1] * self.hop_length, *strides)
        return data.as_strided(size=size, stride=stride)

    def backward(self, data: Tensor) -> Tensor:
        # extract dimension to squash down and all remaining
        num_frames, window_length, *other_dims = data.shape
        # we don't have the original number of samples, and we cannot know how many
        # were discarded (within this scope), but we can use what we have available
        # to remove the overlap, we calculate the difference between the window length
        # and the hop length times the number of frames - 1
        num_samples = (
            (num_frames * self.window_length) -
            int((num_frames - 1) * (self.window_length - self.hop_length))
        )
        size = (num_samples, *other_dims)
        strides = data.stride()
        stride = strides[1:]
        return data.as_strided(size=size, stride=stride)
