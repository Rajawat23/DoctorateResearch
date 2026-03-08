import torch
from torch import nn
import torch.nn.functional as F

from . import common


class Masksembles2D(nn.Module):
    """
    :class:`Masksembles2D` is high-level class that implements Masksembles approach
    for 2-dimensional inputs (similar to :class:`torch.nn.Dropout2d`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C, H, W)
        * Output: (N, C, H, W) (same shape as input)

    Examples:

    >>> m = Masksembles2D(16, 4, 2.0)
    >>> input = torch.ones([4, 16, 28, 28])
    >>> output = m(input)

    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):
        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale

        masks = common.generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).double()

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2, 3, 4])
        x = x * self.masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return x.squeeze(0).float()


class Masksembles1D(nn.Module):
    """
    :class:`Masksembles1D` is high-level class that implements Masksembles approach
    for 1-dimensional inputs (similar to :class:`torch.nn.Dropout`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C)
        * Output: (N, C) (same shape as input)

    Examples:

    >>> m = Masksembles1D(16, 4, 2.0)
    >>> input = torch.ones([4, 16])
    >>> output = m(input)


    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):

        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale

        masks = common.generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).double()

    def forward(self, inputs):
        batch = inputs.shape[0]
        """
        Proposed changes
        ----------------
        * Prediction Case (batch == 1):

            The single sample is expanded to simulate a batch 
            with all masks applied. Output shape will be [n, channels],
            where n is the number of masks, allowing you to get predictions from all masks.

        * Training Case (batch > 1):
            The original behaviour of authors where batch is split into 
            self.n chunks, with each chunk processed by a mask.
            Original splitting, masking, and recombination behavior is preserved.
        """
        if self.training:
            if batch == 1:
                if inputs.dim() == 2:
                    x = inputs.unsqueeze(0)  # [1, 1, C]
                    x = x * self.masks.unsqueeze(1).float()  # [n, 1, C]
                    x = x.permute(1, 0, 2)  # [1, n, C]
                elif inputs.dim() == 3 and inputs.shape[1] == self.n:
                    x = inputs  # Already masked: [1, n, C]
                    x = x * self.masks.unsqueeze(0).float()  # [1, n, C]
                else:
                    raise ValueError(f"Unexpected input shape in batch==1: {inputs.shape}")
                return x.to(dtype=torch.float32)
                # # Prediction case: No splitting, apply all masks to single sample
                # x = inputs.unsqueeze(0) # Add mask dimension, shape: [1, batch, channels]
                # x = x * self.masks.unsqueeze(1) # Apply all masks, shape: [n, 1, channels]
                # # x = x.permute(1,0,2).reshape(batch * self.n,-1) # Reshape: [n, channels]
                # x = torch.cat(torch.split(x, 1, dim=0), dim=1)
                # return x.squeeze(0).to(dtype=torch.float32)
                # return x.to(dtype=torch.float32)
            else:
                original_batch_size = batch
                chunks = torch.chunk(inputs.unsqueeze(1), self.n, dim=0)
                max_size = max(chunk.size(0) for chunk in chunks)
                padded_chunks = [
                    F.pad(chunk, (0, 0, 0, 0, 0, max_size - chunk.size(0)))
                    for chunk in chunks
                ]
                x = torch.cat(padded_chunks, dim=1).permute([1, 0, 2])  # [n, max, C]
                x = x * self.masks.unsqueeze(1)  # [n, max, C]
                x = torch.cat(torch.split(x, 1, dim=0), dim=1).squeeze(0)  # [max * n, C]
                x = x[:original_batch_size]  # Trim any padded elements
                return x.to(dtype=torch.float32)
        else:
            if inputs.dim() == 2:
                # inputs: [B, C]
                x = inputs.unsqueeze(1)  # → [B, 1, C]
            elif inputs.dim() == 3 and inputs.shape[1] == self.n:
                # inputs: [B, n, C], leave as-is
                x = inputs
            else:
                raise ValueError(f"Unexpected input shape in eval: {inputs.shape}")

            x = x * self.masks.unsqueeze(0).float()  # [B, n, C]
            # print("Masksembles1D inference output:", x.shape)
            return x
            # do here make original shape
            # then return
