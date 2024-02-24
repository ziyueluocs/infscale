"""Stage class."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    import torch.fx as fx
    from torch import Tensor


class Stage(nn.Module):
    """Stage class."""

    def __init__(self, stage_id: str, layers: list[fx.GraphModule]):
        """Initialize stage class instance."""
        super().__init__()

        self.id = stage_id
        self.layers = deepcopy(layers)

    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor]:
        """Run layers in the stage."""
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs
