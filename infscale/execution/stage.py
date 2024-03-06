"""Stage class."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Union

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device
from infscale import get_logger

if TYPE_CHECKING:
    import torch.fx as fx
    from torch import Tensor


logger = get_logger()


class Stage(nn.Module):
    """Stage class."""

    def __init__(
        self,
        stage_id: str,
        layers: list[fx.GraphModule],
        device: torch.device = torch.device("cpu"),
        output_parser: Callable = None,
    ):
        """Initialize stage class instance."""
        super().__init__()

        self.id = stage_id
        self.layers = deepcopy(layers)
        self.device = device

        self._init_layers()

        # An output parser is only useful for the last stage.
        # The outputs from the last stage need to be sent back to the inference
        # server. Therefore they need to be sent back as a list of tensors.
        # But if the output is a dictionary of tensors. This leads to comm
        # error. Also, in the inference, other values such as loss may not be
        # important. So, a way to manipulate the outputs is provided.
        self._output_parser: Union[Callable, None] = output_parser

    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor]:
        """Run layers in the stage."""
        logger.debug(f"calling forward with inputs of type {type(inputs)}")
        for layer in self.layers:
            inputs = layer(*inputs)

        outputs = self._output_parser(inputs) if self._output_parser else inputs

        return outputs

    def _init_layers(self):
        """Initialize meta layers and move them to a device."""
        for layer in self.layers:
            self._init_tensors(layer)

    def _init_tensors(self, layer: torch.fx.GraphModule):
        """Initialize meta tensors and move them to a device."""
        # FIXME: need to update values from pretrained model
        #        currently random initialization is applied
        for param_name, param in layer.named_parameters():
            set_module_tensor_to_device(
                layer, param_name, self.device, torch.rand(param.shape)
            )

        for buffer_name, buffer in layer.named_buffers():
            set_module_tensor_to_device(
                layer, buffer_name, self.device, torch.rand(buffer.shape)
            )
