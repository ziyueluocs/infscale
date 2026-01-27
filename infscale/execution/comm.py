# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Communication module."""

import os

import torch
from multiworld.communicator import WorldCommunicator

from infscale.execution.control import MSG_MODE_ACK, Channel, ControlMessage


class TensorSender:
    """Tensor sender class."""

    def __init__(
        self,
        communicator: WorldCommunicator,
        channel: Channel,
        world_name: str,
        rank: int,
        device: torch.device,
    ):
        """Initialize tensor sender instance."""
        self.communicator = communicator
        self.channel = channel
        self.world_name = world_name
        self.rank = rank  # destination's rank
        self.device = device

    async def send(self, tensors: dict[str, torch.Tensor], seqno: int) -> None:
        """Send tensors to destination rank.

        tensors: represented as dictionary where key is string and value is tensor
        seqno: the seqno of a tensor; will be used to keep track of tensors
        traversing a pipeline.
        """
        # to minimize the overhead of busy-waiting by communicator's operations
        # we coordinate send/recv via control channel
        _ = await self.channel.sync(self.rank, seqno=seqno, tensors=tensors)

        for _, tensor in tensors.items():
            await self.communicator.send(tensor, self.rank, self.world_name)

    def is_broken(self) -> bool:
        """Check if world is broken or not."""
        return self.communicator.is_broken(self.world_name)


class TensorReceiver:
    """TensorReceiver class."""

    def __init__(
        self,
        communicator: WorldCommunicator,
        channel: Channel,
        world_name: str,
        rank: int,
        device: torch.device,
    ):
        """Initialize communication instance."""
        self.communicator = communicator
        self.channel = channel
        self.world_name = world_name
        self.rank = rank  # source's rank
        self.device = device

        self.buffer: dict[str, torch.Tensor] = None
        self.metas: dict = None  # Track message metadata

    def _allocate_buffer(self, ctrl_msg: ControlMessage) -> None:
        """Allocate or reallocate buffer based on incoming message metadata."""
        if ctrl_msg.ditto:
            return
        
        # Save metadata for future messages
        self.metas = ctrl_msg.metas

        #max_mem_size = torch.cuda.max_memory_allocated(device=None) / 1024**2
        #if max_mem_size > 0:
        #    for k, v in ctrl_msg.metas.items():
        #        print(f"{os.getpid()} k: {k}, v: {v}")

        #    print(
        #        f"{os.getpid()} -before recv memalloc: gpu used max: {max_mem_size}"
        #    )

        # since there is change in tensor shape, reallocate buffer
        if self.buffer is None:
            self.buffer = {}

        for k, v in ctrl_msg.metas.items():
            msg_shape = v.shape
            
            # Check if buffer exists for this key and compare shapes
            if k in self.buffer:
                buffer_shape = self.buffer[k].shape
                
                # Check dimensions compatibility
                if len(msg_shape) != len(buffer_shape):
                    # Different number of dimensions, reallocate
                    tensor = torch.zeros(
                        msg_shape,
                        device=self.device,
                        dtype=v.dtype,
                        requires_grad=False,
                    )
                    self.buffer[k] = tensor
                elif any(msg_shape[i] > buffer_shape[i] for i in range(len(msg_shape))):
                    # Message shape is larger in at least one dimension
                    # For each dimension, use max(buffer_shape[i] * 2, msg_shape[i])
                    # when buffer is smaller, otherwise keep buffer size
                    new_shape = []
                    for i in range(len(buffer_shape)):
                        if buffer_shape[i] < msg_shape[i]:
                            new_shape.append(max(buffer_shape[i] * 2, msg_shape[i]))
                        else:
                            new_shape.append(buffer_shape[i])
                    
                    # Reallocate buffer with new shape
                    tensor = torch.zeros(
                        tuple(new_shape),
                        device=self.device,
                        dtype=v.dtype,
                        requires_grad=False,
                    )
                    self.buffer[k] = tensor
                # else: buffer is already large enough, no reallocation needed
            else:
                # New key, create buffer
                tensor = torch.zeros(
                    msg_shape,
                    device=self.device,
                    dtype=v.dtype,
                    requires_grad=False,
                )
                self.buffer[k] = tensor
        
        #max_mem_size = torch.cuda.max_memory_allocated(device=None) / 1024**2
        #if max_mem_size > 0:
        #    print(
        #        f"{os.getpid()} +after recv memalloc: gpu used max: {max_mem_size}"
        #    )

    async def recv(self) -> tuple[dict[str, torch.Tensor], int]:
        """Receive tensors from source rank.

        seqno: the seqno of a tensor; will be used to keep track of tensors
        traversing a pipeline
        """
        # to minimize the overhead of busy-waiting by communicator's operations
        # we coordinate send/recv via control channel
        ctrl_msg: ControlMessage = await self.channel.sync(self.rank, mode=MSG_MODE_ACK)
        self._allocate_buffer(ctrl_msg)

        # Slice buffers to match actual message shape and receive into sliced views
        recv_views = {}
        for k, v in self.metas.items():
            msg_shape = v.shape
            slices = tuple(slice(0, size) for size in msg_shape)
            recv_views[k] = self.buffer[k][slices]

        for _, tensor in recv_views.items():
            await self.communicator.recv(tensor, self.rank, self.world_name)

        seqno = ctrl_msg.seqno

        # Clone tensors to avoid corruption when buffer is reused
        recvd = {}
        for k, v in recv_views.items():
            recvd[k] = v.clone()

        return recvd, seqno

    def is_broken(self) -> bool:
        """Check if world is broken or not."""
        return self.communicator.is_broken(self.world_name)
