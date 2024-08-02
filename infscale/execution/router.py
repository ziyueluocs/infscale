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

"""Router class."""
import asyncio
import random
from collections import deque

import torch
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.comm import TensorReceiver, TensorSender
from infscale.execution.world import WorldInfo
from torch.distributed import WorldManager

DEFAULT_QUEUE_SIZE = 3
DEFAULT_SLEEP_TIME = 0.1  # 100ms
QUEUE_WAIT_PERIOD = 0.1  # 100ms


logger = get_logger()


class Router:
    """Router class."""

    def __init__(
        self,
        world_manager: WorldManager,
        world_info_list: list[WorldInfo],
        spec: ServeConfig,
        device=torch.device("cpu"),
    ):
        """Initialize Router instance."""
        self.world_manager = world_manager
        self.device = device

        self._rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline
        self._tx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline

        # a collection of receivers that receive data from me
        self.receivers: list[WorldInfo] = []
        self.__tx_qs: dict[WorldInfo, asyncio.Queue] = {}

        # a collection of senders that send data to me
        self.senders: list[WorldInfo] = []
        self.__rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)

        self.orphan_dq: deque = deque()
        for world_info in world_info_list:
            if world_info.me == 0:  # I am a receiver from other
                self.senders.append(world_info)
            else:  # I am a sender to other
                self.receivers.append(world_info)
                self.__tx_qs[world_info] = asyncio.Queue(DEFAULT_QUEUE_SIZE)

    @property
    def rx_q(self) -> asyncio.Queue:
        """Return receiver queue."""
        return self._rx_q

    @property
    def tx_q(self) -> asyncio.Queue:
        """Return transmit queue."""
        return self._tx_q

    def prepare(self) -> None:
        """Create asyncio tasks for sending and receiving."""
        _ = asyncio.create_task(self._send_arbiter())
        _ = asyncio.create_task(self._recv_arbiter())

        for world_info in self.receivers:
            _ = asyncio.create_task(self._send(world_info))

        for world_info in self.senders:
            _ = asyncio.create_task(self._recv(world_info))

    async def _recv(self, world_info: WorldInfo) -> None:
        logger.debug(
            f"start to receive tensors from {world_info.other} in world {world_info.name}"
        )
        receiver = TensorReceiver(
            self.world_manager.communicator,
            world_info.name,
            world_info.other,
            self.device,
        )
        logger.debug("created tensor receiver")

        while True:
            logger.debug("calling receiver.recv")
            try:
                tensors, index = await receiver.recv()
            except Exception as e:
                logger.warn(f"{world_info.name} error: {e}")
                break

            logger.debug(f"received tensor {index}")
            await self.__rx_q.put((tensors, index))
            logger.debug(f"put tensors {index} into __rx_q")

        logger.warn(f"done with recv task for {world_info.name}")

    async def _send(self, world_info: WorldInfo) -> None:
        logger.debug(
            f"start to send tensors to {world_info.other} in {world_info.name}"
        )
        sender = TensorSender(
            self.world_manager.communicator,
            world_info.name,
            world_info.other,
            self.device,
        )
        logger.debug("created tensor sender")
        tx_q = self.__tx_qs[world_info]
        logger.debug("acquired tx q")

        while True:
            try:
                tensor, seqno = await asyncio.wait_for(tx_q.get(), QUEUE_WAIT_PERIOD)
            except asyncio.TimeoutError:
                if not sender.is_broken():
                    continue
                logger.warn(f"{world_info.name} is broken")
                break

            logger.debug(f"got tensor {seqno} from __tx_q")
            try:
                await sender.send(tensor, seqno)
            except Exception as e:
                logger.warn(f"{world_info.name} error: {e}")
                break
            logger.debug(f"sent tensor {seqno}")

        # remove tx queue for the world
        del self.__tx_qs[world_info]

        await self._handle_orphan_data(tx_q)

        logger.warn(f"done with send task for {world_info.name}")

    async def _handle_orphan_data(self, queue: asyncio.Queue) -> None:
        async def _drain_and_put_inner():
            while not queue.empty():
                # drain queue
                data = await queue.get()
                # put data into orphan deque so that data can be resent
                self.orphan_dq.append(data)

        await _drain_and_put_inner()

        # we give some time for _send_arbiter to finish any blocked "put" call
        await asyncio.sleep(DEFAULT_SLEEP_TIME)

        # we call this one more time
        await _drain_and_put_inner()

    async def _recv_arbiter(self) -> None:
        logger.debug("start recv_arbiter")
        while True:
            tensor, seqno = await self.__rx_q.get()
            logger.debug(f"fetched tensor {seqno} from __rx_q")
            # TODO: introduce a prioritization policy
            await self._rx_q.put((tensor, seqno))
            logger.debug("put tensor to _rx_q")

    async def _send_arbiter(self) -> None:
        logger.debug("start send_arbiter")
        while True:
            try:
                tensor, seqno = self.orphan_dq.popleft()
                logger.debug(f"fetched tensor {seqno} from orhpan_dq")
            except IndexError:
                # In case of this error, we know that there is no orphan
                # data, so we can proceed to fetch data from _tx_q
                tensor, seqno = await self._tx_q.get()
                logger.debug(f"fetched tensor {seqno} from _tx_q")

            while len(self.__tx_qs) == 0:
                logger.warn("no worker to send data")
                await asyncio.sleep(DEFAULT_SLEEP_TIME)

            # TODO: introduce a prioritization policy
            #       current default policy is to choose receiving rank randomly

            # TODO: choosing a rank randomly by converting dictionary keys into
            #       a list can be a performance bottleneck; look into it later.
            world_info = random.choice(list(self.__tx_qs.keys()))
            logger.debug(f"world name: {world_info.name}")
            logger.debug(f"receiver rank: {world_info.other}")

            # FIXME: this creates a head-of-line blocking issue
            #        if the current queue is full. We need to fix it.
            await self.__tx_qs[world_info].put((tensor, seqno))
            logger.debug(f"put tensor {seqno} to __tx_q for {world_info.other}")
