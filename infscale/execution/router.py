"""Router class."""
import asyncio
import random

import torch
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.comm import TensorReceiver, TensorSender

DEFAULT_QUEUE_SIZE = 3

logger = get_logger()


class Router:
    """Router class."""

    def __init__(self, spec: ServeConfig):
        """Initialize Router instance."""
        self._rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline
        self._tx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline

        # a collection of ranks that receive data from me
        self.receiver_ranks: dict[int] = []
        self.__tx_qs: dict[int, asyncio.Queue] = {}

        # a collection of ranks that send data to me
        self.sender_ranks: list[int] = []
        self.__rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)

        my_id = spec.stage.id
        self.rank = spec.rank_map[my_id]
        for k, v in spec.flow_graph.items():
            if my_id == k:  # I am a sender to v
                for other_id in v:
                    rank = spec.rank_map[other_id]
                    self.receiver_ranks.append(rank)
                    self.__tx_qs[rank] = asyncio.Queue(DEFAULT_QUEUE_SIZE)
            elif my_id in v:  # I am a receiver from k
                self.sender_ranks.append(spec.rank_map[k])

    @property
    def rx_q(self):
        """Return receiver queue."""
        return self._rx_q

    @property
    def tx_q(self):
        """Return transmit queue."""
        return self._tx_q

    def prepare(self):
        """Create asyncio tasks for sending and receiving."""
        _ = asyncio.create_task(self._send_arbiter())
        _ = asyncio.create_task(self._recv_arbiter())

        for rank in self.receiver_ranks:
            # TODO: revise hard-coded device
            _ = asyncio.create_task(self._send(rank, torch.device("cpu")))

        for rank in self.sender_ranks:
            # TODO: revise hard-coded device
            _ = asyncio.create_task(self._recv(rank, torch.device("cpu")))

    async def _recv(self, rank: int, device: torch.device):
        logger.debug(f"start to receive tensors from {rank}")
        receiver = TensorReceiver(rank, device)
        logger.debug("created tensor receiver")

        while True:
            logger.debug("calling receiver.recv")
            tensors, index = await receiver.recv()
            logger.debug(f"received tensor {index}")
            await self.__rx_q.put((tensors, index))
            logger.debug(f"put tensors {index} into __rx_q")

    async def _send(self, rank: int, device: torch.device):
        logger.debug(f"start to send tensors to {rank}")
        sender = TensorSender(rank, device)
        logger.debug("created tensor sender")
        tx_q = self.__tx_qs[rank]
        logger.debug("acquired tx q")

        while True:
            tensor, seqno = await tx_q.get()
            logger.debug(f"got tensor {seqno} from __tx_q")
            sender.send(tensor, seqno)
            logger.debug(f"sent tensor {seqno}")

    async def _recv_arbiter(self):
        logger.debug("start recv_arbiter")
        while True:
            tensor, seqno = await self.__rx_q.get()
            logger.debug(f"fetched tensor {seqno} from __rx_q")
            # TODO: introduce a prioritization policy
            await self._rx_q.put((tensor, seqno))
            logger.debug("put tensor to _rx_q")

    async def _send_arbiter(self):
        logger.debug("start send_arbiter")
        while True:
            tensor, seqno = await self._tx_q.get()
            logger.debug(f"fetched tensor {seqno} from _tx_q")
            # TODO: introduce a prioritization policy
            #       current default policy is to choose receiving rank randomly

            # TODO: choosing a rank randomly by converting dictionary keys into
            #       a list can be a performance bottleneck; look into it later.
            rank = random.choice(list(self.__tx_qs.keys()))
            logger.debug(f"selected rank {rank} as a reciever")

            await self.__tx_qs[rank].put((tensor, seqno))
            logger.debug(f"put tensor {seqno} to __tx_q for {rank}")
