"""Pipeline class."""

import asyncio
import os

import torch
import torch.distributed as dist
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.router import Router
from infscale.execution.stage import Stage
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.modelir import ModelIR

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

logger = get_logger()


class Pipeline:
    """Pipeline class."""

    def __init__(
        self,
        spec: ServeConfig,
        modelir: ModelIR,
        dataset: HuggingFaceDataset,
    ):
        """Initialize pipeline instance."""
        self.stage: Stage = None
        self.spec = spec

        self._initialize_distributed()

        if "s" in spec.stage.id:  # it's server
            self.dataset = dataset
            self._run = self._run_server
            logger.info("I am server and master")
        else:
            logger.info("I am a worker")
            self._run = self._run_worker
            self._initialize_worker(spec, modelir)

    def _initialize_distributed(self, backend: str = "gloo"):
        # TODO: revise the hard-coded values and configure it dynamically
        os.environ["MASTER_ADDR"] = MASTER_ADDR
        os.environ["MASTER_PORT"] = MASTER_PORT

        # TODO: using torch.distributed is not fault-tolerant;
        #       replace it with elatic horovod's functionality
        rank = self.spec.rank_map[self.spec.stage.id]
        size = len(self.spec.rank_map)
        dist.init_process_group(backend, rank=rank, world_size=size)
        logger.info("initializing distributed: done")

    def _initialize_worker(self, spec: ServeConfig, modelir: ModelIR):
        (start, end, my_id) = (
            spec.stage.start,
            spec.stage.end,
            spec.stage.id,
        )

        output_parser = modelir.output_parser if spec.stage.is_last else None
        layers = modelir.layers[start : end + 1]

        self.stage = Stage(my_id, layers, output_parser=output_parser)

    async def _server_send(self, router: Router):
        logger.debug("start to send requests")
        seqno = 0
        while True:
            batch = self.dataset.next_batch()
            if batch is None:
                break

            logger.debug(f"sending batch {seqno}")
            # send batch to the first stage
            await router.tx_q.put((batch, seqno))
            seqno += 1

        logger.debug("_server_send task done")

    async def _server_recv(self, router: Router, max_seqno: int = -1):
        """
        Receive inference results from the last stage.

        max_seqno: if it's -1, run forever;
                   come out of loop if seqno is max_seqno
        """
        logger.debug("start to receive responses")
        seqno = -1
        while max_seqno == -1 or max_seqno != seqno:
            logger.debug("waiting for response")
            outputs, seqno = await router.rx_q.get()
            logger.debug(f"received response for {seqno}: {outputs}")

        logger.debug("_server_recv task done")

    async def _run_server(self):
        # create router
        router = Router(self.spec)
        router.prepare()

        # TODO: we read data directly from a dataset right now.
        #       in the future, we need to take dataset from stream as well.
        self.dataset.set_micro_batch_size(self.spec.micro_batch_size)
        max_seqno = self.dataset.num_of_batches() - 1

        # send and recv asynchronously
        send_task = asyncio.create_task(self._server_send(router))
        recv_task = asyncio.create_task(self._server_recv(router, max_seqno))
        logger.debug("created _send_request and _recv_resp tasks")

        await asyncio.gather(*[send_task, recv_task])
        logger.debug("inference serving is done")

    async def _run_worker(self):
        logger.debug("start to run worker")
        router = Router(self.spec)
        router.prepare()
        while True:
            inputs, seqno = await router.rx_q.get()
            logger.debug(f"received input {seqno} from rx_q")

            with torch.inference_mode():
                outputs = self.stage(inputs)
                # if isinstance(inputs, tuple):
                #     logger.debug(f"len(inputs) = {len(inputs)}")
                #     outputs = self.stage(*inputs)
                # else:
                #     outputs = self.stage(inputs)
            logger.debug("got output from stage and put output into tx_q")
            await router.tx_q.put((outputs, seqno))
            logger.debug("put output into tx_q")

    async def run(self):
        """Run pipeline."""
        await self._run()
