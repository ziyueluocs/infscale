"""Pipeline class."""

import asyncio
import os

import torch
import torch.distributed as dist
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.router import Router
from infscale.execution.stage import Stage
from infscale.execution.world import WorldInfo
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.modelir import ModelIR

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
        self.world_manager = dist.WorldManager()

        # TODO: need to revisit this to set a specific cuda deivce
        #       e.g., cuda:0, cuda:1, etc (based on local rank)
        #       we may need to set device id dynamically based on
        #       availability of gpu resources
        device_type = "cuda" if self.spec.backend == "nccl" else "cpu"
        self.device = torch.device(device_type)

        self.world_info_list: list[WorldInfo] = list()

        self._initialize_multiworld()

        if "s" in spec.stage.id:  # it's server
            self.dataset = dataset
            self._run = self._run_server
            logger.info("I am server and master")
        else:
            logger.info("I am a worker")
            self._run = self._run_worker
            self._initialize_worker(spec, modelir)

    def _initialize_multiworld(self):
        my_id = self.spec.stage.id

        world_idx = 0
        for k, v in self.spec.flow_graph.items():
            for wrk_info in v:
                if my_id == k:
                    my_rank = 0
                    other_rank = 1
                elif my_id == wrk_info.peer:
                    my_rank = 1
                    other_rank = 0
                else:
                    continue

                world_name = f"w{world_idx}"
                logger.info(f"initializing world {world_name} with my rank {my_rank}")
                logger.info(f"master addr={wrk_info.addr}, port={wrk_info.port}")
                self.world_manager.initialize_world(
                    world_name,
                    my_rank,
                    2,
                    backend=self.spec.backend,
                    addr=wrk_info.addr,
                    port=wrk_info.port,
                )
                data = {"name": world_name, "me": my_rank, "other": other_rank}
                world_info = WorldInfo(**data)
                self.world_info_list.append(world_info)
            world_idx += 1

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
        logger.info("start to send requests")
        seqno = 0
        while True:
            batch = self.dataset.next_batch()
            if batch is None:
                break

            logger.info(f"sending batch {seqno}")
            # send batch to the first stage
            await router.tx_q.put((batch, seqno))
            seqno += 1

        logger.info("_server_send task done")

    async def _server_recv(self, router: Router, max_seqno: int = -1):
        """
        Receive inference results from the last stage.

        max_seqno: if it's -1, run forever;
                   come out of loop if seqno is max_seqno
        """
        logger.info("start to receive responses")
        seqno = -1
        while max_seqno == -1 or max_seqno != seqno:
            logger.info("waiting for response")
            outputs, seqno = await router.rx_q.get()
            logger.info(f"received response for {seqno}: {type(outputs)}")

        logger.info("_server_recv task done")

    async def _run_server(self):
        # create router
        router = Router(
            self.world_manager, self.world_info_list, self.spec, self.device
        )
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
        router = Router(
            self.world_manager, self.world_info_list, self.spec, self.device
        )
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
