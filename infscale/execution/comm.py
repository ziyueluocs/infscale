"""Communication module."""
import asyncio
from asyncio import Queue as AsyncQ
from queue import Queue as SyncQ
from threading import Thread

import torch
import torch.distributed as dist
from infscale import get_logger
from infscale.utils import run_async

# from https://github.com/SymbioticLab/Oobleck/blob/develop/oobleck/execution/utils.py#L4-L18
ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

KEY_LOOP = "asyncio_event_loop"
KEY_THD_RX_Q = "thd_rx_q"
KEY_THD_TX_Q = "thd_tx_q"

logger = get_logger()


class TensorSender:
    """Tensor sender class.

    This class maintains state related to sending tensors and sends tensors.
    """

    def __init__(self, rank: int, device: torch.device):
        """Initialize tensor sender instance."""
        self.rank = rank  # destination's rank
        self.device = device

        self.sent_tensor_meta = False

    def send(self, tensors: tuple[torch.Tensor], seqno: int) -> None:
        """Send tensors to destination rank.

        seqno: the seqno of a tensor; will be used to keep track of tensors
        traversing a pipeline.
        """

        def _send_tensor_meta(tensors: tuple[torch.Tensor]) -> None:
            """
            Send meta data for tensor.

            sending order of the meta data:
            t_dim -> t_dtype -> t_shape
            """
            count = torch.LongTensor(data=[len(tensors)]).to(self.device)
            dist.send(count, self.rank)

            for tensor in tensors:
                dim = len(tensor.size())
                t_dim = torch.LongTensor(data=[dim]).to(self.device)

                dtype = DTYPE_TO_ID[tensor.dtype]
                t_dtype = torch.LongTensor(data=[dtype]).to(self.device)

                shape = tensor.size()
                t_shape = torch.LongTensor(data=shape).to(self.device)

                # TODO: Make send asynchronous
                dist.send(t_dim, self.rank)
                dist.send(t_dtype, self.rank)
                dist.send(t_shape, self.rank)

        logger.debug("calling send")
        if not self.sent_tensor_meta:
            logger.debug("sending tensor meta data")
            # we only send meta data once
            _send_tensor_meta(tensors)
            self.sent_tensor_meta = True
            logger.debug("done tensor meta data tx")

        logger.debug("sending tensors")
        for tensor in tensors:
            dist.send(tensor, self.rank)
        logger.debug("sent tensors")

        seqno = torch.tensor([seqno], dtype=torch.int).to(self.device)
        dist.send(seqno, self.rank)
        logger.debug(f"sent seqno {seqno}")


class FutureThread(Thread):
    """A dedicated thread for waiting for future from irecv call.

    We implement this dedicated thread instead of using run_in_executor
    in asyncio aling with concurrent.futures.ThreadPoolExecutor().
    The rationale to this decision is that ThreadPoolExecutor causes
    the creatation of a thread pool every time it's called.
    Thus, it can be expensive. Since we only use this FutureThread for I/O
    bounded operations, a dedicated single thread may be sufficient.
    TODO: need to test the above assumption.
    """

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs=None,
        *,
        daemon=None,
    ):
        """Initialize an instance."""
        # call parent constructure method
        super().__init__(group, target, name, daemon=daemon)

        self._loop = kwargs[KEY_LOOP]
        self._rx_q: SyncQ = kwargs[KEY_THD_RX_Q]
        self._tx_q: AsyncQ = kwargs[KEY_THD_TX_Q]

        self._done = False

    def stop(self):
        """Set a flag to stop the thread."""
        self._done = True

    def run(self):
        """Override run function of Thread.

        The function calls wait() method of the Work object and
        once the message arrives, it returns the message back to
        the main thread via asyncio's queue.

        The main purpose of doing this is to allow the main thread
        to get scheduled via asyncio's loop.
        """
        while not self._done:
            work = self._rx_q.get()
            # blocked until future becomes available
            res = work.wait()
            self._rx_q.task_done()

            # res is the rank of src
            _, _ = run_async(self._tx_q.put(res), self._loop)


class TensorReceiver:
    """TensorReceiver class."""

    def __init__(self, rank: int, device: torch.device):
        """Initialize communication instance."""
        self.rank = rank  # source's rank
        self.device = device

        self.buffer: torch.Tensor = None

        self._thd_rx_q = SyncQ()  # regular synchronized queue
        self._thd_tx_q = AsyncQ()  # asyncio queue

        kwargs = {
            KEY_LOOP: asyncio.get_running_loop(),
            KEY_THD_RX_Q: self._thd_rx_q,
            KEY_THD_TX_Q: self._thd_tx_q,
        }

        future_thd = FutureThread(kwargs=kwargs, daemon=True)
        future_thd.start()

    async def _recv(self, tensor: torch.LongTensor):
        work = dist.irecv(tensor, self.rank)
        self._thd_rx_q.put(work)
        _ = await self._thd_tx_q.get()

    async def recv(self) -> tuple[tuple[torch.Tensor], int]:
        """Receive tensors from source rank.

        seqno: the seqno of a tensor; will be used to keep track of tensors
        traversing a pipeline
        """

        async def _create_receive_buffer() -> tuple[torch.Tensor]:
            """Receive menta data for tensor and return allocated buffer.

            receiving order of the meta data:
            t_dim -> t_dtype -> t_shape
            """
            count = torch.LongTensor(data=[0]).to(self.device)
            await self._recv(count)
            num_tensors = count.item()
            tensors: list[torch.Tensor] = []

            for _ in range(num_tensors):
                t_dim = torch.LongTensor(data=[0]).to(self.device)
                await self._recv(t_dim)
                t_dim = t_dim.item()

                t_dtype = torch.LongTensor(data=[0]).to(self.device)
                await self._recv(t_dtype)
                t_dtype = ID_TO_DTYPE[t_dtype.item()]

                t_shape = torch.LongTensor([1] * t_dim).to(self.device)
                await self._recv(t_shape)
                t_shape = t_shape.tolist()

                tensor = torch.zeros(
                    t_shape,
                    device=self.device,
                    dtype=t_dtype,
                    requires_grad=False,
                )
                tensors.append(tensor)

            return tuple(tensors)

        logger.debug("calling recv")
        if self.buffer is None:
            logger.debug("creating a recv buffer")
            # allocate buffer once and reuse it
            self.buffer = await _create_receive_buffer()
            logger.debug("done recv buffer creation")

        recvd: list[torch.Tensor | None] = [None] * len(self.buffer)
        for idx, tensor in enumerate(self.buffer):
            logger.debug(f"receiving tensor {idx}")
            assert torch.is_tensor(tensor)
            await self._recv(tensor)
            recvd[idx] = tensor.clone().detach()

        seqno = torch.LongTensor(data=[0]).to(self.device)
        await self._recv(seqno)
        seqno = seqno.item()
        logger.debug(f"received tensors of seqno {seqno}")

        return tuple(recvd), seqno
