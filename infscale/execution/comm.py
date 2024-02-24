"""Communication module."""
import torch
import torch.distributed as dist

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


class TensorSender:
    """Tensor sender class.

    This class maintains state related to sending tensors and sends tensors.
    """

    def __init__(self, rank: int, device: torch.device):
        """Initialize tensor sender instance."""
        self.rank = rank  # destination's rank
        self.device = device

        self.sent_tensor_meta = False

    def send(self, tensor: torch.Tensor, index: int) -> None:
        """Send tensors to destination rank."""

        def _send_tensor_meta(tensor: torch.Tensor, index: int) -> None:
            """
            Send menta data for tensor.

            index: the index of a tensor; will be used to keep track of tensors
                   traversing a pipeline

            sending order of the meta data:
            index -> t_dim -> t_dtype -> t_shape
            """
            index = torch.tensor([index], dtype=torch.int).to(self.device)
            t_dim = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
            t_dtype = torch.LongTensor(data=[DTYPE_TO_ID[tensor.dtype]]).to(self.device)
            t_shape = torch.LongTensor(data=tensor.size()).to(self.device)

            # TODO: Make send asynchronous
            dist.send(index, self.rank)
            dist.send(t_dim, self.rank)
            dist.send(t_dtype, self.rank)
            dist.send(t_shape, self.rank)

        if not self.sent_tensor_meta:
            # we only send meta data once
            _send_tensor_meta(tensor, index)
            self.sent_tensor_meta = True

        dist.send(tensor, self.rank)


class TensorReceiver:
    """TensorReceiver class."""

    def __init__(self, rank: int, device: torch.device):
        """Initialize communication instance."""
        self.rank = rank  # source's rank
        self.device = device

        self.buffer: torch.Tensor = None

    def recv(self) -> tuple[torch.Tensor, int]:
        """Receive tensors from source rank."""

        def _create_receive_buffer() -> tuple[torch.Tensor, int]:
            """Receive menta data for tensor and return allocated buffer.

            index: the index of a tensor; will be used to keep track of tensors
                   traversing a pipeline

            receiving order of the meta data:
            index -> t_dim -> t_dtype -> t_shape
            """
            index = torch.LongTensor(data=[0]).to(self.device)
            dist.recv(index, self.rank)
            index = index.item()

            t_dim = torch.LongTensor(data=[0]).to(self.device)
            dist.recv(t_dim, self.rank)
            t_dim = t_dim.item()

            t_dtype = torch.LongTensor(data=[0]).to(self.device)
            dist.recv(t_dtype, self.rank)
            t_dtype = ID_TO_DTYPE[t_dtype.item()]

            t_shape = torch.LongTensor([1] * t_dim).to(self.device)
            dist.recv(t_shape, self.rank)
            t_shape = t_shape.tolist()

            buffer = torch.zeros(t_shape, device=self.device, dtype=t_dtype)

            return buffer, index

        if not self.buffer:
            # allocate buffer once and reuse it
            self.buffer, index = _create_receive_buffer()

        dist.recv(self.buffer, self.rank)

        # copy buffer to a new detached tensor
        tensor = self.buffer.clone().detach()

        return tensor, index
