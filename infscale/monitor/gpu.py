"""GPU monitoring class."""
import asyncio
from dataclasses import dataclass
from enum import Enum

from infscale import get_logger
from pynvml import (nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetCount,
                    nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlDeviceGetName, nvmlDeviceGetUtilizationRates, nvmlInit)

DEFAULT_INTERVAL = 10  # 10 seconds

logger = get_logger()


class GpuType(str, Enum):
    """Gpu Type enum class."""

    def __str__(self):
        """__str__ dunder method."""
        return str(self.value)

    UNKNOWN = "unknown"
    T4 = "T4"
    V100 = "V100"
    A100 = "A100"


@dataclass
class GpuStat:
    """GPU statistics."""

    id: int
    type: GpuType
    used: bool
    utilization: int


@dataclass
class VramStat:
    """VRAM statistics."""

    id: int
    total: int
    used: int


class GpuMonitor:
    """GpuMonitor class."""

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        """Initialize GpuMonitor instance."""
        self.interval = interval

        self.util_ewma = 0
        self.mem_ewma = 0

    async def start(self):
        """Start to monitor GPU statistics - utilization and vram usage.

        utilization reports device's utilization.
        it's not a per-application metric.
        vram usage is also an aggregated metric, not a per-application metric.
        """
        try:
            nvmlInit()
        except Exception as e:
            logger.debug("failed to initialize gpustat.nvml.pynvml")
            logger.debug(f"exception: {e}")
            return

        while True:
            count = nvmlDeviceGetCount()
            mems = [None] * count
            computes = [None] * count
            for i in range(count):
                try:
                    handle = nvmlDeviceGetHandleByIndex(i)

                    # gpu memory information
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    # device name
                    name = nvmlDeviceGetName(handle)
                    # GPU utilization
                    util = nvmlDeviceGetUtilizationRates(handle).gpu
                    # processes running on gpu
                    processes = nvmlDeviceGetComputeRunningProcesses(handle)
                except Exception as e:
                    logger.debug(f"failed to retrieve info for GPU {i}")
                    logger.debug(f"exception: {e}")
                    continue

                mems[i] = VramStat(i, mem_info.total, mem_info.used)

                dev_type = GpuType.UNKNOWN
                for t in GpuType:
                    if str(t) in name:
                        dev_type = t
                        break

                used = len(processes) > 0
                computes[i] = GpuStat(i, dev_type, used, util)

            # TODO: need to work on reporting the stats to controller
            logger.debug(f"mems: {mems}")
            logger.debug(f"computes: {computes}")

            await asyncio.sleep(self.interval)
