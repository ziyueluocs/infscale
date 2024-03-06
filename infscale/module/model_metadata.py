"""ModelMetaData."""
import os
from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Union

from accelerate import disk_offload
from infscale import get_logger
from transformers import (AutoModelForCausalLM,
                          AutoModelForImageClassification,
                          AutoModelForPreTraining, PretrainedConfig,
                          PreTrainedModel)

AutoModelType = (
    AutoModelForPreTraining | AutoModelForCausalLM | AutoModelForImageClassification
)

OFFLOAD_FOLDER_PREFIX = "/tmp/infscale/offload/"

logger = get_logger()


class ModelGroup(str, Enum):
    """Model class enum class."""

    UNKNOWN = "unknown"
    IMAGE = "image"
    LANG = "language"


class BaseModelMetaData:
    """Base class for model meta data implementation."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        self.name: str = name
        self.model_group = ModelGroup.UNKNOWN
        self.config: PretrainedConfig = config

        self.model: AutoModelType = None
        self.split_points: List[str] = None

        self.pid = os.getpid()

    def _offload_model_to_disk(self, auto_model_type: AutoModelType) -> PreTrainedModel:
        model = auto_model_type.from_pretrained(
            self.name,
            device_map="cpu",
            offload_folder=OFFLOAD_FOLDER_PREFIX + self.name,
            low_cpu_mem_usage=True,
        ).cpu()

        offload_foler_path = os.path.join(
            OFFLOAD_FOLDER_PREFIX,
            str(self.pid),
            self.name,
        )
        # offload model to disk
        disk_offload(model, offload_dir=offload_foler_path)

        return model

    @abstractmethod
    def get_model(self) -> PreTrainedModel:
        """Abstract method to get model."""

    @abstractmethod
    def get_split_points(self) -> List[str]:
        """Abstract method to get split points."""

    @abstractmethod
    def get_output_parser(self) -> Union[Callable, None]:
        """Abstract method to return function to parse output."""


class Gpt2ModelMetaData(BaseModelMetaData):
    """Gpt2 model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.LANG

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._offload_model_to_disk(AutoModelForPreTraining)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"transformer.h.{i}")
        self.split_points.append("transformer.ln_f")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""
        return None


class BertModelMetaData(BaseModelMetaData):
    """Bert model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.LANG

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._offload_model_to_disk(AutoModelForCausalLM)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"bert.encoder.layer.{i}")
        self.split_points.append("cls")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""
        return None


class T5ModelMetaData(BaseModelMetaData):
    """T5 model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.LANG

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._offload_model_to_disk(AutoModelForPreTraining)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_layers):
            self.split_points.append(f"encoder.block.{i}")
        for i in range(self.config.num_decoder_layers):
            self.split_points.append(f"decoder.block.{i}")
        self.split_points.append("lm_head")

        logger.debug(f"#layers = {self.config.num_layers}")
        logger.debug(f"#decoder_layers = {self.config.num_decoder_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""
        return None


class VitModelMetaData(BaseModelMetaData):
    """Vit model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.IMAGE

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._offload_model_to_disk(AutoModelForImageClassification)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        # Sharding for the Google's HuggingFace ViT model
        # e.g. google/vit-base-patch16-224 (https://huggingface.co/google/vit-base-patch16-224)
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"vit.encoder.layer.{i}")
        self.split_points.append("vit.layernorm")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""
        return None


class ResnetModelMetaData(BaseModelMetaData):
    """Resnet model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.IMAGE

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._offload_model_to_disk(AutoModelForImageClassification)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        # Sharding for the Microsoft's HuggingFace ResNet model
        # e.g. microsoft/resnet-152 (https://huggingface.co/microsoft/resnet-152)
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i, depth in enumerate(self.config.depths):
            for j in range(depth):
                self.split_points.append(f"resnet.encoder.stages.{i}.layers.{j}")

        self.split_points.append("resnet.pooler")

        logger.debug(f"#depths = {sum(self.config.depths)}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""

        def inner(outputs):
            return outputs["logits"]

        return inner
