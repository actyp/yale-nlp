from langfun.core.llms import openai_compatible
import langfun.core as lf
import pyglove as pg
import functools


class LocalModelInfo(lf.ModelInfo):
    description = "Locally hosted LLMs"
    provider = "vllm"


SUPPORTED_MODELS = [
    LocalModelInfo(
        model_id="m-a-p/OpenCodeInterpreter-DS-33B",
        model_type="instruction-tuned",
        url="https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-33B",
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=12_000,
            max_output_tokens=4_384,
        ),
    ),
    LocalModelInfo(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        model_type="instruction-tuned",
        url="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=129_024,
            max_output_tokens=8_192,
        ),
    ),
    LocalModelInfo(
        model_id="bigcode/starcoder2-15b-instruct-v0.1",
        model_type="instruction-tuned",
        url="https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1",
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=12_000,
            max_output_tokens=4_384,
        ),
    ),
    LocalModelInfo(
        model_id="codellama/CodeLlama-13b-Instruct-hf",
        model_type="instruction-tuned",
        url="https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf",
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=3_004,
            max_output_tokens=1_092,
        ),
    ),
    LocalModelInfo(
        model_id="NTQAI/Nxcode-CQ-7B-orpo",
        model_type="instruction-tuned",
        url="https://huggingface.co/NTQAI/Nxcode-CQ-7B-orpo",
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=63_488,
            max_output_tokens=6_144,
        ),
    ),
    LocalModelInfo(
        model_id="Artigenz/Artigenz-Coder-DS-6.7B",
        model_type="instruction-tuned",
        url="https://huggingface.co/Artigenz/Artigenz-Coder-DS-6.7B",
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=12_000,
            max_output_tokens=4_384,
        ),
    ),
]

_SUPPORTED_MODELS_BY_ID = {m.model_id: m for m in SUPPORTED_MODELS}


@lf.use_init_args(["model"])
class LocalModel(openai_compatible.OpenAICompatible):
    """Language model served locally with vllm."""

    model: pg.typing.Annotated[
        pg.typing.Enum(
            pg.MISSING_VALUE,
            [m.model_id for m in SUPPORTED_MODELS]
        ),
        "The name of the model to use.",
    ]

    api_endpoint: str = "http://localhost:8000/v1/chat/completions"

    @functools.cached_property
    def model_info(self) -> LocalModelInfo:
        return _SUPPORTED_MODELS_BY_ID[self.model]

    @classmethod
    def dir(cls):
        return [m.model_id for m in SUPPORTED_MODELS]


class OpenCodeInterpreter_DS_33B(LocalModel):
    """m-a-p/OpenCodeInterpreter-DS-33B model."""

    model = "m-a-p/OpenCodeInterpreter-DS-33B"


class Qwen25_Coder_32B_Instruct(LocalModel):
    """Qwen/Qwen2.5-Coder-32B-Instruct model."""

    model = "Qwen/Qwen2.5-Coder-32B-Instruct"


class Starcoder2_15B_Instruct_v01(LocalModel):
    """bigcode/starcoder2-15b-instruct-v0.1 model."""

    model = "bigcode/starcoder2-15b-instruct-v0.1"


class CodeLlama_13B_Instruct_hf(LocalModel):
    """codellama/CodeLlama-13b-Instruct-hf model."""

    model = "codellama/CodeLlama-13b-Instruct-hf"


class Nxcode_CQ_7B_orpo(LocalModel):
    """NTQAI/Nxcode-CQ-7B-orpo model."""

    model = "NTQAI/Nxcode-CQ-7B-orpo"


class Artigenz_Coder_DS_67B(LocalModel):
    """Artigenz/Artigenz-Coder-DS-6.7B model."""

    model = "Artigenz/Artigenz-Coder-DS-6.7B"


def _register_local_models():
    """Registers local models."""
    for m in SUPPORTED_MODELS:
        lf.LanguageModel.register(m.model_id, LocalModel)


_register_local_models()
