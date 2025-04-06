from langfun.core.llms import openai_compatible
import langfun.core as lf
import pyglove as pg
import functools


class LocalModelInfo(lf.ModelInfo):
    description = "Locally hosted LLMs"
    provider = "vllm"


SUPPORTED_MODELS = [
    LocalModelInfo(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_type="instruction-tuned",
        url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=129_024,
            max_output_tokens=8_192,
        ),
    ),
]

_SUPPORTED_MODELS_BY_ID = {m.model_id: m for m in SUPPORTED_MODELS}


@lf.use_init_args(["model"])
class LocalModel(openai_compatible.OpenAICompatible):
    """Language model served locally with vllm."""

    model: pg.typing.Annotated[
        pg.typing.Enum(pg.MISSING_VALUE, [m.model_id for m in SUPPORTED_MODELS]),
        "The name of the model to use.",
    ]

    api_endpoint: str = "http://localhost:8000/v1/chat/completions"

    @functools.cached_property
    def model_info(self) -> LocalModelInfo:
        return _SUPPORTED_MODELS_BY_ID[self.model]

    @classmethod
    def dir(cls):
        return [m.model_id for m in SUPPORTED_MODELS]


class Qwen25_7B_Instruct(LocalModel):
    """Qwen2.5 7B instruction-tuned model."""

    model = "Qwen/Qwen2.5-7B-Instruct"


def _register_local_models():
    """Registers local models."""
    for m in SUPPORTED_MODELS:
        lf.LanguageModel.register(m.model_id, LocalModel)


_register_local_models()
