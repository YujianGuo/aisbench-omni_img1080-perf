"""AISBench streaming perf: 1 image + 200 text tokens, up to 400 output tokens."""

from pathlib import Path

from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as vllm_models,
    )
    from ais_bench.benchmark.configs.datasets.mm_custom.mm_custom_gen import (
        mm_custom_datasets,
    )
    from ais_bench.benchmark.configs.summarizers.example import summarizer

models = vllm_models
bundle_dir = Path(__file__).resolve().parent
models[0]["path"] = "/data/Qwen3.5-35B-A3B"
models[0]["model"] = "qwen35"
models[0]["host_ip"] = "127.0.0.1"
models[0]["host_port"] = 9004
models[0]["api_key"] = "EMPTY"
models[0]["stream"] = True
models[0]["batch_size"] = 1
models[0]["request_rate"] = 0
models[0]["max_out_len"] = 400
models[0]["generation_kwargs"] = dict(
    temperature=0.01,
    ignore_eos=True,
    chat_template_kwargs=dict(enable_thinking=False),
)

datasets = mm_custom_datasets
datasets[0]["abbr"] = "omni_img1080_1img_200in"
datasets[0]["path"] = str(bundle_dir / "datasets" / "omni_img1080_1img_200in.jsonl")

work_dir = str(bundle_dir / "outputs" / "omni_img1080_1img_200in_400out")
