# omni_img1080 AISBench streaming performance set

This directory contains the image samples, AISBench JSONL manifests, a dataset
builder, and two independent streaming performance configurations.

## Workloads

| Config | Images per request | Text tokens per request | Output tokens | Max concurrency |
| --- | ---: | ---: | ---: | ---: |
| `omni_img1080_1img_200in_400out_perf.py` | 1 | 200 | 400 | 1 |
| `omni_img1080_5img_200in_400out_perf.py` | 5 | 200 | 400 | 1 |

The prompt is counted with the Qwen tokenizer using `add_special_tokens=False`.
The 200-token count covers the question text only; chat-template and image tokens
are additional. The configs use `max_out_len=400` and `ignore_eos=True` to drive
generation to the 400-token limit. They use `batch_size=1` and streaming output.

## Dataset contents

`datasets/omni_img1080/` contains the 9 image files currently present in the
source directory. The requested full set is 128 images, but the other 119 images
were not present when this bundle was prepared. The included manifests therefore
contain 9 one-image requests and 1 five-image request. Add the remaining images
to `datasets/omni_img1080/` and rebuild the manifests before measuring the full
set. With all 128 images, the five-image workload uses 125 images in 25 complete
requests and reports the three-image remainder as skipped.

## Requirements

- Run inside an environment where AISBench is installed.
- Install `transformers` and make the Qwen tokenizer files available locally.
- Ensure the vLLM-compatible streaming endpoint is reachable from the AISBench
  environment. Defaults are `127.0.0.1:9004` and model path
  `/data/Qwen3.5-35B-A3B`.

The tokenizer builder accepts `AISBENCH_TOKENIZER_PATH` as an environment
override. If the serving address or model path differs, edit the corresponding
`host_ip`, `host_port`, or `path` field in both config files before running.

## Build manifests and run

From this directory, build both manifests using the images included here:

```bash
python3 prepare_omni_img1080_fixed_tokens.py --images-per-request 1 --limit 9
python3 prepare_omni_img1080_fixed_tokens.py --images-per-request 5 --limit 5
```

Then run each workload:

```bash
ais_bench omni_img1080_1img_200in_400out_perf.py --mode perf
ais_bench omni_img1080_5img_200in_400out_perf.py --mode perf
```

For the complete 128-image set, omit `--limit`; the builder checks that exactly
128 images are available. It writes a JSONL and a `.meta.json` sidecar for each
workload. The two configs read separate JSONL files and write results to separate
subdirectories under `outputs/`.

The test workloads have not been run as part of packaging; this repository
contains the inputs and configurations for running them.
