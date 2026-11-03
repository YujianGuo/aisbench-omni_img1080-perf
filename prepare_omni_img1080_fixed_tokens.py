#!/usr/bin/env python3
"""Create fixed-token AISBench JSONL datasets for omni_img1080."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = str(BUNDLE_DIR / "datasets" / "omni_img1080")
DEFAULT_TOKENIZER = os.getenv("AISBENCH_TOKENIZER_PATH", "/data/Qwen3.5-35B-A3B")
DEFAULT_OUTPUT_DIR = str(BUNDLE_DIR / "datasets")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PROMPT_UNIT = (
    "Analyze the image carefully. Describe visible objects, people, colors, actions, "
    "spatial relationships, background, lighting, and readable text. Be factual and "
    "specific, and do not guess details that are not visible."
)


def encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def make_prompt(tokenizer: Any, token_count: int) -> str:
    """Return readable text that round-trips to exactly token_count tokenizer IDs."""
    unit_tokens = encode(tokenizer, PROMPT_UNIT)
    repetitions = max(1, math.ceil(token_count / len(unit_tokens)))
    candidate = " ".join([PROMPT_UNIT] * repetitions)
    candidate_ids = encode(tokenizer, candidate)
    candidate = tokenizer.decode(
        candidate_ids[:token_count],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Tokenizer decode is normally lossless for this model. The loop handles
    # boundary normalization by trimming or extending with a short neutral word.
    for _ in range(4096):
        actual_count = len(encode(tokenizer, candidate))
        if actual_count == token_count:
            return candidate
        if actual_count > token_count:
            candidate = candidate[:-1]
        else:
            candidate += " a"
    raise RuntimeError(
        f"Could not construct a prompt with exactly {token_count} tokenizer tokens"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one-image or five-image requests with fixed text-token length."
    )
    parser.add_argument("--images-per-request", type=int, choices=(1, 5), required=True)
    parser.add_argument("--text-tokens", type=int, default=200)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-count", type=int, default=128)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use the first N images (use a multiple of five for five-image requests).",
    )
    args = parser.parse_args()

    if args.text_tokens <= 0:
        raise SystemExit("--text-tokens must be positive")
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {image_dir}")
    images = sorted(
        path.resolve()
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        if len(images) < args.limit:
            raise SystemExit(
                f"--limit requested {args.limit} images, but found only {len(images)}"
            )
        images = images[: args.limit]
    elif len(images) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} images in {image_dir}, found {len(images)}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt = make_prompt(tokenizer, args.text_tokens)
    measured_tokens = len(encode(tokenizer, prompt))
    if measured_tokens != args.text_tokens:
        raise RuntimeError(
            f"Prompt token count mismatch: expected {args.text_tokens}, got {measured_tokens}"
        )

    group_size = args.images_per_request
    usable_count = len(images) - len(images) % group_size
    skipped_images = images[usable_count:]
    images = images[:usable_count]
    if not images:
        raise SystemExit(
            f"Need at least {group_size} images to build a complete request"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"omni_img1080_{group_size}img_{args.text_tokens}in"
    output_path = output_dir / f"{stem}.jsonl"
    metadata_path = output_dir / f"{stem}.meta.json"

    rows = []
    for offset in range(0, len(images), group_size):
        group = images[offset : offset + group_size]
        rows.append(
            {
                "type": "image",
                "path": [str(path) for path in group],
                "question": prompt,
                "answer": "",
            }
        )

    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "dataset": str(output_path),
        "images_per_request": group_size,
        "text_tokens_per_request": measured_tokens,
        "tokenizer_path": str(Path(args.tokenizer_path).expanduser().resolve()),
        "image_count": len(images),
        "request_count": len(rows),
        "skipped_remainder_images": [str(path) for path in skipped_images],
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(
        f"Wrote {len(rows)} requests ({len(images)} images, {group_size} images/request); "
        f"question text is exactly {measured_tokens} tokenizer tokens."
    )
    print(f"Dataset: {output_path}")
    print(f"Metadata: {metadata_path}")
    if skipped_images:
        print(
            f"Skipped {len(skipped_images)} trailing image(s) to keep every request "
            f"at exactly {group_size} images."
        )


if __name__ == "__main__":
    main()
