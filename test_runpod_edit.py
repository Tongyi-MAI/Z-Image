"""Small helper to test Z-Image edit modes through a RunPod endpoint."""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests


DEFAULT_ENDPOINT_ID = "q3sc07erypme32"


def encode_file_base64(path: Path) -> str:
    """Read a file and return its raw base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test img2img/inpaint against RunPod.")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--endpoint-id", default=DEFAULT_ENDPOINT_ID, help="RunPod endpoint ID")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["text2img", "img2img", "inpaint"],
        help="Generation mode to test",
    )
    parser.add_argument("--prompt", required=True, help="Prompt to send")
    parser.add_argument("--image", help="Path to source image for img2img/inpaint")
    parser.add_argument("--mask", help="Path to mask image for inpaint")
    parser.add_argument(
        "--aspect-ratio",
        choices=["16:9", "9:16"],
        help="High-quality aspect ratio preset to request",
    )
    parser.add_argument("--width", type=int, help="Optional output width")
    parser.add_argument("--height", type=int, help="Optional output height")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps")
    parser.add_argument("--seed", type=int, help="Optional random seed")
    parser.add_argument("--strength", type=float, default=0.6, help="Edit strength")
    parser.add_argument("--negative-prompt", help="Optional negative prompt")
    parser.add_argument(
        "--output",
        default="runpod_output.png",
        help="Where to save the generated PNG",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between RunPod status checks",
    )
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict:
    """Construct the RunPod payload from CLI arguments."""
    payload = {
        "input": {
            "mode": args.mode,
            "prompt": args.prompt,
            "steps": args.steps,
            "strength": args.strength,
        }
    }

    if args.width is not None:
        payload["input"]["width"] = args.width
    if args.height is not None:
        payload["input"]["height"] = args.height
    if args.aspect_ratio:
        payload["input"]["aspect_ratio"] = args.aspect_ratio
    if args.seed is not None:
        payload["input"]["seed"] = args.seed
    if args.negative_prompt:
        payload["input"]["negative_prompt"] = args.negative_prompt

    if args.mode in {"img2img", "inpaint"}:
        if not args.image:
            raise ValueError("--image is required for img2img and inpaint")
        payload["input"]["image"] = encode_file_base64(Path(args.image))

    if args.mode == "inpaint":
        if not args.mask:
            raise ValueError("--mask is required for inpaint")
        payload["input"]["mask_image"] = encode_file_base64(Path(args.mask))

    return payload


def wait_for_completion(endpoint_id: str, api_key: str, job_id: str, poll_interval: float) -> dict:
    """Poll RunPod until the job finishes."""
    headers = {"Authorization": f"Bearer {api_key}"}
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    while True:
        response = requests.get(status_url, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        print(f"Status: {status}")

        if status == "COMPLETED":
            return data["output"]
        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RuntimeError(json.dumps(data, indent=2))

        time.sleep(poll_interval)


def save_output_image(output: dict, output_path: Path) -> None:
    """Decode the output image and write it to disk."""
    image_base64 = output.get("image")
    if not image_base64:
        raise RuntimeError(f"No image in output: {json.dumps(output, indent=2)}")

    output_path.write_bytes(base64.b64decode(image_base64))


def main() -> int:
    """CLI entrypoint."""
    try:
        args = parse_args()
        payload = build_payload(args)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        }
        run_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/run"

        response = requests.post(run_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        job = response.json()
        job_id = job["id"]
        print(f"Job submitted: {job_id}")

        output = wait_for_completion(args.endpoint_id, args.api_key, job_id, args.poll_interval)

        output_path = Path(args.output)
        save_output_image(output, output_path)
        print(f"Saved output to: {output_path.resolve()}")
        print(json.dumps({k: v for k, v in output.items() if k != 'image'}, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
