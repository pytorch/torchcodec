import argparse
import os
import time

import torch.utils.benchmark as benchmark

import torchcodec
from torchvision.transforms import Resize


def decode_full_video(video_path, decode_device):
    decoder = torchcodec.decoders._core.create_from_file(video_path)
    if "cuda" in decode_device:
        torchcodec.decoders._core.add_video_stream(
            decoder, stream_index=0, device_string=decode_device, num_threads=1
        )
    else:
        torchcodec.decoders._core.add_video_stream(
            decoder, stream_index=0, device_string=decode_device
        )
    start_time = time.time()
    frame_count = 0
    while True:
        try:
            frame, _, _ = torchcodec.decoders._core.get_next_frame(decoder)
            # the following line is a no-op if frame is already on the GPU.
            # Otherwise it transfers the frame to GPU memory.
            frame = frame.to("cuda:0")
            r = Resize((256, 256))(frame)
            # Just use r[0] so we don't get warnings about unused variables.
            r[0]
            frame_count += 1
        except Exception as e:
            print("EXCEPTION", e)
            break
        print(f"current {frame_count=}", flush=True)
    end_time = time.time()
    elapsed = end_time - start_time
    fps = frame_count / (end_time - start_time)
    print(
        f"****** DECODED full video {decode_device=} {frame_count=} {elapsed=} {fps=}"
    )
    return frame_count, end_time - start_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        default="cuda:0,cpu",
        type=str,
        help="Comma-separated devices to test decoding on.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=os.path.dirname(__file__) + "/../../test/resources/nasa_13013.mp4",
    )
    parser.add_argument(
        "--use_torch_benchmark",
        action="store_true",
        default=True,
        help=(
            "Use pytorch benchmark to measure decode time with warmup and "
            "autorange. Without this we just run one iteration without warmup "
            "to measure the cold start time."
        ),
    )
    args = parser.parse_args()
    video_path = args.video

    if not args.use_torch_benchmark:
        for device in args.devices.split(","):
            print("Testing on", device)
            decode_full_video(video_path, device)
        return

    results = []
    for device in args.devices.split(","):
        print("device", device)
        t = benchmark.Timer(
            stmt="decode_full_video(video_path, device)",
            globals={
                "device": device,
                "video_path": video_path,
                "decode_full_video": decode_full_video,
            },
            label="Decode+Resize Time",
            sub_label=f"video={os.path.basename(video_path)}",
            description=f"decode_device={device}",
        ).blocked_autorange()
        results.append(t)
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
