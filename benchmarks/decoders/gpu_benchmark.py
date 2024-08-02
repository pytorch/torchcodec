import argparse
import os
import time

import torch
import torch.utils.benchmark as benchmark

import torchcodec
from torchvision.transforms import Resize


def transfer_and_resize_frame(frame):
    # This should be a no-op if the frame is already on the GPU.
    frame = frame.to("cuda:0")
    frame = Resize((256, 256))(frame)
    return frame


def decode_full_video(video_path, device_string, do_gpu_preproc):
    decoder = torchcodec.decoders.SimpleVideoDecoder(
        video_path, device=torch.device(device_string)
    )
    start_time = time.time()
    frame_count = 0
    for frame in decoder:
        # You can do a resize to simulate extra preproc work that happens
        # on the GPU by uncommenting the following line:
        if do_gpu_preproc:
            frame = transfer_and_resize_frame(frame)
        frame_count += 1
    end_time = time.time()
    elapsed = end_time - start_time
    fps = frame_count / (end_time - start_time)
    print(
        f"****** DECODED full video {device_string=} {frame_count=} {elapsed=} {fps=}"
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use pytorch benchmark to measure decode time with warmup and "
            "autorange. Without this we just run one iteration without warmup "
            "to measure the cold start time."
        ),
    )
    parser.add_argument(
        "--do_gpu_preproc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Do a transfer to GPU and resize operation after the decode to "
            "simulate a real-world transform."
        ),
    )
    args = parser.parse_args()
    video_path = args.video

    if not args.use_torch_benchmark:
        for device in args.devices.split(","):
            print("Testing on", device)
            decode_full_video(video_path, device)
        return

    label = "Decode"
    if args.do_gpu_preproc:
        label += " + GPU Preproc"
    label += " Time"

    results = []
    for device in args.devices.split(","):
        print("device", device)
        t = benchmark.Timer(
            stmt="decode_full_video(video_path, device, do_gpu_preproc)",
            globals={
                "device": device,
                "video_path": video_path,
                "decode_full_video": decode_full_video,
                "do_gpu_preproc": args.do_gpu_preproc,
            },
            label=label,
            sub_label=f"video={os.path.basename(video_path)}",
            description=f"decode_device={device}",
        ).blocked_autorange()
        results.append(t)
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
