import argparse
import os
import time

import torch.utils.benchmark as benchmark

import torchcodec
from torchvision.transforms import Resize


def transfer_and_resize_frame(frame, device):
    # This should be a no-op if the frame is already on the device.
    frame = frame.to(device)
    frame = Resize((256, 256))(frame)
    return frame


def decode_full_video(video_path, decode_device):
    decoder = torchcodec.decoders._core.create_from_file(video_path)
    num_threads = None
    if "cuda" in decode_device:
        num_threads = 1
    torchcodec.decoders._core.add_video_stream(
        decoder, stream_index=0, device_string=decode_device, num_threads=num_threads
    )
    start_time = time.time()
    frame_count = 0
    while True:
        try:
            frame, *_ = torchcodec.decoders._core.get_next_frame(decoder)
            # You can do a resize to simulate extra preproc work that happens
            # on the GPU by uncommenting the following line:
            # frame = transfer_and_resize_frame(frame, decode_device)

            frame_count += 1
        except Exception as e:
            print("EXCEPTION", e)
            break
        # print(f"current {frame_count=}", flush=True)
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
        action=argparse.BooleanOptionalAction,
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
