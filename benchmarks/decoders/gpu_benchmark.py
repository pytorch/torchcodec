import argparse
import os
import time

import torch.utils.benchmark as benchmark

import torchcodec
from torchvision.transforms import Resize

RESIZED_WIDTH = 256
RESIZED_HEIGHT = 256


def transfer_and_resize_frame(frame, resize_device_string):
    # This should be a no-op if the frame is already on the GPU.
    frame = frame.to(resize_device_string)
    frame = Resize((RESIZED_HEIGHT, RESIZED_WIDTH))(frame)
    return frame


def decode_full_video(video_path, decode_device_string, resize_device_string):
    # We use the core API instead of SimpleVideoDecoder because the core API
    # allows us to natively resize as part of the decode step.
    print(f"{decode_device_string=} {resize_device_string=}")
    decoder = torchcodec.decoders._core.create_from_file(video_path)
    num_threads = None
    if "cuda" in decode_device_string:
        num_threads = 1
    width = None
    height = None
    if "native" in resize_device_string:
        width = RESIZED_WIDTH
        height = RESIZED_HEIGHT
    torchcodec.decoders._core.add_video_stream(
        decoder,
        stream_index=-1,
        device_string=decode_device_string,
        num_threads=num_threads,
        width=width,
        height=height,
    )

    start_time = time.time()
    frame_count = 0
    while True:
        try:
            frame, *_ = torchcodec.decoders._core.get_next_frame(decoder)
            if resize_device_string != "none" and "native" not in resize_device_string:
                frame = transfer_and_resize_frame(frame, resize_device_string)

            frame_count += 1
        except Exception as e:
            print("EXCEPTION", e)
            break

    end_time = time.time()
    elapsed = end_time - start_time
    fps = frame_count / (end_time - start_time)
    print(
        f"****** DECODED full video {decode_device_string=} {frame_count=} {elapsed=} {fps=}"
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
        "--resize_devices",
        default="cuda:0,cpu,native,none",
        type=str,
        help="Comma-separated devices to test preroc (resize) on. Use 'none' to specify no resize.",
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

    resize_devices = args.resize_devices.split(",")
    resize_devices = [d for d in resize_devices if d != ""]
    if len(resize_devices) == 0:
        resize_devices.append("none")

    label = "Decode+Resize Time"

    results = []
    for decode_device_string in args.devices.split(","):
        for resize_device_string in resize_devices:
            decode_label = decode_device_string
            if "cuda" in decode_label:
                # Shorten "cuda:0" to "cuda"
                decode_label = "cuda"
            resize_label = resize_device_string
            if "cuda" in resize_device_string:
                # Shorten "cuda:0" to "cuda"
                resize_label = "cuda"
            print("decode_device", decode_device_string)
            print("resize_device", resize_device_string)
            t = benchmark.Timer(
                stmt="decode_full_video(video_path, decode_device_string, resize_device_string)",
                globals={
                    "decode_device_string": decode_device_string,
                    "video_path": video_path,
                    "decode_full_video": decode_full_video,
                    "resize_device_string": resize_device_string,
                },
                label=label,
                sub_label=f"video={os.path.basename(video_path)}",
                description=f"D={decode_label},R={resize_label}",
            ).blocked_autorange()
            results.append(t)
    compare = benchmark.Compare(results)
    compare.print()
    print("Key: D=Decode, R=Resize")
    print("Native resize is done as part of the decode step")
    print("none resize means there is no resize step -- native or otherwise")


if __name__ == "__main__":
    main()
