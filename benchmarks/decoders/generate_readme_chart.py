# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

from pathlib import Path

from benchmark_decoders_library import plot_data


def main() -> None:
    data_json = Path(__file__).parent / "benchmark_readme_data.json"
    with open(data_json, "r") as read_file:
        data_from_file = json.load(read_file)

    output_png = Path(__file__).parent / "benchmark_readme_chart.png"
    plot_data(data_from_file, output_png)


if __name__ == "__main__":
    main()
