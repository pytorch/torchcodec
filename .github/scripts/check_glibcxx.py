"""
The goal of this script is to ensure that the .so files we ship do not contain
symbol versions from libstdc++ that are too recent.

Why this is needed: during development, we observed the following general
scenario in various local development setups:
- torchcodec is compiled with a given (somewhat recent) c++ toolchain (say
  gcc11)
- because the toolchain is recent, some recent symbol versions from libstdc++
  are added as dependencies in the torchcodec?.so files, e.g. GLIBCXX_3.4.29
  (this is normal)
- at runtime, for whatever reason, the libstdc++.so that gets loaded is *not*
  the one that was used when building. The libstdc++.so that is loaded can be
  older than the toolchain one, and it doesn't contain the more recent symbols
  that torchcodec?.so depends on, which leads to a runtime error.

The reasons why a different libstdc++.so is loaded at runtime can be multiple
(and mysterious! https://hackmd.io/@_NznxihTSmC-IgW4cgnlyQ/HJXc4BEHR).

This script doesn't try to prevent *that* (it's impossible anyway, as we don't
control users' environments). Instead, it prevents the dependency of torchcodec
on recent symbol versions, which ensures that torchcodec can run on both recent
*and* older runtimes.
The most recent symbol on the manylinux torch.2.3.1 wheel is
GLIBCXX_3.4.19, so as long as torchcodec doesn't ship a symbol that is higher
than that, torchcodec should be fine.

Note that the easiest way to avoid recent symbols is simply to use an old-enough
toolchain. In July 2024, pytorch libraries (and torchcodec) are built with gcc
9.
"""

import sys
import re

if len(sys.argv) != 2:
    raise ValueError("Wrong usage: python check_glibcxx.py <str_with_symbols>.")

MAX_MINOR_ALLOWED = 19

all_symbols = set()
max_minor_version = float("-inf")
for line in sys.argv[1].split("\n"):
    # We search for GLIBCXX_3.4.X where X is the minor version with 1 or 2 digits.
    if match := re.search(r"GLIBCXX_3\.4\.(\d{1,2})", line):
        all_symbols.add(match.group(0))
        max_minor_version = max(max_minor_version, int(match.group(1)))

if not all_symbols:
    raise ValueError("No GLIBCXX symbols found. Something is wrong.")

print(f"Found the following GLIBCXX symbol versions: {all_symbols}.")
print(
    f"The max minor version is {max_minor_version}. Max allowed is {MAX_MINOR_ALLOWED}."
)

if max_minor_version > MAX_MINOR_ALLOWED:
    raise AssertionError(
        "The max minor version is greater than the max allowed! "
        "That may leads to compatibility issues. "
        "Was the wheel compiled with an old-enough toolchain?"
    )

print("All good.")
