import sys
import re

if len(sys.argv) != 2:
    print("Wrong usage, git grep where this script is used to figure it out.")
    sys.exit(1)

MAX_MINOR_ALLOWED = 19


all_symbols = set()
max_minor_version = float("-inf")
for line in sys.argv[1].split("\n"):
    # We search for GLIBCXX_3.4.X where X is the minor version with 1 or 2 digits.
    if match := re.search(r"GLIBCXX_3\.4\.(\d{1,2})", line):
        all_symbols.add(match.group(0))
        max_minor_version = max(max_minor_version, int(match.group(1)))

print(f"Found the following GLIBCXX symbol versions: {all_symbols}.")
print(f"The max minor version is {max_minor_version}")

if max_minor_version > MAX_MINOR_ALLOWED:
    raise AssertionError(
        f"The max minor version {max_minor_version} is greater than {MAX_MINOR_ALLOWED = }. "
        "That may leads to compatibility issues. Was the wheel compiled with an old-enough toolchain?"
    )

print("All good.")
