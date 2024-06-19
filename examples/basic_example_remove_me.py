"""
=============
Basic Example
=============

Remove this!
"""

# %%
print("This is a cell that gets executerd")
# %%
print("And another cell.")

# %%
# We can write rst cells too!
# ---------------------------
#
# As if we're writing normal docs/docstrings. Let's reference the
# :class:`~torchcodec.decoders.SimpleVideoDecoder` class. Click on this and it
# should bring you to its docstring!! In the docstring, you should see a
# backreference to this example as well!
#
# And of course we can write normal code

# %%
from torchcodec.decoders import SimpleVideoDecoder

try:
    SimpleVideoDecoder("bad_path")
except ValueError as e:
    print(f"Ooops:\n {e}")

# %%
print("All good!")
