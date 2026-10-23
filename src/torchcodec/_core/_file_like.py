# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expose Python file-like objects to the C++ FileLikeIO through ctypes.

None of our compiled libraries use the CPython C API, which is what allows us to
ship a single Python-agnostic wheel. Instead, C++ calls back into the file-like
object through the C function pointers defined here. See FileLikeIO.h for the
C++ side of the contract.
"""

import ctypes
import operator
from typing import Any

# Must match kFileLikeCallbackError in FileLikeIO.h.
_CALLBACK_ERROR = -(2**63)

_READ_FN = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64
)
_WRITE_FN = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64
)
_SEEK_FN = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
)
_RELEASE_FN = ctypes.CFUNCTYPE(None, ctypes.c_int64)
_GET_ERROR_FN = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64
)


# Must match the layout of FileLikeCallbacks in FileLikeIO.h.
class _FileLikeCallbacks(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_int64),
        ("read", _READ_FN),
        ("write", _WRITE_FN),
        ("seek", _SEEK_FN),
        ("release", _RELEASE_FN),
        ("get_error", _GET_ERROR_FN),
    ]


class _Entry:
    __slots__ = ("file_like", "callbacks", "error")

    def __init__(self, file_like: Any):
        self.file_like = file_like
        self.callbacks: _FileLikeCallbacks | None = None
        self.error: BaseException | None = None


# Keeps the file-like objects alive until C++ calls release(). Keyed by
# id(entry), which is unique for as long as the entry is in this dict.
_entries: dict[int, _Entry] = {}


# An exception escaping a ctypes callback is only printed, and the callback then
# returns 0, which C++ would interpret as EOF. So every callback catches
# everything and reports the error through _CALLBACK_ERROR + get_error instead.
def _read(handle: int, buf: int, size: int) -> int:
    entry = _entries[handle]
    try:
        data = entry.file_like.read(size)
        if not isinstance(data, bytes):
            raise TypeError(
                f"read() must return bytes, got {type(data).__name__} instead."
            )
        num_bytes = len(data)
        if num_bytes > size:
            raise ValueError(
                f"Requested up to {size} bytes but, received {num_bytes} bytes. "
                "The given object does not conform to read protocol of file object."
            )
        ctypes.memmove(buf, data, num_bytes)
        return num_bytes
    except BaseException as e:
        entry.error = e
        return _CALLBACK_ERROR


def _write(handle: int, buf: int, size: int) -> int:
    entry = _entries[handle]
    try:
        return operator.index(entry.file_like.write(ctypes.string_at(buf, size)))
    except BaseException as e:
        entry.error = e
        return _CALLBACK_ERROR


def _seek(handle: int, offset: int, whence: int) -> int:
    entry = _entries[handle]
    try:
        return operator.index(entry.file_like.seek(offset, whence))
    except BaseException as e:
        entry.error = e
        return _CALLBACK_ERROR


def _release(handle: int) -> None:
    _entries.pop(handle, None)


def _get_error(handle: int, buf: int, buf_len: int) -> int:
    entry = _entries[handle]
    error, entry.error = entry.error, None
    message = f"{type(error).__name__}: {error}".encode()
    # Don't truncate in the middle of a multi-byte UTF-8 sequence: the message
    # must remain valid UTF-8 to be turned back into a Python str.
    message = message[:buf_len].decode(errors="ignore").encode()
    ctypes.memmove(buf, message, len(message))
    return len(message)


# These must stay alive for as long as C++ may call them, i.e. forever.
_read_fn = _READ_FN(_read)
_write_fn = _WRITE_FN(_write)
_seek_fn = _SEEK_FN(_seek)
_release_fn = _RELEASE_FN(_release)
_get_error_fn = _GET_ERROR_FN(_get_error)


def create_file_like_context(file_like: Any, is_for_writing: bool) -> int:
    """Return an int to pass as the ``file_like_context`` of a custom op.

    The custom op takes ownership: it releases the file-like object once it's
    done with it. The returned value must be passed to exactly one custom op.
    """
    if is_for_writing:
        if not hasattr(file_like, "write"):
            raise RuntimeError(
                "File like object must implement a write method for writing."
            )
    elif not hasattr(file_like, "read"):
        raise RuntimeError("File like object must implement a read method for reading.")
    if not hasattr(file_like, "seek"):
        raise RuntimeError("File like object must implement a seek method.")

    entry = _Entry(file_like)
    handle = id(entry)
    entry.callbacks = _FileLikeCallbacks(
        handle, _read_fn, _write_fn, _seek_fn, _release_fn, _get_error_fn
    )
    _entries[handle] = entry
    return ctypes.addressof(entry.callbacks)
