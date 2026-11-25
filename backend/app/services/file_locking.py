"""Small cross-platform non-blocking exclusive file locks.

The lock is advisory and coordinates cooperating processes that share the same
local lock directory. POSIX uses ``flock``; Windows uses a one-byte ``msvcrt``
range lock. The operating system releases the lock when the owning handle or
process exits.
"""

from __future__ import annotations

import os
from typing import BinaryIO


class FileLockUnavailable(OSError):
    """Raised when another process currently holds the requested lock."""


def try_lock_exclusive(file_obj: BinaryIO) -> None:
    """Acquire an exclusive lock without waiting; raise if it is already held."""
    if os.name == "nt":
        import msvcrt

        # msvcrt locks a byte range starting at the current file position.
        # Ensure the slot file contains that byte before attempting the lock.
        file_obj.seek(0, os.SEEK_END)
        if file_obj.tell() == 0:
            file_obj.write(b"\0")
            file_obj.flush()
        file_obj.seek(0)
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise FileLockUnavailable(str(exc)) from exc
        return

    import fcntl

    try:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise FileLockUnavailable(str(exc)) from exc


def unlock_exclusive(file_obj: BinaryIO) -> None:
    """Release a lock previously acquired with :func:`try_lock_exclusive`."""
    if os.name == "nt":
        import msvcrt

        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
