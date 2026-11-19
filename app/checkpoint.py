"""Atomic checkpoint persistence, optionally AES-256-GCM encrypted.

A checkpoint is a single ``checkpoint.pt`` file inside the runs directory.
Saves are atomic on POSIX filesystems:

1. serialize (and, when a key is configured, encrypt) entirely in memory,
2. write the result to ``.checkpoint.pt.tmp.<uuid>`` in the same directory —
   with encryption enabled this file contains **only ciphertext**,
3. flush + fsync the file,
4. ``os.replace()`` it over ``checkpoint.pt`` (atomic rename),
5. fsync the directory so the rename itself is durable.

If the process is killed at any point, ``checkpoint.pt`` is therefore either
the complete previous checkpoint or the complete new one — never a torn file.
A failed save (e.g. disk full, encryption error) leaves the previous
checkpoint untouched so it can be used for a retry or for recovery after a
restart.

Encryption mode
---------------
Passing a 32-byte ``key`` (see :mod:`app.crypto`) enables AES-256-GCM.  The
plaintext torch-serialized checkpoint is never written to disk: both the temp
file and the final file hold the self-describing encrypted frame
(``RPCKENC1`` magic + fresh nonce + tag + ciphertext).  Loading an encrypted
checkpoint without a key, with the wrong key, or encountering a plaintext
checkpoint while encryption is enabled, all raise an explicit error and never
modify the on-disk file.  Without a key, legacy plaintext checkpoints keep
working exactly as before.

TEST-ONLY HOOK: if a file named ``INJECT_SAVE_FAILURES`` exists in the runs
directory, every save fails *before touching anything* (the old checkpoint
stays intact).  ``scripts/verify_repro.py`` uses this to exercise the
save-failure path; it is never present in normal operation.
"""
from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path

import torch

from .crypto import (
    CheckpointFormatError,
    decrypt_blob,
    encrypt_blob,
    is_encrypted_blob,
)

CHECKPOINT_FILENAME = "checkpoint.pt"
INJECT_FAILURES_FILENAME = "INJECT_SAVE_FAILURES"


class CheckpointError(IOError):
    """Raised when a checkpoint could not be persisted."""


def checkpoint_path(runs_dir) -> Path:
    return Path(runs_dir) / CHECKPOINT_FILENAME


def save_checkpoint(runs_dir, payload: dict, key: bytes | None = None,
                    save_delay_seconds: float = 0.0) -> Path:
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)

    # --- test-only fault injection (see module docstring) ---
    if (runs / INJECT_FAILURES_FILENAME).exists():
        raise CheckpointError(
            f"checkpoint save failed: injected fault "
            f"('{INJECT_FAILURES_FILENAME}' file present in {runs})")
    # ---------------------------------------------------------

    final = runs / CHECKPOINT_FILENAME
    tmp = runs / f".{CHECKPOINT_FILENAME}.tmp.{uuid.uuid4().hex}"
    try:
        # Serialize fully in memory so plaintext never lands on disk.
        plaintext = _serialize(payload)
        if key is not None:
            blob = encrypt_blob(plaintext, key)   # fresh nonce every save
        else:
            blob = plaintext
        with open(tmp, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        if save_delay_seconds > 0:
            # Simulates slow storage.  The window between "new file fully
            # written" and "atomic rename" is where a kill must still leave
            # the OLD checkpoint intact.
            time.sleep(save_delay_seconds)
        os.replace(tmp, final)
        _fsync_dir(runs)
    except CheckpointError:
        raise
    except Exception as exc:  # noqa: BLE001 - report any persistence failure
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise CheckpointError(f"failed to persist checkpoint to {final}: {exc!r}") from exc
    return final


def load_checkpoint(runs_dir, key: bytes | None = None) -> dict | None:
    """Load the checkpoint if one exists; ``None`` otherwise.

    With ``key`` set, only encrypted checkpoints are accepted; a plaintext
    checkpoint raises :class:`CheckpointFormatError` and is left untouched.
    Without a key, encrypted checkpoints raise
    :class:`~app.crypto.CheckpointKeyError` and plaintext ones load normally.
    Only complete checkpoints can be observed thanks to the atomic rename.
    """
    final = checkpoint_path(runs_dir)
    if not final.exists():
        return None
    blob = final.read_bytes()
    if is_encrypted_blob(blob):
        plaintext = decrypt_blob(blob, key)   # wrong/missing key raises here
    else:
        if key is not None:
            # Refuse to silently consume a plaintext point in encrypted mode;
            # the caller must decide (e.g. reject start / recovery).
            raise CheckpointFormatError(
                "encryption is enabled but the existing checkpoint.pt is a "
                "plaintext (unencrypted) file; refusing to load or overwrite "
                "it. Remove or rotate it out of the runs directory before "
                "starting an encrypted run")
        plaintext = blob
    return torch.load(io.BytesIO(plaintext), map_location="cpu",
                      weights_only=True)


def checkpoint_is_encrypted(runs_dir) -> bool | None:
    """Return whether the on-disk checkpoint is an encrypted frame.

    ``None`` when no checkpoint exists.  Only inspects the magic header.
    """
    final = checkpoint_path(runs_dir)
    if not final.exists():
        return None
    with open(final, "rb") as f:
        head = f.read(8)
    return is_encrypted_blob(head)


def _serialize(payload: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def cleanup_stale_tmp_files(runs_dir) -> list[str]:
    """Remove leftover temp files from saves interrupted by a kill."""
    removed = []
    runs = Path(runs_dir)
    if runs.is_dir():
        for tmp in runs.glob(f".{CHECKPOINT_FILENAME}.tmp.*"):
            try:
                tmp.unlink()
                removed.append(tmp.name)
            except OSError:
                pass
    return removed


def _fsync_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
