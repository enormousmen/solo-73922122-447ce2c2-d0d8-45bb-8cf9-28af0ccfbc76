"""AES-256-GCM static encryption for checkpoints.

When encryption is enabled, the *entire* checkpoint (model parameters,
optimizer state, RNG state, loss history, ...) is authenticated-encrypted with
AES-256-GCM before anything touches disk.  Plaintext bytes only ever live in
process memory; both the temporary file and the final checkpoint file contain
ciphertext exclusively.

Key material
------------
The key is read from the environment variable named
:data:`KEY_ENV_VAR` and must be the Base64 encoding of **exactly 32 raw bytes**
(256 bits).  Generate one locally with::

    python3 -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"

An unset / empty variable means encryption is disabled and legacy plaintext
checkpoints keep working unchanged.  A variable that is set but does not decode
to exactly 32 bytes is a hard configuration error (fail fast at startup).

On-disk frame (all multi-byte integers are big-endian)::

    magic      8 bytes  b"RPCKENC1"
    nonce      12 bytes fresh ``os.urandom`` per save (never reused)
    tag        16 bytes GCM authentication tag
    ciphertext N bytes  AES-256-GCM(plaintext, nonce, aad=magic)

The fixed magic header doubles as a plaintext/ciphertext discriminator: a
legacy torch-saved checkpoint is a ZIP container whose first bytes are
``PK\\x03\\x04`` and therefore never collides with the magic.
"""
from __future__ import annotations

import base64
import binascii
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_ENV_VAR = "CHECKPOINT_ENCRYPTION_KEY"

KEY_LENGTH_BYTES = 32      # AES-256
NONCE_LENGTH_BYTES = 12    # GCM standard nonce
TAG_LENGTH_BYTES = 16      # GCM standard tag
MAGIC = b"RPCKENC1"
HEADER_LENGTH = len(MAGIC) + NONCE_LENGTH_BYTES + TAG_LENGTH_BYTES


class CheckpointCryptoError(Exception):
    """Base class for checkpoint encryption / decryption failures."""


class CheckpointKeyError(CheckpointCryptoError):
    """No usable key is available for an encrypted checkpoint."""


class CheckpointFormatError(CheckpointCryptoError):
    """A checkpoint cannot be read under the current encryption mode
    (plaintext checkpoint found while encryption is enabled, malformed
    encrypted frame, ...).  The on-disk file is never modified."""


class CheckpointDecryptError(CheckpointCryptoError):
    """Authentication failed (wrong key, corrupted or tampered frame)."""


def load_encryption_key(env: dict | None = None) -> bytes | None:
    """Return the 32-byte key from :data:`KEY_ENV_VAR`, or ``None`` if the
    variable is unset / empty (encryption disabled).

    Raises :class:`CheckpointKeyError` if a key is configured but it is not
    valid Base64 or does not decode to exactly 32 bytes.
    """
    source = os.environ if env is None else env
    raw = source.get(KEY_ENV_VAR, "")
    if raw is None or raw.strip() == "":
        return None
    try:
        key = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CheckpointKeyError(
            f"{KEY_ENV_VAR} must be valid Base64 encoding of exactly "
            f"{KEY_LENGTH_BYTES} bytes: {exc}") from exc
    if len(key) != KEY_LENGTH_BYTES:
        raise CheckpointKeyError(
            f"{KEY_ENV_VAR} must decode to exactly {KEY_LENGTH_BYTES} bytes "
            f"(got {len(key)} bytes)")
    return key


def is_encrypted_blob(blob: bytes) -> bool:
    """True if the bytes carry the encrypted-frame magic header."""
    return blob[: len(MAGIC)] == MAGIC


def encrypt_blob(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt one checkpoint blob with a fresh random nonce."""
    if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_LENGTH_BYTES:
        raise CheckpointKeyError(
            f"encryption requires a {KEY_LENGTH_BYTES}-byte key")
    nonce = os.urandom(NONCE_LENGTH_BYTES)
    aesgcm = AESGCM(bytes(key))
    # AESGCM.encrypt appends the 16-byte tag to the ciphertext.  The magic is
    # bound as AAD so a frame cannot be silently transplanted/relabelled.
    ct_and_tag = aesgcm.encrypt(nonce, plaintext, MAGIC)
    return MAGIC + nonce + ct_and_tag


def decrypt_blob(blob: bytes, key: bytes | None) -> bytes:
    """Authenticate and decrypt an encrypted frame.

    Raises :class:`CheckpointFormatError` for a missing/invalid header and
    :class:`CheckpointKeyError` / :class:`CheckpointDecryptError` for a missing
    key or a failed GCM authentication (wrong key, corruption, tampering).
    """
    if not is_encrypted_blob(blob):
        raise CheckpointFormatError(
            "checkpoint is a plaintext (unencrypted) file but encryption is "
            "enabled; refusing to load or overwrite it")
    if len(blob) < HEADER_LENGTH:
        raise CheckpointFormatError(
            f"encrypted checkpoint is truncated ({len(blob)} bytes; header "
            f"needs {HEADER_LENGTH})")
    if key is None:
        raise CheckpointKeyError(
            f"checkpoint is encrypted but no key is set in "
            f"{KEY_ENV_VAR}; refusing to load")
    if len(key) != KEY_LENGTH_BYTES:
        raise CheckpointKeyError(
            f"{KEY_ENV_VAR} must decode to exactly {KEY_LENGTH_BYTES} bytes")
    nonce = blob[len(MAGIC): len(MAGIC) + NONCE_LENGTH_BYTES]
    ct_and_tag = blob[len(MAGIC) + NONCE_LENGTH_BYTES:]
    aesgcm = AESGCM(bytes(key))
    try:
        return aesgcm.decrypt(nonce, ct_and_tag, MAGIC)
    except Exception as exc:  # InvalidTag etc. -> explicit, safe failure
        raise CheckpointDecryptError(
            "failed to decrypt and authenticate checkpoint: wrong key, "
            "corrupted data or tampered file") from exc
