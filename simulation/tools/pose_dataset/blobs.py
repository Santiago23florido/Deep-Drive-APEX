"""Binary encodings of the per-scan LiDAR arrays stored in SQLite BLOBs.

``ranges_blob`` (column ``lidar_scans.ranges_encoding`` tells which one):

* ``f32le``       raw IEEE-754 float32, little-endian, one value per beam, in
                  beam order (``angle_min + i * angle_increment``);
                  ``4 * beam_count`` bytes, no header, no compression.
* ``f32le+zlib``  the same byte string compressed with zlib (RFC 1950).

Range values follow REP-117: finite metres for returns, ``+inf`` for "no
return" (dropout or beyond ``range_max``), ``-inf`` for returns closer than
``range_min``. ``NaN`` is never stored.

``valid_mask_blob``: ``numpy.packbits(mask, bitorder="little")``; bit ``i %
8`` of byte ``i // 8`` is 1 when beam ``i`` is a valid return (finite and
within ``[range_min, range_max]``); ``ceil(beam_count / 8)`` bytes, padding
bits are 0.

``lidar_beam_truth.outcome_blob``: one ``uint8`` per beam with the true
outcome of the noise model (0 no_truth, 1 hit, 2 dropout, 3 short outlier,
4 random outlier). It is a label, never a model input.
"""

from __future__ import annotations

import zlib

import numpy as np

RANGES_F32LE = "f32le"
RANGES_F32LE_ZLIB = "f32le+zlib"
_F32LE = np.dtype("<f4")


def encode_ranges(ranges: np.ndarray, compress: bool = False) -> tuple[bytes, str]:
    raw = np.ascontiguousarray(np.asarray(ranges, dtype=_F32LE)).tobytes()
    if np.isnan(np.frombuffer(raw, dtype=_F32LE)).any():
        raise ValueError("NaN ranges are not allowed (use +inf / -inf, REP-117)")
    if compress:
        return zlib.compress(raw, 6), RANGES_F32LE_ZLIB
    return raw, RANGES_F32LE


def decode_ranges(blob: bytes, encoding: str = RANGES_F32LE, beam_count: int | None = None) -> np.ndarray:
    if encoding == RANGES_F32LE_ZLIB:
        blob = zlib.decompress(blob)
    elif encoding != RANGES_F32LE:
        raise ValueError(f"unknown ranges encoding {encoding!r}")
    out = np.frombuffer(blob, dtype=_F32LE).astype(np.float32)
    if beam_count is not None and out.size != beam_count:
        raise ValueError(f"blob has {out.size} beams, expected {beam_count}")
    return out


def encode_mask(mask: np.ndarray) -> bytes:
    return np.packbits(np.asarray(mask, dtype=bool), bitorder="little").tobytes()


def decode_mask(blob: bytes, beam_count: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")
    return bits[:beam_count].astype(bool)


def encode_outcome(outcome: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(outcome, dtype=np.uint8)).tobytes()


def decode_outcome(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.uint8).copy()
