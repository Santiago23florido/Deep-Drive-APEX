import numpy as np
import pytest

from pose_dataset.blobs import decode_mask, decode_outcome, decode_ranges, encode_mask, encode_outcome, encode_ranges


def test_ranges_roundtrip_raw_and_zlib():
    r = np.array([1.0, 2.5, np.inf, -np.inf, 0.123456789], dtype=np.float64)
    for compress in (False, True):
        blob, enc = encode_ranges(r, compress=compress)
        out = decode_ranges(blob, enc, beam_count=5)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, r.astype(np.float32))
    blob, enc = encode_ranges(r)
    assert enc == "f32le" and len(blob) == 4 * 5


def test_ranges_are_little_endian_float32():
    blob, _ = encode_ranges(np.array([1.0]))
    assert blob == b"\x00\x00\x80\x3f"


def test_nan_rejected():
    with pytest.raises(ValueError):
        encode_ranges(np.array([1.0, np.nan]))


def test_mask_packbits_little_bitorder():
    mask = np.zeros(360, dtype=bool)
    mask[[0, 9, 359]] = True
    blob = encode_mask(mask)
    assert len(blob) == 45
    assert blob[0] == 0b00000001 and blob[1] == 0b00000010
    np.testing.assert_array_equal(decode_mask(blob, 360), mask)
    odd = np.ones(13, dtype=bool)
    np.testing.assert_array_equal(decode_mask(encode_mask(odd), 13), odd)


def test_outcome_roundtrip():
    o = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    np.testing.assert_array_equal(decode_outcome(encode_outcome(o)), o)
