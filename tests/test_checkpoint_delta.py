import pytest

from watching.checkpoint_delta import decode, encode


def test_multiblock_round_trip_and_wrong_base_rejected(tmp_path):
    base, target = tmp_path / "base", tmp_path / "target"
    base.write_bytes(bytes(range(256)) * 5000)
    target.write_bytes(b"changed" + base.read_bytes()[7:-3] + b"end")
    delta = tmp_path / "delta"
    encode(base, target, delta)
    restored = tmp_path / "restored"
    decode(base, delta, restored)
    assert restored.read_bytes() == target.read_bytes()
    base.write_bytes(b"x" + base.read_bytes()[1:])
    invalid = tmp_path / "invalid"
    with pytest.raises(ValueError, match="checksum"):
        decode(base, delta, invalid)
    assert not invalid.exists()
    assert not (tmp_path / "invalid.partial").exists()


def test_different_checkpoint_lengths_rejected(tmp_path):
    base, target = tmp_path / "base", tmp_path / "target"
    base.write_bytes(b"abc")
    target.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="lengths"):
        encode(base, target, tmp_path / "delta")
