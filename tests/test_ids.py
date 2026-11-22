import unittest

import numpy

from dedupe import ids
from dedupe.ids import (
    RecordIDTooLongError,
    decode_id,
    decode_pair,
    encode_record_id,
    maybe_encode_record_id,
    needs_encoding,
)


class NeedsEncodingTest(unittest.TestCase):
    def test_homogeneous(self):
        assert not needs_encoding([1, 2, 3])
        assert not needs_encoding(["1", "2"])
        assert not needs_encoding(iter([1, 2]))

    def test_mixed_int_and_str(self):
        assert needs_encoding([1, "1"])
        assert needs_encoding(["1", 1])

    def test_composite(self):
        assert needs_encoding([("A", 1)])
        assert needs_encoding([("A", 1), 1])
        assert needs_encoding([("A", 1), ("B", "1")])

    def test_invalid(self):
        for bad in (1.5, None, b"1", True):
            with self.assertRaises(TypeError):
                needs_encoding([bad])


class EncodeDecodeTest(unittest.TestCase):
    def test_composite_round_trip(self):
        cases = [
            ("A", 1),
            ("A", "1"),
            (7, "x:t:1:z"),
            ("", ""),
            ("orders", 123456),
            ("src", "a" * 200),
        ]
        for record_id in cases:
            encoded = encode_record_id(record_id)
            assert isinstance(encoded, str)
            assert decode_id(encoded) == record_id, record_id
            # types are restored, not just values
            decoded = decode_id(encoded)
            assert type(decoded[0]) is type(record_id[0])
            assert type(decoded[1]) is type(record_id[1])

    def test_distinct_identities_never_collide(self):
        # same bare id, different source
        assert encode_record_id(("A", 1)) != encode_record_id(("B", 1))
        # int vs str original id
        assert encode_record_id(("A", 1)) != encode_record_id(("A", "1"))
        # tagged scalars distinguish int 1 from str "1"
        assert encode_record_id(1) != encode_record_id("1")
        assert decode_id(encode_record_id(1)) == 1
        assert decode_id(encode_record_id("1")) == "1"

    def test_deterministic(self):
        assert encode_record_id(("A", 1)) == encode_record_id(("A", 1))

    def test_sorting(self):
        # within a source, order follows the original value
        assert encode_record_id(("A", 1)) < encode_record_id(("A", 10))
        # source is the primary ordering key
        assert encode_record_id(("A", 9)) < encode_record_id(("B", 1))

    def test_maybe_encode(self):
        assert maybe_encode_record_id(1, False) == 1
        assert maybe_encode_record_id("1", False) == "1"
        assert maybe_encode_record_id(1, True) == "i:1"
        assert maybe_encode_record_id("1", True) == "s:1:1"

    def test_decode_passthrough_and_idempotent(self):
        assert decode_id(5) == 5
        assert decode_id("5") == "5"
        assert decode_id(("A", 1)) == ("A", 1)
        once = decode_id(encode_record_id(("A", 1)))
        assert decode_id(once) == once
        # numpy integral and string scalars
        assert decode_id(numpy.int64(5)) == 5
        assert isinstance(decode_id(numpy.str_("5")), str)

    def test_decode_pair_numpy_row(self):
        encoded = numpy.array(
            [encode_record_id(("A", 1)), encode_record_id(("A", 10))],
            dtype="U256",
        )
        assert decode_pair(encoded) == (("A", 1), ("A", 10))

    def test_too_long(self):
        with self.assertRaises(RecordIDTooLongError):
            encode_record_id(("A", "x" * 300))

    def test_invalid_ids(self):
        for bad in (1.5, None, b"1", True):
            with self.assertRaises(TypeError):
                encode_record_id(bad)

    def test_wrong_tuple_length(self):
        with self.assertRaises(ValueError):
            encode_record_id(("A", 1, 2))


if __name__ == "__main__":
    unittest.main()
