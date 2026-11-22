#!/usr/bin/python
"""
Stable identities for records that may come from different sources.

Records from different tables frequently reuse the same local ids -- two
unrelated records can both be ``1`` -- and those local ids may be a mix of
integers and strings. Storing the bare ids in a single SQLite table or a
single numpy array collapses those distinct identities: SQLite applies
integer affinity and turns the string ``"1"`` into the integer ``1``, and a
numpy integer column cannot hold strings at all.

The functions in this module build one lossless, unambiguous textual
identity out of a source and an original value, and turn it back into the
exact object the caller passed in.

Three kinds of record ids are supported:

* an integer, such as ``1``
* a string, such as ``"1"``
* a composite id, a ``(source, original_id)`` tuple, such as
  ``("orders", 1)``

Every composite id is rendered as text with a type tag so that distinct
identities never collide:

* ``1`` is stored as the integer ``1`` (unchanged)
* ``"1"`` is stored as the string ``"1"`` (unchanged)
* ``("orders", 1)`` becomes ``"t:s:6:orders:i:1"``
* ``("orders", "1")`` becomes ``"t:s:6:orders:s:1:1"``

The ``t:`` prefix marks composite encodings, which can never be produced by
a bare string id, and the length-prefixed ``s:``/tagged ``i:`` payloads make
the encoding self delimiting: no payload can be confused with a tag or a
separator, regardless of the characters it contains, and integer and string
payloads stay distinguishable. Composite encodings sort as
``(source, original_id)`` pairs -- ids sharing a source sort by their
original value -- so a record can never be ordered against itself.

Encoding is a pure function of the id, so the same id always produces the
same text and distinct ids always produce distinct text. Decoding restores
the original tuple exactly, including the types of its parts. Plain integer
and string ids need no encoding and are returned by the decoder unchanged,
which keeps all existing single-table call sites working as before.
"""
from __future__ import annotations

import numbers
from typing import Any, Tuple, Union

#: A record id as supplied by a caller.
RawRecordID = Union[int, str, "Tuple[Any, Any]"]

#: A composite record id: the source (table, file, ...) and the original id
#: within that source.
CompositeRecordID = Tuple[Union[int, str], Union[int, str]]

#: Every record id accepted by the library.
RecordIDLike = Union[int, str, CompositeRecordID]

# Width of the numpy unicode column used to carry record ids. Kept in sync
# with the (str, 256) dtype historically produced by core.sniff_id_type so
# that score files remain compatible with older callers.
MAX_ID_LENGTH = 256


class RecordIDTooLongError(ValueError):
    """Raised when an encoded record id cannot fit in the score array."""


def is_composite_id(record_id: Any) -> bool:
    """Return whether ``record_id`` is a ``(source, original_id)`` tuple."""
    return isinstance(record_id, tuple)


def needs_encoding(data_keys: Any) -> bool:
    """Whether a record collection must encode its ids.

    A homogeneous collection of plain integers or plain strings can be
    stored and compared as-is without losing information, so it returns
    ``False`` to preserve the historical single-table behavior. A
    collection that mixes integers and strings, or that contains composite
    ``(source, original_id)`` ids, must encode every id: otherwise SQLite
    integer affinity and numpy column types would merge distinct
    identities (for example, the integer ``1`` and the string ``"1"``).
    """
    seen_int = False
    seen_str = False
    for record_id in data_keys:
        if is_composite_id(record_id):
            return True
        if isinstance(record_id, numbers.Integral) and not isinstance(
            record_id, bool
        ):
            seen_int = True
        elif isinstance(record_id, str):
            seen_str = True
        else:
            raise TypeError(
                "Record ids must be integers, strings, or "
                f"(source, original_id) tuples; got {record_id!r}"
            )
        if seen_int and seen_str:
            return True

    return False


def pair_needs_encoding(record_pairs: Any) -> bool:
    """Whether the ids carried by a batch of record pairs need encoding.

    Mirrors :func:`needs_encoding` for the nested pair-of-pairs shape used
    by the scorers: returns ``True`` if any id is composite or if the batch
    mixes integer and string ids.
    """
    seen_int = False
    seen_str = False
    for pair in record_pairs:
        for record_id in pair:
            if is_composite_id(record_id):
                return True
            if isinstance(record_id, numbers.Integral) and not isinstance(
                record_id, bool
            ):
                seen_int = True
            elif isinstance(record_id, str):
                seen_str = True
            else:
                raise TypeError(
                    "Record ids must be integers, strings, or "
                    f"(source, original_id) tuples; got {record_id!r}"
                )
            if seen_int and seen_str:
                return True
    return False


def _check_scalar(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (numbers.Integral, str)):
        raise TypeError(
            "Record ids and sources must be integers or strings, "
            f"got {value!r} of type {type(value).__name__}"
        )


def _encode_scalar(value: Union[int, str]) -> str:
    _check_scalar(value)
    if isinstance(value, str):
        return f"s:{len(value)}:{value}"
    return f"i:{value}"


def _check_length(encoded: str, record_id: RecordIDLike) -> None:
    if len(encoded) > MAX_ID_LENGTH:
        raise RecordIDTooLongError(
            f"Encoded record id {record_id!r} is {len(encoded)} characters "
            f"long but the maximum is {MAX_ID_LENGTH}. Use shorter source "
            "names or record ids."
        )


def encode_record_id(record_id: RecordIDLike) -> Any:
    """Encode a record id for storage in a heterogeneous collection.

    Composite ``(source, original_id)`` ids become a ``t:`` text encoding,
    and -- because a mixed table needs every id tagged to keep the integer
    ``1`` distinct from the string ``"1"`` -- plain scalars become ``i:``
    or ``s:`` text as well. Homogeneous single-table collections call
    :func:`maybe_encode_record_id` instead, which leaves plain int and str
    ids untouched for backward compatibility.
    """
    if is_composite_id(record_id):
        if len(record_id) != 2:
            raise ValueError(
                "Composite record ids must be (source, original_id) pairs "
                f"of length 2, got {record_id!r}"
            )
        source, original_id = record_id
        encoded = (
            "t:" + _encode_scalar(source) + ":" + _encode_scalar(original_id)
        )
    else:
        encoded = _encode_scalar(record_id)

    _check_length(encoded, record_id)
    return encoded


def _as_tuple(record_id: Any) -> RecordIDLike:
    """Convert a numpy composite row back into a plain tuple if needed.

    Reading composite ids through a numpy object array can hand back a
    ``numpy.ndarray`` instead of the original tuple; the encoder only
    accepts tuples.
    """
    if isinstance(record_id, tuple):
        return record_id
    if hasattr(record_id, "tolist"):
        converted = record_id.tolist()
        if isinstance(converted, list):
            return tuple(converted)
    return record_id


def maybe_encode_record_id(record_id: RecordIDLike, encode: bool) -> Any:
    """Encode ``record_id`` when its collection requires it.

    When ``encode`` is ``False`` (a homogeneous table of plain integers or
    plain strings), the id is returned unchanged so existing single-table
    behavior is preserved. When ``True``, the id gets its full lossless
    text encoding.
    """
    if not encode:
        return record_id

    return encode_record_id(_as_tuple(record_id))


def _decode_scalar(encoded: str, marker: int) -> tuple[Union[int, str], int]:
    """Decode one scalar beginning at ``marker``.

    Returns the decoded scalar and the position immediately after it
    (the separating colon, if there is one, is skipped).
    """
    tag = encoded[marker]
    if tag == "i":
        value_start = marker + 2
        end = encoded.find(":", value_start)
        if end == -1:
            # last scalar in the encoding: the digits run to the end
            end = len(encoded)
            next_marker = end
        else:
            next_marker = end + 1
        value: Union[int, str] = int(encoded[value_start:end])
        return value, next_marker
    elif tag == "s":
        length_end = encoded.index(":", marker + 2)
        length = int(encoded[marker + 2 : length_end])
        value_start = length_end + 1
        value_end = value_start + length
        value = encoded[value_start:value_end]
        if value_end < len(encoded) and encoded[value_end] == ":":
            next_marker = value_end + 1
        else:
            next_marker = value_end
        return value, next_marker
    else:  # pragma: no cover - defensive, inputs come from encode_record_id
        raise ValueError(f"Invalid encoded record id: {encoded!r}")


def decode_id(encoded: Any) -> RecordIDLike:
    """Restore a record id produced by the storage boundary.

    The inverse of :func:`maybe_encode_record_id`:

    * composite ``t:`` encodings come back as ``(source, original_id)``
      tuples,
    * tagged ``i:`` / ``s:`` scalars come back as integers / strings,
    * untagged integers and strings (homogeneous tables that were never
      encoded) are returned unchanged,
    * an already decoded composite tuple is returned unchanged, so
      decoding is idempotent.

    A bare string that happens to start with ``i:`` or ``s:`` cannot be
    produced by the boundary encoder, since every string in a collection
    that uses encoding is tagged. Numpy's ``str_`` is a ``str`` subclass,
    so values read from score arrays decode the same way.
    """
    if isinstance(encoded, numbers.Integral) and not isinstance(encoded, bool):
        return encoded

    if is_composite_id(encoded):
        return encoded

    if not isinstance(encoded, str):
        raise TypeError(
            "Record ids must be integers, strings, or "
            "(source, original_id) tuples; got "
            f"{encoded!r} of type {type(encoded).__name__}"
        )

    tag = encoded[:2]
    if tag == "t:":
        source, marker = _decode_scalar(encoded, 2)
        original_id, _ = _decode_scalar(encoded, marker)
        return (source, original_id)
    if tag in ("i:", "s:"):
        value, _ = _decode_scalar(encoded, 0)
        return value

    return encoded


def decode_pair(encoded_pair: Any) -> tuple[RecordIDLike, RecordIDLike]:
    """Decode a pair of encoded record ids.

    Accepts a plain tuple or a numpy row (``numpy.void``/``numpy.ndarray``)
    of encoded ids.
    """
    a, b = tuple(encoded_pair)
    return decode_id(a), decode_id(b)
