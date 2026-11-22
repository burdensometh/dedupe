"""The comparison-feature contract for fields declared as possibly missing.

A field declared with ``has_missing=True`` may show up in a record in three
different ways, and those ways are *not* interchangeable:

``NOT_PROVIDED``
    The record does not carry the field at all (the key is absent). This is
    the common case for a source whose schema has no such field.

``EMPTY``
    The record carries the field, but its value is blank (``None``, ``""``,
    an empty set, an empty sequence, ...). No value was invented to stand in
    for the empty value.

``COMPARABLE``
    The record carries a real value that can be fed to the field comparator.

A comparison feature for such a field has to be able to tell those three
cases apart, and it has to keep them apart from a genuine comparison whose
value happens to be zero.  This module defines the small vocabulary that the
:class:`dedupe.datamodel.DataModel` uses to encode that contract.  Interaction
terms that contain a declared-missing field use the same vocabulary, combining
the states of their constituent fields.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from dedupe._typing import RecordDict


#: Bumped whenever the layout of the missing-data comparison features changes.
#: Models pickled before this version keep using the legacy contract (a single
#: "both present" indicator per declared field).
CONTRACT_VERSION = 3


class MissingState(IntEnum):
    """How one record presents a field declared as possibly missing.

    The values are ordered by severity so that the state of a pair, or the
    state of an interaction, is simply the most severe state of its parts.
    """

    NOT_PROVIDED = 0
    EMPTY = 1
    COMPARABLE = 2


#: Sentinel meaning "the record does not carry this field at all".
ABSENT = object()


def is_empty_value(value: Any) -> bool:
    """Is ``value`` a present-but-blank value?

    ``None`` counts as blank: the key was provided, but it carries no value.
    Numeric zeroes are *not* blank: a price of ``0`` is a value, not a blank.
    """
    if value is None:
        return True
    if isinstance(value, (str, bytes, bytearray, list, tuple, set, frozenset, dict)):
        return len(value) == 0
    return False


def value_state(value: Any) -> MissingState:
    """Classify a single record's value for a declared-missing field."""
    if value is ABSENT:
        return MissingState.NOT_PROVIDED
    if is_empty_value(value):
        return MissingState.EMPTY
    return MissingState.COMPARABLE


def record_state(record: RecordDict, field: str) -> MissingState:
    """Classify how ``record`` presents ``field``."""
    try:
        value = record[field]
    except KeyError:
        return MissingState.NOT_PROVIDED
    return value_state(value)


def record_value(record: RecordDict, field: str, default: Any = None) -> Any:
    """Read ``field`` from ``record``, treating an omitted key as missing.

    Blocking, indexing, and display code reads record fields directly. A field
    that the variable definition declared as possibly missing may be omitted
    from a record entirely, so those readers must not assume the key exists.
    """
    try:
        return record[field]
    except KeyError:
        return default


#: The canonical, unordered pair states encoded by one comparison-feature block.
#: Each block is a one-hot vector over this list, so a pair whose real text
#: comparison happens to be zero is never confused with a pair that is absent
#: or blank: they land in different slots.
PAIR_STATES: Tuple[Tuple[MissingState, MissingState], ...] = (
    (MissingState.NOT_PROVIDED, MissingState.NOT_PROVIDED),
    (MissingState.NOT_PROVIDED, MissingState.EMPTY),
    (MissingState.NOT_PROVIDED, MissingState.COMPARABLE),
    (MissingState.EMPTY, MissingState.EMPTY),
    (MissingState.EMPTY, MissingState.COMPARABLE),
    (MissingState.COMPARABLE, MissingState.COMPARABLE),
)

#: Number of comparison features contributed by one declared-missing unit.
PAIR_FEATURE_WIDTH = len(PAIR_STATES)

_PAIR_STATE_INDEX = {state: i for i, state in enumerate(PAIR_STATES)}


def canonical_pair(
    state_1: MissingState, state_2: MissingState
) -> Tuple[MissingState, MissingState]:
    """Order the two sides of a pair deterministically."""
    if state_1 <= state_2:
        return (state_1, state_2)
    return (state_2, state_1)


def pair_state_index(state_1: MissingState, state_2: MissingState) -> int:
    """Index of the one-hot feature for this pair of states."""
    return _PAIR_STATE_INDEX[canonical_pair(state_1, state_2)]


def pair_features(state_1: MissingState, state_2: MissingState) -> list[int]:
    """The one-hot comparison features for a pair of states."""
    features = [0] * PAIR_FEATURE_WIDTH
    features[pair_state_index(state_1, state_2)] = 1
    return features

