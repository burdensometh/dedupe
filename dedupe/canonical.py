from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy
import numpy.typing
from affinegap import normalizedAffineGapDistance as affine

from dedupe._typing import Comparator, RecordDict

# Strategies for picking the canonical value of one field out of the
# non-empty values supplied by the members of a duplicate cluster.
CENTROID = "centroid"
AUTHORITY = "authority"
LATEST = "latest"

Strategy = str

# Sentinel used when no member supplies a usable value. Callers may
# override it, e.g. with None, through the ``missing_marker`` argument.
DEFAULT_MISSING_MARKER = ""


class CanonicalRecord(dict):
    """
    The canonical (merged) representation of a duplicate cluster.

    Behaves exactly like an ordinary record dictionary, so it can be
    handed straight to :meth:`dedupe.Gazetteer.index` and used as a
    Gazetteer index: new queries are blocked and compared against the
    field values that were actually selected.

    The ``provenance`` attribute records, for every field, which member
    of the cluster the selected value came from. When an ``id_field``
    was supplied the provenance values are member ids, otherwise they
    are the positions of the members in the input sequence (use an
    ``id_field`` if you need provenance that is independent of input
    ordering). Fields that were filled with a missing or conflict
    marker have provenance ``None``.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> None:
        super().__init__(data)
        self.provenance: dict = dict(provenance)

    def __repr__(self) -> str:
        return (
            f"CanonicalRecord({dict.__repr__(self)}, "
            f"provenance={self.provenance!r})"
        )


@dataclass(frozen=True)
class FieldRule:
    """
    Explicit priority rule for selecting the canonical value of one
    field.

    Args:
        field: name of the record field the rule applies to.
        strategy: how to choose among the non-empty member values:

            ``"authority"``
                take the value from the most trusted source. Sources
                are ranked by ``source_priority`` (highest rank first);
                sources not listed there rank below every listed source.
                Ties between equally trusted sources are broken by the
                most recent effective date and then deterministically
                by the value itself, so the result never depends on the
                order of the input records.
            ``"latest"``
                take the value with the most recent effective date.
                Ties are broken deterministically by the value itself.
            ``"centroid"`` (default)
                the medoid of the values under the affine-gap
                comparator, i.e. the value with the smallest average
                string distance to the others. Ties are broken
                deterministically by the value itself.
        source_field: name of the field holding each record's source
            identifier. Only used by the ``"authority"`` strategy.
        source_priority: ordered mapping of source identifiers to
            ranks; larger numbers are more authoritative. A source that
            is absent from the mapping is treated as rank
            ``-infinity``.
        date_field: name of the field holding each record's effective
            date. Used by the ``"latest"`` strategy and to break
            authority ties. Values may be ``datetime.date`` /
            ``datetime.datetime`` objects or ISO-8601 strings.
    """

    field: str
    strategy: Strategy = CENTROID
    source_field: Optional[str] = None
    source_priority: Mapping[Any, int] = field(default_factory=dict)
    date_field: Optional[str] = None

    def __post_init__(self) -> None:
        if self.strategy not in (CENTROID, AUTHORITY, LATEST):
            raise ValueError(
                f"Unknown canonicalization strategy {self.strategy!r} "
                f"for field {self.field!r}; expected one of "
                f"{CENTROID!r}, {AUTHORITY!r}, {LATEST!r}"
            )
        if self.strategy == AUTHORITY and not self.source_field:
            raise ValueError(
                f"FieldRule for {self.field!r} uses the "
                f"{AUTHORITY!r} strategy but no source_field was given"
            )


# A rule may be passed in as a ready-made FieldRule or as a plain
# mapping of its constructor arguments.
FieldRuleLike = Union[FieldRule, Mapping[str, Any]]


def getCentroid(attribute_variants: Sequence[str], comparator: Comparator) -> str:
    """
    Takes in a list of attribute values for a field,
    evaluates the centroid using the comparator,
    & returns the centroid (i.e. the 'best' value for the field)
    """

    n = len(attribute_variants)

    distance_matrix = numpy.zeros([n, n])

    # populate distance matrix by looping through elements of matrix triangle
    for i in range(0, n):
        for j in range(0, i):
            distance = comparator(attribute_variants[i], attribute_variants[j])
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    average_distance = distance_matrix.mean(0)

    # there can be ties for minimum, average distance string
    min_dist_indices: numpy.typing.NDArray[numpy.int_]
    min_dist_indices = numpy.where(average_distance == average_distance.min())[0]

    if len(min_dist_indices) > 1:
        centroid = breakCentroidTie(attribute_variants, min_dist_indices)
    else:
        centroid_index = min_dist_indices[0]
        centroid = attribute_variants[centroid_index]

    return centroid


def breakCentroidTie(
    attribute_variants: Sequence[str],
    min_dist_indices: numpy.typing.NDArray[numpy.int_],
) -> str:
    """
    Finds centroid when there are multiple values w/ min avg distance
    (e.g. any dupe cluster of 2). The tie is broken by the
    lexicographically smallest string so that the canonical value is
    independent of the order records are presented in.
    """
    return min(attribute_variants[i] for i in min_dist_indices)


def _coerce_rule(rule: FieldRuleLike) -> FieldRule:
    if isinstance(rule, FieldRule):
        return rule
    return FieldRule(**dict(rule))


def _parse_date(value: Any) -> Optional[datetime.date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _member_id(record: RecordDict, position: int, id_field: Optional[str]) -> Any:
    if id_field is not None:
        return record.get(id_field)
    return position


def _source_rank(source: Any, rule: FieldRule) -> float:
    # Every unlisted source is less trusted than every listed source.
    return rule.source_priority.get(source, float("-inf"))


def _tie_key(value: Any, member_id: Any) -> Tuple[str, str]:
    return (str(value), str(member_id))


def _select_authority(
    members: Sequence[Tuple[Any, Any, Any, Optional[datetime.date]]],
    rule: FieldRule,
) -> Tuple[Any, Any]:
    def rank_key(member: Tuple[Any, Any, Any, Optional[datetime.date]]):
        _, _, source, effective_date = member
        # authority rank and effective date are maximized; missing
        # dates sort first.
        return (
            _source_rank(source, rule),
            effective_date is not None,
            effective_date or datetime.date.min,
        )

    best_rank = max(rank_key(member) for member in members)
    # Fully-tied members (same source rank and effective date) are
    # resolved deterministically by the smallest value, then member id,
    # so the canonical value never depends on input order.
    member_id, value, _, _ = min(
        (member for member in members if rank_key(member) == best_rank),
        key=lambda member: _tie_key(member[1], member[0]),
    )
    return value, member_id


def _select_latest(
    members: Sequence[Tuple[Any, Any, Any, Optional[datetime.date]]],
) -> Tuple[Any, Any]:
    dated = [member for member in members if member[3] is not None]
    if dated:
        latest = max(member[3] for member in dated)
        # Same-date ties are broken deterministically.
        member_id, value, _, _ = min(
            (member for member in dated if member[3] == latest),
            key=lambda member: _tie_key(member[1], member[0]),
        )
        return value, member_id
    # No member carries a usable effective date: fall back to a
    # deterministic choice instead of silently depending on input order.
    member_id, value, _, _ = min(members, key=lambda member: str(member[1]))
    return value, member_id


def _select_centroid(
    members: Sequence[Tuple[Any, Any, Any, Optional[datetime.date]]],
) -> Tuple[Any, Any]:
    # Distances are computed on string forms, but the selected value is
    # returned as the member supplied it (so an integer stays an integer).
    values = [str(member[1]) for member in members]
    centroid = getCentroid(values, affine)
    # Several members may carry the same winning value; pick among them
    # deterministically (by member id) so input order never matters.
    winning = [
        member
        for member in members
        if str(member[1]) == centroid
    ]
    member_id, value, _, _ = min(winning, key=lambda member: str(member[0]))
    return value, member_id


def getCanonicalRep(
    record_cluster: Sequence[RecordDict],
    field_rules: Union[Sequence[FieldRuleLike], Mapping[str, FieldRuleLike], None] = None,
    id_field: Optional[str] = None,
    missing_marker: Any = DEFAULT_MISSING_MARKER,
    conflict_marker: Any = None,
) -> CanonicalRecord:
    """
    Given a list of records within a duplicate cluster, constructs a
    canonical representation of the cluster by finding canonical
    values for each field.

    By default each field is the affine-gap centroid of its non-empty
    member values. Explicit :class:`FieldRule` objects override that
    behavior per field, allowing the canonical value to be chosen by
    authoritative source, effective date, or simply by preferring any
    real non-empty value.

    Args:
        record_cluster: the member records of one duplicate cluster.
        field_rules: an optional sequence (or field-name keyed mapping)
            of :class:`FieldRule` objects / constructor argument
            mappings. Fields without a rule keep the centroid behavior.
        id_field: field whose value identifies each member (typically a
            record id). When supplied, provenance is reported with
            these ids; otherwise member positions are used.
        missing_marker: value written into a field for which every
            member is empty. Defaults to the empty string.
        conflict_marker: value written when a rule cannot resolve the
            field (e.g. an ``"authority"`` rule whose members carry no
            usable source, or a ``"latest"`` rule with no parseable
            date). Defaults to ``missing_marker``.

    Returns:
        A :class:`CanonicalRecord` -- a plain record dict usable
        directly as a Gazetteer index record -- whose ``provenance``
        attribute maps each field name to the member the selected value
        came from (or ``None`` for marker-filled fields).

    The result depends only on the contents of the member records, not
    on their order: every internal tie is broken deterministically.
    """
    if conflict_marker is None:
        conflict_marker = missing_marker

    rules = _coerce_rules(field_rules)

    if not record_cluster:
        raise ValueError("Cannot canonicalize an empty record cluster")

    # Union of every member's keys so fields present on only some
    # records are still represented in the canonical record.
    keys = set()
    for record in record_cluster:
        keys.update(record.keys())

    canonical_rep: dict = {}
    provenance: dict = {}

    for key in sorted(keys):
        rule = rules.get(key)

        # (member id, non-empty value, source, effective date)
        members = []
        for position, record in enumerate(record_cluster):
            value = record.get(key)
            if not value:
                # non-empty values always take precedence over empty
                continue
            source = record.get(rule.source_field) if rule is not None else None
            effective_date = (
                _parse_date(record.get(rule.date_field))
                if rule is not None
                else None
            )
            members.append(
                (
                    _member_id(record, position, id_field),
                    value,
                    source,
                    effective_date,
                )
            )

        if not members:
            canonical_rep[key] = missing_marker
            provenance[key] = None
            continue

        if rule is None or rule.strategy == CENTROID:
            canonical_rep[key], provenance[key] = _select_centroid(members)
        elif rule.strategy == AUTHORITY:
            if all(source is None for _, _, source, _ in members):
                # Declared mapping cannot be applied: no member carries
                # a source, so the conflict is surfaced rather than
                # guessed at.
                canonical_rep[key] = conflict_marker
                provenance[key] = None
            else:
                canonical_rep[key], provenance[key] = _select_authority(
                    members, rule
                )
        elif rule.strategy == LATEST:
            if all(effective_date is None for *_, effective_date in members):
                canonical_rep[key] = conflict_marker
                provenance[key] = None
            else:
                canonical_rep[key], provenance[key] = _select_latest(members)

    return CanonicalRecord(canonical_rep, provenance)


def _coerce_rules(
    field_rules: Union[Sequence[FieldRuleLike], Mapping[str, FieldRuleLike], None],
) -> dict:
    if field_rules is None:
        return {}
    if isinstance(field_rules, Mapping):
        items = field_rules.items()
    else:
        items = ((None, rule) for rule in field_rules)
    rules = {}
    for _, rule_like in items:
        rule = _coerce_rule(rule_like)
        rules[rule.field] = rule
    return rules
