#!/usr/bin/python
"""
Temporal (bi-temporal) gazetteer matching.

Reference records are organized by the business entity they describe and
by the time interval during which they were the valid version of that
entity. A query is matched at its own event time: only reference versions
that were valid at that point in time can become scoring candidates, so a
version that became valid later can never be selected over the version that
was in effect when the event happened, no matter how high its score.

Reference data are supplied as a nested mapping, keyed first by business
entity and then by reference version identity::

    {
        entity_id: {
            version_id: {
                "record": {...},          # fields used for matching
                "valid_from": datetime,   # inclusive start of validity
                "valid_to": datetime,     # exclusive end (None = open ended)
                "priority": 0,            # tie-breaker for overlaps
            },
            ...
        },
        ...
    }

Search results report both the business entity and the specific reference
version that was selected. Messy records that have no candidate valid at
their event time are returned unmatched.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Union

import dedupe.api as api
import dedupe.core as core

if TYPE_CHECKING:
    from typing import Any, Dict, Iterable, Iterator, MutableMapping, Sequence

    from dedupe._typing import (
        ArrayLinks,
        Blocks,
        Data,
        RecordDict,
        RecordID,
    )

    # A point in time may be given as a datetime/date, an ISO 8601 string,
    # or a POSIX timestamp.
    EventTime = Union[_dt.datetime, _dt.date, str, float, int]
    EventTimes = Union[EventTime, Mapping[RecordID, EventTime]]

    TemporalData = Mapping[RecordID, Mapping[RecordID, "TemporalReference"]]

    TemporalLookupResult = tuple[
        RecordID, tuple[tuple[RecordID, RecordID, float], ...]
    ]
    TemporalLookupResults = Iterable[TemporalLookupResult]


def normalize_time(value: EventTime) -> _dt.datetime:
    """Coerce a supported time value into a naive, UTC-normalized datetime.

    Naive datetimes are returned unchanged. Timezone-aware datetimes are
    converted to UTC and made naive so every value stored and compared by
    this module uses the same representation. Strings are parsed as ISO 8601
    (``Z`` suffix accepted) and numbers are treated as POSIX timestamps.
    """
    if isinstance(value, _dt.datetime):
        parsed = value
    elif isinstance(value, _dt.date):
        parsed = _dt.datetime(value.year, value.month, value.day)
    elif isinstance(value, (int, float)):
        parsed = _dt.datetime.fromtimestamp(value, tz=_dt.timezone.utc)
    elif isinstance(value, str):
        parsed = _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise TypeError(
            "times must be datetime, date, ISO 8601 strings, or timestamps, "
            f"got {type(value)!r}"
        )

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    return parsed


def _event_key(value: _dt.datetime) -> str:
    """Stable, lexicographically ordered representation of an event time."""
    return value.isoformat(sep=" ")


@dataclass
class TemporalReference:
    """One time-bounded version of a business entity.

    Args:
        record: the reference record fields used for blocking and scoring.
        valid_from: inclusive start of the interval of validity.
        valid_to: exclusive end of the interval of validity, or ``None``
            for an interval that is still open (valid indefinitely).
        priority: explicit priority used when another version of the same
            entity is valid at the same point in time; higher wins. Ties
            are broken by the earlier ``valid_from`` and then the smaller
            version id, so overlap resolution is always deterministic.
    """

    record: RecordDict
    valid_from: EventTime
    valid_to: EventTime | None = None
    priority: int = 0

    def normalized(self) -> "TemporalVersion":
        start = normalize_time(self.valid_from)
        end = normalize_time(self.valid_to) if self.valid_to is not None else None
        if end is not None and end <= start:
            raise ValueError(
                "valid_to must be later than valid_from "
                f"(got {start!r} to {end!r})"
            )
        return TemporalVersion(
            record=self.record,
            valid_from=start,
            valid_to=end,
            priority=self.priority,
        )


@dataclass
class TemporalVersion:
    """A :class:`TemporalReference` with parsed, normalized interval bounds."""

    record: RecordDict
    valid_from: _dt.datetime
    valid_to: _dt.datetime | None
    priority: int = 0

    def is_valid(self, when: _dt.datetime) -> bool:
        if when < self.valid_from:
            return False
        if self.valid_to is not None and when >= self.valid_to:
            return False
        return True


def resolve_version(
    versions: "Sequence[TemporalVersion]",
    version_ids: "Sequence[RecordID]",
    when: _dt.datetime,
) -> tuple[RecordID, TemporalVersion] | None:
    """Pick the single winning version of an entity at a point in time.

    Only versions whose half-open interval ``[valid_from, valid_to)``
    contains ``when`` are eligible. Among eligible versions the highest
    ``priority`` wins; ties are broken by the earliest ``valid_from`` and
    then the smallest version id. Returns ``None`` when no version is valid
    at ``when``.
    """
    candidates = [
        (version_id, version)
        for version_id, version in zip(version_ids, versions)
        if version.is_valid(when)
    ]
    if not candidates:
        return None

    def sort_key(item: tuple[RecordID, TemporalVersion]) -> tuple[Any, ...]:
        version_id, version = item
        return (
            -version.priority,
            version.valid_from,
            _version_sort_key(version_id),
        )

    return min(candidates, key=sort_key)


def _version_sort_key(version_id: RecordID) -> tuple[int, Any]:
    # Keep numeric ids ordered numerically and everything else by string
    # representation, while never comparing ints and strs directly.
    if isinstance(version_id, (int, float)):
        return (0, version_id)
    return (1, str(version_id))


class TemporalGazetteerMatching(api.GazetteerMatching):
    """Gazetteer matching against versioned, time-bounded reference data.

    Reference versions are indexed once and queried repeatedly. Every query
    carries an event time; candidates are drawn exclusively from versions
    valid at that event time. Overlapping intervals for the same entity are
    collapsed to a single candidate using an explicit priority rule, so each
    entity contributes at most one candidate per query and future versions
    can never out-score the version in effect at the event time.
    """

    def __init__(
        self, num_cores: int | None, in_memory: bool = False, **kwargs
    ) -> None:
        super().__init__(num_cores, in_memory, **kwargs)

        # Temporal matching opens a fresh sqlite connection per query. An
        # ordinary ":memory:" database is private to each connection, so for
        # in-memory use we switch to a shared-cache in-memory database that
        # every connection to this matcher can see. Shared-cache databases
        # vanish when their last connection closes, so hold one open for the
        # matcher's lifetime.
        if self.in_memory:
            self.db = (
                "file:dedupe-temporal-gazetteer-"
                f"{id(self)}?mode=memory&cache=shared"
            )
            self._memory_uri = True
            self._db_holder = sqlite3.connect(self.db, uri=True)
        else:
            self._memory_uri = False
            self._db_holder = None

        # version id -> normalized version (including the matchable record)
        self.versions: MutableMapping[RecordID, TemporalVersion] = {}
        # version id -> the business entity it belongs to
        self.version_entities: MutableMapping[RecordID, RecordID] = {}
        # entity id -> ordered list of its version ids
        self.entity_versions: MutableMapping[
            RecordID, list[RecordID]
        ] = {}

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db, uri=self._memory_uri)

    def _close(self) -> None:
        if self._db_holder is not None:
            self._db_holder.close()
            self._db_holder = None
        super()._close()

    def index(self, data: TemporalData) -> None:
        """Add time-bounded reference versions to the index.

        Args:
            data: a nested mapping of
                ``{entity_id: {version_id: TemporalReference}}``. Each
                version id must be globally unique; re-indexing a version id
                replaces the previously indexed version.
        """
        flat_records: Dict[RecordID, RecordDict] = {}
        new_versions: Dict[RecordID, TemporalVersion] = {}
        new_entities: Dict[RecordID, RecordID] = {}

        if not data:
            raise ValueError("no temporal reference data were given")

        for entity_id, entity_versions in data.items():
            if not entity_versions:
                raise ValueError(
                    f"entity {entity_id!r} has no reference versions"
                )
            self.entity_versions.setdefault(entity_id, [])
            for version_id, reference in entity_versions.items():
                if version_id in flat_records:
                    raise ValueError(
                        f"version id {version_id!r} is used more than once "
                        "within the data being indexed"
                    )
                version = self._coerce_reference(reference)
                if version_id in self.versions and (
                    self.version_entities.get(version_id) != entity_id
                ):
                    raise ValueError(
                        f"version id {version_id!r} already belongs to "
                        f"entity {self.version_entities[version_id]!r}, "
                        f"cannot re-index it under {entity_id!r}"
                    )

                flat_records[version_id] = version.record
                new_versions[version_id] = version
                new_entities[version_id] = entity_id
                if version_id not in self.entity_versions[entity_id]:
                    self.entity_versions[entity_id].append(version_id)

        self.fingerprinter.index_all(flat_records)

        id_type = core.sqlite_id_type({**self.indexed_data, **flat_records})

        con = self._connect()
        con.execute("pragma journal_mode=wal")
        con.execute(
            f"""CREATE TABLE IF NOT EXISTS indexed_records
                       (block_key text,
                        record_id {id_type},
                        valid_from text NOT NULL,
                        valid_to text,
                        UNIQUE(block_key, record_id))"""
        )

        con.executemany(
            "REPLACE INTO indexed_records VALUES (?, ?, ?, ?)",
            (
                (
                    block_key,
                    version_id,
                    _event_key(new_versions[version_id].valid_from),
                    (
                        _event_key(new_versions[version_id].valid_to)
                        if new_versions[version_id].valid_to is not None
                        else None
                    ),
                )
                for block_key, version_id in self.fingerprinter(
                    flat_records.items(), target=True
                )
            ),
        )

        con.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS
                       indexed_records_block_key_idx
                       ON indexed_records
                       (block_key, record_id)"""
        )
        con.execute("ANALYZE")
        con.commit()
        con.close()

        self.versions.update(new_versions)
        self.version_entities.update(new_entities)
        self.indexed_data.update(flat_records)

    @staticmethod
    def _coerce_reference(reference: Any) -> TemporalVersion:
        if isinstance(reference, TemporalReference):
            return reference.normalized()
        if isinstance(reference, TemporalVersion):
            return reference
        if isinstance(reference, Mapping):
            try:
                return TemporalReference(
                    record=reference["record"],
                    valid_from=reference["valid_from"],
                    valid_to=reference.get("valid_to"),
                    priority=reference.get("priority", 0),
                ).normalized()
            except KeyError as exc:
                raise ValueError(
                    "temporal reference entries require "
                    f"'record' and 'valid_from' keys, missing {exc!s}"
                ) from exc
        raise TypeError(
            "temporal reference entries must be TemporalReference "
            f"objects or mappings, got {type(reference)!r}"
        )

    def unindex(self, data: TemporalData) -> None:
        """Remove reference versions (and, when emptied, entities) from index.

        Args:
            data: the same nested structure passed to :meth:`index`. Only
                the entity and version ids are used; versions not currently
                indexed are ignored.
        """
        version_ids = [
            version_id
            for entity_versions in data.values()
            for version_id in entity_versions
        ]

        # Nothing has ever been indexed; nothing to remove.
        if not version_ids or not self.versions:
            return

        for field in self.fingerprinter.index_fields:
            self.fingerprinter.unindex(
                {
                    self.versions[version_id].record[field]
                    for version_id in version_ids
                    if version_id in self.versions
                    and self.versions[version_id].record[field]
                },
                field,
            )

        con = self._connect()
        con.executemany(
            "DELETE FROM indexed_records WHERE record_id = ?",
            ((version_id,) for version_id in version_ids),
        )
        con.commit()
        con.close()

        for entity_id, entity_versions in data.items():
            member_ids = self.entity_versions.get(entity_id)
            if member_ids is None:
                continue
            for version_id in entity_versions:
                self.versions.pop(version_id, None)
                self.version_entities.pop(version_id, None)
                self.indexed_data.pop(version_id, None)
                if version_id in member_ids:
                    member_ids.remove(version_id)
            if not member_ids:
                del self.entity_versions[entity_id]

    # ------------------------------------------------------------------
    # blocking at an event time
    # ------------------------------------------------------------------
    def blocks(
        self,
        data: Data,
        event_times: EventTimes | None = None,
    ) -> Blocks:
        """Yield blocks of candidate pairs valid at each record's event time.

        Each block contains one messy record paired with the single winning
        version of every entity that (a) shares a fingerprint and (b) has a
        version valid at the messy record's event time. Versions that were
        not yet valid, or had already expired, never appear, and overlapping
        intervals are resolved to one candidate per entity.

        Args:
            data: messy records keyed by record id.
            event_times: the event time to match at. A scalar applies the
                same time to every record; a mapping gives a per-record time.
        """
        when_by_id = self._event_times_by_id(data, event_times)

        if not data:
            return

        id_type = core.sqlite_id_type(data)

        # Interval semantics are half-open: valid_from <= event < valid_to.
        # Comparing the normalized ISO strings in SQL is equivalent to
        # comparing the datetimes, and keeps future/expired versions out of
        # the candidate set entirely.
        yield from self._blocks_by_event(data, when_by_id, id_type)

    def _blocks_by_event(
        self,
        data: Data,
        when_by_id: Mapping[RecordID, _dt.datetime],
        id_type: str,
    ) -> Iterator[list]:
        con = self._connect()
        con.execute("BEGIN")
        con.execute(
            f"CREATE TEMPORARY TABLE blocking_map "
            f"(block_key text, record_id {id_type})"
        )
        con.executemany(
            "INSERT INTO blocking_map VALUES (?, ?)",
            self.fingerprinter(data.items()),
        )

        winner_cache: dict[tuple[str, RecordID], RecordID | None] = {}

        try:
            for record_id in data:
                when = when_by_id[record_id]
                event_key = _event_key(when)

                rows = con.execute(
                    """SELECT DISTINCT b.record_id
                       FROM blocking_map a
                       INNER JOIN indexed_records b
                       USING (block_key)
                       WHERE a.record_id = ?
                         AND b.valid_from <= ?
                         AND (b.valid_to IS NULL OR b.valid_to > ?)""",
                    (record_id, event_key, event_key),
                )

                # Resolve overlapping intervals to one winning version per
                # entity before any scoring happens.
                chosen_versions: dict[RecordID, RecordID] = {}
                for (version_id,) in rows:
                    entity_id = self.version_entities[version_id]
                    cache_key = (event_key, entity_id)
                    winner = winner_cache.get(cache_key)
                    if winner is None and cache_key not in winner_cache:
                        member_ids = self.entity_versions[entity_id]
                        resolved = resolve_version(
                            [self.versions[v] for v in member_ids],
                            member_ids,
                            when,
                        )
                        winner = resolved[0] if resolved is not None else None
                        winner_cache[cache_key] = winner
                    if winner is not None and version_id == winner:
                        chosen_versions[entity_id] = version_id

                if chosen_versions:
                    yield [
                        (
                            (record_id, data[record_id]),
                            (version_id, self.indexed_data[version_id]),
                        )
                        for version_id in chosen_versions.values()
                    ]
        finally:
            con.execute("ROLLBACK")
            con.close()

    def _event_times_by_id(
        self, data: Data, event_times: EventTimes | None
    ) -> Dict[RecordID, _dt.datetime]:
        if event_times is None:
            raise ValueError(
                "event_times is required: every query must specify the "
                "point in time at which it should be matched"
            )

        if isinstance(event_times, Mapping):
            missing = data.keys() - event_times.keys()
            if missing:
                raise ValueError(
                    "missing event times for records: "
                    + ", ".join(repr(k) for k in sorted(missing, key=str))
                )
            extra = event_times.keys() - data.keys()
            if extra:
                raise ValueError(
                    "event times supplied for records not in the query data: "
                    + ", ".join(repr(k) for k in sorted(extra, key=str))
                )
            return {
                record_id: normalize_time(event_times[record_id])
                for record_id in data
            }

        when = normalize_time(event_times)
        return {record_id: when for record_id in data}

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def search(
        self,
        data: Data,
        event_times: EventTimes | None = None,
        threshold: float = 0.0,
        n_matches: int = 1,
        generator: bool = False,
    ):
        """Match messy records against reference versions valid at event time.

        Args:
            data: messy records keyed by record id.
            event_times: scalar event time applied to every record, or a
                mapping of record id to event time.
            threshold: minimum predicted probability for a candidate.
            n_matches: maximum matches returned per record. ``None`` returns
                every candidate above the threshold.
            generator: return a generator instead of a materialized list.

        Returns:
            ``(record_id, ((entity_id, version_id, score), ...))`` for each
            record, in the order of the input records. Records without a
            valid candidate are returned with an empty tuple.
        """
        blocks = self.blocks(data, event_times)
        pair_scores = self.score(blocks)
        search_results = self.many_to_n(pair_scores, threshold, n_matches)

        results = self._format_search_results(data, search_results)

        if generator:
            return results
        return list(results)

    def _format_search_results(
        self,
        search_d: Data,
        results: ArrayLinks,
    ) -> Iterator[TemporalLookupResult]:
        seen: set[RecordID] = set()

        for result in results:
            messy_id: RecordID | None = None
            prepared: list[tuple[RecordID, RecordID, float]] = []
            for (a, b), score in result:
                messy_id = a
                prepared.append((self.version_entities[b], b, float(score)))

            assert messy_id is not None
            yield messy_id, tuple(prepared)
            seen.add(messy_id)

        # Preserve input ordering and keep records with no candidate as
        # explicit unmatched results.
        for record_id in search_d:
            if record_id not in seen:
                yield record_id, ()


class TemporalGazetteer(api.Link, TemporalGazetteerMatching):
    """Active-learning temporal gazetteer.

    Train this exactly like a regular :class:`dedupe.Gazetteer`; the learned
    variables, classifier and predicates are time-agnostic. Time semantics
    enter at :meth:`~TemporalGazetteerMatching.index` and
    :meth:`~TemporalGazetteerMatching.search`, where reference versions are
    organized by entity and validity interval and queries are matched at
    their own event times.
    """


class StaticTemporalGazetteer(api.StaticMatching, TemporalGazetteerMatching):
    """Temporal gazetteer initialized from saved, previously trained settings."""


__all__ = (
    "TemporalGazetteer",
    "TemporalGazetteerMatching",
    "TemporalReference",
    "TemporalVersion",
    "StaticTemporalGazetteer",
    "normalize_time",
    "resolve_version",
)
