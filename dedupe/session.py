"""Persist the state of an interactive, manual labeling session.

The active-learning API is stateful: pairs are drawn from a sampled pool,
shown to a person, and then labeled.  If the process driving the labeling
is interrupted, both pieces of state are at risk: the labels that were
already supplied, and the pairs that were shown but not yet answered.

This module implements an append-only session document that makes it
possible to resume both.  It deliberately does not change the format of
files written by :func:`dedupe.write_training`; training files remain
readable, and a session is a separate artifact.

A session binds itself to the exact input data it was created against by
storing a digest of the record ids and record values, and by keying every
pair on the stable record ids of both sides (or on a canonical digest of
the record contents when ids are not available).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import dedupe.serializer as serializer

if TYPE_CHECKING:
    from typing import Iterable, Optional, TextIO, Union

    SessionFile = Union[str, "os.PathLike[str]", TextIO]

logger = logging.getLogger(__name__)

SESSION_TYPE = "dedupe-labeling-session"
SESSION_VERSION = 1

LABELS = ("match", "distinct", "unsure")


class SessionFileLoadingException(ValueError):
    """The session file could not be understood."""


class SessionMismatchError(ValueError):
    """The session was created against a different input or workflow."""


class SessionConflictWarning(UserWarning):
    """Two contradictory labels were supplied for the same record pair."""


class SessionDataWarning(UserWarning):
    """A stored session entry could not be matched to the current input."""


# ---------------------------------------------------------------------------
# stable identifiers
# ---------------------------------------------------------------------------


def _id_sort_key(value: Any) -> tuple:
    """A total ordering for record ids of mixed types."""
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, str):
        return (2, value)
    if isinstance(value, bytes):
        return (3, value)
    return (4, repr(value))


def _id_tag(value: Any) -> Any:
    """A JSON-friendly representation that keeps ints and strs distinct."""
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", list(value)]
    return ["repr", repr(value)]


def pair_key(ids: tuple, mode: str) -> str:
    """Return a stable key for a pair of record ids.

    Dedupe pairs are unordered, while link pairs are directional because
    the two sides come from different datasets.
    """
    tags = [_id_tag(ids[0]), _id_tag(ids[1])]
    if mode != "link":
        tags.sort(key=_dumps)
    return json.dumps(["ids", tags], ensure_ascii=True, separators=(",", ":"))


def _normalize(value: Any) -> Any:
    """Recursively normalize a value for canonical JSON.

    Sets are sorted so that a digest does not depend on hash randomisation,
    and tuples are distinguished from lists.
    """
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize(item) for item in value]
        normalized.sort(key=_dumps)
        return {"__set__": normalized}
    if isinstance(value, tuple):
        return {"__tuple__": [_normalize(item) for item in value]}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__repr__": repr(value)}


def _dumps(value: Any) -> str:
    return json.dumps(
        _normalize(value), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )


def record_content_key(record: Mapping) -> str:
    """A content digest for a record, used when a record has no stable id."""
    return _dumps(record)


def pair_content_key(record_pair: Sequence[Mapping], mode: str) -> str:
    """A content digest for a pair, used when ids cannot be resolved."""
    keys = [record_content_key(record_pair[0]), record_content_key(record_pair[1])]
    if mode != "link":
        keys.sort()
    return json.dumps(["content", keys], ensure_ascii=True, separators=(",", ":"))


def data_fingerprint(
    datasets: Sequence[Mapping], schema: Optional[Sequence[str]] = None
) -> dict:
    """Fingerprint the exact input data a session is bound to.

    The fingerprint includes record ids, record values and (optionally) the
    ordered field names, so changing either the membership or the contents
    of the input is detected when the session is resumed.
    """
    digest = hashlib.sha256()
    digest.update(SESSION_TYPE.encode("utf-8"))
    if schema is not None:
        digest.update(json.dumps(list(schema), ensure_ascii=True).encode("utf-8"))
    digest.update(str(len(datasets)).encode("utf-8"))

    n_records = []
    for side, data in enumerate(datasets):
        items = sorted(data.items(), key=lambda item: _id_sort_key(item[0]))
        n_records.append(len(items))
        digest.update(b"\x00side:")
        digest.update(str(side).encode("utf-8"))
        for record_id, record in items:
            tag = json.dumps(_id_tag(record_id), sort_keys=True)
            digest.update(tag.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(record_content_key(record).encode("utf-8"))
            digest.update(b"\n")

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "n_records": n_records,
    }


# ---------------------------------------------------------------------------
# resolving record ids
# ---------------------------------------------------------------------------


class RecordResolver:
    """Resolve record dictionaries back to the ids they were supplied under.

    Interactive labels are keyed on record ids so that a session survives
    changes in sampling order.  Records can reach us as the original dict
    objects (the common case) or as equal-but-distinct dicts supplied by a
    caller, so both object identity and record content are considered.
    Content matching is only used when it is unambiguous.
    """

    def __init__(
        self, datasets: Sequence[Mapping], schema: Optional[Sequence[str]] = None
    ) -> None:
        self.datasets = [dict(data) for data in datasets]
        self.schema = list(schema) if schema is not None else None
        self._by_content: list[dict] = []
        self._by_object: dict = {}

        for side, data in enumerate(self.datasets):
            content_map: dict = {}
            for record_id, record in data.items():
                content_map.setdefault(record_content_key(record), []).append(record_id)
                self._by_object.setdefault(id(record), []).append((side, record_id))
            self._by_content.append(content_map)

    def _id_on_side(self, record: Mapping, side: int) -> Optional[Any]:
        side_ids = {
            record_id
            for entry_side, record_id in self._by_object.get(id(record), ())
            if entry_side == side
        }
        if len(side_ids) == 1:
            return side_ids.pop()
        if side_ids:
            return None

        ids = self._by_content[side].get(record_content_key(record), [])
        if len(ids) == 1:
            return ids[0]
        return None

    def _id_any_side(self, record: Mapping) -> Optional[Any]:
        by_object = self._by_object.get(id(record), ())
        object_ids = {record_id for _, record_id in by_object}
        if len(object_ids) == 1:
            return object_ids.pop()
        if object_ids:
            # The same dict object is registered under several ids; don't guess.
            return None

        found = set()
        for content_map in self._by_content:
            found.update(content_map.get(record_content_key(record), []))
        if len(found) == 1:
            return found.pop()
        return None

    def ids_for_pair(self, pair: Sequence[Mapping], mode: str) -> Optional[tuple]:
        if mode == "link" and len(self.datasets) == 2:
            first = self._id_on_side(pair[0], 0)
            second = self._id_on_side(pair[1], 1)
        else:
            first = self._id_any_side(pair[0])
            second = self._id_any_side(pair[1])
        if first is None or second is None:
            return None
        return (first, second)

    def key_for_pair(self, pair: Sequence[Mapping], mode: str) -> str:
        ids = self.ids_for_pair(pair, mode)
        if ids is not None:
            return pair_key(ids, mode)
        return pair_content_key(pair, mode)

    def records_for_ids(self, ids: Sequence[Any], mode: str) -> Optional[tuple]:
        if mode == "link" and len(self.datasets) == 2:
            datasets = (self.datasets[0], self.datasets[1])
        else:
            datasets = (self.datasets[0], self.datasets[0])
        try:
            return (datasets[0][ids[0]], datasets[1][ids[1]])
        except (KeyError, IndexError, TypeError):
            return None


# ---------------------------------------------------------------------------
# record encoding
# ---------------------------------------------------------------------------


def encode_records(records: Sequence[Mapping]) -> Optional[list]:
    """Encode record dicts into a JSON-friendly structure.

    Returns ``None`` when the records cannot be serialized, in which case
    the caller should rely on stable record ids instead.
    """
    try:
        return json.loads(json.dumps(list(records), cls=serializer.TupleEncoder))
    except (TypeError, ValueError):
        return None


def decode_records(value: Any) -> Optional[tuple]:
    """Inverse of :func:`encode_records`."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (_decode(value[0]), _decode(value[1]))


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__class__") == "tuple":
            return tuple(_decode(item) for item in value.get("__value__", []))
        if value.get("__class__") == "frozenset":
            return frozenset(_decode(item) for item in value.get("__value__", []))
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------


class LabelingSession:
    """An append-only log of labels plus the queue of shown-but-unlabeled pairs.

    The document carries:

    ``input``
        The workflow mode and a fingerprint of the data the session was
        started against.
    ``pending``
        Ordered pairs that were shown to the user but not yet labeled.
    ``events``
        Every label submission, in order.  ``previous`` records the label
        it replaced, which makes relabeling traceable.
    """

    def __init__(self, source: Any, mode: str) -> None:
        self.source = source
        self.mode = mode
        self.pending: list[dict] = []
        self.events: list[dict] = []
        self.latest: dict[str, dict] = {}
        self.conflicts: list[dict] = []
        self.input: dict = {"mode": mode}
        self.created = self._now()
        self.dirty = False

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # -- construction -------------------------------------------------

    @classmethod
    def open(
        cls,
        session_file: SessionFile,
        mode: str,
        fingerprint: Optional[dict] = None,
    ) -> "LabelingSession":
        raw, source = _read_source(session_file)
        if raw is None:
            session = cls(source, mode)
            if fingerprint is not None:
                session.input["fingerprint"] = fingerprint
                session.dirty = True
            return session

        if not isinstance(raw, dict):
            raise SessionFileLoadingException(
                "A labeling session file must contain a JSON object."
            )

        if raw.get("type") != SESSION_TYPE:
            if "match" in raw or "distinct" in raw:
                raise SessionFileLoadingException(
                    "This looks like a training file, not a labeling session. "
                    "Pass training files to `prepare_training(training_file=...)`."
                )
            raise SessionFileLoadingException(
                "This is not a dedupe labeling session file."
            )

        version = raw.get("version")
        if version != SESSION_VERSION:
            raise SessionFileLoadingException(
                f"Unsupported labeling session version: {version!r}."
            )

        session = cls(source, mode)
        session.input = raw.get("input") or {"mode": mode}
        session.created = raw.get("created", session.created)
        session.pending = list(raw.get("pending", []))
        session.events = list(raw.get("events", []))
        session._load_derived()
        session._check_input(mode, fingerprint)
        return session

    def _load_derived(self) -> None:
        self.latest = {}
        self.conflicts = []
        for event in self.events:
            key = event.get("pair")
            label = event.get("label")
            if key is None or label not in LABELS:
                continue
            self.latest[key] = event
            if event.get("conflict"):
                self.conflicts.append(
                    {
                        "pair": key,
                        "previous": event.get("previous"),
                        "label": label,
                        "seq": event.get("seq"),
                    }
                )

    def _check_input(self, mode: str, fingerprint: Optional[dict]) -> None:
        stored_mode = self.input.get("mode")
        if stored_mode and stored_mode != mode:
            raise SessionMismatchError(
                "This labeling session was created for "
                f"{stored_mode!r} matching but is being resumed for {mode!r}."
            )

        stored = self.input.get("fingerprint")
        if fingerprint is None:
            if stored is not None:
                warnings.warn(
                    "The labeling session is bound to an input dataset, but the "
                    "current matcher has no input data, so the session's record "
                    "ids cannot be verified.",
                    SessionDataWarning,
                )
            return

        if stored is None:
            # A minimal session: adopt the current input.  There is nothing
            # older to compare against.
            self.input["fingerprint"] = fingerprint
            self.dirty = True
            return

        if stored.get("digest") != fingerprint.get("digest"):
            raise SessionMismatchError(
                "This labeling session was created against different input "
                "data or fields. Resuming it would silently apply labels to "
                "the wrong records; use a new session file or restore the "
                "original data."
            )

    # -- persistence --------------------------------------------------

    def save(self) -> None:
        document = {
            "type": SESSION_TYPE,
            "version": SESSION_VERSION,
            "created": self.created,
            "updated": self._now(),
            "input": self.input,
            "pending": self.pending,
            "events": self.events,
        }
        _write_source(self.source, document)
        self.dirty = False

    # -- pending pairs ------------------------------------------------

    def find_pending(self, key: str) -> Optional[dict]:
        for entry in self.pending:
            if entry.get("pair") == key:
                return entry
        return None

    def add_pending(
        self, key: str, ids: Optional[tuple], records: Optional[Sequence[Mapping]]
    ) -> None:
        if key in self.latest or self.find_pending(key) is not None:
            return
        entry: dict = {
            "pair": key,
            "ids": _ids_to_json(ids),
            "seq": len(self.pending) + 1,
        }
        # Without stable ids, store the records themselves so the pair can
        # still be reconstructed on resume. With ids, the input data is the
        # source of truth and the file stays small.
        if ids is None and records is not None:
            encoded = encode_records(records)
            if encoded is not None:
                entry["records"] = encoded
        self.pending.append(entry)
        self.dirty = True

    def remove_pending(self, key: str) -> None:
        kept = [entry for entry in self.pending if entry.get("pair") != key]
        if len(kept) != len(self.pending):
            self.pending = kept
            self.dirty = True

    # -- events -------------------------------------------------------

    def record_label(
        self,
        key: str,
        ids: Optional[tuple],
        label: str,
        records: Optional[Sequence[Mapping]] = None,
        warn: bool = True,
    ) -> str:
        """Record one label submission.

        Returns ``"duplicate"`` when the pair already has this label (the
        operation is idempotent), ``"new"`` for a first label and
        ``"changed"`` when an existing label was replaced.  A contradictory
        label emits :class:`SessionConflictWarning` unless ``warn`` is false;
        either way, both submissions are retained in the event log.
        """
        if label not in LABELS:
            raise ValueError(f"Unknown label: {label!r}")

        previous_event = self.latest.get(key)
        previous = previous_event.get("label") if previous_event is not None else None

        if previous == label:
            self.remove_pending(key)
            return "duplicate"

        conflict = {previous, label} == {"match", "distinct"}
        event: dict = {
            "seq": len(self.events) + 1,
            "pair": key,
            "ids": _ids_to_json(ids),
            "label": label,
            "previous": previous,
            "time": self._now(),
            "conflict": conflict,
        }
        if ids is None and records is not None:
            encoded = encode_records(records)
            if encoded is not None:
                event["records"] = encoded

        self.events.append(event)
        self.latest[key] = event
        self.remove_pending(key)
        self.dirty = True

        if conflict:
            self.conflicts.append(
                {
                    "pair": key,
                    "previous": previous,
                    "label": label,
                    "seq": event["seq"],
                }
            )
            message = (
                f"Conflicting labels for record pair {_display_pair(ids, key)}: "
                f"{previous!r} was already recorded and is now {label!r}. "
                "Both submissions are kept in the session log."
            )
            logger.warning(message)
            if warn:
                warnings.warn(message, SessionConflictWarning)

        return "new" if previous is None else "changed"

    def resolved(self) -> Iterable[dict]:
        """Latest event for each pair, in the order the latest one appeared."""
        return [
            event for event in self.events if self.latest.get(event["pair"]) is event
        ]

    def history(self, key: str) -> list[dict]:
        """All label submissions for a pair, oldest first."""
        return [event for event in self.events if event.get("pair") == key]


def _ids_to_json(ids: Optional[tuple]) -> Optional[list]:
    return list(ids) if ids is not None else None


def _display_pair(ids: Optional[tuple], key: str) -> str:
    return repr(ids) if ids is not None else key


# ---------------------------------------------------------------------------
# source handling
# ---------------------------------------------------------------------------


def _normalize_path(value: Any) -> str:
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    return os.path.abspath(os.fspath(value))


def _read_source(session_file: Any) -> tuple[Any, Any]:
    if isinstance(session_file, (str, bytes, os.PathLike)):
        path = _normalize_path(session_file)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except FileNotFoundError:
            return None, path
        except UnicodeDecodeError as exc:
            raise SessionFileLoadingException(
                f"The labeling session {path!r} is not valid UTF-8 text."
            ) from exc

        if not text.strip():
            return None, path
        return _parse(text, path), path

    try:
        session_file.seek(0)
        text = session_file.read()
    except (AttributeError, OSError, UnicodeDecodeError) as exc:
        raise SessionFileLoadingException(
            "A session file object must be readable and seekable."
        ) from exc

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SessionFileLoadingException(
                "The session file object is not valid UTF-8 text."
            ) from exc

    if not text or not text.strip():
        return None, session_file
    return _parse(text, "session file object"), session_file


def _parse(text: str, origin: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise SessionFileLoadingException(
            f"Could not read the labeling session {origin!r}: {exc}"
        ) from exc


def _write_source(source: Any, document: Mapping) -> None:
    text = json.dumps(document, ensure_ascii=True, sort_keys=True)

    if not isinstance(source, (str, bytes, os.PathLike)):
        source.seek(0)
        source.truncate()
        source.write(text)
        try:
            source.flush()
        except (AttributeError, OSError):
            pass
        return

    path = _normalize_path(source)
    directory = os.path.dirname(path) or "."
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def open_session(
    session_file: SessionFile,
    mode: str,
    fingerprint: Optional[dict] = None,
) -> LabelingSession:
    """Open an existing session, or create one bound to ``fingerprint``."""
    return LabelingSession.open(session_file, mode, fingerprint)


def source_matches(source: Any, session_file: Any) -> bool:
    """Whether ``session_file`` refers to the same session as ``source``."""
    if isinstance(source, (str, bytes, os.PathLike)):
        return isinstance(session_file, (str, bytes, os.PathLike)) and (
            _normalize_path(source) == _normalize_path(session_file)
        )
    return source is session_file
