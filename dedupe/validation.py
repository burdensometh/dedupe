"""Entity-aware training / validation splitting for dedupe models.

The ordinary way of making training data treats every pair of records that
does not share a known identifier as a *distinct* pair.  When training and
validation pairs are drawn from the same records, the two sets share
entities: the model has already seen one side of a validation pair (or a
near-identical sibling) during training, so reported generalisation is
optimistic even when no single record pair is repeated.

This module implements a stricter protocol:

1. **Resolve entity membership first.**  Records sharing a known identifier
   (and records connected by known positive pairs) are placed in the same
   entity.  Entity membership is never split.
2. **Assign whole entities to folds.**  Validation records and training
   records therefore cannot share an entity.
3. **Generate fold-specific material.**  For every fold we build real
   training pairs (within-entity matches plus sampled cross-entity distinct
   pairs), compute the training distances, and score the real candidate
   pairs that the fold model's blocking rules generate on the held-out
   records.  Match quality is measured against held-out entity identities.
4. **Never silently mislabel a cross-fold positive.**  Distinct training
   pairs are always sampled across *different* training entities, and no
   validation record appears in training.  A record-level fold assignment
   that splits an entity is rejected with :class:`EntityLeakageError`,
   which carries the offending positive pairs; :func:`audit_record_folds`
   reports them without building a plan.
5. **Keep validation inputs reproducible.**  Splits are deterministic for a
   given ``seed`` and can be persisted with :meth:`EntityPlan.save`, so the
   exact validation inputs can be replayed against a fully retrained model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy

import dedupe.core as core

if TYPE_CHECKING:
    from dedupe._typing import TrainingData


class EntityLeakageError(ValueError):
    """Raised when a fold assignment splits a known entity across folds."""

    def __init__(self, message: str, pairs: Sequence[tuple[Any, Any]] = ()) -> None:
        super().__init__(message)
        self.pairs = list(pairs)


# ---------------------------------------------------------------------------
# entity resolution
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[Any, Any] = {}

    def add(self, item: Any) -> None:
        if item not in self._parent:
            self._parent[item] = item

    def find(self, item: Any) -> Any:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: Any, b: Any) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        # deterministic root choice, independent of insertion order
        if repr(root_b) < repr(root_a):
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a


def entity_ids(
    data: Mapping[Any, Mapping[str, Any]],
    entity_key: str | None = None,
    known_pairs: Iterable[tuple[Any, Any]] = (),
) -> dict[Any, Any]:
    """Return a mapping ``record_id -> entity_id``.

    Records that share the same non-null ``entity_key`` value are the same
    entity.  Records connected (transitively) by ``known_pairs`` are also the
    same entity.  This is the step that must happen *before* any split: it
    defines what "the same entity" means for the whole protocol.

    Entity labels are deterministic: a singleton entity is labelled by its
    record id, and a multi-record entity is labelled by the smallest member
    under :func:`repr` ordering.
    """
    uf = _UnionFind()

    buckets: dict[Any, list[Any]] = {}
    for record_id, record in data.items():
        uf.add(record_id)
        if entity_key is not None:
            key = record.get(entity_key)
            if key is not None and key != "":
                buckets.setdefault(key, []).append(record_id)

    for record_ids in buckets.values():
        first = record_ids[0]
        for other in record_ids[1:]:
            uf.union(first, other)

    for a, b in known_pairs:
        uf.union(a, b)

    members: dict[Any, list[Any]] = {}
    for record_id in data:
        members.setdefault(uf.find(record_id), []).append(record_id)

    entity_of: dict[Any, Any] = {}
    for record_ids in members.values():
        label = min(record_ids, key=repr)
        for record_id in record_ids:
            entity_of[record_id] = label
    return entity_of


def entity_groups(entity_of: Mapping[Any, Any]) -> dict[Any, list[Any]]:
    """Group record ids by entity, with records sorted deterministically."""
    groups: dict[Any, list[Any]] = {}
    for record_id, entity in entity_of.items():
        groups.setdefault(entity, []).append(record_id)
    return {entity: sorted(ids, key=repr) for entity, ids in groups.items()}


def positive_pairs(entity_of: Mapping[Any, Any]) -> list[tuple[Any, Any]]:
    """All within-entity record pairs, deterministic and de-duplicated."""
    pairs: list[tuple[Any, Any]] = []
    for record_ids in entity_groups(entity_of).values():
        for i, a in enumerate(record_ids):
            for b in record_ids[i + 1 :]:
                pairs.append((a, b))
    pairs.sort(key=repr)
    return pairs


def straddling_pairs(
    entity_of: Mapping[Any, Any],
    record_fold: Mapping[Any, int],
) -> list[tuple[Any, Any]]:
    """Known positive pairs whose endpoints land in different folds.

    A non-empty result means the record-level assignment would force a true
    match to be treated as a negative (or would leak an entity across the
    split).  Such pairs must not be silently relabelled.
    """
    result = []
    for a, b in positive_pairs(entity_of):
        fold_a = record_fold.get(a)
        fold_b = record_fold.get(b)
        if fold_a is not None and fold_b is not None and fold_a != fold_b:
            result.append((a, b))
    return result


@dataclass
class FoldAudit:
    """Coverage and leakage for a proposed record-level fold assignment."""

    folds: list[dict[str, Any]]
    straddling_pairs: list[tuple[Any, Any]]

    @property
    def is_entity_pure(self) -> bool:
        return not self.straddling_pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_entity_pure": self.is_entity_pure,
            "folds": self.folds,
            "straddling_pairs": [[repr(a), repr(b)] for a, b in self.straddling_pairs],
        }


def audit_record_folds(
    entity_of: Mapping[Any, Any],
    record_fold: Mapping[Any, int],
) -> FoldAudit:
    """Report fold coverage and any cross-fold positive pairs.

    This never mutates or silently repairs the assignment; callers see the
    leakage explicitly and can decide to rebuild folds at the entity level.
    """
    total_records = len(entity_of)
    all_positive = len(positive_pairs(entity_of))
    rows = []
    for fold in sorted(set(record_fold.values())):
        record_ids = [r for r, f in record_fold.items() if f == fold]
        held = set(record_ids)
        entities = {entity_of[r] for r in record_ids}
        inside = sum(1 for a, b in positive_pairs(entity_of) if a in held and b in held)
        rows.append(
            {
                "fold": fold,
                "validation_records": len(record_ids),
                "validation_entities": len(entities),
                "validation_record_share": len(record_ids) / max(total_records, 1),
                "validation_positive_pairs": inside,
                "validation_positive_share": inside / max(all_positive, 1),
            }
        )
    return FoldAudit(rows, straddling_pairs(entity_of, record_fold))


# ---------------------------------------------------------------------------
# split plan
# ---------------------------------------------------------------------------
@dataclass
class EntityPlan:
    """A reproducible, entity-pure fold assignment."""

    data: Mapping[Any, Mapping[str, Any]]
    entity_of: dict[Any, Any]
    folds: dict[int, tuple[Any, ...]]
    seed: int = 0
    name: str = ""

    # -- construction -----------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: Mapping[Any, Mapping[str, Any]],
        entity_key: str | None = None,
        known_pairs: Iterable[tuple[Any, Any]] = (),
        n_folds: int = 5,
        seed: int = 0,
        name: str = "",
    ) -> "EntityPlan":
        entity_of = entity_ids(data, entity_key=entity_key, known_pairs=known_pairs)
        return cls.from_entity_map(
            data, entity_of, n_folds=n_folds, seed=seed, name=name
        )

    @classmethod
    def from_entity_map(
        cls,
        data: Mapping[Any, Mapping[str, Any]],
        entity_of: Mapping[Any, Any],
        n_folds: int = 5,
        seed: int = 0,
        name: str = "",
    ) -> "EntityPlan":
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        entities = sorted(set(entity_of.values()), key=repr)
        if n_folds > len(entities):
            raise ValueError(
                f"n_folds ({n_folds}) cannot exceed the number of entities "
                f"({len(entities)})"
            )
        rng = numpy.random.default_rng(seed)
        shuffled = [entities[i] for i in rng.permutation(len(entities))]
        folds = {fold: tuple(shuffled[fold::n_folds]) for fold in range(n_folds)}
        return cls(
            data=dict(data),
            entity_of=dict(entity_of),
            folds=folds,
            seed=seed,
            name=name,
        )

    @classmethod
    def from_record_folds(
        cls,
        data: Mapping[Any, Mapping[str, Any]],
        record_fold: Mapping[Any, int],
        entity_of: Mapping[Any, Any] | None = None,
        on_straddling: str = "error",
        seed: int = 0,
        name: str = "",
    ) -> "EntityPlan":
        """Build a plan from a record-level assignment.

        A record-level assignment is usable only when it is entity-pure.  If
        any known positive pair straddles folds:

        * ``on_straddling="error"`` (default) raises
          :class:`EntityLeakageError` with those pairs attached.  The pairs
          are never converted into negative examples.
        * ``on_straddling="union"`` resolves the conflict by moving the
          whole entity into the smallest fold among its records.  This keeps
          the assignment entity-pure without ever turning a positive pair
          into a negative one; the resulting plan records no leakage.
        """
        if on_straddling not in {"error", "union"}:
            raise ValueError(
                "on_straddling must be 'error' or 'union', " f"got {on_straddling!r}"
            )
        if entity_of is None:
            entity_of = entity_ids(data)
        leaked = straddling_pairs(entity_of, record_fold)
        if leaked and on_straddling == "error":
            raise EntityLeakageError(
                f"{len(leaked)} known positive pair(s) straddle folds; "
                "refusing to treat them as negatives. Build entity-level "
                "folds with EntityPlan.from_data, inspect the assignment "
                "with audit_record_folds, or pass on_straddling='union' to "
                "move whole entities into a single fold.",
                leaked,
            )

        fold_of_entity: dict[Any, int] = {}
        for record_id, fold in record_fold.items():
            entity = entity_of[record_id]
            if entity in fold_of_entity:
                # union mode: keep the entity in the smallest fold
                fold_of_entity[entity] = min(fold_of_entity[entity], fold)
            else:
                fold_of_entity[entity] = fold
        folds: dict[int, list[Any]] = {
            fold: [] for fold in sorted(set(record_fold.values()))
        }
        for entity, fold in fold_of_entity.items():
            folds.setdefault(fold, []).append(entity)
        # union resolution can empty a fold; drop it rather than report a
        # validation split with nothing held out
        folds = {fold: values for fold, values in folds.items() if values}
        plan = cls(
            data=dict(data),
            entity_of=dict(entity_of),
            folds={
                fold: tuple(sorted(values, key=repr))
                for fold, values in sorted(folds.items())
            },
            seed=seed,
            name=name,
        )
        # sanity: a union-resolved plan must be entity-pure
        resolved_folds = {
            record: fold_of_entity[entity] for record, entity in plan.entity_of.items()
        }
        assert not straddling_pairs(plan.entity_of, resolved_folds)
        return plan

    # -- fold accessors ---------------------------------------------------
    @property
    def fold_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.folds))

    def validation_entities(self, fold: int) -> frozenset[Any]:
        return frozenset(self.folds[fold])

    def training_entities(self, fold: int) -> frozenset[Any]:
        held_out = self.validation_entities(fold)
        return frozenset(
            entity for entity in set(self.entity_of.values()) if entity not in held_out
        )

    def validation_records(self, fold: int) -> list[Any]:
        held_out = self.validation_entities(fold)
        return [
            record_id
            for record_id, entity in self.entity_of.items()
            if entity in held_out
        ]

    def training_records(self, fold: int) -> list[Any]:
        held_out = self.validation_entities(fold)
        return [
            record_id
            for record_id, entity in self.entity_of.items()
            if entity not in held_out
        ]

    def fold_data(self, fold: int) -> tuple[dict[Any, Any], dict[Any, Any]]:
        train_ids = self.training_records(fold)
        valid_ids = self.validation_records(fold)
        train = {record_id: self.data[record_id] for record_id in train_ids}
        valid = {record_id: self.data[record_id] for record_id in valid_ids}
        return train, valid

    # -- coverage ---------------------------------------------------------
    def coverage(self) -> list[dict[str, Any]]:
        """Report how much record / entity / positive mass each fold holds."""
        total_records = len(self.data)
        total_entities = len(set(self.entity_of.values()))
        all_positive = len(positive_pairs(self.entity_of))
        rows = []
        for fold in self.fold_indices:
            valid_ids = set(self.validation_records(fold))
            valid_entities = self.validation_entities(fold)
            positive = sum(
                1
                for a, b in positive_pairs(self.entity_of)
                if a in valid_ids and b in valid_ids
            )
            rows.append(
                {
                    "fold": fold,
                    "validation_records": len(valid_ids),
                    "validation_entities": len(valid_entities),
                    "validation_record_share": len(valid_ids) / max(total_records, 1),
                    "validation_entity_share": len(valid_entities)
                    / max(total_entities, 1),
                    "validation_positive_pairs": positive,
                    "validation_positive_share": positive / max(all_positive, 1),
                }
            )
        return rows

    # -- training material ------------------------------------------------
    def training_pairs(
        self,
        fold: int,
        n_distinct: int = 2000,
        seed: int | None = None,
    ) -> "TrainingData":
        """Real training pairs for a fold, with no validation entity present.

        Matches are all within-entity pairs among *training* entities.
        Distinct pairs are sampled from *different* training entities.  No
        pair involving a validation record is emitted, so a true cross-fold
        match can never be presented as a negative example.
        """
        rng = numpy.random.default_rng(self.seed if seed is None else seed)
        train_ids = self.training_records(fold)
        entity_of = self.entity_of

        by_entity: dict[Any, list[Any]] = {}
        for record_id in train_ids:
            by_entity.setdefault(entity_of[record_id], []).append(record_id)
        by_entity = {entity: sorted(ids, key=repr) for entity, ids in by_entity.items()}

        matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for record_ids in by_entity.values():
            for i, a in enumerate(record_ids):
                for b in record_ids[i + 1 :]:
                    matches.append((self.data[a], self.data[b]))

        distinct = _sample_distinct(self.data, by_entity, n_distinct, rng)
        training: "TrainingData" = {"match": matches, "distinct": distinct}
        return training

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "entity_of": {
                repr(record_id): repr(entity)
                for record_id, entity in sorted(
                    self.entity_of.items(), key=lambda kv: repr(kv[0])
                )
            },
            "records": [repr(r) for r in sorted(self.data, key=repr)],
            "folds": {
                str(fold): [repr(entity) for entity in entities]
                for fold, entities in sorted(self.folds.items())
            },
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(
        cls,
        path: str,
        data: Mapping[Any, Mapping[str, Any]],
        id_lookup: Mapping[str, Any] | None = None,
    ) -> "EntityPlan":
        """Reload a plan against freshly loaded data.

        Record and entity ids are stored as ``repr`` strings; by default they
        are parsed back with :func:`ast.literal_eval`, which round-trips int
        and str ids exactly.  Pass ``id_lookup`` to resolve tokens yourself.
        """
        import ast

        def resolve(token: str) -> Any:
            if id_lookup is not None:
                return id_lookup[token]
            return ast.literal_eval(token)

        with open(path) as f:
            payload = json.load(f)
        entity_of = {
            resolve(record): resolve(entity)
            for record, entity in payload["entity_of"].items()
        }
        folds = {
            int(fold): tuple(resolve(entity) for entity in entities)
            for fold, entities in payload["folds"].items()
        }
        return cls(
            data=dict(data),
            entity_of=entity_of,
            folds=folds,
            seed=payload["seed"],
            name=payload["name"],
        )


def _sample_distinct(
    data: Mapping[Any, Mapping[str, Any]],
    by_entity: Mapping[Any, list[Any]],
    n_distinct: int,
    rng: numpy.random.Generator,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Sample distinct pairs from different entity buckets."""
    entities = sorted(by_entity, key=repr)
    distinct: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    if len(entities) < 2 or n_distinct <= 0:
        return distinct

    seen: set[tuple[Any, Any]] = set()
    max_attempts = max(20 * n_distinct, 1000)
    attempts = 0
    while len(distinct) < n_distinct and attempts < max_attempts:
        attempts += 1
        entity_a = entities[int(rng.integers(0, len(entities)))]
        entity_b = entities[int(rng.integers(0, len(entities)))]
        if entity_a == entity_b:
            continue
        ids_a, ids_b = by_entity[entity_a], by_entity[entity_b]
        a = ids_a[int(rng.integers(0, len(ids_a)))]
        b = ids_b[int(rng.integers(0, len(ids_b)))]
        key = (a, b) if repr(a) <= repr(b) else (b, a)
        if key in seen:
            continue
        seen.add(key)
        distinct.append((data[a], data[b]))
    return distinct


# ---------------------------------------------------------------------------
# distances / link-quality evaluation
# ---------------------------------------------------------------------------
def training_distances(data_model: Any, training: "TrainingData"):
    """Compute ``(X, y)`` for a fold's real training pairs."""
    examples = list(training["match"]) + list(training["distinct"])
    y = numpy.array(
        [1] * len(training["match"]) + [0] * len(training["distinct"]),
        dtype=int,
    )
    if not examples:
        return numpy.empty((0, len(data_model)), dtype="f4"), y
    return data_model.distances(examples), y


@dataclass
class LinkQuality:
    """Pair-level match quality against held-out entity identities."""

    n_truth: int = 0
    n_candidates: int = 0
    n_predicted: int = 0
    true_positives: int = 0
    candidate_recall: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    @classmethod
    def from_pairs(
        cls,
        truth: set[Any],
        candidates: set[Any],
        predicted: set[Any],
    ) -> "LinkQuality":
        """Compare predicted pairs with truth and blocking candidates.

        For dedupe, pairs are ``frozenset`` of the two record ids.  For
        record linkage they are ``(id_from_data_1, id_from_data_2)`` tuples.
        ``candidate_recall`` is the share of true pairs that blocking even
        presented to the classifier -- the ceiling on recall.
        """
        true_positives = len(predicted & truth)
        precision = true_positives / len(predicted) if predicted else 0.0
        recall = true_positives / len(truth) if truth else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return cls(
            n_truth=len(truth),
            n_candidates=len(candidates),
            n_predicted=len(predicted),
            true_positives=true_positives,
            candidate_recall=(len(candidates & truth) / len(truth) if truth else 0.0),
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_truth": self.n_truth,
            "n_candidates": self.n_candidates,
            "n_predicted": self.n_predicted,
            "true_positives": self.true_positives,
            "candidate_recall": self.candidate_recall,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def truth_pairs(
    entity_of: Mapping[Any, Any],
    record_ids: Iterable[Any],
) -> set[frozenset[Any]]:
    """Within-entity pairs restricted to ``record_ids``."""
    wanted = set(record_ids)
    groups: dict[Any, list[Any]] = {}
    for record_id in wanted:
        groups.setdefault(entity_of[record_id], []).append(record_id)
    truth: set[frozenset[Any]] = set()
    for record_ids_in_entity in groups.values():
        ordered = sorted(record_ids_in_entity, key=repr)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                truth.add(frozenset((a, b)))
    return truth


def predicted_pairs_from_clusters(
    clusters: Iterable[Any],
) -> set[frozenset[Any]]:
    """Convert :meth:`Dedupe.partition` output to a set of pairs."""
    import itertools

    predicted: set[frozenset[Any]] = set()
    for record_ids, _scores in clusters:
        for pair in itertools.combinations(record_ids, 2):
            predicted.add(frozenset(pair))
    return predicted


def predicted_pairs_from_links(
    links: Iterable[Any],
) -> set[tuple[Any, Any]]:
    """Convert :meth:`RecordLink.join` output to side-aware pair keys.

    A tuple ``(id_from_data_1, id_from_data_2)`` is used rather than a
    ``frozenset``: the two datasets may share id values, and the side is
    part of the identity of a link.
    """
    return {(pair[0], pair[1]) for pair, _score in links}


def candidate_pairs(scores: Any) -> set[frozenset[Any]]:
    """Convert a scored-pairs array to a set of pairs (blocking candidates)."""
    result: set[frozenset[Any]] = set()
    for a, b in scores["pairs"]:
        result.add(frozenset((a.item(), b.item())))
    return result


def link_candidate_pairs(scores: Any) -> set[tuple[Any, Any]]:
    """Side-aware candidate pairs for record linkage scores."""
    return {(a.item(), b.item()) for a, b in scores["pairs"]}


def score_validation(
    matcher: Any,
    validation_data: Mapping[Any, Mapping[str, Any]],
) -> tuple[Any, set[frozenset[Any]]]:
    """Generate the *real* blocking candidates on held-out records.

    Returns the scored-pairs array and the candidate pair set.  The
    candidates are exactly what the trained blocking rules emit for the
    validation records; nothing is sampled or fabricated.  If blocking
    produces no pairs at all, an empty score array is returned rather than
    raising, so a fold with zero candidates is reported honestly (recall 0)
    instead of aborting the run.
    """
    try:
        scores = matcher.score(matcher.pairs(validation_data))
    except core.BlockingError:
        scores = numpy.array([], dtype=[("pairs", "i4", 2), ("score", "f4")])
    return scores, candidate_pairs(scores)


def evaluate_dedupe(
    matcher: Any,
    validation_data: Mapping[Any, Mapping[str, Any]],
    truth: set[Any],
    threshold: float = 0.5,
) -> LinkQuality:
    """Score a dedupe matcher on held-out records against entity truth.

    ``truth`` uses ``frozenset`` pairs, matching :func:`truth_pairs`.
    """
    if not validation_data:
        return LinkQuality.from_pairs(truth, set(), set())
    scores, candidates = score_validation(matcher, validation_data)
    if len(scores) == 0:
        return LinkQuality.from_pairs(truth, candidates, set())
    clusters = matcher.cluster(scores, threshold)
    predicted = predicted_pairs_from_clusters(clusters)
    return LinkQuality.from_pairs(truth, candidates, predicted)


def evaluate_link(
    matcher: Any,
    validation_data_1: Mapping[Any, Mapping[str, Any]],
    validation_data_2: Mapping[Any, Mapping[str, Any]],
    truth: set[Any],
    threshold: float = 0.5,
    constraint: str = "one-to-one",
) -> LinkQuality:
    """Score a record-linkage matcher on held-out records against truth.

    ``truth`` and the returned candidate / predicted pairs are side-aware
    ``(id_from_data_1, id_from_data_2)`` tuples.
    """
    if constraint not in {"one-to-one", "many-to-one", "many-to-many"}:
        raise ValueError(
            "%s is an invalid constraint option. Valid options include "
            "one-to-one, many-to-one, or many-to-many" % constraint
        )
    if not validation_data_1 or not validation_data_2:
        return LinkQuality.from_pairs(truth, set(), set())
    try:
        scores = matcher.score(matcher.pairs(validation_data_1, validation_data_2))
    except core.BlockingError:
        scores = numpy.array([], dtype=[("pairs", "i4", 2), ("score", "f4")])
    candidates = link_candidate_pairs(scores)
    if len(scores) == 0:
        return LinkQuality.from_pairs(truth, candidates, set())
    if constraint == "one-to-one":
        links = matcher.one_to_one(scores, threshold)
    elif constraint == "many-to-one":
        links = matcher.many_to_one(scores, threshold)
    else:
        links = scores[scores["score"] > threshold]
    predicted = predicted_pairs_from_links(links)
    return LinkQuality.from_pairs(truth, candidates, predicted)


def distance_digest(data_model: Any, training: "TrainingData") -> dict[str, Any]:
    """Summary of the fold's real training distances.

    Useful for spotting a fold whose training pairs are degenerate (for
    example, all matches land in one comparator column) without persisting
    the full matrix.
    """
    X, y = training_distances(data_model, training)
    if len(X) == 0:
        return {"n_rows": 0, "n_features": len(data_model)}
    positives = X[y == 1]
    negatives = X[y == 0]
    return {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_match": int(int((y == 1).sum())),
        "n_distinct": int(int((y == 0).sum())),
        "match_distance_mean": _column_mean(positives),
        "distinct_distance_mean": _column_mean(negatives),
    }


def _column_mean(X: Any) -> list[float]:
    if len(X) == 0:
        return []
    return [float(value) for value in numpy.asarray(X, dtype="f8").mean(axis=0)]


def input_digest(payload: Any) -> str:
    """Stable digest of a validation input, for reproducibility checks.

    Two runs with the same seed should produce the same digest; compare it
    before trusting a replayed report.
    """
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_digest(training: "TrainingData") -> str:
    """Digest of a fold's real training pairs (not the fitted model)."""
    return input_digest(
        {
            "match": [list(map(repr, pair)) for pair in training["match"]],
            "distinct": [list(map(repr, pair)) for pair in training["distinct"]],
        }
    )


@dataclass
class CrossValidationReport:
    """Aggregate of per-fold reports, with split coverage attached."""

    folds: list[dict[str, Any]] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": self.folds,
            "coverage": self.coverage,
            "aggregate": self.aggregate,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_quality(
        cls,
        coverage: list[dict[str, Any]],
        qualities: Sequence[LinkQuality],
        fold_indices: Sequence[int],
        details: Sequence[Mapping[str, Any]] | None = None,
    ) -> "CrossValidationReport":
        details = list(details or [])
        folds = []
        for index, (fold, quality) in enumerate(zip(fold_indices, qualities)):
            row = {"fold": fold}
            row.update(quality.to_dict())
            if index < len(details):
                row.update(details[index])
            folds.append(row)
        n_truth = sum(q.n_truth for q in qualities)
        n_predicted = sum(q.n_predicted for q in qualities)
        true_positives = sum(q.true_positives for q in qualities)
        n_candidates = sum(q.n_candidates for q in qualities)
        precision = true_positives / n_predicted if n_predicted else 0.0
        recall = true_positives / n_truth if n_truth else 0.0
        candidate_recall = (
            sum(q.candidate_recall * q.n_truth for q in qualities) / n_truth
            if n_truth
            else 0.0
        )
        aggregate = {
            "n_folds": len(qualities),
            "n_truth": n_truth,
            "n_candidates": n_candidates,
            "n_predicted": n_predicted,
            "true_positives": true_positives,
            "candidate_recall": candidate_recall,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "mean_fold_precision": _mean([q.precision for q in qualities]),
            "mean_fold_recall": _mean([q.recall for q in qualities]),
            "mean_fold_f1": _mean([q.f1 for q in qualities]),
            "std_fold_f1": _std([q.f1 for q in qualities]),
        }
        return cls(folds=folds, coverage=coverage, aggregate=aggregate)


def replay_dedupe(
    plan: EntityPlan,
    matcher: Any,
    threshold: float = 0.5,
) -> CrossValidationReport:
    """Replay a persisted plan against an already-trained deployment model.

    The split and the held-out records come from ``plan``; ``matcher`` is
    the fully retrained model (see :func:`refit_full_model`).  This measures
    the deployment model on the same validation inputs the folds used,
    without re-training per fold.
    """
    qualities = []
    for fold in plan.fold_indices:
        _train, valid = plan.fold_data(fold)
        truth = truth_pairs(plan.entity_of, plan.validation_records(fold))
        qualities.append(evaluate_dedupe(matcher, valid, truth, threshold))
    return CrossValidationReport.from_quality(
        plan.coverage(), qualities, plan.fold_indices
    )


def replay_link(
    plan: LinkPlan,
    matcher: Any,
    threshold: float = 0.5,
    constraint: str = "one-to-one",
) -> CrossValidationReport:
    """Replay a persisted link plan against a retrained deployment model."""
    fold_indices = tuple(sorted(plan.folds))
    qualities = []
    for fold in fold_indices:
        valid_1, valid_2 = plan.validation_data(fold)
        truth = plan.truth_pairs(fold)
        qualities.append(
            evaluate_link(matcher, valid_1, valid_2, truth, threshold, constraint)
        )
    return CrossValidationReport.from_quality(plan.coverage(), qualities, fold_indices)


def _std(values: Sequence[float]) -> float:
    return float(numpy.std(values)) if values else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(numpy.mean(values)) if values else 0.0


def make_matcher(
    variable_definition: Sequence[Any],
    kind: str = "dedupe",
    num_cores: int = 0,
    **kwargs: Any,
) -> Any:
    """Build a fresh, untrained matcher of the requested kind."""
    import dedupe

    matcher_types = {
        "dedupe": dedupe.Dedupe,
        "recordlink": dedupe.RecordLink,
        "gazetteer": dedupe.Gazetteer,
    }
    try:
        matcher_type = matcher_types[kind]
    except KeyError:
        raise ValueError(
            f"unknown matcher kind {kind!r}; choose from {sorted(matcher_types)}"
        )
    return matcher_type(variable_definition, num_cores=num_cores, **kwargs)


def run_dedupe_cv(
    plan: EntityPlan,
    variable_definition: Sequence[Any],
    threshold: float = 0.5,
    n_distinct: int = 2000,
    num_cores: int = 0,
    train_kwargs: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> CrossValidationReport:
    """Train one model per fold and evaluate on held-out entity identities.

    Every fold gets its own real training pairs, its own fitted classifier /
    blocking rules, and its own validation candidates generated from the
    held-out records.  No model ever sees a validation entity.
    """
    import io

    import dedupe

    train_kwargs = dict(train_kwargs or {})
    qualities = []
    details = []
    for fold in plan.fold_indices:
        train, valid = plan.fold_data(fold)
        if not train:
            raise ValueError(
                f"fold {fold} has no training records; reduce n_folds "
                "or use more data"
            )
        training = plan.training_pairs(fold, n_distinct=n_distinct, seed=seed)
        training_file = io.StringIO()
        dedupe.write_training(training, training_file)
        training_file.seek(0)

        matcher = make_matcher(variable_definition, kind="dedupe", num_cores=num_cores)
        matcher.prepare_training(
            train, training_file=training_file, sample_size=max(len(train), 1)
        )
        matcher.train(**train_kwargs)

        truth = truth_pairs(plan.entity_of, plan.validation_records(fold))
        qualities.append(evaluate_dedupe(matcher, valid, truth, threshold))
        details.append(
            {
                "training_pairs": len(training["match"]) + len(training["distinct"]),
                "training_matches": len(training["match"]),
                "training_digest": training_digest(training),
                "training_distances": distance_digest(matcher.data_model, training),
            }
        )

    return CrossValidationReport.from_quality(
        plan.coverage(), qualities, plan.fold_indices, details
    )


def run_link_cv(
    plan: LinkPlan,
    variable_definition: Sequence[Any],
    threshold: float = 0.5,
    constraint: str = "one-to-one",
    n_distinct: int = 2000,
    num_cores: int = 0,
    train_kwargs: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> CrossValidationReport:
    """Train one record-linkage model per fold and evaluate held-out links."""
    import io

    import dedupe

    train_kwargs = dict(train_kwargs or {})
    qualities = []
    details = []
    fold_indices = tuple(sorted(plan.folds))
    for fold in fold_indices:
        train_1, train_2 = plan.training_records(fold)
        valid_1, valid_2 = plan.validation_data(fold)
        if not train_1 or not train_2:
            raise ValueError(
                f"fold {fold} is missing training records on one side; "
                "reduce n_folds or provide more data"
            )
        training = plan.training_pairs(fold, n_distinct=n_distinct, seed=seed)
        training_file = io.StringIO()
        dedupe.write_training(training, training_file)
        training_file.seek(0)

        matcher = make_matcher(
            variable_definition, kind="recordlink", num_cores=num_cores
        )
        matcher.prepare_training(
            train_1,
            train_2,
            training_file=training_file,
            sample_size=max(len(train_1), len(train_2), 1),
        )
        matcher.train(**train_kwargs)

        truth = plan.truth_pairs(fold)
        qualities.append(
            evaluate_link(matcher, valid_1, valid_2, truth, threshold, constraint)
        )
        details.append(
            {
                "training_pairs": len(training["match"]) + len(training["distinct"]),
                "training_matches": len(training["match"]),
                "training_digest": training_digest(training),
                "training_distances": distance_digest(matcher.data_model, training),
            }
        )

    return CrossValidationReport.from_quality(
        plan.coverage(), qualities, fold_indices, details
    )


# ---------------------------------------------------------------------------
# record linkage plan
# ---------------------------------------------------------------------------
def _side_token(side: int, record_id: Any) -> tuple[str, Any]:
    return (str(side), record_id)


@dataclass
class LinkPlan:
    """Entity-pure fold assignment across two datasets.

    An entity may have records in both ``data_1`` and ``data_2``; the entity
    label is resolved globally so a link's two sides always share a fold.
    """

    data_1: Mapping[Any, Mapping[str, Any]]
    data_2: Mapping[Any, Mapping[str, Any]]
    entity_of: dict[Any, Any]
    folds: dict[int, tuple[Any, ...]]
    seed: int = 0
    name: str = ""

    @classmethod
    def from_data(
        cls,
        data_1: Mapping[Any, Mapping[str, Any]],
        data_2: Mapping[Any, Mapping[str, Any]],
        entity_key: str | None = None,
        known_pairs: Iterable[tuple[Any, Any]] = (),
        n_folds: int = 5,
        seed: int = 0,
        name: str = "",
    ) -> "LinkPlan":
        combined: dict[Any, Mapping[str, Any]] = {}
        for record_id, record in data_1.items():
            combined[_side_token(1, record_id)] = record
        for record_id, record in data_2.items():
            combined[_side_token(2, record_id)] = record

        renamed_pairs = [(_side_token(1, a), _side_token(2, b)) for a, b in known_pairs]
        entity_of_combined = entity_ids(
            combined, entity_key=entity_key, known_pairs=renamed_pairs
        )
        plan = EntityPlan.from_entity_map(
            combined, entity_of_combined, n_folds=n_folds, seed=seed, name=name
        )
        return cls(
            data_1=dict(data_1),
            data_2=dict(data_2),
            entity_of=dict(plan.entity_of),
            folds=plan.folds,
            seed=seed,
            name=name,
        )

    def validation_entities(self, fold: int) -> frozenset[Any]:
        return frozenset(self.folds[fold])

    def validation_records(self, fold: int) -> tuple[list[Any], list[Any]]:
        held_out = self.validation_entities(fold)
        ids_1 = [
            record_id
            for record_id in self.data_1
            if self.entity_of[_side_token(1, record_id)] in held_out
        ]
        ids_2 = [
            record_id
            for record_id in self.data_2
            if self.entity_of[_side_token(2, record_id)] in held_out
        ]
        return ids_1, ids_2

    def validation_data(self, fold: int) -> tuple[dict[Any, Any], dict[Any, Any]]:
        ids_1, ids_2 = self.validation_records(fold)
        return (
            {record_id: self.data_1[record_id] for record_id in ids_1},
            {record_id: self.data_2[record_id] for record_id in ids_2},
        )

    def training_records(self, fold: int) -> tuple[dict[Any, Any], dict[Any, Any]]:
        held_out = self.validation_entities(fold)
        train_1 = {
            record_id: record
            for record_id, record in self.data_1.items()
            if self.entity_of[_side_token(1, record_id)] not in held_out
        }
        train_2 = {
            record_id: record
            for record_id, record in self.data_2.items()
            if self.entity_of[_side_token(2, record_id)] not in held_out
        }
        return train_1, train_2

    def training_pairs(
        self, fold: int, n_distinct: int = 2000, seed: int | None = None
    ) -> "TrainingData":
        """Link training pairs.

        Matches connect records of the same entity across the two datasets.
        Distinct pairs connect *different* entities.  Validation entities are
        absent from both sides, so no cross-fold positive can appear as a
        negative example.
        """
        rng = numpy.random.default_rng(self.seed if seed is None else seed)
        train_1, train_2 = self.training_records(fold)

        by_entity_1: dict[Any, list[Any]] = {}
        by_entity_2: dict[Any, list[Any]] = {}
        for record_id in train_1:
            by_entity_1.setdefault(
                self.entity_of[_side_token(1, record_id)], []
            ).append(record_id)
        for record_id in train_2:
            by_entity_2.setdefault(
                self.entity_of[_side_token(2, record_id)], []
            ).append(record_id)

        matches = []
        for entity in sorted(set(by_entity_1) & set(by_entity_2), key=repr):
            ids_1 = sorted(by_entity_1[entity], key=repr)
            ids_2 = sorted(by_entity_2[entity], key=repr)
            for a in ids_1:
                for b in ids_2:
                    matches.append((train_1[a], train_2[b]))

        distinct = _sample_distinct_link(
            train_1, train_2, by_entity_1, by_entity_2, n_distinct, rng
        )
        training: "TrainingData" = {"match": matches, "distinct": distinct}
        return training

    def coverage(self) -> list[dict[str, Any]]:
        rows = []
        for fold in sorted(self.folds):
            ids_1, ids_2 = self.validation_records(fold)
            rows.append(
                {
                    "fold": fold,
                    "validation_entities": len(self.folds[fold]),
                    "validation_records_1": len(ids_1),
                    "validation_records_2": len(ids_2),
                    "validation_record_share_1": len(ids_1) / max(len(self.data_1), 1),
                    "validation_record_share_2": len(ids_2) / max(len(self.data_2), 1),
                }
            )
        return rows

    def truth_pairs(self, fold: int) -> set[tuple[Any, Any]]:
        ids_1, ids_2 = self.validation_records(fold)
        entity_1 = {
            record_id: self.entity_of[_side_token(1, record_id)] for record_id in ids_1
        }
        entity_2 = {
            record_id: self.entity_of[_side_token(2, record_id)] for record_id in ids_2
        }
        truth: set[tuple[Any, Any]] = set()
        for a, entity_a in entity_1.items():
            for b, entity_b in entity_2.items():
                if entity_a == entity_b:
                    truth.add((a, b))
        return truth

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "entity_of": {
                repr(token): repr(entity)
                for token, entity in sorted(
                    self.entity_of.items(), key=lambda kv: repr(kv[0])
                )
            },
            "folds": {
                str(fold): [repr(entity) for entity in entities]
                for fold, entities in sorted(self.folds.items())
            },
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(
        cls,
        path: str,
        data_1: Mapping[Any, Mapping[str, Any]],
        data_2: Mapping[Any, Mapping[str, Any]],
    ) -> "LinkPlan":
        import ast

        with open(path) as f:
            payload = json.load(f)
        entity_of = {
            ast.literal_eval(token): ast.literal_eval(entity)
            for token, entity in payload["entity_of"].items()
        }
        folds = {
            int(fold): tuple(ast.literal_eval(entity) for entity in entities)
            for fold, entities in payload["folds"].items()
        }
        return cls(
            data_1=dict(data_1),
            data_2=dict(data_2),
            entity_of=entity_of,
            folds=folds,
            seed=payload["seed"],
            name=payload["name"],
        )


def _sample_distinct_link(
    train_1: Mapping[Any, Any],
    train_2: Mapping[Any, Any],
    by_entity_1: Mapping[Any, list[Any]],
    by_entity_2: Mapping[Any, list[Any]],
    n_distinct: int,
    rng: numpy.random.Generator,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    distinct: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    entities_1 = sorted(by_entity_1, key=repr)
    entities_2 = sorted(by_entity_2, key=repr)
    if not entities_1 or not entities_2 or n_distinct <= 0:
        return distinct

    seen: set[tuple[Any, Any]] = set()
    max_attempts = max(20 * n_distinct, 1000)
    attempts = 0
    while len(distinct) < n_distinct and attempts < max_attempts:
        attempts += 1
        entity_a = entities_1[int(rng.integers(0, len(entities_1)))]
        entity_b = entities_2[int(rng.integers(0, len(entities_2)))]
        if entity_a == entity_b:
            continue
        ids_a = by_entity_1[entity_a]
        ids_b = by_entity_2[entity_b]
        a = ids_a[int(rng.integers(0, len(ids_a)))]
        b = ids_b[int(rng.integers(0, len(ids_b)))]
        if (a, b) in seen:
            continue
        seen.add((a, b))
        distinct.append((train_1[a], train_2[b]))
    return distinct


# ---------------------------------------------------------------------------
# full retrain
# ---------------------------------------------------------------------------
def full_training_pairs(
    data: Mapping[Any, Mapping[str, Any]],
    entity_key: str | None = None,
    n_distinct: int = 15000,
    seed: int = 0,
) -> "TrainingData":
    """Training pairs using every entity, for the final deployment model."""
    entity_of = entity_ids(data, entity_key=entity_key)
    by_entity: dict[Any, list[Any]] = {}
    for record_id, entity in entity_of.items():
        by_entity.setdefault(entity, []).append(record_id)
    by_entity = {entity: sorted(ids, key=repr) for entity, ids in by_entity.items()}

    matches = []
    for record_ids in by_entity.values():
        for i, a in enumerate(record_ids):
            for b in record_ids[i + 1 :]:
                matches.append((data[a], data[b]))

    rng = numpy.random.default_rng(seed)
    distinct = _sample_distinct(data, by_entity, n_distinct, rng)
    training: "TrainingData" = {"match": matches, "distinct": distinct}
    return training


def refit_full_model(
    matcher: Any,
    data: Mapping[Any, Mapping[str, Any]],
    entity_key: str | None = None,
    sample_size: int = 15000,
    n_distinct: int = 15000,
    seed: int = 0,
) -> Any:
    """Refit a matcher on *all* entities once the configuration is settled.

    Cross-validation chooses the configuration; this deployment step sees
    every entity.  Pair generation is seeded, so the refit is reproducible.

    The pairs are handed to ``prepare_training`` through a training file
    rather than ``mark_pairs``; that way the active learner indexes every
    example record, including records outside its sampled subset.
    """
    import io

    from dedupe import serializer

    training = full_training_pairs(
        data, entity_key=entity_key, n_distinct=n_distinct, seed=seed
    )
    training_file = io.StringIO()
    serializer.write_training(training, training_file)
    training_file.seek(0)
    matcher.prepare_training(data, training_file=training_file, sample_size=sample_size)
    matcher.train()
    return matcher


def full_link_training_pairs(
    data_1: Mapping[Any, Mapping[str, Any]],
    data_2: Mapping[Any, Mapping[str, Any]],
    entity_of: Mapping[Any, Any],
    n_distinct: int = 15000,
    seed: int = 0,
) -> "TrainingData":
    """Link training pairs using *all* entities, for the deployment model."""
    rng = numpy.random.default_rng(seed)

    by_entity_1: dict[Any, list[Any]] = {}
    by_entity_2: dict[Any, list[Any]] = {}
    for record_id in data_1:
        by_entity_1.setdefault(_side_token(1, record_id), []).append(record_id)
    by_entity_1 = {entity_of[token]: ids for token, ids in by_entity_1.items()}
    for record_id in data_2:
        by_entity_2.setdefault(_side_token(2, record_id), []).append(record_id)
    by_entity_2 = {entity_of[token]: ids for token, ids in by_entity_2.items()}
    by_entity_1 = {entity: sorted(ids, key=repr) for entity, ids in by_entity_1.items()}
    by_entity_2 = {entity: sorted(ids, key=repr) for entity, ids in by_entity_2.items()}

    matches = []
    for entity in sorted(set(by_entity_1) & set(by_entity_2), key=repr):
        for a in by_entity_1[entity]:
            for b in by_entity_2[entity]:
                matches.append((data_1[a], data_2[b]))

    distinct = _sample_distinct_link(
        data_1, data_2, by_entity_1, by_entity_2, n_distinct, rng
    )
    training: "TrainingData" = {"match": matches, "distinct": distinct}
    return training


def refit_full_link(
    matcher: Any,
    data_1: Mapping[Any, Mapping[str, Any]],
    data_2: Mapping[Any, Mapping[str, Any]],
    entity_key: str | None = None,
    known_pairs: Iterable[tuple[Any, Any]] = (),
    sample_size: int = 15000,
    n_distinct: int = 15000,
    seed: int = 0,
) -> Any:
    """Refit a record-linkage matcher on all entities and both datasets."""
    import io

    from dedupe import serializer

    plan = LinkPlan.from_data(
        data_1,
        data_2,
        entity_key=entity_key,
        known_pairs=known_pairs,
        n_folds=2,
        seed=seed,
    )
    training = full_link_training_pairs(
        data_1, data_2, plan.entity_of, n_distinct=n_distinct, seed=seed
    )
    training_file = io.StringIO()
    serializer.write_training(training, training_file)
    training_file.seek(0)
    matcher.prepare_training(
        data_1, data_2, training_file=training_file, sample_size=sample_size
    )
    matcher.train()
    return matcher
