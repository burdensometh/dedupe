#!/usr/bin/python
"""
Stable, cross-run entity identities for clustering results.

Re-running a clustering (for example, the monthly re-cluster of a
deduplication pipeline) does not guarantee that clusters come back in
the same order, or with the same shape. The position of a cluster in
the output is therefore not an entity id.

This module provides :class:`EntityRegistry`, which publishes entity
ids that are stable across clustering rounds by tracking the
*membership* of entities (the sets of underlying record ids):

* a new cluster made up of exactly the members of one old entity
  **retains** that entity's id;
* several old entities whose members now land in a single cluster are
  **merged**: one id survives and the others become aliases that
  resolve to the survivor;
* an old entity whose members are spread over several new clusters is
  **split**: one cluster keeps the old id and the other clusters
  receive freshly minted ids. The split event records which members
  changed identity;
* a cluster that cannot be tied to any previous entity receives a new
  id from a deterministic, content-based strategy, so the same input
  always produces the same id regardless of cluster ordering or
  retries.

The registry is deliberately free of any global counter or random
state: ids are derived from the (sorted) member sets and every
iteration over sets or dictionaries uses explicit ordering, so the
result of publishing a round is a pure function of the persisted
registry state and the cluster memberships.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Collection,
    Iterable,
    Mapping,
    Sequence,
    cast,
)

from dedupe._typing import RecordID

STATE_VERSION = 1

# Event kinds
CREATED = "created"
RETAINED = "retained"
MERGED = "merged"
SPLIT = "split"
RETIRED = "retired"


class EntityRegistryError(Exception):
    """Base class for entity registry errors."""


class UnknownEntity(EntityRegistryError):
    """Raised when an entity id has never been published by the registry."""


class EntityRetired(EntityRegistryError):
    """Raised when resolving an entity whose members no longer exist."""


class StaleRound(EntityRegistryError):
    """Raised when a publish round is older than a published round."""


class AmbiguousRound(EntityRegistryError):
    """Raised when the same round is published with different membership."""


def _id_key(record_id: RecordID) -> tuple[str, str]:
    """Total ordering key that keeps int and str record ids unambiguous."""
    return type(record_id).__name__, str(record_id)


def _sorted(members: Iterable[RecordID]) -> list[RecordID]:
    return sorted(members, key=_id_key)


def _members_of(cluster: object) -> frozenset[RecordID]:
    """Accept either a plain cluster of record ids or a
    ``(record_ids, scores)`` tuple as produced by
    :meth:`dedupe.Dedupe.partition`."""
    if (
        isinstance(cluster, tuple)
        and len(cluster) == 2
        and isinstance(cluster[0], (list, tuple))
    ):
        return frozenset(cast("Sequence[RecordID]", cluster[0]))
    return frozenset(cast("Collection[RecordID]", cluster))


def mint_entity_id(
    members: Collection[RecordID],
    *,
    id_prefix: str = "ent",
    used_ids: Collection[str] = (),
) -> str:
    """Deterministically derive an entity id from a member set.

    The id is ``"<prefix>_<sha256 of the sorted member ids>"`` (the
    first 16 hex digits). It depends solely on membership, never on
    cluster position, so the same members always receive the same id
    even if cluster ordering changes or the publish is retried. In the
    astronomically unlikely event of a digest collision between
    distinct member sets, deterministic numeric suffixes disambiguate.
    """
    # Type-tag each member so that, e.g., the integer 1 and the string
    # "1" produce distinct ids. Members are sorted on a total key that
    # separates ints from strings first.
    tagged = [f"{type(member).__name__}:{member}" for member in _sorted(members)]
    joined = "\u001f".join(tagged)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    base = f"{id_prefix}_{digest}"
    candidate = base
    suffix = 1
    taken = set(used_ids)
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


@dataclass(frozen=True)
class EntityEvent:
    """A description of what happened to an entity in one publish round."""

    kind: str
    #: The current entity id this event is about.
    entity_id: str
    #: Previous entity ids feeding into this entity (merges/splits).
    sources: tuple[str, ...] = ()
    #: Members of ``entity_id`` after the publish.
    members: tuple[RecordID, ...] = ()
    #: Members whose entity id differs from the previous round.
    changed_members: tuple[RecordID, ...] = ()
    #: For split events, the new entities carved out of ``sources[0]``
    #: as ``(entity_id, members)`` pairs, excluding the anchor entity.
    children: tuple[tuple[str, tuple[RecordID, ...]], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "entity_id": self.entity_id,
            "sources": list(self.sources),
            "members": list(self.members),
            "changed_members": list(self.changed_members),
            "children": [
                {"entity_id": child_id, "members": list(child_members)}
                for child_id, child_members in self.children
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EntityEvent":
        children = tuple(
            (
                cast(str, child["entity_id"]),
                tuple(cast("list[RecordID]", child["members"])),
            )
            for child in cast("list[Mapping[str, object]]", data["children"])
        )
        return cls(
            kind=cast(str, data["kind"]),
            entity_id=cast(str, data["entity_id"]),
            sources=tuple(cast("list[str]", data["sources"])),
            members=tuple(cast("list[RecordID]", data["members"])),
            changed_members=tuple(cast("list[RecordID]", data["changed_members"])),
            children=children,
        )


@dataclass(frozen=True)
class PublishResult:
    """The output of :meth:`EntityRegistry.publish`."""

    round: int
    #: ``(entity_id, record_ids)`` in a deterministic order.
    entities: tuple[tuple[str, tuple[RecordID, ...]], ...]
    #: record id -> current entity id
    members: Mapping[RecordID, str]
    #: identity changes recorded in this round
    events: tuple[EntityEvent, ...] = field(default_factory=tuple)

    def as_clusters(self) -> list[tuple[RecordID, ...]]:
        """Return the entities as plain member tuples (dedupe cluster shape)."""
        return [record_ids for _, record_ids in self.entities]


@dataclass
class _Reconciliation:
    entities: list[tuple[str, frozenset[RecordID]]]
    aliases: dict[str, str]
    events: list[EntityEvent]


def _greedy_matches(
    overlaps: Mapping[str, Mapping[int, int]],
    new_clusters: Sequence[frozenset[RecordID]],
) -> dict[str, int]:
    """One-to-one assignment of old entities to new clusters.

    Edges are considered from the largest member overlap downward with
    fully deterministic tie-breaking, so the assignment never depends
    on dictionary or set iteration order.
    """
    edges: list[tuple[int, str, int]] = []
    for old_id, cluster_weights in overlaps.items():
        for cluster_index, weight in cluster_weights.items():
            edges.append((weight, old_id, cluster_index))
    edges.sort(
        key=lambda edge: (
            -edge[0],
            edge[1],
            _id_key(min(new_clusters[edge[2]], key=_id_key)),
        )
    )

    matched_old: set[str] = set()
    matched_cluster: set[int] = set()
    assignment: dict[str, int] = {}
    for _, old_id, cluster_index in edges:
        if old_id in matched_old or cluster_index in matched_cluster:
            continue
        assignment[old_id] = cluster_index
        matched_old.add(old_id)
        matched_cluster.add(cluster_index)
    return assignment


def reconcile_clusters(
    old_entities: Mapping[str, Collection[RecordID]],
    new_clusters: Sequence[object],
    *,
    id_prefix: str = "ent",
    used_ids: Collection[str] = (),
) -> _Reconciliation:
    """Reconcile new clusters against the entities of a previous round.

    Args:
        old_entities: mapping of entity id -> member record ids from
            the previous round.
        new_clusters: the clusters of the current round, either as
            collections of record ids or as ``(record_ids, scores)``
            tuples as returned by :meth:`dedupe.Dedupe.partition`.
        id_prefix: prefix for freshly minted entity ids.
        used_ids: entity ids that must never be re-minted (the
            registry's full history of current ids and aliases).

    Returns:
        A :class:`_Reconciliation` holding the new entity assignments,
        any new aliases and the events describing identity changes.
    """
    normalized: list[frozenset[RecordID]] = []
    seen_members: set[RecordID] = set()
    for cluster in new_clusters:
        members = _members_of(cluster)
        if not members:
            raise ValueError("Clusters must contain at least one record id")
        overlap = members & seen_members
        if overlap:
            example = _sorted(overlap)[0]
            raise ValueError(
                f"Record id {example!r} appears in more than one cluster; "
                "clusters must be disjoint"
            )
        seen_members.update(members)
        normalized.append(members)

    old_membership = {
        old_id: frozenset(members) for old_id, members in old_entities.items()
    }

    # Fast lookups indexed by record id.
    cluster_of_member: dict[RecordID, int] = {}
    for cluster_index, members in enumerate(normalized):
        for member in members:
            cluster_of_member[member] = cluster_index
    previous_entity_of_member: dict[RecordID, str] = {}
    for old_id in sorted(old_membership):
        for member in old_membership[old_id]:
            previous_entity_of_member[member] = old_id

    # Bipartite overlap graph (old entities x new clusters), split into
    # connected components. Overlap weights are shared-member counts.
    overlaps: dict[str, dict[int, int]] = defaultdict(dict)
    for member, old_id in previous_entity_of_member.items():
        member_cluster = cluster_of_member.get(member)
        if member_cluster is not None:
            overlaps[old_id][member_cluster] = (
                overlaps[old_id].get(member_cluster, 0) + 1
            )

    components: list[tuple[set[str], set[int]]] = []
    visited_old: set[str] = set()
    visited_cluster: set[int] = set()
    for seed_old in sorted(overlaps):
        if seed_old in visited_old:
            continue
        component_old: set[str] = set()
        component_cluster: set[int] = set()
        stack = [seed_old]
        while stack:
            old_id = stack.pop()
            if old_id in component_old:
                continue
            component_old.add(old_id)
            for cluster_index in overlaps[old_id]:
                if cluster_index not in component_cluster:
                    component_cluster.add(cluster_index)
                    for neighbor in overlaps:
                        if cluster_index in overlaps[neighbor]:
                            stack.append(neighbor)
        component_old -= visited_old
        component_cluster -= visited_cluster
        visited_old |= component_old
        visited_cluster |= component_cluster
        components.append((component_old, component_cluster))

    taken_ids = set(used_ids) | set(old_membership)
    current_id: dict[int, str] = {}
    aliases: dict[str, str] = {}
    events: list[EntityEvent] = []

    for component_old, component_cluster in components:
        old_ids = sorted(component_old)
        cluster_indices = sorted(
            component_cluster,
            key=lambda index: _id_key(min(normalized[index], key=_id_key)),
        )
        # A connected component need not be fully bipartite: an old
        # entity can overlap only some of the component's clusters, so
        # missing edges are zero-weight rather than absent keys.
        component_overlap = {
            old_id: {index: overlaps[old_id].get(index, 0) for index in cluster_indices}
            for old_id in old_ids
        }
        assignment = _greedy_matches(component_overlap, normalized)
        anchor_of: dict[int, str] = {}

        # 1. Anchors: an old entity matched one-to-one to a cluster keeps
        #    its identity.
        for old_id in old_ids:
            if old_id in assignment:
                cluster_index = assignment[old_id]
                current_id[cluster_index] = old_id
                anchor_of[cluster_index] = old_id

        # 2. Every remaining cluster in the component gets a freshly
        #    minted, deterministic id.
        for cluster_index in cluster_indices:
            if cluster_index not in current_id:
                new_id = mint_entity_id(
                    normalized[cluster_index],
                    id_prefix=id_prefix,
                    used_ids=taken_ids,
                )
                taken_ids.add(new_id)
                current_id[cluster_index] = new_id

        # 3. Old entities without an anchor cluster redirect to the
        #    current entity holding the largest share of their members.
        #    Only clusters actually sharing members are eligible; ties
        #    resolve on the deterministic cluster ordering.
        for old_id in old_ids:
            if old_id not in assignment:
                weights = component_overlap[old_id]
                eligible = [index for index in cluster_indices if weights[index] > 0]
                best_index = max(eligible, key=lambda index: weights[index])
                aliases[old_id] = current_id[best_index]

        if len(old_ids) == 1 and len(cluster_indices) == 1:
            old_id = old_ids[0]
            cluster_index = cluster_indices[0]
            events.append(
                EntityEvent(
                    kind=RETAINED,
                    entity_id=current_id[cluster_index],
                    sources=(old_id,),
                    members=tuple(_sorted(normalized[cluster_index])),
                )
            )

        elif len(cluster_indices) == 1:
            # Many old entities -> one cluster: a merge.
            cluster_index = cluster_indices[0]
            survivor = anchor_of[cluster_index]
            members = normalized[cluster_index]
            changed_records: tuple[RecordID, ...] = tuple(
                _sorted(
                    member
                    for member in members
                    if previous_entity_of_member.get(member) not in (None, survivor)
                )
            )
            events.append(
                EntityEvent(
                    kind=MERGED,
                    entity_id=survivor,
                    sources=tuple(old_id for old_id in old_ids if old_id != survivor),
                    members=tuple(_sorted(members)),
                    changed_members=changed_records,
                )
            )

        elif len(old_ids) == 1:
            # One old entity -> many clusters: a split. The anchor
            # cluster keeps the id; the others are the split children.
            old_id = old_ids[0]
            anchor_index = assignment[old_id]
            anchor_id = current_id[anchor_index]
            split_children: list[tuple[str, tuple[RecordID, ...]]] = []
            split_changed: list[RecordID] = []
            for cluster_index in cluster_indices:
                if cluster_index == anchor_index:
                    continue
                split_children.append(
                    (
                        current_id[cluster_index],
                        tuple(_sorted(normalized[cluster_index])),
                    )
                )
                split_changed.extend(normalized[cluster_index])
            events.append(
                EntityEvent(
                    kind=SPLIT,
                    entity_id=anchor_id,
                    sources=(old_id,),
                    members=tuple(_sorted(normalized[anchor_index])),
                    changed_members=tuple(_sorted(split_changed)),
                    children=tuple(split_children),
                )
            )

        else:
            # Mixed reorganization: both merges and splits inside one
            # connected component. Emit per-cluster events first ...
            for cluster_index in cluster_indices:
                # Only old entities actually sharing members with this
                # cluster are sources; zero-weight component edges are not.
                source_olds = sorted(
                    old_id
                    for old_id in old_ids
                    if component_overlap[old_id][cluster_index] > 0
                )
                members = normalized[cluster_index]
                cluster_survivor: str | None = anchor_of.get(cluster_index)
                if cluster_survivor is not None and len(source_olds) > 1:
                    mixed_changed: tuple[RecordID, ...] = tuple(
                        _sorted(
                            member
                            for member in members
                            if previous_entity_of_member.get(member)
                            not in (None, cluster_survivor)
                        )
                    )
                    events.append(
                        EntityEvent(
                            kind=MERGED,
                            entity_id=cluster_survivor,
                            sources=tuple(
                                old_id
                                for old_id in source_olds
                                if old_id != cluster_survivor
                            ),
                            members=tuple(_sorted(members)),
                            changed_members=mixed_changed,
                        )
                    )
                elif cluster_survivor is not None:
                    events.append(
                        EntityEvent(
                            kind=RETAINED,
                            entity_id=cluster_survivor,
                            sources=(cluster_survivor,),
                            members=tuple(_sorted(members)),
                        )
                    )
                elif not source_olds:
                    # No old entity contributes members: a genuinely new
                    # entity. Split products with old members are
                    # described by the split events below instead.
                    events.append(
                        EntityEvent(
                            kind=CREATED,
                            entity_id=current_id[cluster_index],
                            members=tuple(_sorted(members)),
                            changed_members=tuple(_sorted(members)),
                        )
                    )

            # ... then a split event for every old entity that fanned
            # out over more than one current entity.
            for old_id in old_ids:
                target_indices = sorted(
                    (
                        index
                        for index in component_overlap[old_id]
                        if component_overlap[old_id][index] > 0
                    ),
                    key=lambda index: _id_key(min(normalized[index], key=_id_key)),
                )
                if len(target_indices) <= 1:
                    continue
                mixed_anchor_index: int | None = assignment.get(old_id)
                anchor_id = (
                    current_id[mixed_anchor_index]
                    if mixed_anchor_index is not None
                    else aliases[old_id]
                )
                anchor_members: list[RecordID] = [
                    member
                    for member in old_membership[old_id]
                    if cluster_of_member.get(member) is not None
                    and current_id[cluster_of_member[member]] == anchor_id
                ]
                moved_members: list[RecordID] = [
                    member
                    for member in old_membership[old_id]
                    if cluster_of_member.get(member) is not None
                    and current_id[cluster_of_member[member]] != anchor_id
                ]
                if not moved_members:
                    # Every member stayed under the anchor identity; this
                    # is a merge, not a split.
                    continue
                mixed_children: tuple[tuple[str, tuple[RecordID, ...]], ...] = tuple(
                    (
                        current_id[index],
                        tuple(_sorted(normalized[index])),
                    )
                    for index in target_indices
                    if current_id[index] != anchor_id
                )
                events.append(
                    EntityEvent(
                        kind=SPLIT,
                        entity_id=anchor_id,
                        sources=(old_id,),
                        members=tuple(_sorted(anchor_members)),
                        changed_members=tuple(_sorted(moved_members)),
                        children=mixed_children,
                    )
                )

    # Clusters with no link to any previous entity are brand new.
    for cluster_index, members in enumerate(normalized):
        if cluster_index not in current_id:
            new_id = mint_entity_id(members, id_prefix=id_prefix, used_ids=taken_ids)
            taken_ids.add(new_id)
            current_id[cluster_index] = new_id
            events.append(
                EntityEvent(
                    kind=CREATED,
                    entity_id=new_id,
                    members=tuple(_sorted(members)),
                    changed_members=tuple(_sorted(members)),
                )
            )

    entities = [
        (current_id[index], normalized[index])
        for index in sorted(
            current_id,
            key=lambda index: _id_key(min(normalized[index], key=_id_key)),
        )
    ]
    events.sort(key=lambda event: (event.kind, event.entity_id))
    return _Reconciliation(entities=entities, aliases=aliases, events=events)


class EntityRegistry:
    """Publish stable entity ids across successive clustering rounds.

    Example:
        >>> registry = EntityRegistry()
        >>> result = registry.publish(
        ...     [((1, 2), (0.9, 0.9)), ((3,), (1.0,))], round=1
        ... )
        >>> registry.members[1] == registry.members[2]
        True

    Args:
        id_prefix: prefix for entity ids minted for clusters that
            cannot inherit a previous identity.
    """

    def __init__(self, id_prefix: str = "ent") -> None:
        self.id_prefix = id_prefix
        self._round = 0
        self._entities: dict[str, frozenset[RecordID]] = {}
        self._aliases: dict[str, str] = {}
        self._history: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------
    @property
    def round(self) -> int:
        """The last successfully published round."""
        return self._round

    @property
    def entities(self) -> Mapping[str, tuple[RecordID, ...]]:
        """Current entity id -> sorted member record ids."""
        return {
            entity_id: tuple(_sorted(members))
            for entity_id, members in self._entities.items()
        }

    @property
    def members(self) -> Mapping[RecordID, str]:
        """Current record id -> entity id mapping."""
        mapping: dict[RecordID, str] = {}
        for entity_id, members in self._entities.items():
            for member in members:
                mapping[member] = entity_id
        return mapping

    @property
    def aliases(self) -> Mapping[str, str]:
        """Merge aliases mapped to their (possibly indirect) survivor."""
        return {alias: self.resolve(alias) for alias in sorted(self._aliases)}

    @property
    def history(self) -> tuple[EntityEvent, ...]:
        events: list[EntityEvent] = []
        for entry in self._history:
            events.extend(
                EntityEvent.from_dict(cast("Mapping[str, object]", event))
                for event in cast("list[Mapping[str, object]]", entry["events"])
            )
        return tuple(events)

    def _retired_ids(self) -> set[str]:
        retired: set[str] = set()
        for entry in self._history:
            for event in cast("list[Mapping[str, object]]", entry["events"]):
                if event["kind"] == RETIRED:
                    retired.add(cast(str, event["entity_id"]))
        return retired

    def resolve(self, entity_id: str) -> str:
        """Return the current entity id for a current id or a merge alias.

        Alias chains are followed to their survivor. Raises
        :class:`EntityRetired` for an entity whose members have all
        disappeared and :class:`UnknownEntity` for an id the registry
        never published.
        """
        if entity_id in self._entities:
            return entity_id
        if entity_id in self._aliases:
            seen: set[str] = set()
            current = entity_id
            while current in self._aliases:
                if current in seen:
                    raise EntityRegistryError(f"Alias cycle detected at {current!r}")
                seen.add(current)
                current = self._aliases[current]
            if current not in self._entities:
                raise EntityRetired(
                    f"Entity {entity_id!r} was merged into {current!r}, "
                    "which has since retired"
                )
            return current
        if entity_id in self._retired_ids():
            raise EntityRetired(f"Entity {entity_id!r} has no remaining members")
        raise UnknownEntity(f"Unknown entity id {entity_id!r}")

    def members_for(self, entity_id: str) -> tuple[RecordID, ...]:
        """Return the current members of an entity id or merge alias."""
        return tuple(_sorted(self._entities[self.resolve(entity_id)]))

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------
    def publish(
        self,
        clusters: Iterable[object],
        round: int | None = None,
        *,
        master_path: str | os.PathLike[str] | None = None,
        member_map_path: str | os.PathLike[str] | None = None,
    ) -> PublishResult:
        """Publish a new clustering round and return stable entity ids.

        Args:
            clusters: clusters of the current round, either iterables of
                record ids or ``(record_ids, scores)`` tuples as yielded
                by :meth:`dedupe.Dedupe.partition`.
            round: monotonically increasing round label. Defaults to the
                previous round plus one. Re-publishing the current round
                with identical membership is a retry and changes nothing.
            master_path: if given, write the entity master file here.
            member_map_path: if given, write the record-id -> entity-id
                mapping here. When both paths are given the two files
                are replaced together (each via an atomic rename), so
                readers never see a partial publish.

        Returns:
            A :class:`PublishResult` with the entity assignments, the
            member mapping and the round's events.
        """
        cluster_list = list(clusters)
        proposed = [_members_of(cluster) for cluster in cluster_list]

        if round is None:
            round = self._round + 1
        if round < self._round:
            raise StaleRound(
                f"Round {round} is older than the published round " f"{self._round}"
            )

        current_sets = sorted(
            (frozenset(members) for members in self._entities.values()),
            key=lambda members: _id_key(min(members, key=_id_key)),
        )
        proposed_sets = sorted(
            proposed,
            key=lambda members: _id_key(min(members, key=_id_key)),
        )
        identical_membership = current_sets == proposed_sets

        if round == self._round and self._round > 0:
            if not identical_membership:
                raise AmbiguousRound(
                    f"Round {round} was already published with different "
                    "cluster membership"
                )
            # Idempotent retry: do not mutate any state or history.
            return self._result(round, ())

        if identical_membership and self._round > 0:
            # Same membership at a new round: every entity is retained,
            # no ids are minted and no aliases change.
            events = tuple(
                EntityEvent(
                    kind=RETAINED,
                    entity_id=entity_id,
                    sources=(entity_id,),
                    members=tuple(_sorted(members)),
                )
                for entity_id, members in sorted(self._entities.items())
            )
            self._round = round
            self._history.append(
                {"round": round, "events": [event.to_dict() for event in events]}
            )
            if master_path is not None or member_map_path is not None:
                self._write_files(master_path, member_map_path)
            return self._result(round, events)

        used_ids = set(self._entities) | set(self._aliases) | self._retired_ids()
        reconciliation = reconcile_clusters(
            self._entities,
            cluster_list,
            id_prefix=self.id_prefix,
            used_ids=used_ids,
        )

        # An old entity retires only if none of its members survive in
        # any new cluster. Merged or split entities are aliases or
        # sources and must not be retired.
        surviving_members: set[RecordID] = set()
        for _, members in reconciliation.entities:
            surviving_members.update(members)
        retired_events = [
            EntityEvent(kind=RETIRED, entity_id=old_id)
            for old_id, members in sorted(self._entities.items())
            if not members & surviving_members
        ]

        self._entities = {
            entity_id: members for entity_id, members in reconciliation.entities
        }
        self._aliases.update(reconciliation.aliases)
        events = tuple(reconciliation.events + retired_events)
        self._round = round
        self._history.append(
            {"round": round, "events": [event.to_dict() for event in events]}
        )

        if master_path is not None or member_map_path is not None:
            self._write_files(master_path, member_map_path)
        return self._result(round, events)

    def _result(self, round: int, events: tuple[EntityEvent, ...]) -> PublishResult:
        entities = tuple(
            (entity_id, tuple(_sorted(members)))
            for entity_id, members in sorted(
                self._entities.items(),
                key=lambda item: _id_key(min(item[1], key=_id_key)),
            )
        )
        return PublishResult(
            round=round,
            entities=entities,
            members=dict(self.members),
            events=events,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """Serialize the full registry state (master, aliases, history)."""
        return {
            "version": STATE_VERSION,
            "round": self._round,
            "id_prefix": self.id_prefix,
            "entities": {
                entity_id: list(_sorted(members))
                for entity_id, members in self._entities.items()
            },
            "aliases": dict(self._aliases),
            "history": list(self._history),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EntityRegistry":
        version = cast(int, data["version"])
        if version != STATE_VERSION:
            raise EntityRegistryError(
                f"Unsupported entity registry state version {version}; "
                f"expected {STATE_VERSION}"
            )
        registry = cls(id_prefix=cast(str, data["id_prefix"]))
        registry._round = cast(int, data["round"])
        registry._entities = {
            cast(str, entity_id): frozenset(cast("list[RecordID]", members))
            for entity_id, members in cast(
                "Mapping[str, object]", data["entities"]
            ).items()
        }
        registry._aliases = dict(cast("Mapping[str, str]", data["aliases"]))
        registry._history = list(cast("list[dict[str, object]]", data["history"]))
        return registry

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically write the full registry state to a single file."""
        directory = os.path.dirname(os.path.abspath(path))
        _atomic_write(
            path,
            json.dumps(self.to_dict(), sort_keys=True, indent=2).encode("utf-8"),
            directory,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "EntityRegistry":
        """Load a registry previously written with :meth:`save`."""
        with open(path, "rb") as file_obj:
            data = json.loads(file_obj.read().decode("utf-8"))
        return cls.from_dict(data)

    def master_document(self) -> dict[str, object]:
        """The entity master document written to ``master_path``."""
        return {
            "version": STATE_VERSION,
            "round": self._round,
            "entities": {
                entity_id: list(_sorted(members))
                for entity_id, members in self._entities.items()
            },
            "aliases": dict(self.aliases),
        }

    def member_map_document(self) -> dict[str, object]:
        """The member mapping document written to ``member_map_path``."""
        return {
            "version": STATE_VERSION,
            "round": self._round,
            "members": {
                json.dumps(member): entity_id
                for member, entity_id in sorted(
                    self.members.items(), key=lambda item: _id_key(item[0])
                )
            },
        }

    def _write_files(
        self,
        master_path: str | os.PathLike[str] | None,
        member_map_path: str | os.PathLike[str] | None,
    ) -> None:
        master_payload = json.dumps(
            self.master_document(), sort_keys=True, indent=2
        ).encode("utf-8")
        mapping_payload = json.dumps(
            self.member_map_document(), sort_keys=True, indent=2
        ).encode("utf-8")

        if master_path is not None and member_map_path is not None:
            # Stage both documents, then swap them in together so the
            # master and the member mapping always describe the same
            # round.
            master_dir = os.path.dirname(os.path.abspath(master_path))
            mapping_dir = os.path.dirname(os.path.abspath(member_map_path))
            master_tmp = _stage(master_payload, master_dir)
            try:
                mapping_tmp = _stage(mapping_payload, mapping_dir)
            except BaseException:
                os.unlink(master_tmp)
                raise
            os.replace(mapping_tmp, member_map_path)
            os.replace(master_tmp, master_path)
            return

        if master_path is not None:
            _atomic_write(
                master_path,
                master_payload,
                os.path.dirname(os.path.abspath(master_path)),
            )
        if member_map_path is not None:
            _atomic_write(
                member_map_path,
                mapping_payload,
                os.path.dirname(os.path.abspath(member_map_path)),
            )


def _stage(payload: bytes, directory: str) -> str:
    file_obj = tempfile.NamedTemporaryFile(
        mode="wb", dir=directory, delete=False, prefix=".entity-", suffix=".tmp"
    )
    try:
        file_obj.write(payload)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    finally:
        file_obj.close()
    return file_obj.name


def _atomic_write(path: str | os.PathLike[str], payload: bytes, directory: str) -> None:
    tmp_path = _stage(payload, directory)
    os.replace(tmp_path, path)
