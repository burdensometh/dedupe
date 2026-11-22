import json
import os
import tempfile
import unittest

from dedupe.entity_registry import (
    CREATED,
    MERGED,
    RETAINED,
    SPLIT,
    AmbiguousRound,
    EntityEvent,
    EntityRegistry,
    EntityRetired,
    StaleRound,
    UnknownEntity,
    mint_entity_id,
    reconcile_clusters,
)


def by_member(registry, *members):
    ids = {registry.members[member] for member in members}
    assert len(ids) == 1
    return next(iter(ids))


class MintIdTest(unittest.TestCase):
    def test_deterministic_and_order_invariant(self):
        first = mint_entity_id([1, 2, 3])
        self.assertEqual(first, mint_entity_id([3, 1, 2]))
        self.assertTrue(first.startswith("ent_"))

    def test_prefix(self):
        self.assertTrue(
            mint_entity_id([1], id_prefix="customer").startswith("customer_")
        )

    def test_distinct_members_distinct_ids(self):
        self.assertNotEqual(mint_entity_id([1]), mint_entity_id([2]))
        self.assertNotEqual(mint_entity_id([1, 2]), mint_entity_id([1, 2, 3]))

    def test_collision_suffix_is_deterministic(self):
        minted = mint_entity_id([1], used_ids={mint_entity_id([1])})
        self.assertEqual(minted, mint_entity_id([1]) + "_1")
        minted_again = mint_entity_id(
            [1], used_ids={mint_entity_id([1]), mint_entity_id([1]) + "_1"}
        )
        self.assertEqual(minted_again, mint_entity_id([1]) + "_2")

    def test_string_and_int_ids_are_distinct(self):
        self.assertNotEqual(mint_entity_id([1]), mint_entity_id(["1"]))


class ReconcileTest(unittest.TestCase):
    def test_first_round_creates_all(self):
        result = reconcile_clusters({}, [(1, 2), (3,)])
        kinds = sorted(event.kind for event in result.events)
        self.assertEqual(kinds, [CREATED, CREATED])
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(result.aliases, {})

    def test_retain_identity(self):
        old = {"ent_a": frozenset([1, 2])}
        result = reconcile_clusters(old, [(1, 2)])
        self.assertEqual(result.entities, [("ent_a", frozenset([1, 2]))])
        self.assertEqual(result.events[0].kind, RETAINED)
        self.assertEqual(result.aliases, {})

    def test_retain_accepts_partition_tuples(self):
        old = {"ent_a": frozenset([1, 2])}
        result = reconcile_clusters(old, [((1, 2), (0.9, 0.9))])
        self.assertEqual(result.entities, [("ent_a", frozenset([1, 2]))])

    def test_merge_records_alias_to_survivor(self):
        old = {
            "ent_a": frozenset([1, 2]),
            "ent_b": frozenset([3, 4]),
        }
        result = reconcile_clusters(old, [(1, 2, 3, 4)])
        self.assertEqual(len(result.entities), 1)
        survivor, members = result.entities[0]
        self.assertEqual(members, frozenset([1, 2, 3, 4]))
        self.assertEqual(set(result.aliases.values()), {survivor})
        merged = result.events[0]
        self.assertEqual(merged.kind, MERGED)
        # The members absorbed from the non-surviving entity are called out.
        non_survivor = next(iter(result.aliases))
        self.assertEqual(set(merged.changed_members), set(old[non_survivor]))

    def test_split_states_changed_members_and_children(self):
        old = {"ent_a": frozenset([1, 2, 3, 4])}
        result = reconcile_clusters(old, [(1, 2), (3, 4)])
        ids = {entity_id: members for entity_id, members in result.entities}
        self.assertIn("ent_a", ids)
        split = result.events[0]
        self.assertEqual(split.kind, SPLIT)
        self.assertTrue(split.changed_members)
        self.assertEqual(len(split.children), 1)
        child_id, child_members = split.children[0]
        self.assertIn(child_id, ids)
        self.assertEqual(set(split.changed_members), set(child_members))
        # Anchor keeps the members it retained.
        self.assertEqual(ids["ent_a"], frozenset(split.members))

    def test_mixed_reorganization(self):
        old = {
            "ent_a": frozenset([1, 2]),
            "ent_b": frozenset([3, 4]),
        }
        result = reconcile_clusters(old, [(1, 3), (2, 4)])
        kinds = {event.kind for event in result.events}
        self.assertIn(MERGED, kinds)
        self.assertIn(SPLIT, kinds)
        for event in result.events:
            if event.kind == SPLIT:
                self.assertTrue(event.changed_members)
                self.assertTrue(event.children)
        # Every member ends up assigned exactly once.
        assigned = set()
        for _, members in result.entities:
            assigned |= members
        self.assertEqual(assigned, {1, 2, 3, 4})

    def test_sparse_component_missing_edges_do_not_crash(self):
        # Connected but not fully bipartite: ent_a only meets the first
        # new cluster, ent_b meets both. Zero-weight edges must be handled.
        old = {
            "ent_a": frozenset([1, 2]),
            "ent_b": frozenset([3, 4]),
        }
        result = reconcile_clusters(old, [(1, 2, 3), (4, 5)])
        assigned = set()
        for _, members in result.entities:
            assigned |= members
        self.assertEqual(assigned, {1, 2, 3, 4, 5})
        # The anchored cluster retains ent_a even though it shares the
        # component with ent_b (no spurious merge).
        anchor = dict(result.entities)
        self.assertEqual(anchor["ent_a"], frozenset([1, 2, 3]))

    def test_split_product_is_not_marked_created(self):
        # Split products carrying old members must not be reported as
        # brand-new entities; only an all-unseen-member cluster is created.
        old = {
            "ent_a": frozenset([1, 2]),
            "ent_b": frozenset([3, 4]),
        }
        result = reconcile_clusters(old, [(1, 2, 3), (4, 5), (8, 9)])
        created = [event for event in result.events if event.kind == CREATED]
        self.assertEqual(len(created), 1)
        created_entity = created[0].entity_id
        entities = dict(result.entities)
        self.assertEqual(entities[created_entity], frozenset([8, 9]))
        # Split events only describe entities that actually lost members.
        for event in result.events:
            if event.kind == SPLIT:
                self.assertTrue(event.changed_members)

    def test_new_unmappable_cluster_gets_deterministic_id(self):
        old = {"ent_a": frozenset([1])}
        first = reconcile_clusters(old, [(1,), (8, 9)])
        second = reconcile_clusters(old, [(8, 9), (1,)])
        self.assertEqual(first.entities, second.entities)

    def test_rejects_overlapping_clusters(self):
        with self.assertRaises(ValueError):
            reconcile_clusters({}, [(1, 2), (2, 3)])

    def test_rejects_empty_cluster(self):
        with self.assertRaises(ValueError):
            reconcile_clusters({}, [()])

    def test_partial_overlap_split_does_not_alias_anchor(self):
        # {1,2,3} -> {1,2} and {3,4}: 3 shares an anchor but 4 is new.
        old = {"ent_a": frozenset([1, 2, 3])}
        result = reconcile_clusters(old, [(1, 2), (3, 4)])
        self.assertNotIn("ent_a", result.aliases)
        self.assertIn("ent_a", dict(result.entities))


class RegistryPublishTest(unittest.TestCase):
    def setUp(self):
        self.registry = EntityRegistry()

    def test_publish_returns_master_and_member_mapping_together(self):
        result = self.registry.publish([((1, 2), (0.9, 0.9)), ((3,), (1.0,))], round=1)
        self.assertEqual(self.registry.members[1], self.registry.members[2])
        self.assertNotEqual(self.registry.members[1], self.registry.members[3])
        self.assertEqual(result.round, 1)
        # Result mapping and registry mapping agree.
        self.assertEqual(dict(result.members), dict(self.registry.members))
        self.assertEqual(
            {entity_id for entity_id, _ in result.entities},
            set(self.registry.entities),
        )

    def test_reordering_input_keeps_identity(self):
        clusters = [(1, 2), (3, 4), (5,)]
        first = EntityRegistry().publish(clusters, round=1)
        shuffled = [(5,), (3, 4), (2, 1)]
        second = EntityRegistry().publish(shuffled, round=1)
        self.assertEqual(dict(first.entities), dict(second.entities))
        self.assertEqual(first.members, second.members)

    def test_retry_same_round_is_idempotent(self):
        self.registry.publish([(1, 2), (3,)], round=1)
        before = self.registry.to_dict()
        result = self.registry.publish([(3,), (2, 1)], round=1)
        self.assertEqual(self.registry.to_dict(), before)
        self.assertEqual(result.events, ())

    def test_same_membership_new_round_retains_without_new_ids(self):
        first = self.registry.publish([(1, 2), (3,)], round=1)
        second = self.registry.publish([(3,), (1, 2)], round=2)
        self.assertEqual(
            {entity_id for entity_id, _ in first.entities},
            {entity_id for entity_id, _ in second.entities},
        )
        self.assertTrue(all(event.kind == RETAINED for event in second.events))

    def test_merged_entity_is_queryable_to_current(self):
        self.registry.publish([(1, 2), (3, 4)], round=1)
        absorbed = by_member(self.registry, 3, 4)
        self.registry.publish([(1, 2, 3, 4)], round=2)
        survivor = self.registry.members[1]
        self.assertEqual(self.registry.resolve(absorbed), survivor)
        self.assertEqual(self.registry.members_for(absorbed), (1, 2, 3, 4))

    def test_split_event_explains_membership_changes(self):
        self.registry.publish([(1, 2, 3, 4)], round=1)
        original = self.registry.members[1]
        result = self.registry.publish([(1, 2), (3, 4)], round=2)
        split_events = [event for event in result.events if event.kind == SPLIT]
        self.assertEqual(len(split_events), 1)
        event = split_events[0]
        self.assertEqual(event.entity_id, original)
        changed = set(event.changed_members)
        # The members now under a child id are precisely the changed ones.
        child_members = set()
        for _, members in event.children:
            child_members |= set(members)
        self.assertEqual(changed, child_members)
        # Anchor members kept their identity.
        self.assertEqual(
            self.registry.members[1],
            self.registry.members[2],
            original,
        )

    def test_retired_entity_raises_on_resolve(self):
        self.registry.publish([(1, 2)], round=1)
        retired = self.registry.members[1]
        self.registry.publish([(9,)], round=2)
        with self.assertRaises(EntityRetired):
            self.registry.resolve(retired)

    def test_alias_to_a_survivor_that_later_retires_raises(self):
        self.registry.publish([(1, 2), (3, 4)], round=1)
        absorbed = by_member(self.registry, 3, 4)
        self.registry.publish([(1, 2, 3, 4)], round=2)
        # Every member of the merged entity then disappears.
        self.registry.publish([(9,)], round=3)
        with self.assertRaises(EntityRetired):
            self.registry.resolve(absorbed)

    def test_unknown_entity(self):
        with self.assertRaises(UnknownEntity):
            EntityRegistry().resolve("ent_never")

    def test_stale_and_ambiguous_rounds(self):
        self.registry.publish([(1,)], round=2)
        with self.assertRaises(StaleRound):
            self.registry.publish([(1,)], round=1)
        with self.assertRaises(AmbiguousRound):
            self.registry.publish([(2,)], round=2)

    def test_auto_incrementing_round(self):
        self.registry.publish([(1,)])
        self.registry.publish([(1,)])
        self.assertEqual(self.registry.round, 2)

    def test_history_records_every_round(self):
        self.registry.publish([(1, 2), (3,)], round=1)
        self.registry.publish([(1, 2, 3)], round=2)
        history = self.registry.history
        self.assertTrue(any(event.kind == MERGED for event in history))
        round_events = [entry for entry in self.registry.to_dict()["history"]]
        self.assertEqual([entry["round"] for entry in round_events], [1, 2])

    def test_alias_chain_resolves_to_final_survivor(self):
        self.registry.publish([(1,), (2,), (3,)], round=1)
        first_absorbed = self.registry.members[2]
        self.registry.publish([(1, 2), (3,)], round=2)
        self.registry.publish([(1, 2, 3)], round=3)
        survivor = self.registry.members[1]
        self.assertEqual(self.registry.resolve(first_absorbed), survivor)
        self.assertEqual(self.registry.members_for(first_absorbed), (1, 2, 3))

    def test_string_record_ids(self):
        result = EntityRegistry().publish([("a", "b"), ("c",)], round=1)
        self.assertEqual(set(result.members), {"a", "b", "c"})

    def test_as_clusters_shape(self):
        result = self.registry.publish([(1, 2), (3,)], round=1)
        clusters = result.as_clusters()
        self.assertEqual(
            sorted(sorted(cluster) for cluster in clusters),
            [[1, 2], [3]],
        )


class PersistenceTest(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        registry = EntityRegistry()
        registry.publish([(1, 2), (3, 4)], round=1)
        registry.publish([(1, 2, 3, 4)], round=2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "registry.json")
            registry.save(path)
            loaded = EntityRegistry.load(path)
        self.assertEqual(loaded.to_dict(), registry.to_dict())
        self.assertEqual(loaded.members, registry.members)
        self.assertEqual(loaded.aliases, registry.aliases)

    def test_master_and_member_map_written_together(self):
        registry = EntityRegistry()
        registry.publish([(1, 2), (3,)], round=1)
        with tempfile.TemporaryDirectory() as directory:
            master_path = os.path.join(directory, "master.json")
            mapping_path = os.path.join(directory, "members.json")
            registry.publish(
                [(1, 2, 3)],
                round=2,
                master_path=master_path,
                member_map_path=mapping_path,
            )
            with open(master_path) as file_obj:
                master = json.load(file_obj)
            with open(mapping_path) as file_obj:
                mapping = json.load(file_obj)
        # Both documents describe the same published round.
        self.assertEqual(master["round"], mapping["round"])
        self.assertEqual(set(master["entities"]), set(registry.entities))
        self.assertEqual(mapping["members"]["1"], registry.members[1])
        # Aliases for merged ids are published in the master.
        self.assertEqual(master["aliases"], dict(registry.aliases))

    def test_event_serialization_round_trip(self):
        registry = EntityRegistry()
        registry.publish([(1, 2, 3, 4)], round=1)
        registry.publish([(1, 2), (3, 4)], round=2)
        for event in registry.history:
            restored = EntityEvent.from_dict(event.to_dict())
            self.assertEqual(restored, event)


if __name__ == "__main__":
    unittest.main()
