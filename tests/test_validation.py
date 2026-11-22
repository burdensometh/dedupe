import json
import os
import tempfile
import unittest

import dedupe
import dedupe.validation as validation


def _toy_data():
    # entity "a" has two records (a duplicate pair); everything else is a
    # singleton entity.  "missing" rows exercise the None / "" handling.
    return {
        "a1": {"name": "acme", "entity": "a"},
        "a2": {"name": "acme inc", "entity": "a"},
        "b1": {"name": "beta", "entity": "b"},
        "c1": {"name": "gamma", "entity": "c"},
        "d1": {"name": "delta", "entity": "d"},
        "e1": {"name": "epsilon", "entity": None},
        "f1": {"name": "zeta", "entity": ""},
    }


class EntityIdTest(unittest.TestCase):
    def test_shared_key_is_one_entity(self):
        entity_of = validation.entity_ids(_toy_data(), entity_key="entity")
        assert entity_of["a1"] == entity_of["a2"]
        assert entity_of["a1"] != entity_of["b1"]

    def test_missing_and_blank_keys_are_not_merged(self):
        entity_of = validation.entity_ids(_toy_data(), entity_key="entity")
        assert entity_of["e1"] != entity_of["f1"]
        assert entity_of["e1"] != entity_of["a1"]

    def test_known_pairs_are_transitive(self):
        data = {i: {"name": str(i)} for i in range(5)}
        entity_of = validation.entity_ids(data, known_pairs=[(0, 1), (1, 2)])
        assert entity_of[0] == entity_of[1] == entity_of[2]
        assert entity_of[3] != entity_of[0]

    def test_positive_pairs(self):
        entity_of = validation.entity_ids(_toy_data(), entity_key="entity")
        assert validation.positive_pairs(entity_of) == [("a1", "a2")]


class FoldPlanTest(unittest.TestCase):
    def setUp(self):
        self.data = _toy_data()
        self.plan = validation.EntityPlan.from_data(
            self.data, entity_key="entity", n_folds=2, seed=1
        )

    def test_folds_are_entity_pure(self):
        for fold in self.plan.fold_indices:
            train_entities = {
                self.plan.entity_of[r] for r in self.plan.training_records(fold)
            }
            valid_entities = {
                self.plan.entity_of[r] for r in self.plan.validation_records(fold)
            }
            assert not (train_entities & valid_entities)
            assert train_entities | valid_entities == set(self.plan.entity_of.values())

    def test_coverage_partitions_entities_and_records(self):
        coverage = self.plan.coverage()
        assert sum(row["validation_entities"] for row in coverage) == len(
            set(self.plan.entity_of.values())
        )
        assert sum(row["validation_records"] for row in coverage) == len(self.data)
        for row in coverage:
            assert 0.0 < row["validation_entity_share"] <= 1.0

    def test_seed_is_reproducible(self):
        again = validation.EntityPlan.from_data(
            self.data, entity_key="entity", n_folds=2, seed=1
        )
        assert again.folds == self.plan.folds

    def test_training_pairs_never_touch_validation(self):
        for fold in self.plan.fold_indices:
            valid_ids = set(self.plan.validation_records(fold))
            valid_records = {id(self.data[r]) for r in valid_ids}
            training = self.plan.training_pairs(fold, n_distinct=10)
            for pair in training["match"] + training["distinct"]:
                assert id(pair[0]) not in valid_records
                assert id(pair[1]) not in valid_records

    def test_distinct_pairs_cross_entities(self):
        fold = self.plan.fold_indices[0]
        training = self.plan.training_pairs(fold, n_distinct=20)
        for record_1, record_2 in training["distinct"]:
            id_1 = next(r for r, rec in self.data.items() if rec is record_1)
            id_2 = next(r for r, rec in self.data.items() if rec is record_2)
            assert self.plan.entity_of[id_1] != self.plan.entity_of[id_2]

    def test_round_trip_save_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "plan.json")
            self.plan.save(path)
            loaded = validation.EntityPlan.load(path, self.data)
            assert loaded.folds == self.plan.folds
            assert loaded.entity_of == self.plan.entity_of

    def test_too_many_folds_rejected(self):
        with self.assertRaises(ValueError):
            validation.EntityPlan.from_data(self.data, entity_key="entity", n_folds=100)


class LeakageTest(unittest.TestCase):
    def setUp(self):
        self.data = _toy_data()
        self.entity_of = validation.entity_ids(self.data, entity_key="entity")

    def test_straddling_pair_is_reported(self):
        record_fold = {record: 0 for record in self.data}
        record_fold["a2"] = 1
        audit = validation.audit_record_folds(self.entity_of, record_fold)
        assert not audit.is_entity_pure
        assert audit.straddling_pairs == [("a1", "a2")]

    def test_record_level_plan_rejects_leak(self):
        record_fold = {record: 0 for record in self.data}
        record_fold["a2"] = 1
        with self.assertRaises(validation.EntityLeakageError) as context:
            validation.EntityPlan.from_record_folds(
                self.data, record_fold, entity_of=self.entity_of
            )
        assert ("a1", "a2") in context.exception.pairs

    def test_union_mode_moves_whole_entity(self):
        record_fold = {record: 0 for record in self.data}
        record_fold["a2"] = 1
        plan = validation.EntityPlan.from_record_folds(
            self.data,
            record_fold,
            entity_of=self.entity_of,
            on_straddling="union",
        )
        label = self.entity_of["a1"]
        # the emptied fold is dropped; "a" moved to fold 0 with all records
        assert plan.fold_indices == (0,)
        assert label in plan.validation_entities(0)
        assert set(plan.validation_records(0)) == set(self.data)

    def test_union_mode_positive_pairs_never_negative(self):
        record_fold = {record: 0 for record in self.data}
        record_fold["a2"] = 1
        plan = validation.EntityPlan.from_record_folds(
            self.data, record_fold, entity_of=self.entity_of, on_straddling="union"
        )
        for fold in plan.fold_indices:
            ids = set(plan.validation_records(fold))
            for a, b in validation.positive_pairs(plan.entity_of):
                if a in ids and b in ids:
                    continue
                # a positive pair is never emitted as distinct
                training = plan.training_pairs(fold, n_distinct=50)
                distinct = {frozenset((id(x), id(y))) for x, y in training["distinct"]}
                assert frozenset((id(self.data[a]), id(self.data[b]))) not in distinct

    def test_entity_pure_record_level_plan_accepted(self):
        # assign fold per entity rather than per record
        label = self.entity_of["a1"]
        record_fold = {
            record: int(entity == label) for record, entity in self.entity_of.items()
        }
        plan = validation.EntityPlan.from_record_folds(
            self.data, record_fold, entity_of=self.entity_of
        )
        assert plan.folds[1] == (label,)
        assert plan.validation_entities(1) == frozenset({label})


class QualityTest(unittest.TestCase):
    def test_cross_fold_positive_is_not_a_negative(self):
        # the true match a1/a2 never appears as a distinct example
        plan = validation.EntityPlan.from_data(
            _toy_data(), entity_key="entity", n_folds=2, seed=1
        )
        for fold in plan.fold_indices:
            training = plan.training_pairs(fold, n_distinct=50)
            distinct_pairs = {
                frozenset((record_1["name"], record_2["name"]))
                for record_1, record_2 in training["distinct"]
            }
            assert frozenset(("acme", "acme inc")) not in distinct_pairs

    def test_link_quality_math(self):
        truth = {frozenset((1, 2)), frozenset((3, 4))}
        candidates = {frozenset((1, 2)), frozenset((3, 4)), frozenset((5, 6))}
        predicted = {frozenset((1, 2)), frozenset((5, 6))}
        quality = validation.LinkQuality.from_pairs(truth, candidates, predicted)
        assert quality.true_positives == 1
        assert quality.n_truth == 2
        assert quality.precision == 0.5
        assert quality.recall == 0.5
        assert quality.candidate_recall == 1.0
        assert quality.f1 == 0.5

    def test_report_aggregates(self):
        quality = validation.LinkQuality.from_pairs(
            {frozenset((1, 2))}, {frozenset((1, 2))}, {frozenset((1, 2))}
        )
        report = validation.CrossValidationReport.from_quality(
            [{"fold": 0}], [quality], [0]
        )
        assert report.aggregate["precision"] == 1.0
        assert report.aggregate["recall"] == 1.0
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["aggregate"]["n_folds"] == 1


class LinkPlanTest(unittest.TestCase):
    def setUp(self):
        self.data_1 = {
            "a": {"name": "acme", "entity": "a"},
            "b": {"name": "beta", "entity": "b"},
            "c": {"name": "gamma", "entity": "c"},
        }
        self.data_2 = {
            "x": {"name": "acme", "entity": "a"},
            "y": {"name": "beta", "entity": "b"},
            "z": {"name": "gamma", "entity": "c"},
            "w": {"name": "omega", "entity": "d"},
        }
        self.plan = validation.LinkPlan.from_data(
            self.data_1, self.data_2, entity_key="entity", n_folds=2, seed=3
        )

    def test_entity_resolved_across_datasets(self):
        assert self.plan.entity_of[("1", "a")] == self.plan.entity_of[("2", "x")]

    def test_truth_pairs_only_within_validation_entities(self):
        for fold in self.plan.folds:
            ids_1, ids_2 = self.plan.validation_records(fold)
            for a, b in self.plan.truth_pairs(fold):
                assert a in ids_1 and b in ids_2
                assert self.plan.entity_of[("1", a)] == self.plan.entity_of[("2", b)]

    def test_link_training_pairs_exclude_validation(self):
        for fold in self.plan.folds:
            ids_1, ids_2 = self.plan.validation_records(fold)
            training = self.plan.training_pairs(fold, n_distinct=10)
            names_1 = {self.data_1[r]["name"] for r in ids_1}
            names_2 = {self.data_2[r]["name"] for r in ids_2}
            for record_1, record_2 in training["match"] + training["distinct"]:
                assert record_1["name"] not in names_1
                assert record_2["name"] not in names_2

    def test_plan_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "link-plan.json")
            self.plan.save(path)
            loaded = validation.LinkPlan.load(path, self.data_1, self.data_2)
            assert loaded.folds == self.plan.folds
            assert loaded.entity_of == self.plan.entity_of


class CrossValidationRunTest(unittest.TestCase):
    """End-to-end training on a tiny dataset: real fold models, no leakage."""

    @classmethod
    def setUpClass(cls):
        # 12 records over 6 entities; one entity has 3 near-identical records.
        cls.data = {}
        names = {
            "a": ["acme", "acme inc", "acme incorporated"],
            "b": ["beta", "beta llc"],
            "c": ["gamma", "gamma corp"],
            "d": ["delta"],
            "e": ["epsilon"],
            "f": ["zeta"],
        }
        for entity, variants in names.items():
            for i, name in enumerate(variants):
                cls.data[f"{entity}{i}"] = {"name": name, "entity": entity}
        cls.variables = [dedupe.variables.String("name")]

    def test_run_dedupe_cv(self):
        plan = validation.EntityPlan.from_data(
            self.data, entity_key="entity", n_folds=2, seed=0
        )
        report = validation.run_dedupe_cv(
            plan, self.variables, n_distinct=20, num_cores=0
        )
        assert report.aggregate["n_folds"] == 2
        assert report.aggregate["n_truth"] == (
            len(validation.positive_pairs(plan.entity_of))
        )
        # every reported truth pair is inside a validation entity
        for row in report.coverage:
            fold = row["fold"]
            valid_ids = set(plan.validation_records(fold))
            for a, b in validation.positive_pairs(plan.entity_of):
                if a in valid_ids and b in valid_ids:
                    assert plan.entity_of[a] == plan.entity_of[b]

    def test_run_link_cv(self):
        entities = [f"e{i}" for i in range(12)]
        data_1 = {
            f"{entity}_1": {"name": f"name {entity}", "entity": entity}
            for entity in entities
        }
        data_2 = {
            f"{entity}_2": {"name": f"name {entity}", "entity": entity}
            for entity in entities
        }
        plan = validation.LinkPlan.from_data(
            data_1, data_2, entity_key="entity", n_folds=2, seed=0
        )
        report = validation.run_link_cv(
            plan, self.variables, n_distinct=50, num_cores=0
        )
        assert report.aggregate["n_truth"] == 12
        assert sum(row["n_truth"] for row in report.folds) == 12

    def test_refit_and_replay_matches_saved_plan(self):
        plan = validation.EntityPlan.from_data(
            self.data, entity_key="entity", n_folds=2, seed=0
        )
        matcher = dedupe.Dedupe(self.variables, num_cores=0)
        validation.refit_full_model(
            matcher,
            self.data,
            entity_key="entity",
            sample_size=len(self.data),
            n_distinct=20,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "plan.json")
            plan.save(path)
            loaded = validation.EntityPlan.load(path, self.data)
        first = validation.replay_dedupe(plan, matcher)
        second = validation.replay_dedupe(loaded, matcher)
        assert first.to_dict() == second.to_dict()
        assert first.aggregate["n_truth"] == len(
            validation.positive_pairs(plan.entity_of)
        )

    def test_training_inputs_are_reproducible(self):
        plan = validation.EntityPlan.from_data(
            self.data, entity_key="entity", n_folds=2, seed=0
        )
        digest_a = validation.training_digest(
            plan.training_pairs(0, n_distinct=10, seed=5)
        )
        digest_b = validation.training_digest(
            plan.training_pairs(0, n_distinct=10, seed=5)
        )
        assert digest_a == digest_b

    def test_refit_full_link_uses_all_entities(self):
        entities = [f"e{i}" for i in range(12)]
        data_1 = {
            f"{entity}_1": {"name": f"name {entity}", "entity": entity}
            for entity in entities
        }
        data_2 = {
            f"{entity}_2": {"name": f"name {entity}", "entity": entity}
            for entity in entities
        }
        matcher = dedupe.RecordLink(self.variables, num_cores=0)
        validation.refit_full_link(
            matcher,
            data_1,
            data_2,
            entity_key="entity",
            sample_size=len(data_1),
            n_distinct=50,
        )
        # training pairs cover every entity's cross-source match
        plan = validation.LinkPlan.from_data(
            data_1, data_2, entity_key="entity", n_folds=2, seed=0
        )
        training = validation.full_link_training_pairs(
            data_1, data_2, plan.entity_of, n_distinct=50, seed=0
        )
        assert len(training["match"]) == len(entities)


if __name__ == "__main__":
    unittest.main()
