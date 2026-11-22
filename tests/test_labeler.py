import random
import unittest

import numpy
import pytest

import dedupe
from dedupe import datamodel, labeler
from dedupe._typing import RecordDictPair

SAMPLE = {
    1: {"name": "Meredith", "age": "40"},
    2: {"name": "Sue", "age": "10"},
    3: {"name": "Willy", "age": "35"},
    4: {"name": "William", "age": "35"},
    5: {"name": "Jimmy", "age": "20"},
    6: {"name": "Jimbo", "age": "21"},
}

SOURCED_SAMPLE = {
    **{
        i: {"name": name, "age": str(20 + i), "source": "big"}
        for i, name in enumerate(
            [
                "Meredith",
                "Sue",
                "Willy",
                "William",
                "Pamela",
                "Pam",
                "Sammy",
                "Sam",
            ],
            start=1,
        )
    },
    10: {"name": "Jimmy", "age": "20", "source": "rare_a"},
    11: {"name": "Jimbo", "age": "21", "source": "rare_a"},
    12: {"name": "Georgie", "age": "40", "source": "rare_b"},
    13: {"name": "Georgette", "age": "41", "source": "rare_b"},
}


def freeze_record_pair(record_pair: RecordDictPair):
    rec1, rec2 = record_pair
    return (frozenset(rec1.items()), frozenset(rec2.items()))


def pair_source(record_pair: RecordDictPair):
    source_1, source_2 = record_pair[0]["source"], record_pair[1]["source"]
    if source_1 == source_2:
        return source_1
    return tuple(sorted((source_1, source_2)))


def build_learner(sample=SOURCED_SAMPLE):
    data_model = datamodel.DataModel(
        [dedupe.variables.String("name"), dedupe.variables.String("age")]
    )
    random.seed(1111111111110)
    return labeler.DedupeDisagreementLearner(
        data_model.predicates, data_model.distances, sample, []
    )


class ActiveLearningTest(unittest.TestCase):
    def setUp(self):
        self.data_model = datamodel.DataModel(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )

    def test_AL(self):
        random.seed(1111111111110)
        # Even with random seed, the order of the following seem to be random,
        # so we shouldn't test for exact order.
        EXPECTED_CANDIDATES = [
            ({"name": "Willy", "age": "35"}, {"name": "William", "age": "35"}),
            ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"}),
            ({"name": "Willy", "age": "35"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "William", "age": "35"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "Meredith", "age": "40"}, {"name": "Sue", "age": "10"}),
            ({"name": "Meredith", "age": "40"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "Sue", "age": "10"}, {"name": "Jimmy", "age": "20"}),
            ({"name": "Willy", "age": "35"}, {"name": "Jimbo", "age": "21"}),
            ({"name": "William", "age": "35"}, {"name": "Jimbo", "age": "21"}),
        ]
        EXPECTED_CANDIDATES = {freeze_record_pair(pair) for pair in EXPECTED_CANDIDATES}
        active_learner = labeler.DedupeDisagreementLearner(
            self.data_model.predicates, self.data_model.distances, SAMPLE, []
        )
        actual_candidates = set()
        for i in range(len(EXPECTED_CANDIDATES), 0, -1):
            assert len(active_learner) == i
            record_pair = freeze_record_pair(active_learner.pop())
            actual_candidates.add(record_pair)
        assert actual_candidates == EXPECTED_CANDIDATES
        with pytest.raises(IndexError):
            active_learner.pop()


class QuotaActiveLearningTest(unittest.TestCase):
    def test_source_quotas_are_respected(self):
        learner = build_learner()
        batch = learner.pop_quota({"big": 2, "rare_a": 1}, seed=7)

        counts = {}
        for pair in batch:
            source = pair_source(pair)
            counts[source] = counts.get(source, 0) + 1

        assert counts.get("big", 0) <= 2
        assert counts.get("rare_a", 0) <= 1
        # the starved channel is actually surfaced
        assert "rare_a" in counts

    def test_explicit_bucket_quotas_are_respected(self):
        learner = build_learner()
        probs = learner._candidate_probs()
        bins = learner._difficulty_bins(probs[:, 0], (1.0 / 3.0, 2.0 / 3.0))
        bucket_of = {
            freeze_record_pair(pair): (pair_source(pair), int(bins[i]))
            for i, pair in enumerate(learner.candidates)
        }

        batch = learner.pop_quota(
            {("big", 0): 1, ("big", 1): 1, ("rare_a", 0): 1}, seed=7
        )

        counts = {}
        for pair in batch:
            key = bucket_of[freeze_record_pair(pair)]
            counts[key] = counts.get(key, 0) + 1

        assert counts.get(("big", 0), 0) <= 1
        assert counts.get(("big", 1), 0) <= 1
        assert counts.get(("rare_a", 0), 0) <= 1

    def test_each_pair_consumes_one_budget_only(self):
        learner = build_learner()
        before = len(learner)
        batch = learner.pop_quota(
            {"big": 5, "rare_a": 5, "rare_b": 5}, seed=1
        )

        frozen = [freeze_record_pair(pair) for pair in batch]
        assert len(frozen) == len(set(frozen))
        # candidates and stable ids stay aligned in every learner
        assert len(learner) == before - len(batch)
        assert len(learner.candidate_ids) == len(learner.candidates)
        assert len(learner.matcher.candidate_ids) == len(learner.matcher.candidates)
        assert len(learner.blocker.candidate_ids) == len(learner.blocker.candidates)

    def test_overlapping_buckets_never_double_count_a_pair(self):
        # Force the same candidate into two different buckets by giving both
        # records a source that sorts identically, then claim it through one
        # quota: it must be unavailable to every other quota draw.
        learner = build_learner()
        n = len(learner.candidates)
        weights = numpy.full(n, 1.0 / n)

        shared_id = learner.candidate_ids[0]
        claimed = {shared_id}
        first = learner._sample_bucket(
            list(range(n)), weights, n, claimed, numpy.random.default_rng(0)
        )
        # The pre-claimed pair was not charged to this bucket.
        assert 0 not in first
        assert len(first) == n - 1

        # Every selected id is added to claimed, so a second overlapping draw
        # returns nothing at all.
        second = learner._sample_bucket(
            list(range(n)), weights, n, claimed, numpy.random.default_rng(0)
        )
        assert second == []

    def test_seed_makes_batches_reproducible(self):
        first = build_learner().pop_quota(
            {("big", 0): 2, ("rare_a", 1): 1}, seed=123
        )
        second = build_learner().pop_quota(
            {("big", 0): 2, ("rare_a", 1): 1}, seed=123
        )
        assert first == second

    def test_different_seed_can_change_batch(self):
        # At this early training stage only a couple of candidates carry
        # non-zero uncertainty weight, so test the seeded weighted sampler
        # directly: different seeds produce different weighted draws.
        learner = build_learner()
        n = len(learner.candidates)
        indices = list(range(n))
        weights = numpy.full(n, 1.0 / n)
        draws = {
            tuple(
                learner._sample_bucket(
                    indices, weights, 3, set(),
                    numpy.random.default_rng(seed),
                )
            )
            for seed in range(40)
        }
        assert len(draws) > 1

        # And the same seed reproduces itself.
        one = learner._sample_bucket(
            indices, weights, 3, set(), numpy.random.default_rng(7)
        )
        two = learner._sample_bucket(
            indices, weights, 3, set(), numpy.random.default_rng(7)
        )
        assert one == two

        # A pair already claimed elsewhere cannot consume budget again.
        claimed = {learner.candidate_ids[0]}
        chosen = learner._sample_bucket(
            indices, weights, n, claimed, numpy.random.default_rng(7)
        )
        assert 0 not in chosen
        assert len(chosen) == n - 1

    def test_fallback_none_leaves_quota_unmet(self):
        learner = build_learner()
        available = sum(
            1 for pair in learner.candidates if pair_source(pair) == "rare_a"
        )
        assert available < 50
        batch = learner.pop_quota({"rare_a": 50}, fallback="none")

        assert len(batch) == available
        assert all(pair_source(pair) == "rare_a" for pair in batch)

    def test_fallback_global_backfills_other_sources(self):
        learner = build_learner()
        available = sum(
            1 for pair in learner.candidates if pair_source(pair) == "rare_a"
        )
        batch = learner.pop_quota(
            {"rare_a": 50}, fallback="global", seed=7
        )

        assert len(batch) > available
        assert sum(1 for pair in batch if pair_source(pair) == "rare_a") == available

    def test_fallback_source_stays_within_declared_source(self):
        learner = build_learner()
        batch = learner.pop_quota({"rare_a": 50}, fallback="source", seed=7)
        assert batch  # the source's own bucket had candidates
        assert all(pair_source(pair) == "rare_a" for pair in batch)

    def test_n_caps_batch_size(self):
        learner = build_learner()
        batch = learner.pop_quota({"big": 5, "rare_a": 5}, n=3, seed=5)
        assert len(batch) == 3

    def test_empty_quotas_and_exhausted_pool(self):
        learner = build_learner()
        assert learner.pop_quota({}) == []

        while batch := learner.pop_quota(
            {"big": 1000, "rare_a": 1000, "rare_b": 1000}, fallback="global"
        ):
            pass
        assert learner.pop_quota({"big": 1}, seed=1) == []

    def test_custom_source_callable_and_field(self):
        learner = build_learner()
        batch = learner.pop_quota(
            {"X": 2},
            source_field=lambda record: "X" if record["source"] == "big" else "Y",
            seed=3,
        )
        assert all(pair[0]["source"] == "big" for pair in batch)

        learner = build_learner()
        batch = learner.pop_quota({"big": 1}, source_field="source", seed=3)
        assert all(pair_source(pair) == "big" for pair in batch)

    def test_invalid_arguments(self):
        learner = build_learner()
        with pytest.raises(ValueError):
            learner.pop_quota({"big": -1})
        with pytest.raises(ValueError):
            learner.pop_quota({"big": 1}, bin_edges=(0.8, 0.2))
        with pytest.raises(ValueError):
            learner.pop_quota({"big": 1}, bin_edges=(0.0, 0.5))
        with pytest.raises(ValueError):
            learner.pop_quota({"big": 1}, fallback="bogus")
        with pytest.raises(ValueError):
            learner.pop_quota({"big": 1}, n=-1)
        with pytest.raises(ValueError):
            learner.pop_quota({("big", -1): 1})
        # a bool is a valid (if odd) int bin, but a float-looking bin that is
        # not an int is read as a source, so only genuine negative ints fail
        with pytest.raises(ValueError):
            labeler._normalize_quotas({("big", -2): 1})
        # tuple source with non-int second element is accepted as a source
        assert labeler._normalize_quotas({("big", "x"): 1}) == {
            (("big", "x"), labeler._ANY_BIN): 1
        }

    def test_difficulty_bins_order_easiest_to_hardest(self):
        match_probs = numpy.array([0.0, 0.5, 1.0, 0.4])
        bins = labeler.DisagreementLearner._difficulty_bins(
            match_probs, (1.0 / 3.0, 2.0 / 3.0)
        )
        # uncertainty: 0 (easy), 1 (hard), 0 (easy), 0.8
        assert list(bins) == [0, 2, 0, 2]

    def test_uncertainty_weights_three_regimes(self):
        weights = labeler.DisagreementLearner._uncertainty_weights

        # Uncovered disagreement: only the uncovered, disagreeing pair is
        # weighted, in proportion to the classifier's match probability.
        probs = numpy.array([[0.8, 0.0], [0.2, 1.0], [0.9, 1.0]])
        out = weights(probs)
        assert out[0] == 1.0 and out[1] == 0.0 and out[2] == 0.0

        # Covered regime: weight is spread evenly over covered candidates.
        probs = numpy.array([[0.3, 1.0], [0.7, 1.0]])
        out = weights(probs)
        assert numpy.allclose(out, [0.5, 0.5])

        # No covered pairs: weight follows the std across learner scores.
        probs = numpy.array([[0.4, 0.6], [0.5, 0.5]])
        out = weights(probs)
        assert out[0] > out[1]
        assert numpy.isclose(out.sum(), 1.0)

    def test_stable_ids_match_candidate_pairs(self):
        learner = build_learner()
        id_to_record = {}
        for (id_1, id_2), pair in zip(learner.candidate_ids, learner.candidates):
            id_to_record[id_1] = pair[0]
            id_to_record[id_2] = pair[1]

        for (id_1, id_2), pair in zip(learner.candidate_ids, learner.candidates):
            assert id_to_record[id_1] == pair[0]
            assert id_to_record[id_2] == pair[1]
        # ids are unique per candidate
        assert len(set(learner.candidate_ids)) == len(learner.candidate_ids)

    def test_mark_pairs_accepts_quota_batch(self):
        matcher = dedupe.Dedupe(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )
        matcher.prepare_training(SOURCED_SAMPLE)

        batch = matcher.uncertain_pairs(
            {("big", 0): 2, ("rare_a", 0): 1}, seed=99
        )
        matcher.mark_pairs({"match": [], "distinct": batch})

        follow_up = matcher.uncertain_pairs(
            {("big", 0): 2, ("rare_a", 0): 1}, seed=99
        )
        first = {freeze_record_pair(pair) for pair in batch}
        second = {freeze_record_pair(pair) for pair in follow_up}
        assert not (first & second)

    def test_uncertain_pairs_without_quota_returns_one(self):
        matcher = dedupe.Dedupe(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )
        matcher.prepare_training(SOURCED_SAMPLE)
        assert len(matcher.uncertain_pairs()) == 1

    def test_missing_source_field_is_tolerated(self):
        # Records without the source field group under a None source key.
        sample = {
            i: {"name": f"Name{i}", "age": str(20 + i)} for i in range(1, 9)
        }
        data_model = datamodel.DataModel(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )
        random.seed(1111111111110)
        learner = labeler.DedupeDisagreementLearner(
            data_model.predicates, data_model.distances, sample, []
        )
        batch = learner.pop_quota({None: 2}, seed=3)
        assert len(batch) == 2

    def test_incomparable_source_values_do_not_crash(self):
        # Mixed None/str sources exercise the unsorted-tuple fallback.
        sample = {
            1: {"name": "A", "age": "1", "source": "x"},
            2: {"name": "B", "age": "2"},
            3: {"name": "C", "age": "3", "source": "y"},
            4: {"name": "D", "age": "4"},
        }
        data_model = datamodel.DataModel(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )
        random.seed(1111111111110)
        learner = labeler.DedupeDisagreementLearner(
            data_model.predicates, data_model.distances, sample, []
        )
        batch = learner.pop_quota({"x": 5, None: 5, "y": 5}, seed=3)
        assert batch  # selection completed despite incomparable source keys

    def test_tuple_source_keys_are_distinct_from_bucket_keys(self):
        # A source value that is itself a 2-tuple with an int second element
        # must not be mistaken for a (source, bin) quota key.
        normalized = labeler._normalize_quotas({("web", 2): 3})
        assert normalized == {("web", 2): 3}  # treated as (source, bin)

        normalized = labeler._normalize_quotas({("web", "x"): 3})
        # second element is not a bin -> treated as a source value
        assert normalized == {(("web", "x"), labeler._ANY_BIN): 3}


if __name__ == "__main__":
    unittest.main()
