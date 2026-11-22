import io
import json
import unittest
import unittest.mock

import numpy
import pytest

import dedupe
import dedupe.api
import dedupe.branch_and_bound
import dedupe.labeler
import dedupe.training

VARIABLES = [dedupe.variables.String("name"), dedupe.variables.String("age")]

MATCH_PAIRS = [
    ({"name": "Bob", "age": "50"}, {"name": "Bob B.", "age": "51"}),
    ({"name": "Linda", "age": "50"}, {"name": "linda", "age": "50"}),
]
DISTINCT_PAIRS = [
    ({"name": "Bob", "age": "50"}, {"name": "Sue", "age": "10"}),
    ({"name": "Jimmy", "age": "20"}, {"name": "Willy", "age": "35"}),
]


class FakePredicate:
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.cover_count = 0

    def __call__(self, record, target=False):
        return self.func(record)

    def __repr__(self):
        return self.name


class MarkPairsWeightsTest(unittest.TestCase):
    def setUp(self):
        self.matcher = dedupe.Dedupe(VARIABLES)

    def test_weights_stored_with_pairs(self):
        self.matcher.mark_pairs(
            {
                "match": MATCH_PAIRS,
                "distinct": DISTINCT_PAIRS[:1],
                "weights": {"match": [0.5, 2.0], "distinct": [3.0]},
            }
        )

        assert self.matcher.training_pairs["match"] == MATCH_PAIRS
        assert self.matcher.training_pairs["distinct"] == DISTINCT_PAIRS[:1]
        assert self.matcher.training_pairs["weights"] == {
            "match": [0.5, 2.0],
            "distinct": [3.0],
        }

    def test_no_weights_no_weights_key(self):
        self.matcher.mark_pairs({"match": MATCH_PAIRS, "distinct": DISTINCT_PAIRS})
        assert "weights" not in self.matcher.training_pairs

    def test_missing_weights_get_default(self):
        # unweighted pairs marked before weighted pairs are backfilled
        # with the default weight
        self.matcher.mark_pairs({"match": MATCH_PAIRS[:1], "distinct": []})
        self.matcher.mark_pairs(
            {
                "match": MATCH_PAIRS[1:],
                "distinct": DISTINCT_PAIRS[:1],
                "weights": {"match": [0.5]},
            }
        )
        assert self.matcher.training_pairs["weights"] == {
            "match": [1.0, 0.5],
            "distinct": [1.0],
        }

        # unweighted pairs marked after weighted pairs also get the
        # default weight
        self.matcher.mark_pairs({"match": [], "distinct": DISTINCT_PAIRS[1:]})
        assert self.matcher.training_pairs["weights"] == {
            "match": [1.0, 0.5],
            "distinct": [1.0, 1.0],
        }

    def test_none_weight_gets_default(self):
        self.matcher.mark_pairs(
            {
                "match": MATCH_PAIRS[:1],
                "distinct": [],
                "weights": {"match": [None]},
            }
        )
        assert self.matcher.training_pairs["weights"]["match"] == [1.0]

    def test_zero_weight_stored(self):
        self.matcher.mark_pairs(
            {
                "match": MATCH_PAIRS[:1],
                "distinct": [],
                "weights": {"match": [0.0]},
            }
        )
        assert self.matcher.training_pairs["weights"]["match"] == [0.0]

    def test_bad_weights(self):
        base = {"match": MATCH_PAIRS[:1], "distinct": []}

        with pytest.raises(ValueError):
            # number of weights must match number of pairs
            self.matcher.mark_pairs({**base, "weights": {"match": [1.0, 2.0]}})

        with pytest.raises(ValueError):
            # weights must be non-negative
            self.matcher.mark_pairs({**base, "weights": {"match": [-1.0]}})

        with pytest.raises(ValueError):
            # weights must be numbers
            self.matcher.mark_pairs({**base, "weights": {"match": ["heavy"]}})

        with pytest.raises(ValueError):
            # unknown labels
            self.matcher.mark_pairs({**base, "weights": {"matches": [1.0]}})

        with pytest.raises(ValueError):
            # weights must be a dictionary
            self.matcher.mark_pairs({**base, "weights": [1.0]})

        with pytest.raises(ValueError):
            # per-label weights must be lists of numbers
            self.matcher.mark_pairs({**base, "weights": {"match": 1.0}})


class FlattenTrainingWeightsTest(unittest.TestCase):
    def test_no_weights(self):
        examples, y, weights = dedupe.api.flatten_training(
            {"match": MATCH_PAIRS, "distinct": DISTINCT_PAIRS}
        )
        assert examples == MATCH_PAIRS + DISTINCT_PAIRS
        assert list(y) == [1, 1, 0, 0]
        assert weights is None

    def test_weights_aligned_with_examples(self):
        examples, y, weights = dedupe.api.flatten_training(
            {
                "match": MATCH_PAIRS,
                "distinct": DISTINCT_PAIRS,
                "weights": {"match": [0.5, 1.0], "distinct": [2.0, 0.0]},
            }
        )
        assert examples == MATCH_PAIRS + DISTINCT_PAIRS
        assert list(y) == [1, 1, 0, 0]
        assert list(weights) == [0.5, 1.0, 2.0, 0.0]

    def test_missing_weights_default(self):
        examples, y, weights = dedupe.api.flatten_training(
            {
                "match": MATCH_PAIRS,
                "distinct": DISTINCT_PAIRS,
                "weights": {"match": [0.5, None]},
            }
        )
        assert list(weights) == [0.5, 1.0, 1.0, 1.0]


class SerializerWeightsTest(unittest.TestCase):
    def test_round_trip_preserves_weights_and_pairs(self):
        training_pairs = {
            "match": MATCH_PAIRS,
            "distinct": DISTINCT_PAIRS,
            "weights": {"match": [0.5, 1.0], "distinct": [2.0, 0.0]},
        }

        encoded_file = io.StringIO()
        dedupe.write_training(training_pairs, encoded_file)
        encoded_file.seek(0)
        loaded = dedupe.read_training(encoded_file)

        assert loaded["match"] == training_pairs["match"]
        assert loaded["distinct"] == training_pairs["distinct"]
        assert loaded["weights"] == training_pairs["weights"]

    def test_replay_through_matcher(self):
        matcher = dedupe.Dedupe(VARIABLES)
        matcher.mark_pairs(
            {
                "match": MATCH_PAIRS,
                "distinct": DISTINCT_PAIRS,
                "weights": {"match": [0.5, 1.0], "distinct": [2.0, 0.0]},
            }
        )

        encoded_file = io.StringIO()
        matcher.write_training(encoded_file)

        new_matcher = dedupe.Dedupe(VARIABLES)
        encoded_file.seek(0)
        new_matcher._read_training(encoded_file)

        assert new_matcher.training_pairs == matcher.training_pairs

    def test_weights_key_absent_when_unweighted(self):
        matcher = dedupe.Dedupe(VARIABLES)
        matcher.mark_pairs({"match": MATCH_PAIRS, "distinct": DISTINCT_PAIRS})

        encoded_file = io.StringIO()
        matcher.write_training(encoded_file)

        assert "weights" not in json.loads(encoded_file.getvalue())


class RecordingClassifier:
    """A classifier that records how it was fit."""

    def __init__(self):
        self.fit_args = None

    def fit(self, X, y, **kwargs):
        self.fit_args = (X, y, kwargs)

    def predict_proba(self, X):
        return numpy.tile([0.5, 0.5], (len(X), 1))


class NoWeightClassifier:
    """A classifier whose fit does not accept sample_weight."""

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return numpy.tile([0.5, 0.5], (len(X), 1))


class StubLearner:
    def learn_predicates(self, recall, index_predicates):
        return ()


def make_trained_matcher(weights):
    matcher = dedupe.Dedupe(VARIABLES)
    labeled = {
        "match": MATCH_PAIRS,
        "distinct": DISTINCT_PAIRS,
    }
    if weights is not None:
        labeled["weights"] = weights
    matcher.mark_pairs(labeled)
    matcher.active_learner = StubLearner()
    return matcher


class TrainWeightsTest(unittest.TestCase):
    def test_weights_enter_classifier_fit(self):
        matcher = make_trained_matcher({"match": [1.0, 0.0], "distinct": [2.0, 1.0]})
        recorder = RecordingClassifier()
        matcher.classifier = recorder
        matcher.train()

        assert recorder.fit_args is not None
        _, y, kwargs = recorder.fit_args
        assert list(y) == [1, 1, 0, 0]
        assert list(kwargs["sample_weight"]) == [1.0, 0.0, 2.0, 1.0]

    def test_zero_weight_rows_passed_with_zero_weight(self):
        matcher = make_trained_matcher({"match": [1.0, 0.0], "distinct": [1.0, 0.0]})
        recorder = RecordingClassifier()
        matcher.classifier = recorder
        matcher.train()

        _, _, kwargs = recorder.fit_args
        assert list(kwargs["sample_weight"]) == [1.0, 0.0, 1.0, 0.0]

    def test_no_weights_no_sample_weight(self):
        matcher = make_trained_matcher(None)
        recorder = RecordingClassifier()
        matcher.classifier = recorder
        matcher.train()

        _, _, kwargs = recorder.fit_args
        assert "sample_weight" not in kwargs

    def test_unsupported_classifier_raises(self):
        matcher = make_trained_matcher({"match": [1.0, 0.5]})
        matcher.classifier = NoWeightClassifier()

        with pytest.raises(TypeError, match="sample_weight"):
            matcher.train()

    def test_unsupported_classifier_fine_without_weights(self):
        matcher = make_trained_matcher(None)
        matcher.classifier = NoWeightClassifier()
        matcher.train()

    def test_default_classifier_supports_weights(self):
        matcher = dedupe.Dedupe(VARIABLES)
        assert dedupe.api._fit_supports_sample_weight(matcher.classifier)


class WeightedBlockingTest(unittest.TestCase):
    def setUp(self):
        # p1 covers only the first match pair, p2 covers both, and p2
        # covers many more comparisons so it is the more expensive rule
        self.p1 = FakePredicate(
            "P1",
            lambda record: frozenset({"a"}) if record["name"] == "a" else frozenset(),
        )
        self.p2 = FakePredicate("P2", lambda record: frozenset({"all"}))

        self.learner = dedupe.training.BlockLearner()
        self.learner.blocker = type("Blocker", (), {"predicates": (self.p1, self.p2)})()
        self.learner.comparison_cover = {
            self.p1: frozenset({(1, 2)}),
            self.p2: frozenset({(1, 2), (3, 4), (5, 6)}),
        }
        self.matches = [
            ({"name": "a"}, {"name": "a"}),
            ({"name": "b"}, {"name": "b"}),
        ]

    def test_zero_weight_pair_does_not_contribute(self):
        # the second pair has zero weight, so covering the first pair
        # with the cheaper p1 satisfies full recall
        final = self.learner.learn(self.matches, 1.0, False, weights=[1.0, 0.0])
        assert final == (self.p1,)

    def test_weighted_pairs_must_be_covered(self):
        # with both pairs weighted, p2 is needed to cover both
        final = self.learner.learn(self.matches, 1.0, False, weights=[1.0, 1.0])
        assert final == (self.p2,)

    def test_unweighted_matches_all_count(self):
        final = self.learner.learn(self.matches, 1.0, False)
        assert final == (self.p2,)

    def test_recall_applies_to_total_weight(self):
        # total weight is 2.0, so recall of 0.5 requires covering
        # weight 1.0, which the cheaper p1 achieves
        final = self.learner.learn(self.matches, 0.5, False, weights=[1.0, 1.0])
        assert final == (self.p1,)

    def test_weights_length_mismatch(self):
        with pytest.raises(ValueError):
            self.learner.learn(self.matches, 1.0, False, weights=[1.0])


class BranchAndBoundWeightsTest(unittest.TestCase):
    def setUp(self):
        self.p1 = FakePredicate("P1", lambda record: frozenset())
        self.p2 = FakePredicate("P2", lambda record: frozenset())
        self.p1.cover_count = 1
        self.p2.cover_count = 3
        self.cover = {self.p1: frozenset({0}), self.p2: frozenset({0, 1})}

    def test_weighted_coverage(self):
        # pair 1 has zero weight, so p1 covering pair 0 reaches the target
        final = dedupe.branch_and_bound.search(self.cover, 1.0, 100, weights=[1.0, 0.0])
        assert final == (self.p1,)

    def test_weighted_target(self):
        # target of 1.5 requires covering both pairs
        final = dedupe.branch_and_bound.search(self.cover, 1.5, 100, weights=[1.0, 1.0])
        assert final == (self.p2,)

    def test_unweighted_unchanged(self):
        final = dedupe.branch_and_bound.search(self.cover, 2, 100)
        assert final == (self.p2,)
        final = dedupe.branch_and_bound.search(self.cover, 1, 100)
        assert final == (self.p1,)


class ActiveLearnerWeightsTest(unittest.TestCase):
    def test_mark_stores_weights(self):
        learner = dedupe.labeler.DisagreementLearner()
        fits = []

        class SpyLearner(dedupe.labeler.Learner):
            def fit(self, pairs, y, weights=None):
                fits.append((list(pairs), list(y), weights))

            def candidate_scores(self):
                return numpy.array([[0.5]])

            def remove(self, index):
                pass

        learner.matcher = SpyLearner()
        learner.blocker = SpyLearner()

        learner.mark([("a", "b")], [1])
        assert learner.weights is None
        assert fits[-1][2] is None

        learner.mark([("c", "d")], [0], weights=[2.0])
        # the pair marked before any weights were seen is backfilled
        # with the default weight
        assert list(learner.weights) == [1.0, 2.0]
        assert list(fits[-1][2]) == [1.0, 2.0]

        learner.mark([("e", "f")], [1])
        assert list(learner.weights) == [1.0, 2.0, 1.0]

    def test_mark_weight_length_mismatch(self):
        learner = dedupe.labeler.DisagreementLearner()
        learner.matcher = unittest.mock.Mock()
        learner.blocker = unittest.mock.Mock()

        with pytest.raises(ValueError):
            learner.mark([("a", "b")], [1], weights=[1.0, 2.0])

    def test_block_learner_fit_threads_weights(self):
        learner = dedupe.labeler.BlockLearner()
        learner.block_learner = unittest.mock.Mock()
        learner.block_learner.learn.return_value = ()
        learner._candidates = []

        pairs = [("a", "b"), ("c", "d")]
        learner.fit(pairs, [1, 1], weights=[1.0, 0.0])

        learner.block_learner.learn.assert_called_once_with(
            [("a", "b"), ("c", "d")],
            recall=1.0,
            index_predicates=True,
            candidate_types="simple",
            weights=[1.0, 0.0],
        )

    def test_zero_weight_dupe_does_not_trigger_relearn(self):
        learner = dedupe.labeler.BlockLearner()
        learner.block_learner = unittest.mock.Mock()
        learner.block_learner.learn.return_value = ()
        learner._candidates = []

        learner.fit([("a", "b")], [1], weights=[1.0])
        assert learner.block_learner.learn.call_count == 1

        # a new dupe with zero weight is not covered by the current
        # predicates, but should not trigger relearning
        learner.fit([("a", "b"), ("c", "d")], [1, 1], weights=[1.0, 0.0])
        assert learner.block_learner.learn.call_count == 1

        # a new dupe with positive weight does trigger relearning
        learner.fit(
            [("a", "b"), ("c", "d"), ("e", "f")],
            [1, 1, 1],
            weights=[1.0, 0.0, 1.0],
        )
        assert learner.block_learner.learn.call_count == 2

    def test_match_learner_fit_weights(self):
        featurizer = lambda pairs: numpy.array(  # noqa: E731
            [[float(len(p[0])), float(len(p[1]))] for p in pairs]
        )
        learner = dedupe.labeler.MatchLearner(featurizer, [("a", "b"), ("cc", "d")])
        learner.fit([("a", "b"), ("cc", "d")], [1, 0], weights=[1.0, 0.0])
        assert learner._fitted

        with pytest.raises(ValueError):
            learner.fit([("a", "b")], [1], weights=[1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
