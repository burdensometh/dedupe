import io
import pickle
import random
import unittest

import numpy
import pytest
import scipy.special

import dedupe
from dedupe.calibration import (
    CalibratedClassifier,
    CalibrationError,
    ProbabilityCalibrator,
    calibrate_classifier,
)

DATA = {
    100: {"name": "Bob", "age": "50"},
    105: {"name": "Charlie", "age": "75"},
    110: {"name": "Meredith", "age": "40"},
    115: {"name": "Sue", "age": "10"},
    120: {"name": "Jimmy", "age": "20"},
    125: {"name": "Jimbo", "age": "21"},
    130: {"name": "Willy", "age": "35"},
    135: {"name": "William", "age": "35"},
    140: {"name": "Martha", "age": "19"},
    145: {"name": "Kyle", "age": "27"},
}

# Independent of the pairs used to train the matcher below.
CALIBRATION_PAIRS = [
    ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"}),
    ({"name": "Willy", "age": "35"}, {"name": "William", "age": "35"}),
    ({"name": "Sue", "age": "10"}, {"name": "Martha", "age": "19"}),
    ({"name": "Bob", "age": "50"}, {"name": "Kyle", "age": "27"}),
    ({"name": "Charlie", "age": "75"}, {"name": "Sue", "age": "10"}),
    ({"name": "Meredith", "age": "40"}, {"name": "William", "age": "35"}),
]
CALIBRATION_LABELS = [1, 1, 0, 0, 0, 0]


class MockClassifier:
    """A duck-typed, frozen classifier used to test calibration in isolation."""

    def __init__(self):
        self.weight = -1.0
        self.bias = 1.0

    def predict_proba(self, examples):
        probabilities = scipy.special.expit(examples * self.weight + self.bias)
        return numpy.column_stack((1.0 - probabilities, probabilities))


class ProbabilityCalibratorTest(unittest.TestCase):
    def setUp(self):
        rng = numpy.random.RandomState(123)
        self.raw_scores = rng.uniform(0.01, 0.99, size=200)
        # labels are a noisy monotonic function of the raw score
        self.labels = (self.raw_scores + rng.normal(scale=0.2, size=200) > 0.5).astype(
            int
        )

    def test_isotonic_is_monotonic_and_bounded(self):
        calibrator = ProbabilityCalibrator("isotonic")
        calibrator.fit(self.raw_scores, self.labels)

        grid = numpy.linspace(0.0, 1.0, 50)
        calibrated = calibrator.predict(grid)

        assert numpy.all(numpy.diff(calibrated) >= -1e-12)
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_sigmoid_method(self):
        calibrator = ProbabilityCalibrator("sigmoid")
        calibrator.fit(self.raw_scores, self.labels)

        calibrated = calibrator.predict(numpy.array([0.1, 0.5, 0.9]))
        assert calibrated.shape == (3,)
        assert numpy.all(numpy.diff(calibrated) >= 0)
        assert numpy.all((calibrated >= 0) & (calibrated <= 1))

    def test_sample_weights_are_accepted_and_change_the_fit(self):
        unweighted = ProbabilityCalibrator("isotonic").fit(self.raw_scores, self.labels)
        weights = numpy.ones_like(self.raw_scores, dtype=float)
        weights[self.labels == 1] = 5.0
        weighted = ProbabilityCalibrator("isotonic").fit(
            self.raw_scores, self.labels, sample_weight=weights
        )

        grid = numpy.linspace(0.05, 0.95, 10)
        assert not numpy.allclose(unweighted.predict(grid), weighted.predict(grid))

    def test_single_label_raises_calibration_error(self):
        for label, phrase in ((0, "distinct (0)"), (1, "match (1)")):
            labels = numpy.full(self.raw_scores.shape, label)
            calibrator = ProbabilityCalibrator()
            with pytest.raises(CalibrationError) as excinfo:
                calibrator.fit(self.raw_scores, labels)
            assert phrase in str(excinfo.value)
            assert "both" in str(excinfo.value)

    def test_invalid_arguments(self):
        calibrator = ProbabilityCalibrator()

        with pytest.raises(ValueError):
            ProbabilityCalibrator("platt")
        with pytest.raises(ValueError):
            calibrator.fit(numpy.array([]), numpy.array([]))
        with pytest.raises(ValueError):
            calibrator.fit(self.raw_scores, self.labels[:-1])
        with pytest.raises(ValueError):
            calibrator.fit(self.raw_scores, numpy.full(200, 2))
        with pytest.raises(ValueError):
            calibrator.fit(
                self.raw_scores,
                self.labels,
                sample_weight=numpy.ones(199),
            )
        with pytest.raises(ValueError):
            calibrator.fit(
                self.raw_scores,
                self.labels,
                sample_weight=numpy.full(200, -1.0),
            )

    def test_predict_before_fit(self):
        with pytest.raises(ValueError):
            ProbabilityCalibrator().predict(self.raw_scores)

    def test_picklable(self):
        calibrator = ProbabilityCalibrator("sigmoid").fit(self.raw_scores, self.labels)
        restored = pickle.loads(pickle.dumps(calibrator))
        numpy.testing.assert_allclose(
            restored.predict(self.raw_scores),
            calibrator.predict(self.raw_scores),
        )


class CalibratedClassifierTest(unittest.TestCase):
    def test_wraps_predict_proba_and_preserves_ranking(self):
        rng = numpy.random.RandomState(7)
        features = rng.normal(size=(100, 1))
        classifier = MockClassifier()
        raw = classifier.predict_proba(features)[:, 1]
        labels = (raw + rng.normal(scale=0.1, size=100) > 0.5).astype(int)

        calibrator = ProbabilityCalibrator().fit(raw, labels)
        wrapped = CalibratedClassifier(classifier, calibrator)

        output = wrapped.predict_proba(features)
        assert output.shape == (100, 2)
        numpy.testing.assert_allclose(output.sum(axis=1), 1.0)
        # the monotonic mapping never reverses an ordering of the raw
        # scores: raw i below raw j implies calibrated i is not above
        # calibrated j (they may become tied)
        for i in range(raw.size):
            for j in range(raw.size):
                if raw[i] < raw[j]:
                    assert output[i, 1] <= output[j, 1]
        # picklable so multiprocessing scoring workers can receive it
        pickle.loads(pickle.dumps(wrapped))

    def test_requires_fitted_calibrator(self):
        with pytest.raises(ValueError):
            CalibratedClassifier(MockClassifier(), ProbabilityCalibrator())


class CalibrateClassifierHelperTest(unittest.TestCase):
    def test_scores_calibration_pairs_with_frozen_classifier(self):
        deduper = dedupe.Dedupe([dedupe.variables.String("name")])
        classifier = MockClassifier()

        wrapped = calibrate_classifier(
            classifier,
            CALIBRATION_PAIRS,
            deduper.data_model.distances,
            CALIBRATION_LABELS,
        )

        features = deduper.data_model.distances(CALIBRATION_PAIRS)
        raw = classifier.predict_proba(features)[:, 1]
        numpy.testing.assert_allclose(
            wrapped.predict_proba(features)[:, 1],
            wrapped.calibrator.predict(raw),
        )

    def test_mismatched_lengths_and_empty_input(self):
        deduper = dedupe.Dedupe([dedupe.variables.String("name")])
        with pytest.raises(ValueError):
            calibrate_classifier(
                MockClassifier(),
                CALIBRATION_PAIRS,
                deduper.data_model.distances,
                [1, 0],
            )
        with pytest.raises(ValueError):
            calibrate_classifier(MockClassifier(), [], deduper.data_model.distances, [])


class MatcherCalibrationTest(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        numpy.random.seed(42)

        self.deduper = dedupe.Dedupe(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )
        self.deduper.prepare_training(DATA)
        self.deduper.mark_pairs(
            {
                "match": [
                    (
                        {"name": "Jimmy", "age": "20"},
                        {"name": "Jimbo", "age": "21"},
                    ),
                    (
                        {"name": "Willy", "age": "35"},
                        {"name": "William", "age": "35"},
                    ),
                ]
                * 3,
                "distinct": [
                    ({"name": "Sue", "age": "10"}, {"name": "William", "age": "35"}),
                    (
                        {"name": "Meredith", "age": "40"},
                        {"name": "Kyle", "age": "27"},
                    ),
                    (
                        {"name": "Charlie", "age": "75"},
                        {"name": "Martha", "age": "19"},
                    ),
                ]
                * 2,
            }
        )
        self.deduper.train()

    def test_calibrate_before_training_raises(self):
        untrained = dedupe.Dedupe([dedupe.variables.String("name")])
        with pytest.raises(ValueError):
            untrained.calibrate(CALIBRATION_PAIRS, CALIBRATION_LABELS)

    def test_calibrate_validates_arguments(self):
        with pytest.raises(ValueError):
            self.deduper.calibrate(CALIBRATION_PAIRS, [1, 0])
        with pytest.raises(ValueError):
            self.deduper.calibrate([], [])
        with pytest.raises(ValueError):
            self.deduper.calibrate(CALIBRATION_PAIRS, [1, 1, 2, 0, 0, 0])
        with pytest.raises(ValueError):
            self.deduper.calibrate(
                CALIBRATION_PAIRS,
                CALIBRATION_LABELS,
                sample_weight=[1.0] * 5,
            )

    def test_single_label_calibration_set_raises(self):
        with pytest.raises(CalibrationError):
            self.deduper.calibrate(CALIBRATION_PAIRS[:3], [1, 1, 1])
        with pytest.raises(CalibrationError):
            self.deduper.calibrate(CALIBRATION_PAIRS[2:], [0, 0, 0, 0])

    def test_score_modes_and_ranking_consistency(self):
        pairs = list(self.deduper.pairs(DATA))

        raw = self.deduper.score(iter(pairs), calibrated=False)
        # nothing fitted yet: default and explicit False are identical
        assert self.deduper.calibrator is None
        numpy.testing.assert_allclose(
            self.deduper.score(iter(pairs))["score"], raw["score"]
        )
        with pytest.raises(ValueError):
            self.deduper.score(iter(pairs), calibrated=True)

        self.deduper.calibrate(
            CALIBRATION_PAIRS,
            CALIBRATION_LABELS,
            sample_weight=[2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )

        calibrated = self.deduper.score(iter(pairs))
        # default now uses the mapping
        assert self.deduper.calibrator is not None
        numpy.testing.assert_allclose(
            calibrated["score"],
            self.deduper.score(iter(pairs), calibrated=True)["score"],
        )
        # raw scores are still available and the monotonic calibration
        # mapping never reverses their ordering (distinct raw scores may
        # map onto the same calibrated value)
        numpy.testing.assert_allclose(
            self.deduper.score(iter(pairs), calibrated=False)["score"],
            raw["score"],
        )
        for i in range(raw["score"].size):
            for j in range(raw["score"].size):
                if raw["score"][i] < raw["score"][j]:
                    assert calibrated["score"][i] <= calibrated["score"][j]

    def test_partition_uses_calibrated_probabilities(self):
        self.deduper.calibrate(CALIBRATION_PAIRS, CALIBRATION_LABELS)

        raw_clusters = list(
            self.deduper.partition(DATA, threshold=0.5, calibrated=False)
        )
        calibrated_clusters = list(self.deduper.partition(DATA, threshold=0.5))

        # same records are clustered; reported confidences are calibrated
        assert [c[0] for c in raw_clusters] == [c[0] for c in calibrated_clusters]
        pairs = list(self.deduper.pairs(DATA))
        calibrated_scores = self.deduper.score(iter(pairs))["score"]
        for (_, confidences), expected in zip(calibrated_clusters, calibrated_scores):
            assert any(abs(c - float(expected)) < 1e-5 for c in confidences)

    def test_threshold_is_interpreted_on_calibrated_scale(self):
        # Fit a mapping on an independent synthetic calibration set with
        # a lower empirical positive rate than the classifier's raw
        # scores imply, so the mapping reshapes the probabilities.
        # Thresholds are then interpreted against the calibrated scale
        # rather than the raw classifier scale.
        rng = numpy.random.RandomState(99)
        synth_raw = rng.uniform(0.05, 0.95, size=60)
        synth_labels = (synth_raw > 0.75).astype(int)
        assert 2 <= synth_labels.sum() <= 58

        self.deduper.calibrator = ProbabilityCalibrator("isotonic").fit(
            synth_raw, synth_labels
        )

        pairs = list(self.deduper.pairs(DATA))
        raw = self.deduper.score(iter(pairs), calibrated=False)
        calibrated = self.deduper.score(iter(pairs))
        assert not numpy.allclose(calibrated["score"], raw["score"])

        # for an edge whose two scales differ, a threshold between the
        # raw and calibrated values keeps that edge on one scale and
        # drops it on the other -- the threshold now means a calibrated
        # probability rather than the classifier's raw score
        gap = numpy.abs(calibrated["score"] - raw["score"]).argmax()
        separating = (float(raw["score"][gap]) + float(calibrated["score"][gap])) / 2
        assert (raw["score"][gap] > separating) != (
            calibrated["score"][gap] > separating
        )

        # calibrated probabilities can legitimately all fall at or below
        # a high threshold; clustering must handle the empty edge list
        # rather than raising. Thresholds are strict (>), so 1.0 keeps
        # no edges.
        assert list(self.deduper.cluster(calibrated, threshold=1.0)) == []

    def test_settings_roundtrip_keeps_calibrator(self):
        self.deduper.calibrate(CALIBRATION_PAIRS, CALIBRATION_LABELS)
        pairs = list(self.deduper.pairs(DATA))
        expected = self.deduper.score(iter(pairs))["score"]

        buffer = io.BytesIO()
        self.deduper.write_settings(buffer)
        buffer.seek(0)

        static = dedupe.StaticDedupe(buffer)
        assert isinstance(static.calibrator, ProbabilityCalibrator)
        numpy.testing.assert_allclose(
            static.score(iter(list(static.pairs(DATA))))["score"],
            expected,
            atol=1e-6,
        )

    def test_legacy_settings_file_without_calibrator(self):
        # emulate a pre-calibration settings file: only three pickles
        buffer = io.BytesIO()
        pickle.dump(self.deduper.data_model, buffer)
        pickle.dump(self.deduper.classifier, buffer)
        pickle.dump(self.deduper.predicates, buffer)
        buffer.seek(0)

        static = dedupe.StaticDedupe(buffer)
        assert static.calibrator is None
        # scoring falls back to raw probabilities
        scores = static.score(iter(list(static.pairs(DATA))))
        assert scores["score"].size >= 1


ENTITIES = [
    ("Bob Smith", "50"),
    ("Charlie Brown", "75"),
    ("Meredith Grey", "40"),
    ("Sue Storm", "10"),
    ("Jimmy Dean", "20"),
    ("Jim Beam", "21"),
    ("Willy Wonka", "35"),
    ("William Tell", "35"),
    ("Martha Stewart", "19"),
    ("Kyle Reese", "27"),
]


def _variant(name, age, salt):
    if salt % 3 == 0:
        name = name.lower()
    elif salt % 3 == 1:
        name = name.upper()
    return {"name": name, "age": age}


def _link_datasets():
    data_1 = {i: _variant(n, a, 0) for i, (n, a) in enumerate(ENTITIES)}
    data_2 = {1000 + i: _variant(n, a, i + 1) for i, (n, a) in enumerate(ENTITIES)}
    return data_1, data_2


def _entity_labels(data_1, data_2):
    match = [(data_1[i], data_2[1000 + i]) for i in range(len(ENTITIES))]
    distinct = [
        (data_1[i], data_2[1000 + j]) for i in range(6) for j in range(4) if i != j
    ]
    random.shuffle(distinct)
    return {"match": match, "distinct": distinct[:12]}


class RecordLinkCalibrationTest(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        numpy.random.seed(42)

        self.data_1, self.data_2 = _link_datasets()

        self.linker = dedupe.RecordLink(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )
        self.linker.prepare_training(self.data_1, self.data_2)
        self.linker.mark_pairs(_entity_labels(self.data_1, self.data_2))
        self.linker.train()
        self.linker.calibrate(CALIBRATION_PAIRS, CALIBRATION_LABELS)

    def test_join_modes(self):
        raw_links = self.linker.join(self.data_1, self.data_2, 0.0, calibrated=False)
        calibrated_links = self.linker.join(self.data_1, self.data_2, 0.0)

        # one-to-one matching: the monotonic calibration mapping never
        # reverses a strict ordering, so each side still contributes the
        # same records to the matching (ties may be ordered differently)
        assert {a for (a, _), _ in raw_links} == {a for (a, _), _ in calibrated_links}
        assert {b for (_, b), _ in raw_links} == {b for (_, b), _ in calibrated_links}
        # and the reported link scores are the calibrated probabilities
        calibrated_scores = {tuple(pair): score for pair, score in calibrated_links}
        for pair, score in raw_links:
            assert 0.0 <= calibrated_scores[tuple(pair)] <= 1.0

    def test_settings_roundtrip(self):
        buffer = io.BytesIO()
        self.linker.write_settings(buffer)
        buffer.seek(0)

        static = dedupe.StaticRecordLink(buffer)
        assert isinstance(static.calibrator, ProbabilityCalibrator)
        static.join(self.data_1, self.data_2, 0.0)


class GazetteerCalibrationTest(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        numpy.random.seed(42)

        self.messy, self.canonical = _link_datasets()

        self.gazetteer = dedupe.Gazetteer(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )
        self.gazetteer.prepare_training(self.messy, self.canonical)
        self.gazetteer.mark_pairs(_entity_labels(self.messy, self.canonical))
        self.gazetteer.train()
        self.gazetteer.index(self.canonical)
        self.gazetteer.calibrate(CALIBRATION_PAIRS, CALIBRATION_LABELS)

    def test_search_uses_calibrated_scores(self):
        blocks = list(self.gazetteer.blocks(self.messy))
        assert blocks

        raw_blocks = list(self.gazetteer.score(iter(blocks), calibrated=False))
        calibrated_blocks = list(self.gazetteer.score(iter(blocks)))

        for raw, calibrated in zip(raw_blocks, calibrated_blocks):
            # monotonic mapping never reverses a strict ordering
            for i in range(raw["score"].size):
                for j in range(raw["score"].size):
                    if raw["score"][i] < raw["score"][j]:
                        assert calibrated["score"][i] <= calibrated["score"][j]

        raw_results = self.gazetteer.search(self.messy, 0.0, calibrated=False)
        calibrated_results = self.gazetteer.search(self.messy, 0.0)
        assert [record_id for record_id, _ in raw_results] == [
            record_id for record_id, _ in calibrated_results
        ]

    def test_settings_roundtrip(self):
        buffer = io.BytesIO()
        self.gazetteer.write_settings(buffer)
        buffer.seek(0)

        static = dedupe.StaticGazetteer(buffer)
        assert isinstance(static.calibrator, ProbabilityCalibrator)
        static.index(self.canonical)
        static.search(self.messy, 0.0)


if __name__ == "__main__":
    unittest.main()
