#!/usr/bin/python
"""
Probability calibration for frozen pairwise classifiers.

The pairwise classifier learned during training outputs scores that are
only loosely interpretable as probabilities. Thresholds applied to those
scores therefore do not correspond to any particular probability of a
pair being a true match, which makes thresholds chosen for automatic
merging hard to reason about.

This module fits a mapping from the frozen classifier's raw candidate
scores to calibrated probabilities on an *independent* calibration
set. The calibration set is never the data that is about to be merged:
it is a separate set of labeled candidate pairs that is scored with the
already frozen classifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy
import numpy.typing
import sklearn.isotonic
import sklearn.linear_model

if TYPE_CHECKING:
    from typing import Collection, Literal

    from dedupe._typing import Classifier, RecordDictPair

    CalibrationMethod = Literal["isotonic", "sigmoid"]


class CalibrationError(ValueError):
    """Raised when a probability calibration mapping cannot be fitted."""


class ProbabilityCalibrator:
    """A savable mapping from raw classifier scores to probabilities.

    The mapping is fitted on the raw candidate scores produced by a
    classifier that is already frozen. Two mapping shapes are
    supported:

    * ``"isotonic"``: a non-parametric isotonic regression of the
      observed labels onto the raw scores. The resulting mapping is
      monotonic, so it never changes the ranking of candidates. It is
      a good default when the calibration set is reasonably large.
    * ``"sigmoid"``: Platt scaling, i.e. a logistic regression of the
      labels on the logit of the raw scores. The resulting mapping has
      a sigmoid shape and needs fewer calibration examples.

    Sample weights are honored by both mappings.

    A calibration set that contains only one label (only matches or
    only distinct pairs) cannot identify a calibration mapping. Rather
    than collapsing the mapping to a constant or, worse, silently
    refitting on the data to be merged, :meth:`fit` raises
    :class:`CalibrationError` in that case.
    """

    def __init__(self, method: CalibrationMethod = "isotonic") -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError("method must be 'isotonic' or 'sigmoid', got %r" % method)
        self.method = method
        self.regressor: Any
        self.fitted_: bool = False

    def fit(
        self,
        raw_scores: numpy.typing.NDArray[numpy.float64],
        labels: numpy.typing.NDArray[numpy.int_],
        sample_weight: numpy.typing.NDArray[numpy.float64] | None = None,
    ) -> ProbabilityCalibrator:
        """Fit the mapping using raw scores and labels from a calibration set.

        Args:
            raw_scores: The candidate scores produced by the frozen
                        classifier for each pair in the independent
                        calibration set.
            labels: A one-dimensional array of binary labels, where
                    ``1`` declares a positive example, meaning the two
                    records refer to the same entity (a match / true
                    duplicate), and ``0`` declares a negative example,
                    meaning the records are distinct.
            sample_weight: Optional weight for each calibration
                           example, for instance inverse sampling
                           weights when the calibration set was not
                           drawn with the same prevalence as the
                           population.

        Both labels must be present. A calibration set containing only
        matches or only distinct pairs raises :class:`CalibrationError`.
        """
        raw_scores = numpy.asarray(raw_scores, dtype=numpy.float64)
        labels = numpy.asarray(labels, dtype=numpy.int_)

        if raw_scores.ndim != 1:
            raise ValueError("raw_scores must be a one-dimensional array")
        if labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional array")
        if raw_scores.shape != labels.shape:
            raise ValueError(
                "raw_scores and labels must have the same length, "
                f"got {raw_scores.shape[0]} and {labels.shape[0]}"
            )
        if raw_scores.size == 0:
            raise ValueError("the calibration set must contain at least one pair")
        if numpy.any((labels != 0) & (labels != 1)):
            raise ValueError("labels must contain only 0 (distinct) and 1 (match)")

        unique_labels = numpy.unique(labels)
        if unique_labels.size < 2:
            present = "match (1)" if unique_labels[0] == 1 else "distinct (0)"
            raise CalibrationError(
                "The calibration set only contains %s examples. A "
                "probability calibration mapping needs both distinct "
                "(label 0) and match (label 1) examples. Provide a "
                "labeled calibration set that contains both classes; "
                "a single-label calibration set cannot calibrate "
                "probabilities and the data to be merged must never be "
                "used to refit the mapping." % present
            )

        if sample_weight is not None:
            sample_weight = numpy.asarray(sample_weight, dtype=numpy.float64)
            if sample_weight.shape != raw_scores.shape:
                raise ValueError(
                    "sample_weight must have the same length as raw_scores"
                )
            if numpy.any(sample_weight < 0):
                raise ValueError("sample_weight must not contain negative values")
            if numpy.sum(sample_weight) <= 0:
                raise ValueError("sample_weight must have a positive total weight")

        if self.method == "isotonic":
            self.regressor = sklearn.isotonic.IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            )
            self.regressor.fit(raw_scores, labels, sample_weight=sample_weight)
        else:
            self.regressor = sklearn.linear_model.LogisticRegression(C=1.0e10)
            self.regressor.fit(
                _logit(raw_scores).reshape(-1, 1),
                labels,
                sample_weight=sample_weight,
            )

        self.fitted_ = True
        return self

    def predict(
        self, raw_scores: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Map raw classifier scores to calibrated probabilities."""
        if not self.fitted_:
            raise ValueError(
                "The calibration mapping is not fitted. Call fit with "
                "an independent, labeled calibration set first."
            )

        raw_scores = numpy.asarray(raw_scores, dtype=numpy.float64)

        if self.method == "isotonic":
            calibrated = self.regressor.predict(raw_scores)
        else:
            calibrated = self.regressor.predict_proba(
                _logit(raw_scores).reshape(-1, 1)
            )[:, 1]

        return numpy.clip(calibrated, 0.0, 1.0)


class CalibratedClassifier:
    """Wrap a frozen classifier so that ``predict_proba`` is calibrated.

    The wrapped classifier is used unchanged to produce the raw
    candidate scores; only the output of ``predict_proba`` is replaced
    by the calibrated mapping. The wrapper exposes the same
    ``predict_proba`` interface as the underlying classifier, so it can
    be passed anywhere the frozen classifier is used, including the
    multiprocessing scoring paths. Both the wrapper and its mapping are
    plain picklable objects.
    """

    def __init__(
        self, classifier: Classifier, calibrator: ProbabilityCalibrator
    ) -> None:
        if not calibrator.fitted_:
            raise ValueError(
                "The calibration mapping must be fitted before it can "
                "wrap a classifier."
            )
        self.classifier = classifier
        self.calibrator = calibrator

    def predict_proba(
        self, X: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        raw_probabilities = self.classifier.predict_proba(X)
        calibrated = self.calibrator.predict(raw_probabilities[:, -1])
        probabilities = numpy.column_stack((1.0 - calibrated, calibrated))
        return probabilities.astype(raw_probabilities.dtype, copy=False)


def _logit(scores: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray:
    # The underlying classifier already constrains its scores to [0, 1],
    # but floating point can land exactly on a boundary, which would make
    # the logit blow up. Clip to a small interior interval first.
    eps = numpy.finfo(numpy.float64).eps
    return numpy.log(numpy.clip(scores, eps, 1.0 - eps)) - numpy.log(
        1.0 - numpy.clip(scores, eps, 1.0 - eps)
    )


def calibrate_classifier(
    classifier: Classifier,
    calibration_pairs: Sequence[RecordDictPair],
    featurizer: Any,
    labels: Sequence[int],
    sample_weight: Collection[float] | None = None,
    method: CalibrationMethod = "isotonic",
) -> CalibratedClassifier:
    """Score an independent calibration set with a frozen classifier and
    fit a calibration mapping on those scores.

    Args:
        classifier: The frozen pairwise classifier.
        calibration_pairs: Pairs of records from the independent
                           calibration set.
        featurizer: The data model's distance function used to turn the
                    record pairs into the feature matrix the classifier
                    expects.
        labels: Binary labels aligned with ``calibration_pairs``, where
                ``1`` means the pair is a match (positive example) and
                ``0`` means the pair is distinct.
        sample_weight: Optional per-example sampling weights.
        method: ``"isotonic"`` or ``"sigmoid"``.
    """
    labels_array = numpy.asarray(list(labels), dtype=numpy.int_)
    pairs = list(calibration_pairs)

    if len(pairs) != labels_array.shape[0]:
        raise ValueError(
            "calibration_pairs and labels must have the same length, "
            f"got {len(pairs)} and {labels_array.shape[0]}"
        )
    if not pairs:
        raise ValueError("the calibration set must contain at least one pair")

    features = featurizer(pairs)
    raw_scores = classifier.predict_proba(features)[:, -1]

    weights: numpy.typing.NDArray[numpy.float64] | None = None
    if sample_weight is not None:
        weights = numpy.asarray(list(sample_weight), dtype=numpy.float64)

    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(raw_scores, labels_array, sample_weight=weights)

    return CalibratedClassifier(classifier, calibrator)
