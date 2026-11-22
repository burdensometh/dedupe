from __future__ import annotations

import collections
import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, overload
from warnings import warn

import numpy
import numpy.typing
import sklearn.linear_model

import dedupe.core as core
import dedupe.training as training

if TYPE_CHECKING:
    from typing import Any, Iterable, Literal, Mapping, Sequence

    from dedupe._typing import (
        CandidateID,
        CandidateQuota,
        Data,
        DataInt,
        DataStr,
        FeaturizerFunction,
        Labels,
        LabelsLike,
        RecordDict,
        SourceFunction,
        SourceQuota,
    )
    from dedupe._typing import RecordDictPair as TrainingExample
    from dedupe._typing import RecordDictPairs as TrainingExamples
    from dedupe._typing import RecordIDPair
    from dedupe.predicates import Predicate


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # Fallback policy when a (source, difficulty) bucket does not hold
    # enough candidates to fill its quota.
    QuotaFallback = Literal["global", "source", "none"]


class HasCandidates:
    """Has a list of pairs that we could ask the user to label."""

    _candidates: TrainingExamples

    @property
    def candidates(self) -> TrainingExamples:
        return self._candidates

    def __len__(self) -> int:
        return len(self.candidates)


class Learner(ABC, HasCandidates):
    """A single learner that is used by DisagreementLearner."""

    _fitted: bool = False
    # Stable ids of the records making up each candidate, kept aligned with
    # _candidates index-for-index.
    _candidate_ids: list[CandidateID]

    @property
    def candidate_ids(self) -> list[CandidateID]:
        return self._candidate_ids

    @abstractmethod
    def fit(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        """Train on the given data."""

    @abstractmethod
    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float64]:
        """For each of self.candidates, return our current guess [0,1] of if a match."""

    @abstractmethod
    def remove(self, index: int) -> None:
        """Remove a pair from self.candidates."""

    @staticmethod
    def _verify_fit_args(pairs: TrainingExamples, y: LabelsLike) -> list[Literal[0, 1]]:
        """Helper method to verify the arguments given to fit()"""
        if len(pairs) == 0:
            raise ValueError("pairs must have length of at least 1")
        y = list(y)
        if len(pairs) != len(y):
            raise ValueError(
                f"pairs and y must be same length. Got {len(pairs)} and {len(y)}"
            )
        return y


class MatchLearner(Learner):
    def __init__(
        self,
        featurizer: FeaturizerFunction,
        candidates: TrainingExamples,
        candidate_ids: Sequence[CandidateID],
    ):
        self._featurizer = featurizer
        self._candidates = candidates.copy()
        self._candidate_ids = list(candidate_ids)
        self._classifier = sklearn.linear_model.LogisticRegression()
        self._features = self._featurizer(self.candidates)

    def fit(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        y = self._verify_fit_args(pairs, y)
        self._classifier.fit(self._featurizer(pairs), numpy.array(y))
        self._fitted = True

    def remove(self, index: int) -> None:
        self._candidates.pop(index)
        self._candidate_ids.pop(index)
        self._features = numpy.delete(self._features, index, axis=0)

    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float64]:
        if not self._fitted:
            raise ValueError("Must call fit() before candidate_scores()")
        return self._classifier.predict_proba(self._features)[:, 1].reshape(-1, 1)


class BlockLearner(Learner):
    block_learner: training.BlockLearner

    def __init__(self):
        self.current_predicates: tuple[Predicate, ...] = ()
        self._cached_scores: numpy.typing.NDArray[numpy.float64] | None = None
        self._old_dupes: TrainingExamples = []
        self._candidate_ids: list[CandidateID] = []

    def fit(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        y = self._verify_fit_args(pairs, y)
        dupes = [pair for label, pair in zip(y, pairs) if label]

        new_dupes = [pair for pair in dupes if pair not in self._old_dupes]
        new_uncovered = not all(self._predict(new_dupes))

        if new_uncovered:
            self.current_predicates = self.block_learner.learn(
                dupes, recall=1.0, index_predicates=True, candidate_types="simple"
            )
            self._cached_scores = None
            self._old_dupes = dupes
        self._fitted = True

    def candidate_scores(self) -> numpy.typing.NDArray[numpy.float64]:
        if not self._fitted:
            raise ValueError("Must call fit() before candidate_scores()")
        if self._cached_scores is None:
            labels = self._predict(self.candidates)
            self._cached_scores = numpy.array(labels).reshape(-1, 1)

        return self._cached_scores

    def learn_predicates(
        self, dupes: TrainingExamples, recall: float, index_predicates: bool
    ) -> tuple[Predicate, ...]:
        return self.block_learner.learn(
            dupes,
            recall=recall,
            index_predicates=index_predicates,
            candidate_types="random forest",
        )

    def _predict(self, pairs: TrainingExamples) -> Labels:
        labels: Labels = []
        for record_1, record_2 in pairs:
            for predicate in self.current_predicates:
                keys = predicate(record_2, target=True)
                try:
                    overlap = not keys.isdisjoint(predicate(record_1))
                except AttributeError:
                    warn(
                        f"the predicate {predicate.__name__} is not returning "
                        "a frozen set, this will soon be required behaviour",
                        DeprecationWarning,
                    )
                    overlap = not frozenset(keys).isdisjoint(predicate(record_1))
                if overlap:
                    labels.append(1)
                    break
            else:
                labels.append(0)

        return labels

    def remove(self, index: int) -> None:
        self._candidates.pop(index)
        self._candidate_ids.pop(index)
        if self._cached_scores is not None:
            self._cached_scores = numpy.delete(self._cached_scores, index, axis=0)

    def _sample_indices(
        self, sample_size: int, max_cover: int
    ) -> Iterable[RecordIDPair]:
        weights: dict[RecordIDPair, float] = {}
        for predicate, covered in self.block_learner.comparison_cover.items():
            # each predicate gets to vote for every record pair it covers. the
            # strength of that vote is in inverse proportion to the number of
            # records the predicate covers.
            #
            # if a predicate only covers a few record pairs, the value of
            # the vote it puts on those few pairs will be worth more than
            # a predicate that covers almost all the record pairs
            if len(covered) < max_cover:
                weight = 1 / len(covered)
                for pair in covered:
                    weights[pair] = weights.get(pair, 0.0) + weight

        sample_ids: Iterable[RecordIDPair]
        if sample_size < len(weights):
            # consider using a reservoir sampling strategy, which would
            # be more memory efficient and probably about as fast
            normalized_weights = numpy.fromiter(
                weights.values(), dtype=float, count=len(weights)
            ) / sum(weights.values())
            rng = numpy.random.default_rng()
            sample_indices = rng.choice(
                len(weights), size=sample_size, replace=False, p=normalized_weights
            )
            keys = list(weights.keys())
            sample_ids = ((keys[i][0], keys[i][1]) for i in sample_indices)
        else:
            sample_ids = weights.keys()

        return sample_ids


def _filter_canopy_predicates(
    predicates: Iterable[Predicate], canopies: bool
) -> set[Predicate]:
    result = set()
    for predicate in predicates:
        if hasattr(predicate, "index"):
            is_canopy = hasattr(predicate, "canopy")
            if is_canopy == canopies:
                result.add(predicate)
        else:
            result.add(predicate)
    return result


class DedupeBlockLearner(BlockLearner):
    def __init__(
        self,
        candidate_predicates: Iterable[Predicate],
        data: Data,
        index_include: TrainingExamples,
    ):
        super().__init__()

        N_SAMPLED_RECORDS = 5000
        N_SAMPLED_RECORD_PAIRS = 10000

        index_data = sample_records(data, 50000)
        sampled_records = sample_records(index_data, N_SAMPLED_RECORDS)

        preds = _filter_canopy_predicates(candidate_predicates, canopies=True)
        self.block_learner = training.DedupeBlockLearner(
            preds, sampled_records, index_data
        )

        self._candidates = self._sample(sampled_records, N_SAMPLED_RECORD_PAIRS)
        examples_to_index = self.candidates.copy()

        if index_include:
            examples_to_index += index_include

        self._index_predicates(examples_to_index)

    def _index_predicates(self, candidates: TrainingExamples) -> None:
        blocker = self.block_learner.blocker

        records = core.unique(record for pair in candidates for record in pair)

        for field in blocker.index_fields:
            unique_fields = {record[field] for record in records}
            blocker.index(unique_fields, field)

        for pred in blocker.index_predicates:
            pred.freeze(records)

    @overload
    def _sample(self, data: DataInt, sample_size: int) -> TrainingExamples: ...

    @overload
    def _sample(self, data: DataStr, sample_size: int) -> TrainingExamples: ...

    def _sample(self, data, sample_size):
        sample_indices = list(
            self._sample_indices(
                sample_size, len(data) * (len(data) - 1) // 2
            )
        )
        self._candidate_ids = sample_indices

        sample = [(data[id_1], data[id_2]) for id_1, id_2 in sample_indices]

        return sample


class RecordLinkBlockLearner(BlockLearner):
    def __init__(
        self,
        candidate_predicates: Iterable[Predicate],
        data_1: Data,
        data_2: Data,
        index_include: TrainingExamples,
    ):
        super().__init__()

        N_SAMPLED_RECORDS = 4000
        N_SAMPLED_RECORD_PAIRS = 10000

        sampled_records_1 = sample_records(data_1, N_SAMPLED_RECORDS)
        index_data = sample_records(data_2, 50000)
        sampled_records_2 = sample_records(index_data, N_SAMPLED_RECORDS)

        preds = _filter_canopy_predicates(candidate_predicates, canopies=False)
        self.block_learner = training.RecordLinkBlockLearner(
            preds, sampled_records_1, sampled_records_2, index_data
        )

        self._candidates = self._sample(
            sampled_records_1, sampled_records_2, N_SAMPLED_RECORD_PAIRS
        )

        examples_to_index = self.candidates.copy()
        if index_include:
            examples_to_index += index_include
        self._index_predicates(examples_to_index)

    def _index_predicates(self, candidates: TrainingExamples) -> None:
        blocker = self.block_learner.blocker

        A_full, B_full = zip(*candidates)
        A = core.unique(A_full)
        B = core.unique(B_full)

        for field in blocker.index_fields:
            unique_fields = {record[field] for record in B}
            blocker.index(unique_fields, field)

        for pred in blocker.index_predicates:
            pred.freeze(A, B)

    @overload
    def _sample(
        self, data_1: DataInt, data_2: DataInt, sample_size: int
    ) -> TrainingExamples: ...

    @overload
    def _sample(
        self, data_1: DataStr, data_2: DataStr, sample_size: int
    ) -> TrainingExamples: ...

    def _sample(self, data_1, data_2, sample_size):
        sample_indices = list(
            self._sample_indices(sample_size, len(data_1) * len(data_2))
        )
        self._candidate_ids = sample_indices

        sample = [(data_1[id_1], data_2[id_2]) for id_1, id_2 in sample_indices]

        return sample


class DisagreementLearner(HasCandidates):
    matcher: MatchLearner
    blocker: BlockLearner

    def __init__(self) -> None:
        self.y: numpy.typing.NDArray[numpy.int_] = numpy.array([])
        self.pairs: TrainingExamples = []
        self._candidate_ids: list[CandidateID] = []
        self.rng = numpy.random.default_rng()

    @property
    def candidate_ids(self) -> list[CandidateID]:
        """Stable (record_id, record_id) ids aligned with :attr:`candidates`."""
        return self.blocker.candidate_ids

    def _candidate_probs(self) -> numpy.typing.NDArray[numpy.float64]:
        """Current probability of a match for every candidate from each
        learner, shape (n_candidates, n_learners)."""
        prob_l = [learner.candidate_scores() for learner in self._learners]
        return numpy.concatenate(prob_l, axis=1)

    @staticmethod
    def _uncertainty_weights(
        probs: numpy.typing.NDArray[numpy.float64],
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Weights over candidates derived from the disagreement between
        the learners.

        These are the same weights used by :meth:`pop`, so quota-based
        selection ranks candidates with the exact same uncertainty signal
        as the unconstrained selection.
        """
        n_candidates = probs.shape[0]

        # where do the classifers disagree?
        decisions = probs > 0.5
        uncovered_disagreement = numpy.any(decisions != decisions[:, [0]], axis=1) * (
            probs[:, 1] == 0
        )

        if uncovered_disagreement.any():
            # If there are records that the classifier thinks are
            # matches but we are not covering with a blocking rule
            # then choose one of those, with the weights
            # proportional to the classifier's confidence that it
            # is a match. These are the most important to capture
            # for the best possible recall.
            weights = uncovered_disagreement * probs[:, 0]
        elif (probs[:, 1] == 1).any():
            # Otherwise, sample from records that are covered, uniformly
            # across classifier confidence.
            #
            # We don't sample uniformly across covered records, because
            # negative examples would dominate.
            weights = numpy.zeros(n_candidates)
            covered = (probs[:, 1] == 1).nonzero()[0]
            # Spread the weight evenly over the quantiles of classifier
            # confidence so that, across a batch, candidates are sampled
            # uniformly across confidence rather than all coming from the
            # most common region.
            order = covered[numpy.argsort(probs[covered, 0])]
            weights[order] = 1.0 / len(order)
        else:
            # If there are no uncovered disagreements and no covered pairs, then
            # choose a pair using weights related to the disagreement
            # between the classifiers
            weights = numpy.std(probs, axis=1)

        total = weights.sum()
        if total <= 0:  # pragma: no cover - defensive, degenerate scores
            weights = numpy.full(n_candidates, 1.0 / n_candidates)
        else:
            weights = weights / total
        return weights

    def pop(self) -> TrainingExample:
        if not (n_candidates := len(self.candidates)):
            raise IndexError("No more unlabeled examples to label")

        probs = self._candidate_probs()

        # where do the classifers disagree?
        decisions = probs > 0.5
        uncovered_disagreement = numpy.any(decisions != decisions[:, [0]], axis=1) * (
            probs[:, 1] == 0
        )

        if uncovered_disagreement.any():
            # If there are records that the classifier thinks are
            # matches but we are not covering with a blocking rule
            # then choose one of those, with the weights
            # proportional to the classifier's confidence that it
            # is a match. These are the most important to capture
            # for the best possible recall.
            weights = uncovered_disagreement * probs[:, 0]
            weights /= weights.sum()
            uncertain_index = self.rng.choice(n_candidates, p=weights)
        elif (probs[:, 1] == 1).any():
            # Otherwise, sample from records that are covered, uniformly
            # across classifier confidence.
            #
            # We don't sample uniformly across covered records, because
            # negative examples would dominate.
            covered = (probs[:, 1] == 1).nonzero()[0]
            target = random.random()
            uncertain_index = covered[
                numpy.argmin(numpy.absolute(probs[covered, 0] - target))
            ]
        else:
            # If there are no uncovered disagreements and no covered pairs, then
            # choose a pair using weights related to the disagreement
            # between the classifiers
            weights = numpy.std(probs, axis=1)
            weights /= weights.sum()
            uncertain_index = self.rng.choice(n_candidates, p=weights)

        logger.debug(
            "Classifier: %.2f, Covered: %s",
            probs[uncertain_index][0],
            bool(probs[uncertain_index][1]),
        )

        uncertain_pair: TrainingExample = self.candidates[uncertain_index]
        self._remove(uncertain_index)
        return uncertain_pair

    @property
    def _learners(self) -> tuple[Learner, ...]:
        return (self.matcher, self.blocker)

    def _remove(self, index: int) -> None:
        self._candidates.pop(index)
        self._candidate_ids.pop(index)
        for learner in self._learners:
            learner.remove(index)

    @staticmethod
    def _default_source(record: RecordDict) -> Any:
        return record.get("source")

    def _pair_source(
        self, pair: TrainingExample, source_field: str | SourceFunction | None
    ) -> Any:
        """Stable source key for a candidate pair.

        When both records come from the same source the key is that
        source; otherwise it is a sorted tuple of the two sources so the
        key does not depend on which side of a pair a record is on.
        """
        if source_field is None:
            source_fn: SourceFunction = self._default_source
        elif callable(source_field):
            source_fn = source_field
        else:
            field = source_field

            def source_fn(record: RecordDict) -> Any:
                return record.get(field)

        source_1 = source_fn(pair[0])
        source_2 = source_fn(pair[1])
        if source_1 == source_2:
            return source_1
        # _stable_sort_key type-tags values, so even incomparable or None
        # sources sort deterministically.
        return tuple(sorted((source_1, source_2), key=_stable_sort_key))

    @staticmethod
    def _difficulty_bins(
        match_probs: numpy.typing.NDArray[numpy.float64],
        bin_edges: Sequence[float],
    ) -> numpy.typing.NDArray[numpy.int_]:
        """Assign each candidate an integer difficulty bin.

        Difficulty is the classifier's uncertainty about the pair:
        ``1 - |2 * p(match) - 1``, so an easy pair (probability near 0 or
        1) gets a low bin and a hard pair (probability near 0.5) gets a
        high bin. ``bin_edges`` are the monotonic uncertainty cut points
        between bins.
        """
        bin_edges = tuple(bin_edges)
        if any(not 0.0 < edge < 1.0 for edge in bin_edges) or list(
            bin_edges
        ) != sorted(bin_edges):
            raise ValueError(
                "bin_edges must be monotonically increasing values "
                f"strictly between 0 and 1, got {bin_edges!r}"
            )
        uncertainty = 1.0 - numpy.absolute(2.0 * match_probs - 1.0)
        return numpy.digitize(uncertainty, bin_edges).astype(numpy.int_)

    def pop_quota(
        self,
        quotas: CandidateQuota | SourceQuota,
        *,
        source_field: str | SourceFunction | None = None,
        bin_edges: Sequence[float] = (1.0 / 3.0, 2.0 / 3.0),
        fallback: QuotaFallback = "global",
        n: int | None = None,
        seed: int | None = None,
    ) -> TrainingExamples:
        """Select a batch of uncertain candidates under per-source quotas.

        Candidates are grouped by (source, difficulty bin) and, within
        each group, drawn without replacement using the same uncertainty
        weights as :meth:`pop`. Each selected pair consumes exactly one
        slot of one bucket, even when several blocking rules cover it.
        Buckets that cannot fill their quota are backfilled according to
        ``fallback``.

        Args:
            quotas: Either a mapping of ``(source, difficulty_bin)`` to
                the maximum number of pairs to label from that bucket, or
                a mapping of source to the maximum number of pairs to
                label from that source across *all* of its difficulty
                bins. The two forms may be mixed; the source-level quota
                is spent from whatever capacity the explicit per-bin
                quotas did not use.
            source_field: Name of the record field holding the source, or
                a callable taking a record and returning its source.
                Defaults to the ``"source"`` field.
            bin_edges: Monotonic uncertainty cut points defining the
                difficulty bins (see :meth:`_difficulty_bins`).
            fallback: What to do with quota that a bucket cannot fill:

                * ``"global"`` (default): redistribute the unmet quota to
                  any remaining candidate, highest uncertainty first.
                * ``"source"``: redistribute only to other difficulty bins
                  of the declared sources.
                * ``"none"``: leave the quota unmet and return fewer pairs.
            n: Overall cap on the number of returned pairs. Defaults to
                the sum of all quotas.
            seed: Seed for the random number generator. With the same
                seed and candidate set, the selection is reproducible.

        Returns:
            A list of record pairs in the same form as :meth:`pop`; the
            pairs are removed from the candidate pool and can be passed
            directly to :meth:`mark` (or ``Dedupe.mark_pairs``).
        """
        bucket_quotas = _normalize_quotas(quotas)
        if not bucket_quotas:
            return []
        if fallback not in ("global", "source", "none"):
            raise ValueError(
                "fallback must be one of 'global', 'source' or 'none', "
                f"got {fallback!r}"
            )
        if n is not None and n < 0:
            raise ValueError("n must be non-negative")

        rng = numpy.random.default_rng(seed)

        if not self.candidates:
            return []

        probs = self._candidate_probs()
        weights = self._uncertainty_weights(probs)
        bins = self._difficulty_bins(probs[:, 0], bin_edges)
        sources = [
            self._pair_source(pair, source_field) for pair in self.candidates
        ]

        # Group candidate indices by (source, difficulty) bucket.
        buckets: dict[tuple[Any, int], list[int]] = collections.defaultdict(list)
        for index, (source, bin_) in enumerate(zip(sources, bins)):
            buckets[(source, int(bin_))].append(index)

        # Stable iteration order so results do not depend on dict hashing.
        ordered_buckets = sorted(buckets, key=_stable_sort_key)

        claimed: set[CandidateID] = set()
        selected_indices: list[int] = []

        explicit_quotas = {
            key: count
            for key, count in bucket_quotas.items()
            if key[1] != _ANY_BIN
        }
        wildcard_quotas = {
            key[0]: count
            for key, count in bucket_quotas.items()
            if key[1] == _ANY_BIN
        }
        wildcard_left = dict(wildcard_quotas)

        # Total budget is the quota the caller declared. A bucket that holds
        # fewer candidates than its quota leaves budget unmet after pass 1;
        # whether that budget is redistributed is decided by ``fallback``.
        # Explicit per-bin quotas and a source-level quota for the same
        # source are additive budgets, but the source pool only fills
        # capacity left unused by the explicit quotas during pass 1.
        total_quota = sum(explicit_quotas.values()) + sum(
            wildcard_quotas.values()
        )
        cap = total_quota if n is None else min(n, total_quota)

        # Pass 1: fill each declared quota bucket from its own candidates.
        # Explicit per-bin quotas are spent first; the shared source-level
        # quota fills any remaining capacity of the source's bins. A pair is
        # claimed at most once, so overlapping blocking coverage can never
        # consume multiple units of budget.
        for source, bin_ in ordered_buckets:
            explicit = explicit_quotas.get((source, bin_), 0)
            budget = explicit + wildcard_left.get(source, 0)
            # Never select more than the overall cap in total.
            budget = min(budget, cap - len(selected_indices))
            if budget <= 0:
                continue
            chosen = self._sample_bucket(
                buckets[(source, bin_)], weights, budget, claimed, rng
            )
            selected_indices.extend(chosen)
            wildcard_spent = max(0, len(chosen) - explicit)
            wildcard_left[source] = wildcard_left.get(source, 0) - wildcard_spent

        # Pass 2: backfill quota that individual buckets could not meet.
        if fallback != "none":
            remaining = cap - len(selected_indices)
            if remaining > 0:
                declared_sources = set(wildcard_quotas) | {
                    key[0] for key in explicit_quotas
                }
                eligible: list[int] = []
                for source, bin_ in ordered_buckets:
                    if fallback == "source" and source not in declared_sources:
                        # Source-level fallback may only redistribute quota
                        # to other difficulty bins of declared sources.
                        # Global fallback may take from any remaining bucket.
                        continue
                    eligible.extend(buckets[(source, bin_)])

                backfill = self._sample_bucket(
                    eligible, weights, remaining, claimed, rng
                )
                selected_indices.extend(backfill)

        # Remove from the end first so earlier indices stay valid.
        selected_pairs: TrainingExamples = []
        for index in sorted(selected_indices, reverse=True):
            selected_pairs.append(self.candidates[index])
            self._remove(index)
        selected_pairs.reverse()
        return selected_pairs

    def _sample_bucket(
        self,
        indices: Sequence[int],
        weights: numpy.typing.NDArray[numpy.float64],
        quota: int,
        claimed: set[CandidateID],
        rng: numpy.random.Generator,
    ) -> list[int]:
        """Weighted sampling without replacement within one bucket.

        A pair covered by several blocking rules (and therefore a member
        of multiple candidate groupings) is protected by ``claimed`` so
        it can only ever consume a single unit of budget.
        """
        available = [
            index
            for index in indices
            if self.candidate_ids[index] not in claimed
        ]
        if not available or quota <= 0:
            return []

        take = min(quota, len(available))
        bucket_weights = numpy.array([weights[index] for index in available])

        # Some candidates can carry zero weight (e.g. covered pairs lose to
        # uncovered disagreements); drop them so numpy's weighted choice does
        # not ask for more draws than it has non-zero options.
        positive = bucket_weights > 0
        if not positive.any():
            # Every candidate is tied at zero weight: choose uniformly so the
            # seed still controls which pairs are returned.
            bucket_weights = numpy.full(len(available), 1.0 / len(available))
        else:
            available = [index for index, keep in zip(available, positive) if keep]
            bucket_weights = bucket_weights[positive]
            take = min(take, len(available))
            bucket_weights = bucket_weights / bucket_weights.sum()

        positions = rng.choice(
            len(available), size=take, replace=False, p=bucket_weights
        )
        chosen = [available[pos] for pos in positions]

        for index in chosen:
            claimed.add(self.candidate_ids[index])
        return chosen

    def mark(self, pairs: TrainingExamples, y: LabelsLike) -> None:
        self.y = numpy.concatenate([self.y, y])
        self.pairs.extend(pairs)
        for learner in self._learners:
            learner.fit(self.pairs, self.y)

    def learn_predicates(
        self, recall: float, index_predicates: bool
    ) -> tuple[Predicate, ...]:
        dupes = [pair for label, pair in zip(self.y, self.pairs) if label]
        return self.blocker.learn_predicates(
            dupes, recall=recall, index_predicates=index_predicates
        )


class DedupeDisagreementLearner(DisagreementLearner):
    def __init__(
        self,
        candidate_predicates: Iterable[Predicate],
        featurizer: FeaturizerFunction,
        data: Data,
        index_include: TrainingExamples,
    ):
        super().__init__()
        data = core.index(data)

        random_pair = (
            random.choice(list(data.values())),
            random.choice(list(data.values())),
        )
        exact_match = (random_pair[0], random_pair[0])

        index_include = index_include.copy()
        index_include.append(exact_match)

        self.blocker = DedupeBlockLearner(candidate_predicates, data, index_include)

        self._candidates = self.blocker.candidates.copy()
        self._candidate_ids = self.blocker.candidate_ids.copy()

        self.matcher = MatchLearner(
            featurizer, self.candidates, self.candidate_ids
        )

        examples = [exact_match] * 4 + [random_pair]
        labels: Labels = [1] * 4 + [0]  # type: ignore[assignment]
        self.mark(examples, labels)


class RecordLinkDisagreementLearner(DisagreementLearner):
    def __init__(
        self,
        candidate_predicates: Iterable[Predicate],
        featurizer: FeaturizerFunction,
        data_1: Data,
        data_2: Data,
        index_include: TrainingExamples,
    ):
        super().__init__()
        data_1 = core.index(data_1)

        offset = len(data_1)
        data_2 = core.index(data_2, offset)

        random_pair = (
            random.choice(list(data_1.values())),
            random.choice(list(data_2.values())),
        )
        exact_match = (random_pair[0], random_pair[0])

        index_include = index_include.copy()
        index_include.append(exact_match)

        self.blocker = RecordLinkBlockLearner(
            candidate_predicates, data_1, data_2, index_include
        )
        self._candidates = self.blocker.candidates.copy()
        self._candidate_ids = self.blocker.candidate_ids.copy()

        self.matcher = MatchLearner(
            featurizer, self.candidates, self.candidate_ids
        )

        examples = [exact_match] * 4 + [random_pair]
        labels: Labels = [1] * 4 + [0]  # type: ignore[assignment]
        self.mark(examples, labels)


def sample_records(data: Mapping, sample_size: int) -> dict:
    keys = data.keys()
    if len(data) > sample_size:
        keys = random.sample(tuple(keys), sample_size)  # type: ignore[assignment]
    # Always make a copy to avoid surprises
    return {k: data[k] for k in keys}


# Sentinel marking a source-level quota that applies to every bin.
_ANY_BIN = -1


def _stable_sort_key(value: object) -> tuple:
    """Deterministic sort key that tolerates mixed/None/unhashable values."""
    if value is None:
        return (0, "")
    if isinstance(value, tuple):
        return (2, tuple(_stable_sort_key(item) for item in value))
    # Tag by type name so incomparable types never meet in a comparison.
    return (1, (type(value).__name__, str(value)))


def _normalize_quotas(
    quotas: CandidateQuota | SourceQuota,
) -> dict[tuple[object, int], int]:
    """Normalize the accepted quota shapes into ``(source, bin) -> count``.

    Accepted shapes:

    * ``{(source, bin): count}``: explicit per-bucket quotas.
    * ``{source: count}``: a total quota for the source across all of its
      difficulty bins, tagged with the :data:`_ANY_BIN` sentinel.
    """
    normalized: dict[tuple[object, int], int] = {}
    for key, count in quotas.items():
        if count < 0:
            raise ValueError(f"quota counts must be non-negative, got {count}")
        if isinstance(key, tuple) and len(key) == 2 and not _looks_like_source(key):
            source, bin_ = key
            bin_index = int(bin_)  # type: ignore[arg-type]
            if bin_index < 0:
                raise ValueError(
                    f"difficulty bins must be non-negative, got {bin_!r}"
                )
            normalized[(source, bin_index)] = (
                normalized.get((source, bin_index), 0) + count
            )
        else:
            # A source-level quota applies to every difficulty bin; the bin
            # wildcard is resolved lazily against the bins that exist.
            normalized[(key, _ANY_BIN)] = (
                normalized.get((key, _ANY_BIN), 0) + count
            )
    return normalized


def _looks_like_source(key: tuple) -> bool:
    """Heuristic: a 2-tuple key is a source (e.g. a pair of source values)
    rather than a (source, bin) key when its second element is not an int."""
    return not isinstance(key[1], int)
