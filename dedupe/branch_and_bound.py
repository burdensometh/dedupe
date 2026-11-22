from __future__ import annotations

import functools
from typing import Any, Callable, Collection, Iterable, Mapping, Tuple

from ._typing import Cover, WeightsLike
from .predicates import Predicate

Partial = Tuple[Predicate, ...]


def _reachable(
    dupe_cover: Mapping[Any, frozenset[int]], coverage_size: Callable = len
) -> float:
    return coverage_size(frozenset.union(*dupe_cover.values())) if dupe_cover else 0


def _remove_dominated(coverage: Cover, dominator: Predicate) -> Cover:
    dominant_cover = coverage[dominator]

    return {
        pred: cover
        for pred, cover in coverage.items()
        if not (dominator.cover_count <= pred.cover_count and dominant_cover >= cover)
    }


def _uncovered_by(
    coverage: Mapping[Any, frozenset[int]], covered: frozenset[int]
) -> dict[Any, frozenset[int]]:
    return {
        pred: still_uncovered
        for pred, uncovered in coverage.items()
        if (still_uncovered := uncovered - covered)
    }


def _order_by(
    candidates: Mapping[Predicate, Collection[Any]],
    p: Predicate,
    coverage_size: Callable = len,
) -> tuple[float, float]:
    return (coverage_size(candidates[p]), -p.cover_count)


def _score(partial: Iterable[Predicate]) -> float:
    return sum(p.cover_count for p in partial)


def search(
    original_cover: Cover, target: float, calls: int, weights: WeightsLike = None
) -> Partial:
    """
    Search for the cheapest set of predicates that covers the target
    amount of training pairs.

    Args:
        original_cover: A dictionary mapping predicates to the indices
                        of the training pairs they cover.
        target: The amount of coverage to reach. When ``weights`` is
                given, this is a weighted amount; otherwise it is a
                count of pairs.
        calls: The maximum number of search states to explore.
        weights: Optional per-pair weights. When given, coverage is
                 measured as the sum of the weights of the covered
                 pairs, so pairs with a weight of zero do not
                 contribute. When ``None``, every pair counts equally.
    """
    if weights is None:
        coverage_size = len
    else:
        pair_weights = list(weights)

        def coverage_size(indices: Collection[int]) -> float:  # type: ignore[misc]
            return sum(pair_weights[i] for i in indices)

    def _covered(partial: Partial) -> float:
        return (
            coverage_size(frozenset.union(*(original_cover[p] for p in partial)))
            if partial
            else 0
        )

    cheapest_score = float("inf")
    cheapest: Partial = ()

    start: tuple[Cover, Partial] = (original_cover, ())
    to_explore = [start]

    while to_explore and calls:
        candidates, partial = to_explore.pop()

        covered = _covered(partial)
        score = _score(partial)

        if covered < target:
            window = cheapest_score - score
            candidates = {
                p: cover for p, cover in candidates.items() if p.cover_count < window
            }

            reachable = _reachable(candidates, coverage_size) + covered

            if candidates and reachable >= target:
                order_by = functools.partial(
                    _order_by, candidates, coverage_size=coverage_size
                )
                best = max(candidates, key=order_by)

                reduced = _remove_dominated(candidates, best)
                to_explore.append((reduced, partial))

                remaining = _uncovered_by(candidates, candidates[best])
                to_explore.append((remaining, partial + (best,)))

        elif score < cheapest_score:
            cheapest = partial
            cheapest_score = score

        calls -= 1

    return cheapest
