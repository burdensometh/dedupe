from __future__ import annotations

import functools
from typing import Any, Collection, Iterable, Mapping, Tuple

from ._typing import Cover
from .predicates import Predicate

Partial = Tuple[Predicate, ...]


def _reachable(dupe_cover: Mapping[Any, frozenset[int]]) -> int:
    return len(frozenset.union(*dupe_cover.values())) if dupe_cover else 0


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
    candidates: Mapping[Predicate, Collection[Any]], p: Predicate
) -> tuple[int, float]:
    return (len(candidates[p]), -p.cover_count)


def _score(partial: Iterable[Predicate]) -> float:
    return sum(p.cover_count for p in partial)


def search_budgeted(
    original_cover: Cover,
    comparison_pairs: Mapping[Predicate, frozenset],
    target: int,
    budget: int,
    calls: int,
) -> tuple[Partial, int] | None:
    """
    Find a set of predicates that covers at least `target` matches while
    generating at most `budget` unique comparison pairs.

    Unlike `search`, the cost of a set of predicates is the number of
    unique pairs in the union of their comparison covers, so pairs covered
    by more than one selected predicate are only counted once.

    Returns the selected predicates and the number of unique comparison
    pairs they generate, or None if no set of predicates can cover the
    target within the budget.
    """

    # A predicate that on its own generates more pairs than the budget can
    # never be part of an affordable set of rules.
    candidates = {
        pred: cover
        for pred, cover in original_cover.items()
        if len(comparison_pairs[pred]) <= budget
    }

    cheapest: Partial | None = None
    cheapest_pairs = budget + 1

    # Greedy seed: repeatedly add the predicate that covers the most still
    # uncovered matches among those we can still afford.
    covered: frozenset[int] = frozenset()
    pairs: frozenset = frozenset()
    partial: Partial = ()
    pool = dict(candidates)
    while len(covered) < target and pool:
        best = max(pool, key=lambda p: len(pool[p] - covered))
        best_cover = pool.pop(best)
        if not best_cover - covered:
            break
        new_pairs = pairs | comparison_pairs[best]
        if len(new_pairs) > budget:
            continue
        partial += (best,)
        covered |= best_cover
        pairs = new_pairs
    if len(covered) >= target:
        cheapest = partial
        cheapest_pairs = len(pairs)

    # A node is (remaining candidates, partial, covered matches, pairs),
    # where candidates map predicates to the matches they cover that the
    # partial does not, and pairs is the union of the comparison pairs of
    # the partial.
    Node = Tuple[Cover, Partial, frozenset, frozenset]
    start: Node = (candidates, (), frozenset(), frozenset())
    to_explore: list[Node] = [start]

    while to_explore and calls:
        candidates, partial, covered, pairs = to_explore.pop()

        if len(covered) < target:
            reachable = len(covered) + _reachable(candidates)

            if candidates and reachable >= target:
                order_by = functools.partial(_order_by, candidates)
                best = max(candidates, key=order_by)

                reduced = _remove_dominated(candidates, best)
                to_explore.append((reduced, partial, covered, pairs))

                new_pairs = pairs | comparison_pairs[best]
                if len(new_pairs) < cheapest_pairs:
                    remaining = _uncovered_by(candidates, candidates[best])
                    to_explore.append(
                        (
                            remaining,
                            partial + (best,),
                            covered | original_cover[best],
                            new_pairs,
                        )
                    )

        elif len(pairs) < cheapest_pairs:
            cheapest = partial
            cheapest_pairs = len(pairs)

        calls -= 1

    if cheapest is None:
        return None

    return cheapest, cheapest_pairs


def search(original_cover: Cover, target: int, calls: int) -> Partial:
    def _covered(partial: Partial) -> int:
        return (
            len(frozenset.union(*(original_cover[p] for p in partial)))
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

            reachable = _reachable(candidates) + covered

            if candidates and reachable >= target:
                order_by = functools.partial(_order_by, candidates)
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
