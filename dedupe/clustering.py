#!/usr/bin/python
from __future__ import annotations

import array
import itertools
import logging
import tempfile
from collections import defaultdict
from typing import Generator, Iterable, Sequence, cast

import numpy
import numpy.typing
import scipy.cluster.hierarchy

from dedupe._typing import (
    ArrayLinks,
    CannotLink,
    Clusters,
    RecordID,
    Scores,
    TupleLinks,
)

logger = logging.getLogger(__name__)


def connected_components(edgelist: Scores, max_components: int) -> Generator[Scores]:
    if len(edgelist) == 0:
        raise StopIteration()

    unlabeled_edgelist = edgelist

    # we are going to keep track of the connected components
    # with another field in the record array of the edgelist.
    # unfortunately, it's not straightforward to add another
    # field to a memmapped record array so, we are going to
    # have to create a new memmapped array with all the fields
    # we want and copy things over.
    with tempfile.TemporaryDirectory() as path:
        filename = path + "/unlabeled_edgelist"
        edgelist = numpy.memmap(
            filename,
            dtype=(unlabeled_edgelist.dtype.descr + [("label", "int32")]),
            mode="w+",
            shape=unlabeled_edgelist.shape,
        )

        if hasattr(unlabeled_edgelist, "filename"):
            assert isinstance(unlabeled_edgelist, numpy.memmap)
            copy_mmap_record_arrays(unlabeled_edgelist, edgelist, ["pairs", "score"])
        else:
            copy_to_mmap_record_array(unlabeled_edgelist, edgelist, ["pairs", "score"])

        yield from _connected_components(edgelist, max_components)

        edgelist._mmap.close()  # type: ignore


def _connected_components(edgelist: Scores, max_components: int) -> Generator[Scores]:
    component_stops = union_find(edgelist)

    start = 0
    for stop in component_stops:
        sub_graph = edgelist[start:stop]
        n_edges = stop - start
        start = stop

        needs_filtering = False
        # first we find the upper bound of the
        # number of components given the edgelist
        upper_bound_components = n_edges + 1
        if upper_bound_components > max_components:
            # which we can refine using a more expensive operation
            # if it's possible we have too many components
            n_components = len(numpy.unique(sub_graph["pairs"]))
            if n_components > max_components:
                needs_filtering = True

        if needs_filtering:
            min_score = numpy.min(sub_graph["score"])
            min_score_logit = numpy.log(min_score) - numpy.log(1 - min_score)
            threshold = 1 / (1 + numpy.exp(-min_score_logit - 1))
            logger.warning(
                "A component contained %s elements. "
                "Components larger than %s are "
                "re-filtered. The threshold for this "
                "filtering is %s" % (n_components, max_components, threshold)
            )
            # slices of memmaped arrays are also memmaped arrays,
            # which is what we want. The components should
            # already sorted by score so we can slice as oppose
            # to selecting like `sub_graph[sub_graph['score'] >
            # threshold]`, which would lead to an in memory copy being
            # made
            cut_point = numpy.searchsorted(sub_graph["score"], threshold)
            filtered_sub_graph = sub_graph[max(cut_point, 2) :]

            for sub_graph in _connected_components(filtered_sub_graph, max_components):
                yield sub_graph[["pairs", "score"]]
        else:
            yield sub_graph[["pairs", "score"]]


def union_find(scored_pairs: Scores) -> numpy.typing.NDArray[numpy.int_]:
    root: dict[RecordID, int] = {}

    components = {}

    edgelist = scored_pairs["pairs"]
    labels = scored_pairs["label"]

    it = numpy.nditer(edgelist, ["external_loop"])

    n_edges = len(scored_pairs)

    array_type = "H"
    for array_type in ("H", "I", "L", "Q"):
        if (2 ** (8 * array.array(array_type).itemsize) - 1) > n_edges:
            break
    else:
        raise ValueError(
            "You are making more comparisons than your machine can handle, try raising the threshold"
        )

    a: RecordID
    b: RecordID

    for i, (a, b) in enumerate(it):  # type: ignore
        root_a = root.get(a)
        root_b = root.get(b)

        if root_a is None and root_b is None:
            root[a] = root[b] = i
            components[i] = array.array(array_type, [i])
        elif root_a is None or root_b is None:
            if root_a is None:
                b = a
                root_a = root_b
            root_a = cast(int, root_a)
            components[root_a].append(i)
            root[b] = root_a
        elif root_a != root_b:
            if len(components[root_a]) < len(components[root_b]):
                root_a, root_b = root_b, root_a

            components[root_a].extend(components[root_b])
            components[root_a].append(i)

            component_b = numpy.unique(edgelist[components[root_b]])
            for node in component_b:
                root[node] = root_a

            del components[root_b]

        else:
            components[root_a].append(i)

    for label, component in components.items():
        labels[component] = label

    # we want our selections to remain memmapped arrays so we sort and
    # get the indices where the components change. This will allow us
    # to slice pieces of the memmapped array, and have those slices
    # also be memmaped arrays.
    #
    # Since we we might be calling union_find recursively it would be
    # good to use a stable sort to take advantage of sorting done in a
    # previous recursion.
    #
    # stable sorts in numpy take about n / 2 work space so we won't
    # do it if scored_pairs is too big
    if len(scored_pairs) > 2500000:
        scored_pairs.sort(order=("label", "score"))
    else:
        scored_pairs.sort(order=("label", "score"), kind="stable")

    return numpy.cumsum(numpy.unique(scored_pairs["label"], return_counts=True)[1])


def condensedDistance(
    dupes: Scores,
) -> tuple[dict[int, RecordID], numpy.typing.NDArray[numpy.float64], int]:
    """
    Convert the pairwise list of distances in dupes to "condensed
    distance matrix" required by the hierarchical clustering
    algorithms. Also return a dictionary that maps the distance matrix
    to the record_ids.

    The formula for an index of the condensed matrix is

    index = {N choose 2}-{N-row choose 2} + (col-row-1)
          = N*(N-1)/2 - (N-row)*(N-row-1)/2 + col - row - 1
            ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
          matrix_length       row_step

    where (row,col) is index of an uncondensed square N X N distance matrix.

    See http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html
    """

    candidate_set = numpy.unique(dupes["pairs"])

    i_to_id = dict(enumerate(candidate_set))

    ids = candidate_set.searchsorted(dupes["pairs"])
    row = ids[:, 0]
    col = ids[:, 1]

    N = len(candidate_set)

    # alternate form thanks to wolfram alpa
    index = row * (2 * N - row - 3) // 2 + col - 1

    condensed_distances = numpy.ones(N * (N - 1) // 2, "f4")
    condensed_distances[index] = 1 - dupes["score"]

    return i_to_id, condensed_distances, N


def cluster(
    dupes: Scores,
    threshold: float = 0.5,
    max_components: int = 30000,
    cannot_link: CannotLink | None = None,
) -> Clusters:
    """
    Takes in a list of duplicate pairs and clusters them in to a
    list records that all refer to the same entity based on a given
    threshold

    Keyword arguments:
    threshold -- number between 0 and 1 (default is .5). lowering the
                 number will increase precision, raising it will increase
                 recall
    cannot_link -- an iterable of pairs of record ids that must never
                   be assigned to the same cluster. The constraint is
                   enforced while clusters are actually built: a merge
                   that would place both members of a pair in the same
                   cluster is refused, even if the two records would
                   only be joined indirectly through other records.
                   Records left without a partner are returned as
                   usual (and, in :func:`dedupe.Dedupe.partition`, as
                   singletons). Pairs whose ids are not part of this
                   batch of scored pairs are ignored, since they can
                   never share a cluster here.
    """
    distance_threshold = 1 - threshold
    dupe_sub_graphs = connected_components(dupes, max_components)

    cannot_link_pairs = _normalize_cannot_link(cannot_link)
    forbidden_ids: set[frozenset[RecordID]] = {
        frozenset(pair) for pair in cannot_link_pairs
    }

    for sub_graph in dupe_sub_graphs:
        if len(sub_graph) > 1:
            i_to_id, condensed_distances, N = condensedDistance(sub_graph)

            linkage = scipy.cluster.hierarchy.linkage(
                condensed_distances, method="centroid"
            )

            partition = scipy.cluster.hierarchy.fcluster(
                linkage, distance_threshold, criterion="distance"
            )

            clusters: dict[int, list[int]] = defaultdict(list)

            for i, cluster_id in enumerate(partition):
                clusters[cluster_id].append(i)

            forbidden = _component_cannot_link(cannot_link_pairs, i_to_id)

            squared_distances = condensed_distances**2
            for cluster in clusters.values():
                if len(cluster) > 1:
                    if forbidden and _cluster_violates(cluster, forbidden):
                        # limited-scope constrained re-merge: only
                        # clusters that actually violate a constraint
                        # are reconsidered; every other cluster is left
                        # exactly as the ordinary algorithm produced it.
                        groups = _constrained_merge(
                            cluster,
                            condensed_distances,
                            N,
                            distance_threshold,
                            forbidden,
                        )
                    else:
                        groups = [set(cluster)]

                    for group in groups:
                        if len(group) > 1:
                            ordered = sorted(group)
                            scores = confidences(ordered, squared_distances, N)
                            record_ids = tuple(i_to_id[i] for i in ordered)
                            yield record_ids, scores  # type: ignore[misc]

        else:
            ((ids, score),) = sub_graph
            if score > threshold:
                normalized = frozenset(_normalize_record_id(i) for i in ids)
                if normalized in forbidden_ids:
                    # the only two records in this component are
                    # explicitly forbidden: do not form a pair. partition
                    # will emit them separately as singletons.
                    continue
                yield tuple(ids), (score,) * 2


def _normalize_record_id(record_id: object) -> RecordID:
    """Normalize a record id to a plain python ``int`` or ``str``.

    Record ids read back out of the numpy structured arrays used by
    dedupe can be numpy scalars (``numpy.bytes_`` or ``numpy.str_``)
    rather than the python objects the caller supplied. Normalizing
    both sides makes them comparable.
    """
    if isinstance(record_id, bytes):
        return record_id.decode("utf-8", "surrogateescape")
    if isinstance(record_id, str):
        return record_id
    if isinstance(record_id, numpy.integer):
        return int(record_id)
    return record_id  # type: ignore[return-value]


def _normalize_cannot_link(
    cannot_link: CannotLink | None,
) -> list[tuple[RecordID, RecordID]]:
    """Validate and canonicalize the supplied cannot-link pairs.

    Policy for edge cases:

    * A pair that names the same record twice is contradictory and
      raises :class:`ValueError`.
    * Anything that is not a two-element pair raises :class:`ValueError`.
    * Repeated pairs, in either order, are collapsed.
    * Ids that do not appear among the records being partitioned are
      kept here but simply never match a cluster; see
      :func:`_component_cannot_link`.
    """
    if not cannot_link:
        return []

    pairs: list[tuple[RecordID, RecordID]] = []
    seen: set[frozenset[RecordID]] = set()
    for pair in cannot_link:
        try:
            left, right = pair
        except (TypeError, ValueError):
            raise ValueError(
                "Every cannot_link constraint must be a pair of record ids, "
                "got %r" % (pair,)
            )

        left = _normalize_record_id(left)
        right = _normalize_record_id(right)

        if left == right:
            raise ValueError(
                "A record cannot be constrained not to link to itself: %r" % (left,)
            )

        key = frozenset((left, right))
        if key not in seen:
            seen.add(key)
            pairs.append((left, right))

    return pairs


def _component_cannot_link(
    cannot_link: Sequence[tuple[RecordID, RecordID]],
    i_to_id: dict[int, RecordID],
) -> dict[int, set[int]]:
    """Translate cannot-link pairs to the local indices used by one
    connected component.

    A pair with a member outside of this component cannot be violated
    within it, so it is dropped: that is the explicit policy for unknown
    / out-of-scope ids.
    """
    id_to_i: dict[RecordID, int] = {}
    for i, record_id in i_to_id.items():
        id_to_i.setdefault(_normalize_record_id(record_id), i)

    forbidden: dict[int, set[int]] = {}
    for left, right in cannot_link:
        i = id_to_i.get(left)
        j = id_to_i.get(right)
        if i is None or j is None:
            continue
        forbidden.setdefault(i, set()).add(j)
        forbidden.setdefault(j, set()).add(i)

    return forbidden


def _condensed_index(i: int, j: int, n: int) -> int:
    """Index into scipy's condensed distance matrix.

    ``i`` and ``j`` are local indices into ``i_to_id`` (as produced by
    :func:`condensedDistance`) and must satisfy ``i < j``.
    """
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 3) // 2 + j - 1


def _cluster_violates(cluster: Sequence[int], forbidden: dict[int, set[int]]) -> bool:
    members = set(cluster)
    for record in cluster:
        partners = forbidden.get(record)
        if partners is not None and not partners.isdisjoint(members):
            return True
    return False


def _constrained_merge(
    members: Sequence[int],
    condensed_distances: numpy.typing.NDArray[numpy.float64],
    n: int,
    distance_threshold: float,
    forbidden: dict[int, set[int]],
) -> list[set[int]]:
    """Agglomeratively re-cluster one already-formed cluster so that no
    forbidden pair ends up in the same output cluster.

    This is only run on the *limited scope* of a cluster that already
    violates a constraint; every other cluster is emitted untouched.
    Within that scope, clusters are merged nearest-first using centroid
    linkage (matching :func:`dedupe.clustering.cluster`), except that a
    merge is skipped whenever it would put a forbidden pair together.
    The merge that is chosen instead is the nearest *allowed* one, which
    is what keeps ordinary paths (e.g. ``b--c`` when ``a--b`` is
    forbidden) intact instead of discarding the whole area.

    Because a forbidden pair is never merged, and every member is
    assigned to exactly one output cluster, the result guarantees that
    no output cluster contains both members of a forbidden pair.
    Returns sets of the original local indices.
    """
    ordered = sorted(members)
    k = len(ordered)
    position = {record: i for i, record in enumerate(ordered)}

    # centroid linkage is defined on *squared* distances (Lance-Williams
    # below), while the threshold and the reported distances are plain
    # distances, so we keep squared distances and take square roots when
    # comparing against the threshold.
    squared = numpy.zeros((k, k), dtype=numpy.float64)
    for a in range(k):
        for b in range(a + 1, k):
            distance = condensed_distances[
                _condensed_index(ordered[a], ordered[b], n)
            ]
            squared[a, b] = squared[b, a] = float(distance) ** 2

    forbidden_positions: list[set[int]] = [set() for _ in range(k)]
    for a, record in enumerate(ordered):
        for partner in forbidden.get(record, ()):
            b = position.get(partner)
            if b is not None:
                forbidden_positions[a].add(b)

    clusters: dict[int, set[int]] = {i: {i} for i in range(k)}
    sizes: dict[int, int] = {i: 1 for i in range(k)}
    active: set[int] = set(range(k))

    def violates(left: int, right: int) -> bool:
        smaller, larger = clusters[left], clusters[right]
        if len(smaller) > len(larger):
            smaller, larger = larger, smaller
        for record in smaller:
            if not forbidden_positions[record].isdisjoint(larger):
                return True
        return False

    while len(active) > 1:
        best: tuple[int, int] | None = None
        best_squared = numpy.inf
        ids = sorted(active)
        for a_index in range(len(ids)):
            left = ids[a_index]
            for b_index in range(a_index + 1, len(ids)):
                right = ids[b_index]
                if violates(left, right):
                    continue
                value = squared[left, right]
                if value < best_squared:
                    best_squared = value
                    best = (left, right)

        if best is None or numpy.sqrt(best_squared) > distance_threshold:
            break

        left, right = best
        left_size = sizes[left]
        right_size = sizes[right]
        merged_size = left_size + right_size
        correction = (left_size * right_size) / (merged_size * merged_size)

        for other in active:
            if other == left or other == right:
                continue
            updated = (
                left_size * squared[left, other]
                + right_size * squared[right, other]
            ) / merged_size - correction * best_squared
            squared[left, other] = squared[other, left] = updated

        clusters[left] |= clusters[right]
        sizes[left] = merged_size
        del clusters[right]
        del sizes[right]
        active.discard(right)

    return [{ordered[p] for p in group} for group in clusters.values()]


def confidences(
    cluster: Sequence[int],
    squared_distances: numpy.typing.NDArray[numpy.float64],
    d: int,
) -> numpy.typing.NDArray[numpy.float64]:
    """
    We calculate a per record score that is similar to a standard
    deviation.  The main reason is that these record scores can be
    used to calculate the standard deviation of an entire cluster,
    which is a reasonable metric for clusters.
    """
    scores: numpy.typing.NDArray[numpy.float64]
    scores_d = dict.fromkeys(cluster, 0.0)
    C = 2 * d - 3
    for i, j in itertools.combinations(cluster, 2):
        index = i * (C - i) // 2 + j - 1
        squared_dist = squared_distances[index]
        scores_d[i] += squared_dist
        scores_d[j] += squared_dist
    scores = numpy.array([score for _, score in sorted(scores_d.items())])
    scores /= len(cluster) - 1
    scores = numpy.sqrt(scores)
    scores = 1 - scores
    return scores


def greedyMatching(dupes: Scores) -> TupleLinks:
    A: set[RecordID] = set()
    B: set[RecordID] = set()

    dupes.sort(order="score")
    dupes = dupes[::-1]

    for (a, b), score in dupes:
        if a not in A and b not in B:
            A.add(a)
            B.add(b)

            yield (a, b), score


def gazetteMatching(
    scored_blocks: Iterable[Scores], threshold: float = 0, n_matches: int = 1
) -> ArrayLinks:
    for block in scored_blocks:
        block = block[block["score"] > threshold]
        block.sort(order="score")
        block = block[::-1]

        if len(block):
            if n_matches:
                yield block[:n_matches].copy()
            else:
                yield block.copy()


def pair_gazette_matching(
    scored_pairs: Scores, threshold: float = 0.0, n_matches: int = 1
) -> ArrayLinks:
    scored_pairs.sort(order="pairs")

    group_key = scored_pairs["pairs"][:, 0]
    change_points = numpy.where(numpy.roll(group_key, 1) != group_key)[0]
    scored_blocks = numpy.split(scored_pairs, change_points)

    for match in gazetteMatching(scored_blocks, threshold, n_matches):
        if match:
            yield from match


def copy_to_mmap_record_array(
    source: numpy.ndarray,
    target: numpy.memmap,
    fields: list[str],
    chunksize: int = 100000,
) -> None:
    """
    Writing into a memmapped array allocates memory equivalent to the
    amount that you are writing. With big arrays this is undesirable
    so we write in chunks
    """
    assert target.filename is not None

    start = 0
    stops = itertools.chain(range(chunksize, source.size, chunksize), [source.size])
    for stop in stops:
        shape = (stop - start,)
        source_slice = source[start:stop]
        target_slice: numpy.memmap = numpy.memmap(
            target.filename,
            dtype=target.dtype,
            offset=(start * target.dtype.itemsize),
            shape=shape,
        )
        target_slice[fields] = source_slice[fields]
        start = stop


def copy_mmap_record_arrays(
    source: numpy.memmap,
    target: numpy.memmap,
    fields: list[str],
    chunksize: int = 100000,
) -> None:
    """
    Writing into a memmapped array allocates memory equivalent to the
    amount that you are writing. With big arrays this is undesirable
    so we write in chunks
    """
    assert source.filename is not None
    assert target.filename is not None

    start = 0
    stops = itertools.chain(range(chunksize, source.size, chunksize), [source.size])
    for stop in stops:
        shape = (stop - start,)
        source_slice: numpy.memmap = numpy.memmap(
            source.filename,
            dtype=source.dtype,
            offset=(start * source.dtype.itemsize),
            shape=shape,
        )
        target_slice: numpy.memmap = numpy.memmap(
            target.filename,
            dtype=target.dtype,
            offset=(start * target.dtype.itemsize),
            shape=shape,
        )
        target_slice[fields] = source_slice[fields]

        start = stop
