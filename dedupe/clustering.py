#!/usr/bin/python
from __future__ import annotations

import array
import heapq
import itertools
import logging
import tempfile
from collections import defaultdict
from typing import Generator, Iterable, Mapping, Sequence, cast

import numpy
import numpy.typing
import scipy.cluster.hierarchy

from dedupe._typing import ArrayLinks, Clusters, RecordID, Scores, TupleLinks

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
    dupes: Scores, threshold: float = 0.5, max_components: int = 30000
) -> Clusters:
    """
    Takes in a list of duplicate pairs and clusters them in to a
    list records that all refer to the same entity based on a given
    threshold

    Keyword arguments:
    threshold -- number between 0 and 1 (default is .5). lowering the
                 number will increase precision, raising it will increase
                 recall
    """
    distance_threshold = 1 - threshold
    dupe_sub_graphs = connected_components(dupes, max_components)

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

            squared_distances = condensed_distances**2
            for cluster in clusters.values():
                if len(cluster) > 1:
                    scores = confidences(cluster, squared_distances, N)
                    yield tuple(i_to_id[i] for i in cluster), scores  # type: ignore[misc]

        else:
            ((ids, score),) = sub_graph
            if score > threshold:
                yield tuple(ids), (score,) * 2


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


# Factor used to turn the (0, 1) similarity scores into exact integer
# costs, so the matching never depends on floating point comparisons.
_SCORE_SCALE = 1_000_000


def _edge_sort_key(edge: tuple[RecordID, RecordID, float]) -> tuple:
    # Highest score first, then the lowest source id, then the lowest
    # target id. Using this order everywhere makes every tie-breaking
    # decision (equal scores, equal costs, equal distances) deterministic.
    return (-edge[2], edge[0], edge[1])


def _min_cost_flow(
    head: list[int],
    residual: list[int],
    cost: list[int],
    adjacency: list[list[int]],
    source_node: int,
    sink_node: int,
    max_flow: int,
    stop_on_nonnegative: bool = False,
) -> int:
    """Send up to ``max_flow`` units of minimum-cost flow over the
    residual network described by the paired-edge arrays, mutating
    ``residual`` in place.

    Edges are stored in pairs: edge ``e`` runs from a node to
    ``head[e]`` with residual capacity ``residual[e]`` and ``cost[e]``,
    and its reverse residual edge is always ``e ^ 1``.
    ``adjacency[u]`` lists the edges leaving node ``u``. Successive
    shortest augmenting paths are found with Dijkstra; the original
    costs are non-negative, so Johnson potentials keep every reduced
    cost non-negative.

    When ``stop_on_nonnegative`` is set, augmentation stops as soon as
    the next path would no longer lower the total cost. This is what
    allows a record to stay unlinked: the maximum-weight matching is not,
    in general, the maximum-cardinality matching.

    Returns the amount of flow actually sent.
    """
    n_nodes = len(adjacency)
    potential = [0] * n_nodes
    flow = 0

    while flow < max_flow:
        distances: list[int | None] = [None] * n_nodes
        previous_edge = [-1] * n_nodes
        distances[source_node] = 0
        queue: list[tuple[int, int]] = [(0, source_node)]

        while queue:
            dist, u = heapq.heappop(queue)
            if dist != distances[u]:
                continue
            for edge in adjacency[u]:
                if residual[edge] == 0:
                    continue
                v = head[edge]
                new_dist = dist + cost[edge] + potential[u] - potential[v]
                current = distances[v]
                if current is None or new_dist < current:
                    distances[v] = new_dist
                    previous_edge[v] = edge
                    heapq.heappush(queue, (new_dist, v))

        sink_distance = distances[sink_node]
        if sink_distance is None:
            break

        for node, node_distance in enumerate(distances):
            if node_distance is not None:
                potential[node] += node_distance

        # The sink's potential now holds the true (un-reduced) cost of
        # this augmenting path. A non-negative path would not improve
        # the total score, so leaving the remaining records unlinked is
        # the better choice.
        if stop_on_nonnegative and potential[sink_node] >= 0:
            break

        # Each augmenting path carries a single unit of flow.
        node = sink_node
        while node != source_node:
            edge = previous_edge[node]
            residual[edge] -= 1
            residual[edge ^ 1] += 1
            node = head[edge ^ 1]

        flow += 1

    return flow


def capacitatedMatching(
    dupes: Scores,
    capacity: Mapping[RecordID, int] | int = 1,
    threshold: float = 0.0,
) -> TupleLinks:
    """Find the links that maximize the total similarity score of the
    matched pairs, subject to the capacities of the target records.

    Unlike :func:`greedyMatching`, which commits to edges one at a time
    in descending score order, this function solves the matching over
    all the candidate edges at once. The result is therefore a global
    optimum rather than the outcome of a sequence of local, possibly
    regrettable, decisions.

    Every source record (the first id of a pair) can be linked to at
    most one target record. A target record (the second id of a pair)
    can be linked to as many source records as its capacity allows. If
    ``capacity`` is an integer, every target has that capacity. If it is
    a mapping, each target id present in the mapping uses the capacity
    given there and every other target defaults to a capacity of one.

    Only edges whose score is strictly greater than ``threshold`` are
    eligible, so records with no eligible edge are simply left unlinked.

    The problem is a maximum-weight capacitated bipartite matching,
    solved as a minimum-cost flow on the network

        source -> source records -> target records -> sink

    where an edge's cost is the negative of its score. The optimization
    runs in two phases: the first fixes the maximum achievable total
    score, and the second chooses among only those maximum-score
    matchings the deterministic one that, in order, minimizes the sum of
    the linked source ranks and then the sum of the linked target ranks,
    where ranks are assigned by sorting the record ids. Keeping the
    phases separate guarantees the tie-breaker can never trade away any
    score.

    Args:
        dupes: a numpy structured array with a dtype of
               `[('pairs', id_type, 2), ('score', 'f4')]` holding the
               candidate scored pairs.

        capacity: either a positive integer applied to every target, or
                  a mapping from target record ids to positive integer
                  capacities.

        threshold: Number between 0 and 1. Edges with a predicted
                   probability at or below the threshold are not
                   eligible for the matching.

    Yields:
        Tuples of ``((source_id, target_id), score)`` for every selected
        edge, in the deterministic order described above.
    """
    if isinstance(capacity, int):
        if capacity < 1:
            raise ValueError("capacity must be a positive integer")
        uniform_capacity = capacity
        target_capacity: dict[RecordID, int] = {}
    else:
        uniform_capacity = 1
        target_capacity = dict(capacity)
        for value in target_capacity.values():
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    "Every target capacity must be a positive integer"
                )

    if threshold:
        dupes = dupes[dupes["score"] > threshold]

    if len(dupes) == 0:
        return

    # A pair can be compared more than once upstream; keep the best edge
    # for each (source, target) combination.
    edges: dict[tuple[RecordID, RecordID], float] = {}
    for (a, b), score in dupes:
        key = (a, b)
        previous = edges.get(key)
        if previous is None or score > previous:
            edges[key] = float(score)

    if not edges:
        return

    ordered_edges = sorted(
        ((a, b, score) for (a, b), score in edges.items()), key=_edge_sort_key
    )

    # Ranks follow the sorted record ids, so the tie-breaker always
    # prefers lower ids regardless of the order edges happen to arrive.
    source_rank = {a: i for i, a in enumerate(sorted({a for a, _, _ in ordered_edges}))}
    target_rank = {b: i for i, b in enumerate(sorted({b for _, b, _ in ordered_edges}))}

    n_sources = len(source_rank)
    n_targets = len(target_rank)

    source_node = n_sources + n_targets
    sink_node = source_node + 1
    n_nodes = sink_node + 1
    max_flow = min(
        n_sources,
        sum(target_capacity.get(b, uniform_capacity) for b in target_rank),
    )

    # Lexicographic packing constants. A matching has at most n_sources
    # edges. The per-edge target digit is kept below `target_base`, so the
    # total target contribution stays below `tie_base` and can never carry
    # into the source-rank digit. Likewise the whole tie-break part stays
    # below `score_mult`, so it can never outweigh a single unit of score.
    # The optimized cost therefore orders matchings by total score, then
    # by the sum of source ranks, then by the sum of target ranks, each
    # compared lexicographically.
    target_base = n_targets + 1
    tie_base = n_sources * target_base + 1
    score_mult = n_sources * (n_sources - 1) * tie_base + 1

    def build_graph(score_weight: int, tie_weight: int) -> tuple[
        list[int],
        list[int],
        list[int],
        list[list[int]],
        dict[tuple[RecordID, RecordID], int],
    ]:
        # Residual network as paired-edge arrays. Edge e's reverse is
        # e ^ 1; adjacency[u] holds the edges leaving node u.
        head: list[int] = []
        residual: list[int] = []
        cost: list[int] = []
        adjacency: list[list[int]] = [[] for _ in range(n_nodes)]

        def add_edge(u: int, v: int, edge_capacity: int, edge_cost: int) -> int:
            forward = len(head)
            head.append(v)
            residual.append(edge_capacity)
            cost.append(edge_cost)
            adjacency[u].append(forward)
            head.append(u)
            residual.append(0)
            cost.append(-edge_cost)
            adjacency[v].append(forward ^ 1)
            return forward

        # The source connects to every source record with capacity one,
        # encoding the rule that a record can be linked at most once.
        for index in source_rank.values():
            add_edge(source_node, index, 1, 0)

        # Every target connects to the sink with its configured capacity.
        for b, index in target_rank.items():
            cap = target_capacity.get(b, uniform_capacity)
            add_edge(n_sources + index, sink_node, cap, 0)

        edge_index: dict[tuple[RecordID, RecordID], int] = {}
        for a, b, score in ordered_edges:
            u = source_rank[a]
            target_index = target_rank[b]
            edge_cost = score_weight * (
                -int(round(score * _SCORE_SCALE)) * score_mult
            ) + tie_weight * (u * tie_base + target_index)
            edge_index[(a, b)] = add_edge(
                u, n_sources + target_index, 1, edge_cost
            )

        return head, residual, cost, adjacency, edge_index

    # Phase 1: maximize the total score using score-only costs. Records
    # may stay unlinked, so we stop augmenting once another link would no
    # longer improve the total score rather than forcing full capacity.
    # This pins down both the optimal number of links and the maximum
    # achievable score.
    head, residual, cost, adjacency, edge_index = build_graph(1, 0)
    flow = _min_cost_flow(
        head,
        residual,
        cost,
        adjacency,
        source_node,
        sink_node,
        max_flow,
        stop_on_nonnegative=True,
    )

    if flow == 0:
        return

    # Phase 2: re-solve for exactly that many links with the lexicographic
    # tie-break switched on. The score digit dominates every possible
    # tie-break contribution, so the maximum score is held fixed while the
    # deterministic rank ordering selects among equal-score alternatives.
    head, residual, cost, adjacency, edge_index = build_graph(1, 1)
    _min_cost_flow(
        head, residual, cost, adjacency, source_node, sink_node, flow
    )

    # An edge is selected when its residual capacity has been consumed.
    for a, b, score in ordered_edges:
        edge = edge_index[(a, b)]
        if residual[edge] == 0:
            yield cast(TupleLinks, ((a, b), score))  # type: ignore[misc]


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
