import itertools
import unittest

import numpy

import dedupe
import dedupe.variables

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

DATA_SAMPLE = (
    ({"age": "27", "name": "Kyle"}, {"age": "50", "name": "Bob"}),
    ({"age": "27", "name": "Kyle"}, {"age": "35", "name": "William"}),
    ({"age": "10", "name": "Sue"}, {"age": "35", "name": "William"}),
    ({"age": "27", "name": "Kyle"}, {"age": "20", "name": "Jimmy"}),
    ({"age": "75", "name": "Charlie"}, {"age": "21", "name": "Jimbo"}),
)


class DataModelTest(unittest.TestCase):
    def test_data_model(self):
        DataModel = dedupe.datamodel.DataModel

        self.assertRaises(TypeError, DataModel)

        data_model = DataModel(
            [
                dedupe.variables.String(field="a", name="a"),
                dedupe.variables.String(field="b", name="b"),
                dedupe.variables.Interaction("a", "b"),
            ]
        )

        assert data_model._interaction_indices == [[0, 1]]

        data_model = DataModel(
            [
                dedupe.variables.String(field="a", name="a", has_missing=True),
                dedupe.variables.String(field="b", name="b"),
                dedupe.variables.Interaction("a", "b"),
            ]
        )

        assert data_model._missing_field_indices == [0, 2]

        data_model = DataModel(
            [
                dedupe.variables.String(field="a", name="a", has_missing=False),
                dedupe.variables.String(field="b", name="b"),
                dedupe.variables.Interaction("a", "b"),
            ]
        )

        assert data_model._missing_field_indices == []


class ConnectedComponentsTest(unittest.TestCase):
    def test_components(self):
        G = numpy.array(
            [
                ((1, 2), 0.1),
                ((2, 3), 0.2),
                ((4, 5), 0.2),
                ((4, 6), 0.2),
                ((7, 9), 0.2),
                ((8, 9), 0.2),
                ((10, 11), 0.2),
                ((12, 13), 0.2),
                ((12, 14), 0.5),
                ((11, 12), 0.2),
            ],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )
        components = dedupe.clustering.connected_components
        G_components = {
            frozenset(tuple(edge) for edge, _ in component)
            for component in components(G, 30000)
        }
        assert G_components == {
            frozenset(((1, 2), (2, 3))),
            frozenset(((4, 5), (4, 6))),
            frozenset(((12, 13), (12, 14), (10, 11), (11, 12))),
            frozenset(((7, 9), (8, 9))),
        }


class ClusteringTest(unittest.TestCase):
    def setUp(self):
        # Fully connected star network
        self.dupes = numpy.array(
            [
                ((1, 2), 0.86),
                ((1, 3), 0.72),
                ((1, 4), 0.2),
                ((1, 5), 0.6),
                ((2, 3), 0.86),
                ((2, 4), 0.2),
                ((2, 5), 0.72),
                ((3, 4), 0.3),
                ((3, 5), 0.5),
                ((4, 5), 0.72),
                ((10, 11), 0.9),
            ],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )

        # Dupes with Ids as String
        self.str_dupes = numpy.array(
            [
                (("1", "2"), 0.86),
                (("1", "3"), 0.72),
                (("1", "4"), 0.2),
                (("1", "5"), 0.6),
                (("2", "3"), 0.86),
                (("2", "4"), 0.2),
                (("2", "5"), 0.72),
                (("3", "4"), 0.3),
                (("3", "5"), 0.5),
                (("4", "5"), 0.72),
            ],
            dtype=[("pairs", "S4", 2), ("score", "f4")],
        )

        self.bipartite_dupes = (
            ((1, 5), 0.1),
            ((1, 6), 0.72),
            ((1, 7), 0.2),
            ((1, 8), 0.6),
            ((2, 5), 0.2),
            ((2, 6), 0.2),
            ((2, 7), 0.72),
            ((2, 8), 0.3),
            ((3, 5), 0.24),
            ((3, 6), 0.72),
            ((3, 7), 0.24),
            ((3, 8), 0.65),
            ((4, 5), 0.63),
            ((4, 6), 0.96),
            ((4, 7), 0.23),
            ((5, 8), 0.24),
        )

    def clusterEquals(self, x, y):
        if [] == x == y:
            return True
        if len(x) != len(y):
            return False

        for cluster_a, cluster_b in zip(x, y):
            if cluster_a[0] != cluster_b[0]:
                return False
            for score_a, score_b in zip(cluster_a[1], cluster_b[1]):
                if abs(score_a - score_b) > 0.001:
                    return False
            else:
                return True

    def test_hierarchical(self):
        hierarchical = dedupe.clustering.cluster
        assert self.clusterEquals(list(hierarchical(self.dupes, 1)), [])

        assert self.clusterEquals(
            list(hierarchical(self.dupes, 0.5)),
            [
                ((1, 2, 3), (0.778, 0.860, 0.778)),
                ((4, 5), (0.720, 0.720)),
                ((10, 11), (0.899, 0.899)),
            ],
        )

        print(hierarchical(self.dupes, 0.0))
        assert self.clusterEquals(
            list(hierarchical(self.dupes, 0)),
            [
                ((1, 2, 3, 4, 5), (0.526, 0.564, 0.542, 0.320, 0.623)),
                ((10, 11), (0.899, 0.899)),
            ],
        )

        assert list(hierarchical(self.str_dupes, 1)) == []
        assert list(zip(*hierarchical(self.str_dupes, 0.5)))[0] == (
            (b"1", b"2", b"3"),
            (b"4", b"5"),
        )
        assert list(zip(*hierarchical(self.str_dupes, 0)))[0] == (
            (b"1", b"2", b"3", b"4", b"5"),
        )

    def test_greedy_matching(self):
        greedyMatch = dedupe.clustering.greedyMatching

        bipartite_dupes = numpy.array(
            list(self.bipartite_dupes), dtype=[("ids", int, 2), ("score", float)]
        )

        assert list(greedyMatch(bipartite_dupes)) == [
            ((4, 6), 0.96),
            ((2, 7), 0.72),
            ((3, 8), 0.65),
            ((1, 5), 0.1),
        ]

    def test_gazette_matching(self):
        gazetteMatch = dedupe.clustering.gazetteMatching
        blocked_dupes = itertools.groupby(self.bipartite_dupes, key=lambda x: x[0][0])

        def to_numpy(x):
            return numpy.array(x, dtype=[("ids", int, 2), ("score", float)])

        blocked_dupes = [to_numpy(list(block)) for _, block in blocked_dupes]

        target = [
            (((1, 6), 0.72), ((1, 8), 0.6)),
            (((2, 7), 0.72), ((2, 8), 0.3)),
            (((3, 6), 0.72), ((3, 8), 0.65)),
            (((4, 6), 0.96), ((4, 5), 0.63)),
            (((5, 8), 0.24),),
        ]

        assert [
            tuple((tuple(pair), score) for pair, score in each.tolist())
            for each in gazetteMatch(blocked_dupes, n_matches=2)
        ] == target


class CannotLinkClusteringTest(unittest.TestCase):
    def setUp(self):
        # star network where 1, 2, 3 all belong together at threshold .5,
        # 4 and 5 are a separate pair, and 10 and 11 are another pair.
        self.dupes = numpy.array(
            [
                ((1, 2), 0.86),
                ((1, 3), 0.72),
                ((1, 4), 0.2),
                ((1, 5), 0.6),
                ((2, 3), 0.86),
                ((2, 4), 0.2),
                ((2, 5), 0.72),
                ((3, 4), 0.3),
                ((3, 5), 0.5),
                ((4, 5), 0.72),
                ((10, 11), 0.9),
            ],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )

    def assert_constraint_satisfied(self, clusters, cannot_link):
        for record_ids, _ in clusters:
            for a, b in cannot_link:
                together = a in record_ids and b in record_ids
                assert not together, "cannot_link pair %r together in %r" % (
                    (a, b),
                    record_ids,
                )

    def test_no_constraint_matches_normal_clustering(self):
        unconstrained = list(dedupe.clustering.cluster(self.dupes.copy(), 0.5))
        constrained = list(
            dedupe.clustering.cluster(self.dupes.copy(), 0.5, cannot_link=())
        )
        assert [c[0] for c in unconstrained] == [c[0] for c in constrained]

    def test_direct_forbidden_pair_is_split(self):
        cannot_link = [(1, 2)]
        clusters = list(
            dedupe.clustering.cluster(self.dupes.copy(), 0.5, cannot_link=cannot_link)
        )
        self.assert_constraint_satisfied(clusters, cannot_link)

        record_ids = {c[0] for c in clusters}
        assert (2, 3) in record_ids
        # unrelated clusters are untouched
        assert (4, 5) in record_ids
        assert (10, 11) in record_ids

    def test_constraint_is_transitive(self):
        # A full triangle: every pair is compared and all three records
        # would normally land in one cluster. If an implementation only
        # deleted the (1, 3) edge and then ran ordinary clustering, 1 and
        # 3 would still be merged through 2. The constraint has to bind
        # on the assignment and keep them apart.
        dupes = numpy.array(
            [((1, 2), 0.9), ((2, 3), 0.9), ((1, 3), 0.9)],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )
        unconstrained = list(dedupe.clustering.cluster(dupes.copy(), 0.5))
        assert [c[0] for c in unconstrained] == [(1, 2, 3)]

        cannot_link = [(1, 3)]
        clusters = list(
            dedupe.clustering.cluster(dupes.copy(), 0.5, cannot_link=cannot_link)
        )
        self.assert_constraint_satisfied(clusters, cannot_link)
        # the merge is not simply dropped: an allowed pair still forms
        assert {c[0] for c in clusters} & {(1, 2), (2, 3)}

    def test_forbidden_pair_still_blocks_bridge(self):
        # 1-3 and 3-2 are strong; 1-2 is weak. Without a constraint 1 and
        # 2 could still end together through 3, so merely dropping the
        # weak (1, 2) edge is not a valid implementation.
        dupes = numpy.array(
            [((1, 3), 0.95), ((2, 3), 0.95), ((1, 2), 0.2)],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )
        cannot_link = [(1, 2)]
        clusters = list(
            dedupe.clustering.cluster(dupes.copy(), 0.5, cannot_link=cannot_link)
        )
        self.assert_constraint_satisfied(clusters, cannot_link)

    def test_ordinary_paths_are_preserved(self):
        # 1-2 and 2-3 are both strong. Forbidding only (1, 3) should keep
        # 1 and 2 (or 2 and 3) together rather than dissolving everything.
        dupes = numpy.array(
            [((1, 2), 0.9), ((2, 3), 0.9), ((1, 3), 0.85)],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )
        cannot_link = [(1, 3)]
        clusters = list(
            dedupe.clustering.cluster(dupes.copy(), 0.5, cannot_link=cannot_link)
        )
        self.assert_constraint_satisfied(clusters, cannot_link)
        clustered = {frozenset(record_ids) for record_ids, _ in clusters}
        assert clustered & {frozenset((1, 2)), frozenset((2, 3))}

    def test_string_ids(self):
        dupes = numpy.array(
            [
                (("1", "2"), 0.86),
                (("1", "3"), 0.72),
                (("2", "3"), 0.86),
            ],
            dtype=[("pairs", "S4", 2), ("score", "f4")],
        )
        cannot_link = [("1", "2")]
        clusters = list(
            dedupe.clustering.cluster(dupes.copy(), 0.5, cannot_link=cannot_link)
        )
        for record_ids, _ in clusters:
            assert not (b"1" in record_ids and b"2" in record_ids)
        assert (b"2", b"3") in {c[0] for c in clusters}

    def test_singleton_component_constraint(self):
        dupes = numpy.array(
            [((1, 2), 0.9)],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )
        assert list(dedupe.clustering.cluster(dupes.copy(), 0.5)) == [
            ((1, 2), (dupes["score"][0],) * 2)
        ]
        assert (
            list(dedupe.clustering.cluster(dupes.copy(), 0.5, cannot_link=[(1, 2)]))
            == []
        )

    def test_unknown_ids_are_ignored(self):
        cannot_link = [(1, 999)]
        clusters = list(
            dedupe.clustering.cluster(self.dupes.copy(), 0.5, cannot_link=cannot_link)
        )
        assert (1, 2, 3) in {c[0] for c in clusters}

    def test_self_pair_raises(self):
        with self.assertRaises(ValueError):
            list(
                dedupe.clustering.cluster(self.dupes.copy(), 0.5, cannot_link=[(1, 1)])
            )

    def test_malformed_pair_raises(self):
        with self.assertRaises(ValueError):
            list(
                dedupe.clustering.cluster(
                    self.dupes.copy(), 0.5, cannot_link=[(1, 2, 3)]
                )
            )

    def test_reversed_and_duplicate_pairs_are_collapsed(self):
        cannot_link = [(2, 1), (1, 2), (1, 2)]
        clusters = list(
            dedupe.clustering.cluster(self.dupes.copy(), 0.5, cannot_link=cannot_link)
        )
        self.assert_constraint_satisfied(clusters, [(1, 2)])

    def test_threshold_is_not_lowered(self):
        # raising the threshold to break up (1, 2) would also break up the
        # independent (10, 11) pair. The constrained result must keep it.
        clusters = list(
            dedupe.clustering.cluster(self.dupes.copy(), 0.5, cannot_link=[(1, 2)])
        )
        assert (10, 11) in {c[0] for c in clusters}


if __name__ == "__main__":
    unittest.main()
