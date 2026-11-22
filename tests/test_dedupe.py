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

    def test_capacitated_matching_global_optimum(self):
        # Greedy would commit to (1, 10) at 0.9 and leave source 2 with
        # nothing. The global optimum instead takes both lower edges for
        # a larger total score.
        capacitated = dedupe.clustering.capacitatedMatching
        dupes = numpy.array(
            [((1, 10), 0.9), ((1, 11), 0.8), ((2, 10), 0.8)],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )

        result = list(capacitated(dupes, 1))
        rounded = [(pair, round(score, 6)) for pair, score in result]

        assert rounded == [((1, 11), 0.8), ((2, 10), 0.8)]
        assert sum(score for _, score in result) > 0.9

    def test_capacitated_matching_capacity(self):
        capacitated = dedupe.clustering.capacitatedMatching
        dupes = numpy.array(
            [
                ((1, 10), 0.9),
                ((2, 10), 0.8),
                ((3, 10), 0.7),
                ((1, 11), 0.6),
            ],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )

        def rounded(result):
            return [(pair, round(score, 6)) for pair, score in result]

        # Capacity two on target 10 lets two sources link to it. Giving
        # source 1 target 11 frees target 10 for sources 2 and 3, for a
        # larger total score (2.1 vs 2.0).
        assert rounded(capacitated(dupes, {10: 2, 11: 1})) == [
            ((2, 10), 0.8),
            ((3, 10), 0.7),
            ((1, 11), 0.6),
        ]

        # A uniform integer capacity applies to every target, so all
        # three sources can take the higher-scoring target 10.
        assert rounded(capacitated(dupes, 3)) == [
            ((1, 10), 0.9),
            ((2, 10), 0.8),
            ((3, 10), 0.7),
        ]

        # Targets missing from the mapping default to a capacity of one.
        assert rounded(capacitated(dupes, {10: 2})) == [
            ((2, 10), 0.8),
            ((3, 10), 0.7),
            ((1, 11), 0.6),
        ]

    def test_capacitated_matching_threshold_and_unlinked(self):
        capacitated = dedupe.clustering.capacitatedMatching
        dupes = numpy.array(
            [((1, 10), 0.9), ((2, 10), 0.4), ((3, 10), 0.3)],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )

        def rounded(result):
            return [(pair, round(score, 6)) for pair, score in result]

        # Only edges above the threshold are eligible; the remaining
        # records are left unlinked.
        assert rounded(capacitated(dupes, {10: 2}, threshold=0.5)) == [
            ((1, 10), 0.9),
        ]

        # With capacity two, one record still has no eligible edge.
        assert rounded(capacitated(dupes, {10: 2})) == [
            ((1, 10), 0.9),
            ((2, 10), 0.4),
        ]

    def test_capacitated_matching_stable_ties(self):
        capacitated = dedupe.clustering.capacitatedMatching
        dupes = numpy.array(
            [
                ((1, 10), 0.5),
                ((1, 11), 0.5),
                ((2, 10), 0.5),
                ((2, 11), 0.5),
            ],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )

        # Every matching has the same total score, so the lowest source
        # ids and then target ids win deterministically.
        assert [
            (pair, round(score, 6)) for pair, score in capacitated(dupes, 1)
        ] == [
            ((1, 10), 0.5),
            ((2, 11), 0.5),
        ]

    def test_capacitated_matching_strings_and_empty(self):
        capacitated = dedupe.clustering.capacitatedMatching

        string_dupes = numpy.array(
            [(("a", "x"), 0.9), (("a", "y"), 0.8), (("b", "x"), 0.8)],
            dtype=[("pairs", "S4", 2), ("score", "f4")],
        )
        assert [
            (tuple(pair), round(score, 6)) for pair, score in capacitated(string_dupes, 1)
        ] == [((b"a", b"y"), 0.8), ((b"b", b"x"), 0.8)]

        empty = numpy.array([], dtype=[("pairs", "i4", 2), ("score", "f4")])
        assert list(capacitated(empty, 1)) == []

    def test_capacitated_matching_invalid_capacity(self):
        capacitated = dedupe.clustering.capacitatedMatching
        dupes = numpy.array(
            [((1, 10), 0.9)], dtype=[("pairs", "i4", 2), ("score", "f4")]
        )

        with self.assertRaises(ValueError):
            list(capacitated(dupes, 0))
        with self.assertRaises(ValueError):
            list(capacitated(dupes, {10: 0}))
        with self.assertRaises(ValueError):
            list(capacitated(dupes, {10: 1.5}))

    def test_capacitated_matching_dedupes_repeated_pairs(self):
        capacitated = dedupe.clustering.capacitatedMatching
        # The same (source, target) pair can be produced more than once
        # upstream; only its best score should count.
        dupes = numpy.array(
            [
                ((1, 10), 0.4),
                ((1, 10), 0.9),
                ((1, 11), 0.8),
                ((2, 10), 0.8),
            ],
            dtype=[("pairs", "i4", 2), ("score", "f4")],
        )

        result = list(capacitated(dupes, 1))
        rounded = [(pair, round(score, 6)) for pair, score in result]

        # (1, 10) appears once in the output at its best score, and the
        # global optimum frees target 10 for source 2.
        assert rounded == [((1, 11), 0.8), ((2, 10), 0.8)]
        assert len(result) == 2


if __name__ == "__main__":
    unittest.main()
