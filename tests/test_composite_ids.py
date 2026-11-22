"""End-to-end tests for composite (source, original_id) record identities."""
import unittest

import numpy

import dedupe
from dedupe.predicate_functions import wholeFieldPredicate
from dedupe.predicates import SimplePredicate


class MockClassifier:
    """High match probability for pairs whose name distance is small."""

    def predict_proba(self, features):
        probability = numpy.where(features[:, 0] <= 0.5, 0.95, 0.01)
        return numpy.column_stack([1 - probability, probability])


def _matcher(cls):
    predicate = SimplePredicate(wholeFieldPredicate, "name")
    matcher = cls([dedupe.variables.String("name")])
    matcher.predicates = (predicate,)
    matcher._fingerprinter = dedupe.blocking.Fingerprinter([predicate])
    matcher.classifier = MockClassifier()
    return matcher


# Two sources reuse local integer ids; a third uses string ids. Equal bare
# ids across sources are distinct records.
DATA_A = {
    ("A", 1): {"name": "Acme"},
    ("A", 2): {"name": "Globex"},
    ("A", 3): {"name": "Initech"},
}
DATA_B = {
    ("B", 1): {"name": "Acme"},
    ("B", 2): {"name": "Stark"},
    ("B", 3): {"name": "Initech"},
}
DATA_C = {
    ("C", "1"): {"name": "Wayne"},
    ("C", "2"): {"name": "Globex"},
}


class CompositeDedupeTest(unittest.TestCase):
    def setUp(self):
        self.data = {**DATA_A, **DATA_B, **DATA_C}
        self.matcher = _matcher(dedupe.Dedupe)

    def test_pairs_restore_original_ids_and_no_self_edges(self):
        pairs = list(self.matcher.pairs(self.data))
        for (a_id, a_rec), (b_id, b_rec) in pairs:
            self.assertIsInstance(a_id, tuple)
            self.assertIsInstance(b_id, tuple)
            self.assertNotEqual(a_id, b_id)
            self.assertIs(a_rec, self.data[a_id])
            self.assertIs(b_rec, self.data[b_id])

    def test_clusters_lossless_and_distinct(self):
        clusters = list(self.matcher.partition(self.data, threshold=0.5))
        emitted = {rid for ids, _ in clusters for rid in ids}

        # every record appears, including singletons, as its original tuple
        self.assertEqual(emitted, set(self.data.keys()))

        def cluster_of(record_id):
            for ids, _ in clusters:
                if record_id in ids:
                    return set(ids)
            return set()

        # matching records across sources cluster despite identical bare id
        self.assertEqual(cluster_of(("A", 1)), {("A", 1), ("B", 1)})
        self.assertEqual(cluster_of(("A", 3)), {("A", 3), ("B", 3)})
        # source C's bare "1" never merges with the integer-id 1 records
        self.assertNotIn(("C", "1"), cluster_of(("A", 1)))
        # integer id 2 matches string id "2" while staying distinct
        self.assertEqual(cluster_of(("A", 2)), {("A", 2), ("C", "2")})

    def test_score_array_uses_text_and_decodes(self):
        scores = self.matcher.score(iter(self.matcher.pairs(self.data)))
        self.assertEqual(scores.dtype["pairs"].base.kind, "U")
        for row in scores:
            for cell in row["pairs"]:
                self.assertTrue(str(cell).startswith("t:"))
        # clustering restores tuples
        decoded = list(self.matcher.cluster(scores, 0.5))
        self.assertTrue(
            all(isinstance(rid, tuple) for ids, _ in decoded for rid in ids)
        )


class CompositeRecordLinkTest(unittest.TestCase):
    def setUp(self):
        self.matcher = _matcher(dedupe.RecordLink)

    def test_links_decode_ids(self):
        for constraint in ("one-to-one", "many-to-one", "many-to-many"):
            links = self.matcher.join(
                DATA_A, DATA_B, threshold=0.5, constraint=constraint
            )
            self.assertTrue(links)
            for (a, b), _ in links:
                self.assertIsInstance(a, tuple)
                self.assertIsInstance(b, tuple)
                self.assertIn(a, DATA_A)
                self.assertIn(b, DATA_B)

    def test_self_join_has_no_self_edges(self):
        data = {
            ("A", 1): {"name": "Acme"},
            ("A", 2): {"name": "Acme"},
            ("A", 3): {"name": "Globex"},
        }
        links = self.matcher.join(data, data, threshold=0.0)
        for (a, b), _ in links:
            self.assertNotEqual(a, b)
        # distinct records sharing a value are still compared
        self.assertTrue(
            any({a, b} == {("A", 1), ("A", 2)} for (a, b), _ in links)
        )


class CompositeGazetteerTest(unittest.TestCase):
    def setUp(self):
        self.matcher = _matcher(dedupe.Gazetteer)
        self.matcher.index(DATA_B)

    def test_search_and_unindex(self):
        results = self.matcher.search(DATA_A, threshold=0.5)
        for messy_id, matches in results:
            self.assertIsInstance(messy_id, tuple)
            self.assertIn(messy_id, DATA_A)
            for canon_id, _ in matches:
                self.assertIsInstance(canon_id, tuple)
                self.assertIn(canon_id, DATA_B)

        # an unindexed record no longer matches
        self.matcher.unindex({("B", 1): DATA_B[("B", 1)]})
        after = self.matcher.search(
            {("A", 1): DATA_A[("A", 1)]}, threshold=0.5
        )
        for _, matches in after:
            self.assertNotIn(("B", 1), [canon for canon, _ in matches])

    def test_unmatched_record_returns_empty(self):
        results = self.matcher.search(
            {("A", 99): {"name": "TotallyUniqueName"}}, threshold=0.5
        )
        self.assertEqual(list(results), [(("A", 99), ())])


class BackwardCompatibilityTest(unittest.TestCase):
    def test_plain_int_ids(self):
        data = {1: {"name": "Acme"}, 2: {"name": "Acme"}, 3: {"name": "Globex"}}
        matcher = _matcher(dedupe.Dedupe)
        scores = matcher.score(iter(matcher.pairs(data)))
        self.assertEqual(scores.dtype["pairs"].base.kind, "i")
        clusters = list(matcher.partition(data, 0.5))
        self.assertEqual(
            {int(rid) for ids, _ in clusters for rid in ids}, {1, 2, 3}
        )

    def test_plain_str_ids(self):
        data = {"1": {"name": "Acme"}, "2": {"name": "Acme"}, "3": {"name": "Globex"}}
        matcher = _matcher(dedupe.Dedupe)
        clusters = list(matcher.partition(data, 0.5))
        self.assertEqual(
            {str(rid) for ids, _ in clusters for rid in ids}, {"1", "2", "3"}
        )

    def test_mixed_int_and_str_stay_distinct(self):
        data = {1: {"name": "Acme"}, "1": {"name": "Acme"}, 2: {"name": "Stark"}}
        matcher = _matcher(dedupe.Dedupe)
        pairs = list(matcher.pairs(data))
        # the int 1 and the str "1" form one cross-type pair, not a collision
        self.assertEqual(
            [{p[0][0], p[1][0]} for p in pairs], [{1, "1"}]
        )
        for (a_id, _), (b_id, _) in pairs:
            self.assertIn(type(a_id), (int, str))
            self.assertIn(type(b_id), (int, str))


if __name__ == "__main__":
    unittest.main()
