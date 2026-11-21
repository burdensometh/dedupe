import unittest

import dedupe
import dedupe.branch_and_bound as branch_and_bound
import dedupe.training as training


def group_predicate(mapping, name):
    def func(column):
        if column in mapping:
            return frozenset((mapping[column],))
        return frozenset()

    func.__name__ = name
    return dedupe.predicates.SimplePredicate(func, "name")


class BudgetedLearningTest(unittest.TestCase):
    def setUp(self):
        self.data = {
            0: {"name": "a"},
            1: {"name": "a"},
            2: {"name": "b"},
            3: {"name": "b"},
            4: {"name": "c"},
            5: {"name": "c"},
        }
        # pairs: (0, 1), (2, 3); covers the first match only
        self.p1 = group_predicate({"a": "g1", "b": "g2"}, "p1")
        # pairs: (2, 3), (4, 5); covers the second match only
        self.p2 = group_predicate({"b": "h1", "c": "h2"}, "p2")
        # a high frequency key: pairs (0,1), (0,2), (0,3), (1,2), (1,3),
        # (2,3), (4,5); covers both matches
        self.p_hf = group_predicate({"a": "x", "b": "x", "c": "y"}, "p_hf")

        self.matches = [
            (self.data[0], self.data[1]),
            (self.data[4], self.data[5]),
        ]

    def learner(self, predicates):
        return training.DedupeBlockLearner(predicates, self.data, self.data)

    def learn(self, predicates, budget):
        learner = self.learner(predicates)
        return learner.learn(
            self.matches,
            recall=1.0,
            index_predicates=False,
            candidate_types="simple",
            candidate_budget=budget,
        )

    def test_overlapping_pairs_counted_once(self):
        # p1 and p2 both generate the pair (2, 3). Their combined cost is
        # 3 unique pairs, not 4, so a budget of 3 must be affordable.
        selected = self.learn([self.p1, self.p2], budget=3)
        assert set(selected) == {self.p1, self.p2}

    def test_budget_search_returns_unique_pair_count(self):
        pairs = {
            self.p1: frozenset({(0, 1), (2, 3)}),
            self.p2: frozenset({(2, 3), (4, 5)}),
        }
        cover = {self.p1: frozenset({0}), self.p2: frozenset({1})}
        self.p1.cover_count = 2
        self.p2.cover_count = 2

        selected, n_pairs = branch_and_bound.search_budgeted(
            cover, pairs, target=2, budget=3, calls=100
        )
        assert set(selected) == {self.p1, self.p2}
        assert n_pairs == 3

        assert (
            branch_and_bound.search_budgeted(
                cover, pairs, target=2, budget=2, calls=100
            )
            is None
        )

    def test_expensive_rules_excluded(self):
        # the high frequency key alone blows the budget, so the affordable
        # combination of cheaper rules must be selected instead
        selected = self.learn([self.p1, self.p2, self.p_hf], budget=3)
        assert set(selected) == {self.p1, self.p2}

    def test_conflict_when_budget_infeasible(self):
        # the only rule that covers both matches generates too many pairs
        with self.assertRaises(dedupe.BudgetConflictError):
            self.learn([self.p_hf], budget=3)

        # each rule is affordable alone, but together they exceed the
        # budget and neither covers both matches
        with self.assertRaises(dedupe.BudgetConflictError):
            self.learn([self.p1, self.p2], budget=2)

    def test_no_budget_keeps_original_behavior(self):
        learner = self.learner([self.p1, self.p2, self.p_hf])
        selected = learner.learn(
            self.matches,
            recall=1.0,
            index_predicates=False,
            candidate_types="simple",
        )
        assert set(selected) == {self.p1, self.p2}


class TrainingTest(unittest.TestCase):
    def setUp(self):
        field_definition = [dedupe.variables.String("name")]
        self.data_model = dedupe.Dedupe(field_definition).data_model
        self.training_pairs = {
            "match": [
                ({"name": "Bob", "age": "50"}, {"name": "Bob", "age": "75"}),
                ({"name": "Meredith", "age": "40"}, {"name": "Sue", "age": "10"}),
            ],
            "distinct": [
                ({"name": "Jimmy", "age": "20"}, {"name": "Jimbo", "age": "21"}),
                ({"name": "Willy", "age": "35"}, {"name": "William", "age": "35"}),
                ({"name": "William", "age": "36"}, {"name": "William", "age": "35"}),
            ],
        }

        self.training = self.training_pairs["match"] + self.training_pairs["distinct"]
        self.training_records = []
        for pair in self.training:
            for record in pair:
                if record not in self.training_records:
                    self.training_records.append(record)

        self.simple = lambda x: {str(k) for k in x if "CompoundPredicate" not in str(k)}

        self.block_learner = training.BlockLearner
        self.block_learner.blocker = dedupe.blocking.Fingerprinter(
            self.data_model.predicates
        )
        self.block_learner.blocker.index_all(
            {i: x for i, x in enumerate(self.training_records)}
        )

    def test_dedupe_coverage(self):
        coverage = self.block_learner.cover(self.block_learner, self.training)
        assert self.simple(coverage.keys()).issuperset(
            {
                "SimplePredicate: (tokenFieldPredicate, name)",
                "SimplePredicate: (commonSixGram, name)",
                "TfidfTextCanopyPredicate: (0.4, name)",
                "SimplePredicate: (sortedAcronym, name)",
                "SimplePredicate: (sameThreeCharStartPredicate, name)",
                "TfidfTextCanopyPredicate: (0.2, name)",
                "SimplePredicate: (sameFiveCharStartPredicate, name)",
                "TfidfTextCanopyPredicate: (0.6, name)",
                "SimplePredicate: (wholeFieldPredicate, name)",
                "TfidfTextCanopyPredicate: (0.8, name)",
                "SimplePredicate: (commonFourGram, name)",
                "SimplePredicate: (firstTokenPredicate, name)",
                "SimplePredicate: (sameSevenCharStartPredicate, name)",
            }
        )

    def test_uncovered_by(self):
        before = {1: frozenset({1, 2, 3}), 2: frozenset({1, 2}), 3: frozenset({3})}
        after = {1: frozenset({1, 2}), 2: frozenset({1, 2})}

        before_copy = before.copy()

        assert branch_and_bound._uncovered_by(before, frozenset()) == before
        assert branch_and_bound._uncovered_by(before, frozenset({3})) == after
        assert before == before_copy

    def test_covered_pairs(self):
        p1 = lambda x, target=None: frozenset((1,))  # noqa: E 731

        self.block_learner.blocker.predicates = (p1,)
        cover = self.block_learner.cover(self.block_learner, [("a", "b")] * 2)

        assert cover[p1] == {0, 1}


if __name__ == "__main__":
    unittest.main()
