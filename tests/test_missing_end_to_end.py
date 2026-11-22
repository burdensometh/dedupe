import io
import unittest

import numpy
import sklearn.linear_model

import dedupe


class MissingFieldContractEndToEndTest(unittest.TestCase):
    """Training, scoring, and reloading must agree feature-for-feature."""

    def setUp(self):
        self.variables = [
            dedupe.variables.String(field="name", name="name"),
            dedupe.variables.String(
                field="address", name="address", has_missing=True
            ),
            dedupe.variables.Interaction("name", "address"),
        ]
        self.matcher = dedupe.Dedupe(self.variables, num_cores=0)

        self.match = [
            (
                {"name": "Acme Corp", "address": "1 Main St"},
                {"name": "Acme Corp", "address": "1 Main Street"},
            ),
            (
                {"name": "Beta LLC", "address": ""},
                {"name": "Beta LLC", "address": "9 Elm Ave"},
            ),
            ({"name": "Beta LLC"}, {"name": "Beta LLC", "address": "9 Elm Ave"}),
            (
                {"name": "Acme Corp", "address": "1 Main St"},
                {"name": "Acme Corp", "address": "1 Main St"},
            ),
            ({"name": "Beta LLC"}, {"name": "Beta LLC"}),
        ]
        self.distinct = [
            (
                {"name": "Acme Corp", "address": "1 Main St"},
                {"name": "Beta LLC", "address": ""},
            ),
            (
                {"name": "Gamma", "address": "3 Oak Rd"},
                {"name": "Beta LLC", "address": ""},
            ),
            (
                {"name": "Acme Corp", "address": "1 Main St"},
                {"name": "Gamma", "address": ""},
            ),
            ({"name": "Gamma", "address": "3 Oak Rd"}, {"name": "Beta LLC"}),
            ({"name": "Acme Corp"}, {"name": "Beta LLC"}),
        ]

        self.pairs = [
            ({"name": "Beta LLC", "address": ""}, {"name": "Beta LLC", "address": ""}),
            ({"name": "Beta LLC"}, {"name": "Beta LLC"}),
            ({"name": "Beta LLC"}, {"name": "Beta LLC", "address": ""}),
            (
                {"name": "Beta LLC", "address": "9 Elm Ave"},
                {"name": "Beta LLC", "address": "9 Elm Ave"},
            ),
            (
                {"name": "Acme Corp", "address": "1 Main St"},
                {"name": "Acme Corp", "address": "1 Main St"},
            ),
            ({"name": "Acme Corp", "address": "1 Main St"}, {"name": "Beta LLC"}),
        ]

    def _train(self):
        data_model = self.matcher.data_model
        examples = self.match + self.distinct
        y = numpy.array([1] * len(self.match) + [0] * len(self.distinct))
        classifier = sklearn.linear_model.LogisticRegression(max_iter=2000)
        classifier.fit(data_model.distances(examples), y)

        # Stand in for ActiveMatching.train(), which fits the classifier on the
        # same featurization, then persist the pieces write_settings needs.
        self.matcher.classifier = classifier
        self.matcher.predicates = data_model.predicates
        return classifier

    def _reload(self):
        settings = io.BytesIO()
        self.matcher.write_settings(settings)
        settings.seek(0)
        return dedupe.StaticDedupe(settings, num_cores=0)

    def test_train_and_reload_produce_identical_features(self):
        classifier = self._train()
        data_model = self.matcher.data_model

        trained_features = data_model.distances(self.pairs)
        trained_scores = classifier.predict_proba(trained_features)[:, -1]

        reloaded = self._reload()
        reloaded_features = reloaded.data_model.distances(self.pairs)
        reloaded_scores = reloaded.classifier.predict_proba(reloaded_features)[:, -1]

        self.assertEqual(trained_features.shape, reloaded_features.shape)
        self.assertTrue(numpy.array_equal(trained_features, reloaded_features))
        numpy.testing.assert_allclose(trained_scores, reloaded_scores)

    def test_absent_empty_and_present_stay_distinct(self):
        self._train()
        rows = self.matcher.data_model.distances(self.pairs)
        empty_both = rows[0]
        absent_both = rows[1]
        identical_present = rows[4]

        self.assertFalse(numpy.array_equal(empty_both, absent_both))
        self.assertFalse(numpy.array_equal(empty_both, identical_present))
        self.assertFalse(numpy.array_equal(absent_both, identical_present))


if __name__ == "__main__":
    unittest.main()
