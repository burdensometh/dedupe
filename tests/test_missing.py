import pickle
import unittest

import numpy

import dedupe
import dedupe.datamodel
import dedupe.missing
from dedupe.missing import MissingState


class MissingStateTest(unittest.TestCase):
    def test_classifies_the_three_cases(self):
        self.assertIs(
            dedupe.missing.value_state(dedupe.missing.ABSENT),
            MissingState.NOT_PROVIDED,
        )
        self.assertIs(dedupe.missing.value_state(None), MissingState.EMPTY)
        self.assertIs(dedupe.missing.value_state(""), MissingState.EMPTY)
        self.assertIs(dedupe.missing.value_state([]), MissingState.EMPTY)
        self.assertIs(dedupe.missing.value_state(set()), MissingState.EMPTY)
        self.assertIs(dedupe.missing.value_state("Main St"), MissingState.COMPARABLE)

    def test_numeric_zero_is_a_value_not_a_blank(self):
        self.assertIs(dedupe.missing.value_state(0), MissingState.COMPARABLE)
        self.assertIs(dedupe.missing.value_state(0.0), MissingState.COMPARABLE)

    def test_record_state_treats_omitted_key_as_not_provided(self):
        self.assertIs(
            dedupe.missing.record_state({}, "address"), MissingState.NOT_PROVIDED
        )
        self.assertIs(
            dedupe.missing.record_state({"address": None}, "address"),
            MissingState.EMPTY,
        )
        self.assertIs(
            dedupe.missing.record_state({"address": ""}, "address"),
            MissingState.EMPTY,
        )
        self.assertIs(
            dedupe.missing.record_state({"address": "x"}, "address"),
            MissingState.COMPARABLE,
        )

    def test_pair_features_are_one_hot_and_order_independent(self):
        comparable_empty = dedupe.missing.pair_features(
            MissingState.COMPARABLE, MissingState.EMPTY
        )
        empty_comparable = dedupe.missing.pair_features(
            MissingState.EMPTY, MissingState.COMPARABLE
        )
        self.assertEqual(comparable_empty, empty_comparable)
        self.assertEqual(sum(comparable_empty), 1)
        self.assertEqual(len(comparable_empty), dedupe.missing.PAIR_FEATURE_WIDTH)

    def test_distinct_pairs_land_in_distinct_slots(self):
        states = list(MissingState)
        seen = set()
        for state_1 in states:
            for state_2 in states:
                features = dedupe.missing.pair_features(state_1, state_2)
                seen.add(tuple(features))
        self.assertEqual(len(seen), 6)


class DataModelMissingContractTest(unittest.TestCase):
    def _model(self, *variables):
        return dedupe.datamodel.DataModel(list(variables))

    def _row(self, model, record_1, record_2):
        return model.distances([(record_1, record_2)])[0]

    def test_absent_empty_and_comparable_get_different_features(self):
        model = self._model(
            dedupe.variables.String(field="address", name="address", has_missing=True),
            dedupe.variables.Exact(field="city", name="city"),
        )

        absent = self._row(model, {"city": "X"}, {"city": "X"})
        empty = self._row(model, {"address": "", "city": "X"}, {"city": "X"})
        comparable = self._row(
            model, {"address": "A", "city": "X"}, {"address": "A", "city": "X"}
        )

        self.assertFalse(numpy.array_equal(absent, empty))
        self.assertFalse(numpy.array_equal(absent, comparable))
        self.assertFalse(numpy.array_equal(empty, comparable))

    def test_comparable_text_distance_zero_is_not_confused_with_missing(self):
        model = self._model(
            dedupe.variables.String(field="address", name="address", has_missing=True),
            dedupe.variables.Exact(field="city", name="city"),
        )

        identical = self._row(
            model,
            {"address": "Main St", "city": "X"},
            {"address": "Main St", "city": "X"},
        )
        absent = self._row(model, {"city": "X"}, {"city": "X"})
        empty = self._row(
            model, {"address": "", "city": "X"}, {"address": "", "city": "X"}
        )

        for other in (absent, empty):
            self.assertFalse(numpy.array_equal(identical, other))

    def test_symmetry(self):
        model = self._model(
            dedupe.variables.String(field="address", name="address", has_missing=True),
            dedupe.variables.Exact(field="city", name="city"),
        )
        left = self._row(model, {"city": "X"}, {"address": "", "city": "X"})
        right = self._row(model, {"address": "", "city": "X"}, {"city": "X"})
        self.assertTrue(numpy.array_equal(left, right))

    def test_interaction_uses_same_contract(self):
        model = self._model(
            dedupe.variables.String(field="address", name="address", has_missing=True),
            dedupe.variables.String(field="name", name="name"),
            dedupe.variables.Interaction("address", "name"),
        )

        identical = self._row(
            model,
            {"address": "Main St", "name": "n"},
            {"address": "Main St", "name": "n"},
        )
        absent = self._row(model, {"name": "n"}, {"name": "n"})
        empty = self._row(
            model, {"address": "", "name": "n"}, {"address": "", "name": "n"}
        )
        empty_vs_present = self._row(
            model, {"address": "", "name": "n"}, {"address": "Main St", "name": "n"}
        )

        for other in (absent, empty, empty_vs_present):
            self.assertFalse(numpy.array_equal(identical, other))

    def test_absent_key_is_legitimate_for_declared_missing_field(self):
        model = self._model(
            dedupe.variables.String(field="address", name="address", has_missing=True),
            dedupe.variables.Exact(field="city", name="city"),
        )
        model.check({"address": "x", "city": "c"})
        model.check({"address": "", "city": "c"})
        model.check({"city": "c"})

    def test_undeclared_missing_key_still_raises(self):
        model = self._model(dedupe.variables.Exact(field="city", name="city"))
        with self.assertRaises(ValueError):
            model.check({"street": "x"})

    def test_feature_width_matches_declared_missing_units(self):
        base = self._model(
            dedupe.variables.String(field="address", name="address", has_missing=True),
            dedupe.variables.Exact(field="city", name="city"),
        )
        width = dedupe.missing.PAIR_FEATURE_WIDTH
        # one comparison column for address, one for city, plus one
        # comparison-feature block for the declared-missing address field.
        self.assertEqual(len(base), 2 + width)
        self.assertEqual(base._missing_field_indices, [0])


class DataModelConsistencyTest(unittest.TestCase):
    def _variables(self):
        return [
            dedupe.variables.String(field="address", name="address", has_missing=True),
            dedupe.variables.Exact(field="city", name="city"),
            dedupe.variables.Interaction("address", "city"),
        ]

    def _pairs(self):
        return [
            ({"address": "Main St", "city": "X"}, {"address": "Main St", "city": "X"}),
            ({"city": "X"}, {"city": "X"}),
            ({"address": "", "city": "X"}, {"address": "", "city": "X"}),
            ({"address": "", "city": "X"}, {"address": "Main St", "city": "X"}),
            ({"city": "X"}, {"address": "Main St", "city": "X"}),
            ({"address": "Other", "city": "X"}, {"address": "Main St", "city": "Y"}),
        ]

    def test_pickle_round_trip_preserves_order_and_values(self):
        model = dedupe.datamodel.DataModel(self._variables())
        before = model.distances(self._pairs())

        reloaded = pickle.loads(pickle.dumps(model))
        after = reloaded.distances(self._pairs())

        self.assertEqual(before.shape, after.shape)
        self.assertTrue(numpy.array_equal(before, after))
        self.assertFalse(reloaded._legacy_missing_contract)

    def test_row_order_does_not_change_features(self):
        model = dedupe.datamodel.DataModel(self._variables())
        pairs = self._pairs()

        forward = model.distances(pairs)
        backward = model.distances(list(reversed(pairs)))

        self.assertTrue(numpy.array_equal(forward, backward[::-1]))

    def test_legacy_model_keeps_original_contract_and_dimensions(self):
        model = dedupe.datamodel.DataModel(self._variables())

        # Reconstruct the pickled state that a pre-contract (version 2) model
        # would have written: flat "both present" indicators, narrower width.
        legacy_state = {
            "field_variables": model.field_variables,
            "_derived_start": 2,
            "_missing_field_indices": [0, 2],
            "_interaction_indices": [[0, 1]],
            "_len": 5,
            "object_version": 2,
        }
        legacy = dedupe.datamodel.DataModel.__new__(dedupe.datamodel.DataModel)
        legacy.__setstate__(legacy_state)

        self.assertTrue(legacy._legacy_missing_contract)
        self.assertEqual(len(legacy), 5)

        pairs = [
            ({"address": "A", "city": "X"}, {"address": "A", "city": "X"}),
            ({"address": None, "city": "X"}, {"address": None, "city": "X"}),
            ({"address": "A", "city": "X"}, {"address": None, "city": "X"}),
        ]
        features = legacy.distances(pairs)
        self.assertEqual(features.shape, (3, 5))

        # The legacy contract still treats None as the missing marker.
        numpy.testing.assert_array_equal(features[0, -2:], [1, 1])
        numpy.testing.assert_array_equal(features[1, -2:], [0, 0])


if __name__ == "__main__":
    unittest.main()
