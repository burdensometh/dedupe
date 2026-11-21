import io
import pickle
import unittest
from collections import OrderedDict

import dedupe
from dedupe.adapter import (
    ORIGINAL_RECORD_KEY,
    AdaptationError,
    FieldSpec,
    SourceAdapter,
    original_record,
)


def make_link_adapters(**zip_kwargs):
    defaults = dict(type="str", pad=5)
    defaults.update(zip_kwargs)
    adapter_1 = SourceAdapter(
        name="source_a",
        fields=(
            FieldSpec("zip", "Zip", **defaults),
            FieldSpec("street", "Street", type="str", case="upper"),
            FieldSpec("name", "Name", type="str", case="upper"),
        ),
    )
    adapter_2 = SourceAdapter(
        name="source_b",
        fields=(
            FieldSpec("zipcode", "Zip", **defaults),
            FieldSpec("address", "Street", type="str", case="upper"),
            FieldSpec("name", "Name", type="str", case="upper"),
        ),
    )
    return adapter_1, adapter_2


VARIABLES = [
    dedupe.variables.String("Zip", has_missing=True),
    dedupe.variables.String("Street", has_missing=True),
    dedupe.variables.String("Name"),
]


class FieldSpecTest(unittest.TestCase):
    def test_leading_zeros_are_normalized_from_both_sides(self):
        spec = FieldSpec("zip", "Zip", type="str", pad=5)
        # integer on one side (zeros lost), zero-padded string on other
        self.assertEqual(spec.convert(2138), "02138")
        self.assertEqual(spec.convert("02138"), "02138")
        self.assertEqual(spec.convert("2138"), "02138")

    def test_pad_leaves_longer_values_untouched(self):
        spec = FieldSpec("zip", "Zip", type="str", pad=5)
        self.assertEqual(spec.convert("123456"), "123456")

    def test_case_semantics_are_single_and_explicit(self):
        self.assertEqual(
            FieldSpec("s", "S", case="upper").convert("main St"), "MAIN ST"
        )
        self.assertEqual(
            FieldSpec("s", "S", case="lower").convert("Main St"), "main st"
        )
        self.assertEqual(FieldSpec("s", "S").convert("Main St"), "Main St")

    def test_null_values_become_the_single_none_marker(self):
        spec = FieldSpec("s", "S", null_values=frozenset({None, "", "NA"}))
        for value in (None, "", "NA"):
            self.assertIsNone(spec.convert(value))
        # default nulls
        default_spec = FieldSpec("s", "S")
        self.assertIsNone(default_spec.convert(None))
        self.assertIsNone(default_spec.convert(""))
        self.assertEqual(default_spec.convert("NA"), "NA")

    def test_integer_type_rejects_numeric_strings(self):
        spec = FieldSpec("n", "N", type="int")
        self.assertEqual(spec.convert(5), 5)
        with self.assertRaises(AdaptationError):
            spec.convert("5")
        with self.assertRaises(AdaptationError):
            spec.convert(True)

    def test_float_type(self):
        spec = FieldSpec("p", "P", type="float")
        self.assertEqual(spec.convert(1.5), 1.5)
        self.assertEqual(spec.convert(2), 2)
        with self.assertRaises(AdaptationError):
            spec.convert("1.5")
        with self.assertRaises(AdaptationError):
            spec.convert(False)

    def test_string_type_rejects_floats_and_bools(self):
        spec = FieldSpec("s", "S")
        with self.assertRaises(AdaptationError):
            spec.convert(1.5)
        with self.assertRaises(AdaptationError):
            spec.convert(True)

    def test_invalid_construction_options(self):
        with self.assertRaises(AdaptationError):
            FieldSpec("s", "S", type="number")  # type: ignore[arg-type]
        with self.assertRaises(AdaptationError):
            FieldSpec("s", "S", case="title")  # type: ignore[arg-type]
        with self.assertRaises(AdaptationError):
            FieldSpec("n", "N", type="int", case="upper")
        with self.assertRaises(AdaptationError):
            FieldSpec("n", "N", type="int", pad=3)
        with self.assertRaises(AdaptationError):
            FieldSpec("s", "S", pad=0)
        with self.assertRaises(AdaptationError):
            FieldSpec("s", "S", null_values={None})  # type: ignore[arg-type]
        with self.assertRaises(AdaptationError):
            FieldSpec("", "S")
        with self.assertRaises(AdaptationError):
            FieldSpec("s", "")


class SourceAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter, _ = make_link_adapters()

    def test_transform_maps_onto_common_fields_and_keeps_original(self):
        raw = {"zip": 2138, "street": "main st", "name": "acme"}
        adapted = self.adapter.transform(raw)
        self.assertEqual(
            adapted,
            {
                "Zip": "02138",
                "Street": "MAIN ST",
                "Name": "ACME",
                ORIGINAL_RECORD_KEY: dict(raw),
            },
        )
        self.assertEqual(original_record(adapted), raw)
        # the raw input was not mutated
        self.assertEqual(raw["zip"], 2138)

    def test_transform_is_idempotent(self):
        raw = {"zip": 2138, "street": "main st", "name": "acme"}
        once = self.adapter.transform(raw)
        twice = self.adapter.transform(once)
        self.assertEqual(once, twice)
        self.assertEqual(original_record(twice), raw)

    def test_missing_source_field_is_reported_not_guessed(self):
        with self.assertRaises(AdaptationError):
            self.adapter.transform({"zip": 2138, "name": "acme"})

    def test_duplicate_source_or_target_fields_rejected(self):
        with self.assertRaises(AdaptationError):
            SourceAdapter(
                fields=(
                    FieldSpec("a", "X"),
                    FieldSpec("a", "Y"),
                )
            )
        with self.assertRaises(AdaptationError):
            SourceAdapter(
                fields=(
                    FieldSpec("a", "X"),
                    FieldSpec("b", "X"),
                )
            )

    def test_empty_adapter_rejected(self):
        with self.assertRaises(AdaptationError):
            SourceAdapter(fields=())  # type: ignore[arg-type]

    def test_original_record_helper_rejects_plain_records(self):
        with self.assertRaises(AdaptationError):
            original_record({"name": "no metadata"})


class AdapterValidationTest(unittest.TestCase):
    def test_adapter_must_cover_every_model_field(self):
        adapter = SourceAdapter(
            fields=(
                FieldSpec("zip", "Zip"),
                FieldSpec("street", "Street"),
            )
        )
        with self.assertRaises(AdaptationError):
            dedupe.RecordLink(VARIABLES, adapters=(adapter, adapter))

    def test_unknown_target_field_rejected(self):
        adapter = SourceAdapter(
            fields=(
                FieldSpec("zip", "Zip"),
                FieldSpec("street", "Street"),
                FieldSpec("name", "Name"),
                FieldSpec("extra", "Mystery"),
            )
        )
        with self.assertRaises(AdaptationError):
            dedupe.RecordLink(VARIABLES, adapters=(adapter, adapter))

    def test_conflicting_semantics_across_sources_rejected(self):
        a, b = make_link_adapters()
        b = SourceAdapter(
            name="source_b",
            fields=(
                FieldSpec("zipcode", "Zip", type="str", pad=9),
                FieldSpec("address", "Street", type="str", case="upper"),
                FieldSpec("name", "Name", type="str", case="upper"),
            ),
        )
        with self.assertRaises(AdaptationError):
            dedupe.RecordLink(VARIABLES, adapters=(a, b))

    def test_conflicting_case_across_sources_rejected(self):
        a = SourceAdapter(fields=(FieldSpec("n", "Name", case="upper"),))
        b = SourceAdapter(fields=(FieldSpec("n", "Name", case="lower"),))
        with self.assertRaises(AdaptationError):
            dedupe.RecordLink([dedupe.variables.String("Name")], adapters=(a, b))

    def test_string_variable_requires_string_type(self):
        adapter = SourceAdapter(
            fields=(
                FieldSpec("zip", "Zip", type="int"),
                FieldSpec("street", "Street"),
                FieldSpec("name", "Name"),
            )
        )
        with self.assertRaises(AdaptationError):
            dedupe.RecordLink(VARIABLES, adapters=(adapter, adapter))

    def test_price_variable_requires_numeric_type(self):
        variables = [
            dedupe.variables.Price("cost"),
            dedupe.variables.String("name"),
        ]
        adapter = SourceAdapter(
            fields=(
                FieldSpec("c", "cost", type="str"),
                FieldSpec("n", "name"),
            )
        )
        with self.assertRaises(AdaptationError):
            dedupe.Dedupe(variables, adapters=adapter)

        numeric = SourceAdapter(
            fields=(
                FieldSpec("c", "cost", type="float"),
                FieldSpec("n", "name"),
            )
        )
        matcher = dedupe.Dedupe(variables, adapters=numeric)
        out = matcher._adapt_data({1: {"c": 9.99, "n": "x"}})
        self.assertEqual(out[1]["cost"], 9.99)

    def test_unsupported_variable_types_reported(self):
        variables = [dedupe.variables.LatLong("coord"), dedupe.variables.String("name")]
        adapter = SourceAdapter(
            fields=(
                FieldSpec("c", "coord"),
                FieldSpec("n", "name"),
            )
        )
        with self.assertRaises(AdaptationError):
            dedupe.Dedupe(variables, adapters=adapter)

    def test_adapter_count_must_match_matcher(self):
        adapter, _ = make_link_adapters()
        linker = dedupe.RecordLink(VARIABLES, adapters=(adapter, adapter))
        # a record-link matcher with a single adapter is invalid
        linker.adapters = (adapter,)
        with self.assertRaises(AdaptationError):
            linker._check_adapter_count(2)

        deduper = dedupe.Dedupe(
            [dedupe.variables.String("Name")],
            adapters=SourceAdapter(fields=(FieldSpec("n", "Name"),)),
        )
        # one adapter is fine for single-source dedupe
        deduper._check_adapter_count(1)
        with self.assertRaises(AdaptationError):
            deduper._check_adapter_count(2)


class RecordLinkIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.adapter_1, self.adapter_2 = make_link_adapters()
        streets = ["main st", "oak ave", "pine rd", "elm blvd", "maple dr", "cedar ln"]
        zips = [2138, 1002, 3030, 4040, 5050, 6060]
        names = ["ACME", "GLOBEX", "INITECH", "UMBRELLA", "STARK", "WAYNE"]
        self.data_1 = OrderedDict(
            (
                (
                    f"a{i}",
                    {"zip": zips[i], "street": streets[i], "name": names[i].lower()},
                )
                for i in range(6)
            )
        )
        self.data_2 = OrderedDict(
            (
                (
                    f"b{i}",
                    {
                        "zipcode": str(zips[i]).zfill(5),
                        "address": streets[i].upper(),
                        "name": names[i],
                    },
                )
                for i in range(6)
            )
        )
        self.linker = dedupe.RecordLink(
            VARIABLES, adapters=(self.adapter_1, self.adapter_2)
        )
        self.labeled = {
            "match": [(self.data_1[f"a{i}"], self.data_2[f"b{i}"]) for i in range(6)],
            "distinct": [
                (self.data_1[f"a{i}"], self.data_2[f"b{(i + 1) % 6}"]) for i in range(6)
            ],
        }

    def _trained_linker(self):
        self.linker.prepare_training(self.data_1, self.data_2)
        self.linker.mark_pairs(self.labeled)
        self.linker.train()
        return self.linker

    def test_prepare_training_adapts_both_sources(self):
        self.linker.prepare_training(self.data_1, self.data_2)
        # the active learner only ever sees common-field records
        pair = self.linker.active_learner.pairs[0]
        for record in pair:
            self.assertIn("Zip", record)
            self.assertIn(ORIGINAL_RECORD_KEY, record)
            self.assertNotIn("zipcode", record)
            self.assertNotIn("address", record)

    def test_mark_pairs_adapts_raw_labeled_records(self):
        self.linker.prepare_training(self.data_1, self.data_2)
        self.linker.mark_pairs(self.labeled)
        match = self.linker.training_pairs["match"][0]
        self.assertEqual(match[0]["Zip"], match[1]["Zip"])
        self.assertEqual(match[0]["Street"], match[1]["Street"])

    def test_blocking_uses_common_fields_and_keeps_record_ids(self):
        linker = self._trained_linker()
        pairs = list(linker.pairs(self.data_1, self.data_2))
        self.assertTrue(pairs)
        ids = {(id_a, id_b) for (id_a, _), (id_b, _) in pairs}
        self.assertIn(("a1", "b1"), ids)
        for (id_a, rec_a), (id_b, rec_b) in pairs:
            self.assertEqual(original_record(rec_a), self.data_1[id_a])
            self.assertEqual(original_record(rec_b), self.data_2[id_b])

    def test_settings_roundtrip_preserves_adapters(self):
        linker = self._trained_linker()
        buf = io.BytesIO()
        linker.write_settings(buf)
        buf.seek(0)
        static = dedupe.StaticRecordLink(buf)

        self.assertEqual(static.adapters, (self.adapter_1, self.adapter_2))

        # the static linker applies the saved rules at match time
        links = static.join(self.data_1, self.data_2, threshold=0.0)
        linked = {pair for pair, score in links}
        for i in range(6):
            self.assertIn((f"a{i}", f"b{i}"), linked)

    def test_renamed_source_field_fails_at_match_time(self):
        linker = self._trained_linker()
        broken = {
            f"b{i}": {
                "postal": self.data_2[f"b{i}"]["zipcode"],
                "address": "MAIN ST",
                "name": "ACME",
            }
            for i in range(6)
        }
        with self.assertRaises(AdaptationError):
            list(linker.pairs(self.data_1, broken))

    def test_incompatible_type_fails_at_match_time(self):
        # numeric common field fed a non-numeric string: reported, not parsed
        variables = [dedupe.variables.Price("amount")]
        numeric_adapter = SourceAdapter(
            fields=(FieldSpec("amt", "amount", type="float"),)
        )
        deduper = dedupe.Dedupe(variables, adapters=numeric_adapter)
        with self.assertRaises(AdaptationError):
            deduper._adapt_data({1: {"amt": "9.99"}})


class LegacySettingsTest(unittest.TestCase):
    def test_three_pickle_settings_file_loads_without_adapters(self):
        matcher = dedupe.Dedupe([dedupe.variables.String("name")])
        buf = io.BytesIO()
        pickle.dump(matcher.data_model, buf)
        pickle.dump(matcher.classifier, buf)
        pickle.dump(tuple(matcher.data_model.predicates), buf)
        buf.seek(0)

        static = dedupe.StaticDedupe(buf)
        self.assertIsNone(static.adapters)


if __name__ == "__main__":
    unittest.main()
