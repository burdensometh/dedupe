import datetime
import unittest

import dedupe
import dedupe.canonical
from dedupe.canonical import FieldRule, getCanonicalRep


class CanonicalizationTest(unittest.TestCase):
    def test_get_centroid(self):
        from affinegap import normalizedAffineGapDistance as comparator

        attributeList = [
            "mary crane center",
            "mary crane center north",
            "mary crane league - mary crane - west",
            "mary crane league mary crane center (east)",
            "mary crane league mary crane center (north)",
            "mary crane league mary crane center (west)",
            "mary crane league - mary crane - east",
            "mary crane family and day care center",
            "mary crane west",
            "mary crane center east",
            "mary crane league mary crane center (east)",
            "mary crane league mary crane center (north)",
            "mary crane league mary crane center (west)",
            "mary crane league",
            "mary crane",
            "mary crane east 0-3",
            "mary crane north",
            "mary crane north 0-3",
            "mary crane league - mary crane - west",
            "mary crane league - mary crane - north",
            "mary crane league - mary crane - east",
            "mary crane league - mary crane - west",
            "mary crane league - mary crane - north",
            "mary crane league - mary crane - east",
        ]

        centroid = dedupe.canonical.getCentroid(attributeList, comparator)
        assert centroid == "mary crane"

    def test_get_canonical_rep(self):
        record_list = [
            {"name": "mary crane", "address": "123 main st", "zip": "12345"},
            {"name": "mary crane east", "address": "123 main street", "zip": ""},
            {"name": "mary crane west", "address": "123 man st", "zip": ""},
        ]

        rep = dedupe.canonical.getCanonicalRep(record_list)
        assert rep == {
            "name": "mary crane",
            "address": "123 main street",
            "zip": "12345",
        }

        rep = dedupe.canonical.getCanonicalRep(record_list[0:2])
        assert rep == {"name": "mary crane", "address": "123 main st", "zip": "12345"}

        rep = dedupe.canonical.getCanonicalRep(record_list[0:1])
        assert rep == {"name": "mary crane", "address": "123 main st", "zip": "12345"}

    def test_centroid_tie_is_order_independent(self):
        # a two-record cluster is always a centroid tie; the canonical
        # value must not depend on which record comes first
        a = {"name": "aaa", "address": "zzz"}
        b = {"name": "bbb", "address": "mmm"}

        rep_ab = getCanonicalRep([a, b])
        rep_ba = getCanonicalRep([b, a])

        assert rep_ab == rep_ba
        assert rep_ab["name"] == "aaa"
        assert rep_ab["address"] == "mmm"

    def test_reordering_does_not_change_result(self):
        records = [
            {
                "id": "r1",
                "name": "acme corp",
                "source": "state",
                "effective": "2020-01-01",
            },
            {
                "id": "r2",
                "name": "acme corporation",
                "source": "web",
                "effective": "2024-05-01",
            },
            {
                "id": "r3",
                "name": "acme",
                "source": "state",
                "effective": "2023-12-31",
            },
        ]
        rules = [
            FieldRule(
                "name",
                strategy="authority",
                source_field="source",
                source_priority={"state": 2, "web": 1},
                date_field="effective",
            ),
            FieldRule("effective", strategy="latest", date_field="effective"),
        ]

        rep_forward = getCanonicalRep(records, rules, id_field="id")
        rep_reversed = getCanonicalRep(list(reversed(records)), rules, id_field="id")

        assert dict(rep_forward) == dict(rep_reversed)
        assert rep_forward.provenance == rep_reversed.provenance

    def test_authority_rule_prefers_trusted_source(self):
        records = [
            {
                "id": "r1",
                "name": "acme web",
                "source": "web",
                "effective": "2024-01-01",
            },
            {
                "id": "r2",
                "name": "acme official",
                "source": "state",
                "effective": "2010-01-01",
            },
        ]
        rule = FieldRule(
            "name",
            strategy="authority",
            source_field="source",
            source_priority={"state": 10, "web": 1},
            date_field="effective",
        )

        rep = getCanonicalRep(records, [rule], id_field="id")

        # authority beats both recency and string similarity
        assert rep["name"] == "acme official"
        assert rep.provenance["name"] == "r2"

    def test_authority_tie_breaks_on_latest_then_value(self):
        records = [
            {
                "id": "r1",
                "name": "beta name",
                "source": "state",
                "effective": "2020-01-01",
            },
            {
                "id": "r2",
                "name": "alpha name",
                "source": "state",
                "effective": "2020-01-01",
            },
            {
                "id": "r3",
                "name": "gamma name",
                "source": "state",
                "effective": "2022-01-01",
            },
        ]
        rule = FieldRule(
            "name",
            strategy="authority",
            source_field="source",
            source_priority={"state": 1},
            date_field="effective",
        )

        rep = getCanonicalRep(records, [rule], id_field="id")
        # same source -> most recent effective date wins
        assert rep["name"] == "gamma name"
        assert rep.provenance["name"] == "r3"

        records[2]["effective"] = "2020-01-01"
        rep = getCanonicalRep(records, [rule], id_field="id")
        # fully tied -> deterministic lexicographic choice
        assert rep["name"] == "alpha name"
        assert rep.provenance["name"] == "r2"

    def test_unlisted_source_ranks_below_listed_sources(self):
        records = [
            {"id": "r1", "name": "scraped", "source": "random-site"},
            {"id": "r2", "name": "registered", "source": "state"},
        ]
        rule = FieldRule(
            "name",
            strategy="authority",
            source_field="source",
            source_priority={"state": 1},
        )

        rep = getCanonicalRep(records, [rule], id_field="id")
        assert rep["name"] == "registered"
        assert rep.provenance["name"] == "r2"

    def test_latest_rule_uses_effective_date(self):
        records = [
            {"id": "r1", "name": "old name", "effective": "2018-03-01"},
            {"id": "r2", "name": "new name", "effective": "2024-06-15"},
            {"id": "r3", "name": "middle name", "effective": "2020-09-09"},
        ]
        rule = FieldRule("name", strategy="latest", date_field="effective")

        rep = getCanonicalRep(records, [rule], id_field="id")
        assert rep["name"] == "new name"
        assert rep.provenance["name"] == "r2"

    def test_latest_accepts_date_objects(self):
        records = [
            {
                "id": "r1",
                "name": "old name",
                "effective": datetime.date(2018, 3, 1),
            },
            {
                "id": "r2",
                "name": "new name",
                "effective": datetime.datetime(2024, 6, 15, 12, 0),
            },
        ]
        rule = FieldRule("name", strategy="latest", date_field="effective")

        rep = getCanonicalRep(records, [rule], id_field="id")
        assert rep["name"] == "new name"

    def test_latest_tie_is_deterministic(self):
        records = [
            {"id": "r1", "name": "zeta", "effective": "2024-01-01"},
            {"id": "r2", "name": "alpha", "effective": "2024-01-01"},
        ]
        rule = FieldRule("name", strategy="latest", date_field="effective")

        rep = getCanonicalRep(records, [rule], id_field="id")
        assert rep["name"] == "alpha"

    def test_non_empty_values_always_preferred(self):
        records = [
            {"id": "r1", "name": "", "source": "state"},
            {"id": "r2", "name": "only real value", "source": "web"},
        ]
        rule = FieldRule(
            "name",
            strategy="authority",
            source_field="source",
            source_priority={"state": 99},
        )

        rep = getCanonicalRep(records, [rule], id_field="id")
        assert rep["name"] == "only real value"
        assert rep.provenance["name"] == "r2"

    def test_all_missing_uses_missing_marker(self):
        records = [
            {"id": "r1", "name": "acme", "zip": ""},
            {"id": "r2", "name": "acme co", "zip": None},
        ]

        rep = getCanonicalRep(records, id_field="id")
        assert rep["zip"] == ""
        assert rep.provenance["zip"] is None

        rep = getCanonicalRep(records, id_field="id", missing_marker=None)
        assert rep["zip"] is None

    def test_authority_without_sources_uses_conflict_marker(self):
        records = [
            {"id": "r1", "name": "one"},
            {"id": "r2", "name": "two"},
        ]
        rule = FieldRule(
            "name",
            strategy="authority",
            source_field="source",
            source_priority={"state": 1},
        )

        rep = getCanonicalRep(records, [rule], id_field="id")
        # defaults to the missing marker
        assert rep["name"] == ""
        assert rep.provenance["name"] is None

        rep = getCanonicalRep(
            records, [rule], id_field="id", conflict_marker="UNRESOLVED"
        )
        assert rep["name"] == "UNRESOLVED"

    def test_latest_without_dates_uses_conflict_marker(self):
        records = [
            {"id": "r1", "name": "one", "effective": ""},
            {"id": "r2", "name": "two", "effective": "not a date"},
        ]
        rule = FieldRule("name", strategy="latest", date_field="effective")

        rep = getCanonicalRep(
            records, [rule], id_field="id", conflict_marker="UNRESOLVED"
        )
        assert rep["name"] == "UNRESOLVED"
        assert rep.provenance["name"] is None

    def test_provenance_uses_positions_without_id_field(self):
        records = [
            {"name": "aaa"},
            {"name": "bbb"},
        ]

        rep = getCanonicalRep(records)
        assert rep.provenance["name"] == 0

    def test_rules_accepted_as_plain_mappings(self):
        records = [
            {"id": "r1", "name": "web", "source": "web"},
            {"id": "r2", "name": "official", "source": "state"},
        ]
        rules = [
            {
                "field": "name",
                "strategy": "authority",
                "source_field": "source",
                "source_priority": {"state": 1, "web": 0},
            }
        ]

        rep = getCanonicalRep(records, rules, id_field="id")
        assert rep["name"] == "official"
        assert rep.provenance["name"] == "r2"

        # field-name keyed mapping form as well
        rep = getCanonicalRep(records, {"name": rules[0]}, id_field="id")
        assert rep["name"] == "official"

    def test_invalid_rule_arguments_raise(self):
        with self.assertRaises(ValueError):
            FieldRule("name", strategy="bogus")

        with self.assertRaises(ValueError):
            FieldRule("name", strategy="authority")

    def test_empty_cluster_raises(self):
        with self.assertRaises(ValueError):
            getCanonicalRep([])

    def test_canonical_record_is_plain_dict_for_gazetteer_index(self):
        records = [
            {"id": "r1", "name": "acme", "address": "1 main st", "source": "web"},
            {
                "id": "r2",
                "name": "acme inc",
                "address": "1 main street",
                "source": "state",
            },
        ]
        rule = FieldRule(
            "name",
            strategy="authority",
            source_field="source",
            source_priority={"state": 1},
        )

        rep = getCanonicalRep(records, [rule], id_field="id")

        # the selected master is a real dict: serializable and indexable
        assert isinstance(rep, dict)
        plain = dict(rep)
        assert plain == {
            "name": "acme inc",
            "address": "1 main st",
            "id": "r1",
            "source": "state",
        }
        # provenance rides alongside without polluting the record fields
        assert set(rep.keys()) == set(plain.keys())
        assert "provenance" not in rep
        assert rep.provenance == {
            "name": "r2",
            "address": "r1",
            "id": "r1",
            "source": "r2",
        }

        # and the actually selected field value is what a new query hits
        indexed = {"canonical-1": plain}
        assert indexed["canonical-1"]["name"] == "acme inc"

    def test_top_level_canonicalize_helper(self):
        records = [
            {"id": "r1", "name": "web name", "source": "web"},
            {"id": "r2", "name": "state name", "source": "state"},
        ]
        rep = dedupe.canonicalize(
            records,
            field_rules=[
                FieldRule(
                    "name",
                    strategy="authority",
                    source_field="source",
                    source_priority={"state": 1},
                )
            ],
            id_field="id",
        )
        assert rep["name"] == "state name"
        assert rep.provenance["name"] == "r2"

    def test_fields_only_present_on_some_members_are_merged(self):
        records = [
            {"id": "r1", "name": "acme"},
            {"id": "r2", "name": "acme", "phone": "555-1234"},
        ]

        rep = getCanonicalRep(records, id_field="id")
        assert rep["phone"] == "555-1234"
        assert rep.provenance["phone"] == "r2"

    def test_non_string_values_keep_their_type(self):
        records = [
            {"id": "r1", "price": 100, "as_of": "2020-01-01"},
            {"id": "r2", "price": 250, "as_of": "2024-01-01"},
        ]

        rep = getCanonicalRep(records, id_field="id")
        assert rep["price"] == 100
        assert isinstance(rep["price"], int)

        rule = FieldRule("price", strategy="latest", date_field="as_of")
        rep = getCanonicalRep(records, [rule], id_field="id")
        assert rep["price"] == 250
        assert isinstance(rep["price"], int)


if __name__ == "__main__":
    unittest.main()
