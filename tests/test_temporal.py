import datetime as dt
import io
import unittest

import numpy

import dedupe
import dedupe.blocking as blocking
import dedupe.datamodel as datamodel
import dedupe.temporal as temporal
from dedupe.temporal import (
    TemporalReference,
    normalize_time,
    resolve_version,
)


class FakeClassifier:
    """Deterministic stand-in for a learned pairwise classifier.

    The affine-gap distance convention used by the String variable assigns
    0.5 to identical strings and larger values to more distant strings, so we
    treat anything at or near 0.5 as a confident match (0.9) and everything
    else as a low score (0.1).
    """

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        distances = numpy.asarray(X, dtype=float)[:, 0]
        scores = numpy.where(distances <= 0.51, 0.9, 0.1)
        return numpy.column_stack([1.0 - scores, scores])


def make_matcher(num_cores=0, in_memory=True):
    """Build an initialized StaticTemporalGazetteer without a settings file."""
    matcher = object.__new__(dedupe.StaticTemporalGazetteer)
    temporal.TemporalGazetteerMatching.__init__(
        matcher, num_cores=num_cores, in_memory=in_memory
    )
    variables = [dedupe.variables.String("name")]
    matcher.data_model = datamodel.DataModel(variables)
    matcher.classifier = FakeClassifier()
    matcher.predicates = matcher.data_model.predicates
    matcher._fingerprinter = blocking.Fingerprinter(matcher.predicates)
    return matcher


JAN = dt.datetime(2020, 1, 1)
JUN = dt.datetime(2020, 6, 1)
JAN21 = dt.datetime(2021, 1, 1)
JAN22 = dt.datetime(2022, 1, 1)


class NormalizeTimeTest(unittest.TestCase):
    def test_naive_datetime_passthrough(self):
        assert normalize_time(JUN) == JUN

    def test_date_becomes_midnight_datetime(self):
        assert normalize_time(dt.date(2020, 6, 1)) == JUN

    def test_iso_string(self):
        assert normalize_time("2020-06-01T00:00:00") == JUN
        assert normalize_time("2020-06-01") == JUN
        assert normalize_time("2020-06-01T00:00:00Z") == JUN

    def test_aware_datetime_converted_to_utc_naive(self):
        eastern = dt.timezone(dt.timedelta(hours=-5))
        value = dt.datetime(2020, 5, 31, 19, 0, tzinfo=eastern)
        assert normalize_time(value) == JUN
        assert normalize_time(value).tzinfo is None

    def test_timestamp(self):
        assert normalize_time(0) == dt.datetime(1970, 1, 1)

    def test_bad_type(self):
        with self.assertRaises(TypeError):
            normalize_time([2020])  # type: ignore[arg-type]


class ResolveVersionTest(unittest.TestCase):
    def setUp(self):
        self.v1 = TemporalReference(
            record={"name": "Old Name"}, valid_from=JAN, valid_to=JAN21
        ).normalized()
        self.v2 = TemporalReference(
            record={"name": "New Name"}, valid_from=JAN21, valid_to=None
        ).normalized()

    def test_picks_version_in_effect_at_event_time(self):
        ids = ["v1", "v2"]
        chosen = resolve_version([self.v1, self.v2], ids, JUN)
        assert chosen is not None and chosen[0] == "v1"

        chosen = resolve_version([self.v1, self.v2], ids, JAN21)
        assert chosen is not None and chosen[0] == "v2"

    def test_half_open_interval_excludes_boundary(self):
        # At exactly valid_to, the old version is no longer valid.
        ids = ["v1", "v2"]
        chosen = resolve_version([self.v1, self.v2], ids, JAN21)
        assert chosen is not None and chosen[0] == "v2"

    def test_no_candidate_before_first_validity(self):
        chosen = resolve_version(
            [self.v1, self.v2], ["v1", "v2"], dt.datetime(2019, 1, 1)
        )
        assert chosen is None

    def test_no_candidate_after_expiry(self):
        expired = TemporalReference(
            record={"name": "Gone"}, valid_from=JAN, valid_to=JUN
        ).normalized()
        chosen = resolve_version([expired], ["x"], JAN21)
        assert chosen is None

    def test_overlap_resolved_by_priority(self):
        low = TemporalReference(
            record={"name": "A"}, valid_from=JAN, valid_to=None, priority=0
        ).normalized()
        high = TemporalReference(
            record={"name": "B"}, valid_from=JUN, valid_to=None, priority=5
        ).normalized()
        chosen = resolve_version([low, high], ["low", "high"], JAN21)
        assert chosen is not None and chosen[0] == "high"

    def test_overlap_priority_tie_earliest_valid_from_wins(self):
        early = TemporalReference(
            record={"name": "A"}, valid_from=JAN, valid_to=None, priority=1
        ).normalized()
        later = TemporalReference(
            record={"name": "B"}, valid_from=JUN, valid_to=None, priority=1
        ).normalized()
        chosen = resolve_version([early, later], ["early", "later"], JAN21)
        assert chosen is not None and chosen[0] == "early"

    def test_overlap_full_tie_smaller_version_id_wins(self):
        a = TemporalReference(
            record={"name": "A"}, valid_from=JAN, valid_to=None, priority=1
        ).normalized()
        b = TemporalReference(
            record={"name": "B"}, valid_from=JAN, valid_to=None, priority=1
        ).normalized()
        assert resolve_version([a, b], [2, 10], JUN)[0] == 2
        assert resolve_version([a, b], ["z", "a"], JUN)[0] == "a"

    def test_invalid_interval_rejected(self):
        with self.assertRaises(ValueError):
            TemporalReference(
                record={"name": "X"}, valid_from=JAN21, valid_to=JAN
            ).normalized()


def reference_data():
    return {
        "cust-1": {
            "v-old": TemporalReference(
                record={"name": "Acme Trading Co"},
                valid_from=dt.datetime(2018, 1, 1),
                valid_to=dt.datetime(2021, 1, 1),
            ),
            "v-new": TemporalReference(
                record={"name": "Acme Global"},
                valid_from=dt.datetime(2021, 1, 1),
                valid_to=None,
            ),
        },
        "cust-2": {
            "v-only": TemporalReference(
                record={"name": "Beta Ltd"},
                valid_from=dt.datetime(2021, 6, 1),
                valid_to=dt.datetime(2022, 1, 1),
            ),
        },
    }


class TemporalGazetteerTest(unittest.TestCase):
    def setUp(self):
        self.matcher = make_matcher()
        self.matcher.index(reference_data())

    # ------------------------------------------------------------------
    # core requirement: queries use their own event time
    # ------------------------------------------------------------------
    def test_old_order_matches_old_name_not_current_name(self):
        messy = {"m1": {"name": "Acme Trading Co"}}
        results = dict(self.matcher.search(messy, dt.datetime(2020, 3, 1)))
        assert results["m1"]
        entity_id, version_id, score = results["m1"][0]
        assert entity_id == "cust-1"
        assert version_id == "v-old"
        assert score > 0.5

    def test_recent_order_matches_current_version(self):
        messy = {"m1": {"name": "Acme Global"}}
        results = dict(
            self.matcher.search(messy, dt.datetime(2021, 6, 1))
        )
        assert results["m1"]
        entity_id, version_id, _ = results["m1"][0]
        assert (entity_id, version_id) == ("cust-1", "v-new")

    def test_future_version_never_selected_even_if_name_matches(self):
        # The messy record looks exactly like the future "Acme Global",
        # but at the 2020 event time that version did not exist yet.
        messy = {"m1": {"name": "Acme Global"}}
        results = dict(self.matcher.search(messy, dt.datetime(2020, 3, 1)))
        for entity_id, version_id, _ in results["m1"]:
            assert version_id != "v-new"
            assert (entity_id, version_id) == ("cust-1", "v-old")

    def test_result_identity_includes_entity_and_version(self):
        messy = {"m7": {"name": "Acme Trading Co"}}
        results = dict(self.matcher.search(messy, JUN))
        entity_id, version_id, score = results["m7"][0]
        assert entity_id == "cust-1"
        assert version_id == "v-old"
        assert isinstance(score, float)

    def test_unmatched_when_no_version_valid_at_event_time(self):
        # cust-2 only exists from mid-2021; querying before that yields no
        # candidate and the record must be retained as an explicit unmatched.
        messy = {"m2": {"name": "Beta Ltd"}}
        results = dict(self.matcher.search(messy, JAN21))
        assert results["m2"] == ()

    def test_expired_version_is_not_a_candidate(self):
        messy = {"m2": {"name": "Beta Ltd"}}
        results = dict(self.matcher.search(messy, JAN22))
        assert results["m2"] == ()

    # ------------------------------------------------------------------
    # per-record event times
    # ------------------------------------------------------------------
    def test_per_record_event_times(self):
        messy = {
            "m1": {"name": "Acme Trading Co"},
            "m2": {"name": "Acme Global"},
        }
        event_times = {
            "m1": dt.datetime(2020, 3, 1),
            "m2": dt.datetime(2021, 6, 1),
        }
        results = dict(self.matcher.search(messy, event_times))
        assert results["m1"][0][1] == "v-old"
        assert results["m2"][0][1] == "v-new"

    def test_missing_event_time_raises(self):
        with self.assertRaises(ValueError):
            self.matcher.search({"m1": {"name": "Acme"}}, {"m2": JUN})

    def test_extra_event_time_raises(self):
        with self.assertRaises(ValueError):
            self.matcher.search(
                {"m1": {"name": "Acme"}}, {"m1": JUN, "m2": JUN}
            )

    def test_event_times_required(self):
        with self.assertRaises(ValueError):
            self.matcher.search({"m1": {"name": "Acme"}})

    def test_iso_string_event_time(self):
        messy = {"m1": {"name": "Acme Trading Co"}}
        results = dict(self.matcher.search(messy, "2020-03-01T00:00:00"))
        assert results["m1"][0][1] == "v-old"

    # ------------------------------------------------------------------
    # overlap / conflict rules
    # ------------------------------------------------------------------
    def test_overlapping_intervals_one_candidate_per_entity_by_priority(self):
        data = {
            "e": {
                "low": TemporalReference(
                    record={"name": "Same Name"},
                    valid_from=JAN,
                    valid_to=None,
                    priority=0,
                ),
                "high": TemporalReference(
                    record={"name": "Same Name"},
                    valid_from=JUN,
                    valid_to=None,
                    priority=10,
                ),
            }
        }
        matcher = make_matcher()
        matcher.index(data)
        blocks = list(matcher.blocks({"m1": {"name": "Same Name"}}, JAN21))
        assert len(blocks) == 1
        version_ids = [pair[1][0] for pair in blocks[0]]
        assert version_ids == ["high"]

    def test_each_entity_contributes_at_most_one_candidate(self):
        blocks = list(
            self.matcher.blocks(
                {"m1": {"name": "Acme Trading Co"}}, dt.datetime(2020, 3, 1)
            )
        )
        assert len(blocks) == 1
        entities = [
            self.matcher.version_entities[pair[1][0]] for pair in blocks[0]
        ]
        assert len(entities) == len(set(entities))

    # ------------------------------------------------------------------
    # threshold, n_matches, unmatched bookkeeping
    # ------------------------------------------------------------------
    def test_threshold_filters_candidates_but_keeps_unmatched_record(self):
        messy = {
            "m1": {"name": "Acme Trading Co"},
            "m2": {"name": "Nothing Matches"},
        }
        results = dict(self.matcher.search(messy, JUN, threshold=0.95))
        assert results["m1"] == ()
        assert results["m2"] == ()
        assert set(results) == {"m1", "m2"}

    def test_result_order_tracks_input_and_includes_unmatched(self):
        messy = {
            "m1": {"name": "Acme Trading Co"},
            "m2": {"name": "Zzz Qqqqqq"},
        }
        results = self.matcher.search(messy, JUN)
        assert [record_id for record_id, _ in results] == ["m1", "m2"]
        assert dict(results)["m2"] == ()

    # ------------------------------------------------------------------
    # index management
    # ------------------------------------------------------------------
    def test_duplicate_version_id_rejected(self):
        bad = {
            "cust-9": {
                "v-old": TemporalReference(
                    record={"name": "X"}, valid_from=JAN, valid_to=None
                )
            }
        }
        with self.assertRaises(ValueError):
            self.matcher.index(bad)

    def test_unindex_removes_versions(self):
        self.matcher.unindex({"cust-2": {"v-only": None}})  # type: ignore[dict-item]
        assert "v-only" not in self.matcher.versions
        assert "cust-2" not in self.matcher.entity_versions
        messy = {"m2": {"name": "Beta Ltd"}}
        assert dict(
            self.matcher.search(messy, dt.datetime(2021, 8, 1))
        ) == {"m2": ()}

    def test_unindex_unknown_entity_is_a_noop(self):
        # The entity was never indexed; the cleanup loop must skip it.
        self.matcher.unindex({"cust-unknown": {"v-x": None}})  # type: ignore[dict-item]
        assert "cust-1" in self.matcher.entity_versions

    def test_reindex_replaces_version(self):
        updated = {
            "cust-2": {
                "v-only": TemporalReference(
                    record={"name": "Beta Ltd"},
                    valid_from=dt.datetime(2019, 1, 1),
                    valid_to=None,
                )
            }
        }
        self.matcher.index(updated)
        messy = {"m2": {"name": "Beta Ltd"}}
        results = dict(self.matcher.search(messy, JAN21))
        assert results["m2"] and results["m2"][0][1] == "v-only"

    def test_mapping_style_reference_entries(self):
        data = {
            "e1": {
                "v10": {
                    "record": {"name": "Plain Dict Version"},
                    "valid_from": "2020-01-01",
                    "valid_to": None,
                    "priority": 2,
                }
            }
        }
        matcher = make_matcher()
        matcher.index(data)
        results = dict(
            matcher.search({"m1": {"name": "Plain Dict Version"}}, JUN)
        )
        assert results["m1"][0] == ("e1", "v10", results["m1"][0][2])

    def test_malformed_reference_entry(self):
        matcher = make_matcher()
        with self.assertRaises(ValueError):
            matcher.index({"e1": {"v1": {"record": {"name": "x"}}}})

    def test_entity_without_versions_rejected(self):
        with self.assertRaises(ValueError):
            self.matcher.index({"e-empty": {}})

    def test_duplicate_version_id_within_one_call_rejected(self):
        data = {
            "e1": {
                "v1": TemporalReference({"name": "A"}, JAN)
            },
            "e2": {
                "v1": TemporalReference({"name": "B"}, JAN)
            },
        }
        with self.assertRaises(ValueError):
            self.matcher.index(data)

    def test_pre_normalized_temporal_version_accepted(self):
        version = TemporalReference({"name": "Prebuilt"}, JAN).normalized()
        matcher = make_matcher()
        matcher.index({"e": {"v": version}})
        results = dict(matcher.search({"m": {"name": "Prebuilt"}}, JUN))
        assert results["m"][0][:2] == ("e", "v")

    def test_wrong_reference_type_rejected(self):
        matcher = make_matcher()
        with self.assertRaises(TypeError):
            matcher.index({"e": {"v": "not a reference"}})  # type: ignore[dict-item]

    def test_empty_index_rejected(self):
        matcher = make_matcher()
        with self.assertRaises(ValueError):
            matcher.index({})

    def test_unindex_before_index_is_a_noop(self):
        matcher = make_matcher()
        # Must not raise even though no table or versions exist yet.
        matcher.unindex({"e1": {"v1": None}})  # type: ignore[dict-item]
        assert matcher.versions == {}

    def test_empty_query_returns_empty(self):
        assert self.matcher.search({}, JUN) == []

    def test_query_event_time_before_any_entity_keeps_all_unmatched(self):
        messy = {
            "m1": {"name": "Acme Trading Co"},
            "m2": {"name": "Beta Ltd"},
        }
        results = dict(
            self.matcher.search(messy, dt.datetime(2010, 1, 1))
        )
        assert results == {"m1": (), "m2": ()}


class OnDiskTemporalGazetteerTest(unittest.TestCase):
    def test_file_backed_database(self):
        matcher = make_matcher(in_memory=False)
        try:
            matcher.index(reference_data())
            messy = {"m1": {"name": "Acme Trading Co"}}
            results = dict(matcher.search(messy, JUN))
            assert results["m1"][0][1] == "v-old"
        finally:
            matcher._close()


class EndToEndTrainingTest(unittest.TestCase):
    def test_trained_temporal_gazetteer_round_trip(self):
        variables = [dedupe.variables.String("name")]
        gaz = dedupe.TemporalGazetteer(variables, num_cores=0)

        canonical = reference_data()
        flat = {
            version_id: version.record
            for entity in canonical.values()
            for version_id, version in entity.items()
        }
        messy = {
            "m1": {"name": "Acme Trading Co"},
            "m2": {"name": "Acme Global"},
            "m3": {"name": "Unrelated Customer"},
        }

        gaz.prepare_training(messy, flat)
        match = [
            ({"name": "Acme Trading Co"}, {"name": "Acme Trading Co"}),
            ({"name": "Acme Global"}, {"name": "Acme Global"}),
        ]
        distinct = [
            ({"name": "Acme Trading Co"}, {"name": "Acme Global"}),
            ({"name": "Beta Ltd"}, {"name": "Acme Global"}),
            ({"name": "Acme Trading Co"}, {"name": "Beta Ltd"}),
            ({"name": "Acme Global"}, {"name": "Beta Ltd"}),
        ]
        # Replicate examples so the default cross-validation has enough
        # folds; labels remain perfectly separable.
        gaz.mark_pairs(
            {
                "match": match * 4,
                "distinct": distinct * 2,
            }
        )
        gaz.train(index_predicates=False)
        gaz.index(canonical)

        settings = io.BytesIO()
        gaz.write_settings(settings)
        settings.seek(0)
        static = dedupe.StaticTemporalGazetteer(
            settings, num_cores=0, in_memory=True
        )
        static.index(canonical)

        old = dict(static.search({"m1": {"name": "Acme Trading Co"}}, JUN))
        new = dict(
            static.search({"m2": {"name": "Acme Global"}}, dt.datetime(2021, 6, 1))
        )
        none = dict(
            static.search({"m3": {"name": "Unrelated Customer"}}, JUN)
        )

        assert old["m1"][0][0] == "cust-1" and old["m1"][0][1] == "v-old"
        assert new["m2"][0][1] == "v-new"
        assert none["m3"] == ()


if __name__ == "__main__":
    unittest.main()
