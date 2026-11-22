import io
import json
import os
import tempfile
import unittest
import warnings
from collections import OrderedDict

import dedupe
import dedupe.session as session_io

DATA = OrderedDict(
    (
        (0, {"name": "Bob", "age": "51"}),
        (1, {"name": "Linda", "age": "50"}),
        (2, {"name": "Gene", "age": "12"}),
        (3, {"name": "Tina", "age": "15"}),
        (4, {"name": "Bob B.", "age": "51"}),
        (5, {"name": "bob belcher", "age": "51"}),
        (6, {"name": "linda ", "age": "50"}),
        (7, {"name": "BOB", "age": "51"}),
        (8, {"name": "LINDA", "age": "50"}),
        (9, {"name": "GENE", "age": "12"}),
        (10, {"name": "TINA", "age": "15"}),
        (11, {"name": "BOB B.", "age": "51"}),
    )
)

DATA_1 = OrderedDict(
    (k, v) for k, v in DATA.items() if k < 6
)
DATA_2 = OrderedDict(
    (k + 100, v) for k, v in DATA.items() if k >= 6
)


def make_dedupe():
    deduper = dedupe.Dedupe(
        [dedupe.variables.String("name"), dedupe.variables.String("age")]
    )
    deduper.classifier.cv = False
    return deduper


def make_link():
    linker = dedupe.RecordLink(
        [dedupe.variables.String("name"), dedupe.variables.String("age")]
    )
    linker.classifier.cv = False
    return linker


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.session_path = os.path.join(self.directory, "labeling.json")

    def tearDown(self):
        for name in os.listdir(self.directory):
            os.unlink(os.path.join(self.directory, name))
        os.rmdir(self.directory)

    # -- persistence --------------------------------------------------

    def test_shown_pair_is_recorded_and_restored(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)

        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]

        resumed = make_dedupe()
        resumed.prepare_training(DATA)
        assert resumed.resume_session(self.session_path) == 0

        # The pair that was on screen is offered again before anything new.
        assert resumed.uncertain_pairs(session_file=self.session_path) == [pair]

    def test_completed_labels_are_restored(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]
        deduper.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )

        resumed = make_dedupe()
        resumed.prepare_training(DATA)
        assert resumed.resume_session(self.session_path) == 1
        assert resumed.training_pairs["match"] == [pair]
        assert resumed.training_pairs["distinct"] == []

    def test_session_is_valid_json_and_document(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]
        deduper.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )

        with open(self.session_path) as handle:
            document = json.load(handle)

        assert document["type"] == session_io.SESSION_TYPE
        assert document["version"] == session_io.SESSION_VERSION
        assert document["input"]["mode"] == "dedupe"
        assert document["pending"] == []
        assert len(document["events"]) == 1
        assert document["events"][0]["label"] == "match"
        assert len(document["events"][0]["ids"]) == 2

    def test_file_object_session(self):
        handle = io.StringIO()
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=handle)[0]
        deduper.mark_pairs({"match": [pair], "distinct": []}, session_file=handle)

        handle.seek(0)
        resumed = make_dedupe()
        resumed.prepare_training(DATA)
        assert resumed.resume_session(handle) == 1
        assert resumed.training_pairs["match"] == [pair]

    # -- idempotency and traceability ---------------------------------

    def test_resubmitting_same_label_is_idempotent(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]

        deduper.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )
        deduper.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )

        assert deduper.training_pairs["match"] == [pair]
        assert deduper.training_pairs["distinct"] == []

        with open(self.session_path) as handle:
            document = json.load(handle)
        assert len(document["events"]) == 1

    def test_changing_a_label_is_traceable(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]

        deduper.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )
        deduper.mark_pairs(
            {"match": [], "distinct": [pair]}, session_file=self.session_path
        )

        assert deduper.training_pairs["match"] == []
        assert deduper.training_pairs["distinct"] == [pair]

        with open(self.session_path) as handle:
            events = json.load(handle)["events"]

        assert [event["label"] for event in events] == ["match", "distinct"]
        assert events[0]["previous"] is None
        assert events[1]["previous"] == "match"
        assert events[1]["conflict"] is True

    def test_conflicting_relabel_warns(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]
        deduper.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deduper.mark_pairs(
                {"match": [], "distinct": [pair]}, session_file=self.session_path
            )

        assert any(
            issubclass(warning.category, session_io.SessionConflictWarning)
            for warning in caught
        )

    def test_unsure_is_stored_as_a_single_event(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]

        deduper.mark_pairs(
            {"match": [pair], "distinct": [pair]}, session_file=self.session_path
        )

        assert deduper.training_pairs["match"] == [pair]
        assert deduper.training_pairs["distinct"] == [pair]

        with open(self.session_path) as handle:
            events = json.load(handle)["events"]
        assert len(events) == 1
        assert events[0]["label"] == "unsure"

    # -- input binding ------------------------------------------------

    def test_mismatched_data_is_rejected(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        deduper.uncertain_pairs(session_file=self.session_path)

        other_data = OrderedDict(DATA)
        other_data[99] = {"name": "New", "age": "1"}

        resumed = make_dedupe()
        resumed.prepare_training(other_data)
        with self.assertRaises(session_io.SessionMismatchError):
            resumed.resume_session(self.session_path)

    def test_changed_field_values_are_rejected(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        deduper.uncertain_pairs(session_file=self.session_path)

        other_data = OrderedDict(DATA)
        other_data[0] = {"name": "Robert", "age": "51"}

        resumed = make_dedupe()
        resumed.prepare_training(other_data)
        with self.assertRaises(session_io.SessionMismatchError):
            resumed.resume_session(self.session_path)

    def test_workflow_mismatch_is_rejected(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        deduper.uncertain_pairs(session_file=self.session_path)

        linker = make_link()
        linker.prepare_training(DATA_1, DATA_2)
        with self.assertRaises(session_io.SessionMismatchError):
            linker.resume_session(self.session_path)

    def test_reordered_data_is_accepted(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]
        deduper.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )

        reordered = OrderedDict(reversed(list(DATA.items())))
        resumed = make_dedupe()
        resumed.prepare_training(reordered)
        assert resumed.resume_session(self.session_path) == 1
        assert resumed.training_pairs["match"] == [pair]

    def test_link_is_directional(self):
        linker = make_link()
        linker.prepare_training(DATA_1, DATA_2)
        pair = linker.uncertain_pairs(session_file=self.session_path)[0]
        linker.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )

        swapped = make_link()
        swapped.prepare_training(DATA_2, DATA_1)
        with self.assertRaises(session_io.SessionMismatchError):
            swapped.resume_session(self.session_path)

        resumed = make_link()
        resumed.prepare_training(DATA_1, DATA_2)
        assert resumed.resume_session(self.session_path) == 1
        assert resumed.training_pairs["match"] == [pair]

    # -- compatibility ------------------------------------------------

    def test_without_session_file_behavior_is_unchanged(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs()[0]
        deduper.mark_pairs({"match": [pair], "distinct": []})

        assert deduper.session is None
        assert deduper.training_pairs["match"] == [pair]

    def test_old_training_file_is_still_readable(self):
        training = io.StringIO()
        training_pairs = {"match": [(DATA[0], DATA[7])], "distinct": []}
        dedupe.write_training(training_pairs, training)
        training.seek(0)

        deduper = make_dedupe()
        deduper.prepare_training(DATA, training_file=training)

        assert deduper.training_pairs["match"] == training_pairs["match"]

    def test_session_replay_deduplicates_training_file(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        pair = deduper.uncertain_pairs(session_file=self.session_path)[0]
        deduper.mark_pairs(
            {"match": [pair], "distinct": []}, session_file=self.session_path
        )

        training = io.StringIO()
        dedupe.write_training({"match": [pair], "distinct": []}, training)
        training.seek(0)

        resumed = make_dedupe()
        resumed.prepare_training(DATA, training_file=training)
        resumed.resume_session(self.session_path)

        assert resumed.training_pairs["match"] == [pair]

    def test_training_file_is_not_treated_as_a_session(self):
        training = io.StringIO()
        dedupe.write_training({"match": [], "distinct": []}, training)
        training.seek(0)

        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        with self.assertRaises(session_io.SessionFileLoadingException):
            deduper.resume_session(training)

    def test_bad_session_file_raises(self):
        deduper = make_dedupe()
        deduper.prepare_training(DATA)

        with self.assertRaises(session_io.SessionFileLoadingException):
            deduper.resume_session(io.StringIO("not json"))


class ConsoleLabelSessionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.session_path = os.path.join(self.directory, "labeling.json")

    def tearDown(self):
        for name in os.listdir(self.directory):
            os.unlink(os.path.join(self.directory, name))
        os.rmdir(self.directory)

    def test_console_label_resumes_and_persists(self):
        # First run: answer the very first question, then "finish".
        responses = iter(["y", "f"])
        deduper = make_dedupe()
        deduper.prepare_training(DATA)
        with _patch_input(responses), _patch_print():
            dedupe.console_label(deduper, session_file=self.session_path)

        with open(self.session_path) as handle:
            document = json.load(handle)
        assert len(document["events"]) == 1
        assert document["events"][0]["label"] == "match"

        # Second run, fresh matcher: the earlier label is restored and the
        # "finish" answer ends it without hitting stdin again.
        responses = iter(["f"])
        resumed = make_dedupe()
        resumed.prepare_training(DATA)
        with _patch_input(responses), _patch_print():
            dedupe.console_label(resumed, session_file=self.session_path)

        assert len(resumed.training_pairs["match"]) == 1
        assert resumed.training_pairs["distinct"] == []


class _patch_input:
    def __init__(self, responses):
        self.responses = responses

    def __enter__(self):
        import builtins

        self._original = builtins.input
        builtins.input = lambda *args, **kwargs: next(self.responses)

    def __exit__(self, *exc):
        import builtins

        builtins.input = self._original


class _patch_print:
    def __enter__(self):
        import dedupe.convenience as convenience

        self._original = convenience._print
        convenience._print = lambda *args, **kwargs: None

    def __exit__(self, *exc):
        import dedupe.convenience as convenience

        convenience._print = self._original


class LabelerResetTest(unittest.TestCase):
    def test_reset_replaces_history_and_keeps_bootstrap(self):
        from dedupe import datamodel, labeler

        sample = {
            1: {"name": "Meredith", "age": "40"},
            2: {"name": "Sue", "age": "10"},
            3: {"name": "Willy", "age": "35"},
            4: {"name": "William", "age": "35"},
            5: {"name": "Jimmy", "age": "20"},
            6: {"name": "Jimbo", "age": "21"},
        }
        model = datamodel.DataModel(
            [dedupe.variables.String("name"), dedupe.variables.String("age")]
        )
        learner = labeler.DedupeDisagreementLearner(
            model.predicates, model.distances, sample, []
        )
        n_bootstrap = len(learner.pairs)

        first = learner.pop()
        learner.mark([first], [1])
        second = learner.pop()
        learner.mark([second], [0])
        assert len(learner.pairs) == n_bootstrap + 2

        # Relabel `first` as distinct: the old match label must be dropped.
        learner.reset([first, second], [0, 0])
        assert len(learner.pairs) == n_bootstrap + 2
        assert list(learner.y) == [1, 1, 1, 1, 0, 0, 0]
        assert learner.pairs[:n_bootstrap] == learner._bootstrap_pairs


class RecordResolverTest(unittest.TestCase):
    def setUp(self):
        self.datasets = [DATA_1, DATA_2]

    def test_link_pairs_are_resolved_per_side(self):
        resolver = session_io.RecordResolver(self.datasets, ["name", "age"])
        ids = resolver.ids_for_pair((DATA_1[1], DATA_2[107]), "link")
        assert ids == (1, 107)

    def test_dedupe_ids_are_available_regardless_of_order(self):
        resolver = session_io.RecordResolver([DATA], ["name", "age"])
        assert resolver.ids_for_pair((DATA[4], DATA[0]), "dedupe") == (4, 0)

    def test_pair_key_is_order_independent_for_dedupe(self):
        assert session_io.pair_key((4, 7), "dedupe") == session_io.pair_key(
            (7, 4), "dedupe"
        )
        assert session_io.pair_key((4, 7), "link") != session_io.pair_key(
            (7, 4), "link"
        )

    def test_pair_key_distinguishes_int_and_str_ids(self):
        assert session_io.pair_key((1, 2), "dedupe") != session_io.pair_key(
            ("1", 2), "dedupe"
        )

    def test_content_fallback_when_ids_unknown(self):
        resolver = session_io.RecordResolver([DATA], ["name", "age"])
        unknown = ({"name": "Z", "age": "9"}, {"name": "W", "age": "3"})
        assert resolver.ids_for_pair(unknown, "dedupe") is None
        key = resolver.key_for_pair(unknown, "dedupe")
        assert key.startswith('["content"')


if __name__ == "__main__":
    unittest.main()
