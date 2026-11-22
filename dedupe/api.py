#!/usr/bin/python
"""
dedupe provides the main user interface for the library the
Dedupe class
"""
from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
import pickle
import sqlite3
import tempfile
import warnings
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy
import sklearn.linear_model
import sklearn.model_selection

import dedupe.blocking as blocking
import dedupe.clustering as clustering
import dedupe.core as core
import dedupe.datamodel as datamodel
import dedupe.labeler as labeler
import dedupe.predicates
import dedupe.serializer as serializer
import dedupe.session as session_io

if TYPE_CHECKING:
    from typing import BinaryIO, Collection, Generator, Iterable, MutableMapping, TextIO

    import numpy.typing

    from dedupe._typing import (
        ArrayLinks,
        Blocks,
        BlocksInt,
        BlocksStr,
        Classifier,
        Clusters,
        ClustersInt,
        ClustersStr,
        Data,
        DataInt,
        DataStr,
        JoinConstraint,
        LabelsLike,
        Links,
        LookupResultsInt,
        LookupResultsStr,
        PathLike,
        RecordDict,
    )
    from dedupe._typing import RecordDictPair as TrainingExample
    from dedupe._typing import RecordDictPairs as TrainingExamples
    from dedupe._typing import (
        RecordID,
        RecordPairs,
        Scores,
        TrainingData,
        TupleLinks,
        Variable,
    )
    from dedupe.session import SessionFile

logger = logging.getLogger(__name__)


class Matching:
    """
    Base Class for Record Matching Classes
    """

    def __init__(
        self, num_cores: int | None, in_memory: bool = False, **kwargs
    ) -> None:
        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = num_cores

        self.in_memory = in_memory
        self._fingerprinter: blocking.Fingerprinter | None = None
        self.data_model: datamodel.DataModel
        self.classifier: Classifier
        self.predicates: Collection[dedupe.predicates.Predicate]

    @property
    def fingerprinter(self) -> blocking.Fingerprinter:
        if self._fingerprinter is None:
            raise ValueError(
                "the record fingerprinter is not intialized, "
                "please run the train method"
            )

        return self._fingerprinter


class IntegralMatching(Matching):
    """
    This class is for linking class where we need to score all possible
    pairs before deciding on any matches
    """

    def score(self, pairs: RecordPairs) -> Scores:
        """
        Scores pairs of records. Returns pairs of tuples of records id and
        associated probabilities that the pair of records are match

        Args:
            pairs: Iterator of pairs of records

        """
        try:
            matches = core.scoreDuplicates(
                pairs, self.data_model.distances, self.classifier, self.num_cores
            )
        except RuntimeError:
            raise RuntimeError(
                """
                You need to either turn off multiprocessing or protect
                the calls to the Dedupe methods with a
                `if __name__ == '__main__'` in your main module, see
                https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods"""
            )

        return matches


class DedupeMatching(IntegralMatching):
    """
    Class for Deduplication, extends Matching.

    Use DedupeMatching when you have a dataset that can contain
    multiple references to the same entity.

    """

    @overload
    def partition(
        self, data: DataInt, threshold: float = 0.5
    ) -> ClustersInt:  # pragma: no cover
        ...

    @overload
    def partition(
        self, data: DataStr, threshold: float = 0.5
    ) -> ClustersStr:  # pragma: no cover
        ...

    def partition(self, data, threshold=0.5):  # pragma: no cover
        """
        Identifies records that all refer to the same entity, returns
        tuples containing a sequence of record ids and corresponding
        sequence of confidence score as a float between 0 and 1. The
        record_ids within each set should refer to the same entity and the
        confidence score is a measure of our confidence a particular entity
        belongs in the cluster.

        For details on the confidence score, see :func:`dedupe.Dedupe.cluster`.

        This method should only used for small to moderately sized
        datasets for larger data, you need may need to generate your
        own pairs of records and feed them to :func:`~score`.

        Args:
            data: Dictionary of records, where the keys are record_ids
                  and the values are dictionaries with the keys being
                  field names

            threshold: Number between 0 and 1.  We
                       will only consider put together records into
                       clusters if the `cophenetic similarity
                       <https://en.wikipedia.org/wiki/Cophenetic>`_ of
                       the cluster is greater than the threshold.

                       Lowering the number will increase recall,
                       raising it will increase precision

        Examples:
            >>> duplicates = matcher.partition(data, threshold=0.5)
            >>> duplicates
            [
                ((1, 2, 3), (0.790, 0.860, 0.790)),
                ((4, 5), (0.720, 0.720)),
                ((10, 11), (0.899, 0.899)),
            ]
        """
        pairs = self.pairs(data)
        pair_scores = self.score(pairs)
        clusters = self.cluster(pair_scores, threshold)
        clusters = self._add_singletons(data.keys(), clusters)
        clusters_eval = list(clusters)
        _cleanup_scores(pair_scores)
        return clusters_eval

    @overload
    @staticmethod
    def _add_singletons(
        all_ids: Iterable[int], clusters: ClustersInt
    ) -> ClustersInt: ...

    @overload
    @staticmethod
    def _add_singletons(
        all_ids: Iterable[str], clusters: ClustersStr
    ) -> ClustersStr: ...

    @staticmethod
    def _add_singletons(all_ids, clusters):
        singletons = set(all_ids)

        for record_ids, score in clusters:
            singletons.difference_update(record_ids)
            yield (record_ids, score)

        for singleton in singletons:
            yield (singleton,), (1.0,)

    def pairs(self, data: Data) -> RecordPairs:
        """
        Yield pairs of records that share common fingerprints.

        Each pair will occur at most once. If you override this
        method, you need to take care to ensure that this remains
        true, as downstream methods, particularly :func:`cluster`, assumes
        that every pair of records is compared no more than once.

        Args:
            data: Dictionary of records, where the keys are record_ids
                  and the values are dictionaries with the keys being
                  field names

        Examples:
            >>> pairs = matcher.pairs(data)
            >>> list(pairs)
            [
                (
                    (1, {"name": "Pat", "address": "123 Main"}),
                    (2, {"name": "Pat", "address": "123 Main"}),
                ),
                (
                    (1, {"name": "Pat", "address": "123 Main"}),
                    (3, {"name": "Sam", "address": "123 Main"}),
                ),
            ]
        """

        self.fingerprinter.index_all(data)

        id_type = core.sqlite_id_type(data)

        # Blocking and pair generation are typically the first memory
        # bottlenecks, so we'll use sqlite3 to avoid doing them in memory
        with tempfile.TemporaryDirectory() as temp_dir:
            if self.in_memory:
                con = sqlite3.connect(":memory:")
            else:
                con = sqlite3.connect(temp_dir + "/blocks.db")

            # Set journal mode to WAL.
            con.execute("pragma journal_mode=off")
            con.execute(
                f"CREATE TABLE blocking_map (block_key text, record_id {id_type})"
            )
            con.executemany(
                "INSERT INTO blocking_map values (?, ?)",
                self.fingerprinter(data.items()),
            )

            self.fingerprinter.reset_indices()

            con.execute(
                """CREATE UNIQUE INDEX record_id_block_key_idx
                           ON blocking_map (record_id, block_key)"""
            )
            con.execute(
                """CREATE INDEX block_key_idx
                           ON blocking_map (block_key)"""
            )
            con.execute("""ANALYZE""")
            pairs = con.execute(
                """SELECT DISTINCT a.record_id, b.record_id
                                   FROM blocking_map a
                                   INNER JOIN blocking_map b
                                   USING (block_key)
                                   WHERE a.record_id < b.record_id"""
            )

            for a_record_id, b_record_id in pairs:
                yield (
                    (a_record_id, data[a_record_id]),
                    (b_record_id, data[b_record_id]),
                )

            pairs.close()
            con.close()

    def cluster(self, scores: Scores, threshold: float = 0.5) -> Clusters:
        r"""From the similarity scores of pairs of records, decide which groups
        of records are all referring to the same entity.

        Yields tuples containing a sequence of record ids and corresponding
        sequence of confidence score as a float between 0 and 1. The
        record_ids within each set should refer to the same entity and the
        confidence score is a measure of our confidence a particular entity
        belongs in the cluster.

        Each confidence scores is a measure of how similar the record is
        to the other records in the cluster. Let :math:`\phi(i,j)` be the pair-wise
        similarity between records :math:`i` and :math:`j`. Let :math:`N` be the number of records in the cluster.

        .. math::

           \text{confidence score}_i = 1 - \sqrt {\frac{\sum_{j}^N (1 - \phi(i,j))^2}{N -1}}

        This measure is similar to the average squared distance
        between the focal record and the other records in the
        cluster. These scores can be `combined to give a total score
        for the cluster
        <https://en.wikipedia.org/wiki/Variance#Discrete_random_variable>`_.

        .. math::

           \text{cluster score} = 1 - \sqrt { \frac{\sum_i^N(1 - \mathrm{score}_i)^2 \cdot (N - 1) } { 2 N^2}}

        Args:
            scores: a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ with a dtype of `[('pairs', id_type, 2),
                    ('score', 'f4')]` where dtype is either a str
                    or int, and score is a number between 0 and
                    1. The 'pairs' column contains pairs of ids of
                    the records compared and the 'score' column
                    should contains the similarity score for that
                    pair of records.

                    For each pair, the smaller id should be first.

            threshold: Number between 0 and 1. We will only consider
                       put together records into clusters if the
                       `cophenetic similarity
                       <https://en.wikipedia.org/wiki/Cophenetic>`_ of
                       the cluster is greater than the threshold.

                       Lowering the number will increase recall,
                       raising it will increase precision

        Examples:
            >>> pairs = matcher.pairs(data)
            >>> scores = matcher.scores(pairs)
            >>> clusters = matcher.cluster(scores)
            >>> list(clusters)
            [
                ((1, 2, 3), (0.790, 0.860, 0.790)),
                ((4, 5), (0.720, 0.720)),
                ((10, 11), (0.899, 0.899)),
            ]

        """

        logger.debug("matching done, begin clustering")

        yield from clustering.cluster(scores, threshold)


class RecordLinkMatching(IntegralMatching):
    """
    Class for Record Linkage, extends Matching.

    Use RecordLinkMatching when you have two datasets that you want to merge
    """

    def pairs(self, data_1: Data, data_2: Data) -> RecordPairs:
        """
        Yield pairs of records that share common fingerprints.

        Each pair will occur at most once. If you override this
        method, you need to take care to ensure that this remains
        true, as downstream methods, particularly :func:`one_to_one`,
        and :func:`many_to_one` assumes that every pair of records is
        compared no more than once.

        Args:
            data_1: Dictionary of records from first dataset, where the
                    keys are record_ids and the values are dictionaries
                    with the keys being field names
            data_2: Dictionary of records from second dataset, same
                    form as data_1

        Examples:
            >>> pairs = matcher.pairs(data_1, data_2)
            >>> list(pairs)
            [
                (
                    (1, {"name": "Pat", "address": "123 Main"}),
                    (2, {"name": "Pat", "address": "123 Main"}),
                ),
                (
                    (1, {"name": "Pat", "address": "123 Main"}),
                    (3, {"name": "Sam", "address": "123 Main"}),
                ),
            ]
        """

        self.fingerprinter.index_all(data_2)

        id_type_a = core.sqlite_id_type(data_1)
        id_type_b = core.sqlite_id_type(data_2)

        # Blocking and pair generation are typically the first memory
        # bottlenecks, so we'll use sqlite3 to avoid doing them in memory
        with tempfile.TemporaryDirectory() as temp_dir:
            if self.in_memory:
                con = sqlite3.connect(":memory:")
            else:
                con = sqlite3.connect(temp_dir + "/blocks.db")

            con.execute("pragma journal_mode=off")

            con.executescript(
                f"""CREATE TABLE blocking_map_a
                                 (block_key text, record_id {id_type_a});

                                 CREATE TABLE blocking_map_b
                                 (block_key text, record_id {id_type_b});"""
            )

            con.executemany(
                "INSERT INTO blocking_map_a values (?, ?)",
                self.fingerprinter(data_1.items()),
            )

            con.executemany(
                "INSERT INTO blocking_map_b values (?, ?)",
                self.fingerprinter(data_2.items(), target=True),
            )

            self.fingerprinter.reset_indices()

            con.executescript(
                """CREATE UNIQUE INDEX block_key_a_idx
                                 ON blocking_map_a (record_id, block_key);

                   CREATE UNIQUE INDEX block_key_b_idx
                                 ON blocking_map_b (block_key, record_id);"""
            )
            con.execute("""ANALYZE""")

            pairs = con.execute(
                """SELECT DISTINCT a.record_id, b.record_id
                                   FROM blocking_map_a a
                                   INNER JOIN blocking_map_b b
                                   USING (block_key)"""
            )

            for a_record_id, b_record_id in pairs:
                yield (
                    (a_record_id, data_1[a_record_id]),
                    (b_record_id, data_2[b_record_id]),
                )

            pairs.close()
            con.close()

    def join(
        self,
        data_1: Data,
        data_2: Data,
        threshold: float = 0.5,
        constraint: JoinConstraint = "one-to-one",
    ) -> Links:
        """
        Identifies pairs of records that refer to the same entity.

        Returns pairs of record ids with a confidence score as a float
        between 0 and 1. The record_ids within the pair should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        This method should only used for small to moderately sized
        datasets for larger data, you need may need to generate your
        own pairs of records and feed them to the :func:`~score`.

        Args:
            data_1: Dictionary of records from first dataset, where the
                    keys are record_ids and the values are dictionaries
                    with the keys being field names

            data_2: Dictionary of records from second dataset, same form
                    as data_1

            threshold: Number between 0 and 1. We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision

            constraint: What type of constraint to put on a join.

                        'one-to-one'
                              Every record in data_1 can match at most
                              one record from data_2 and every record
                              from data_2 can match at most one record
                              from data_1. This is good for when both
                              data_1 and data_2 are from different
                              sources and you are interested in
                              matching across the sources. If,
                              individually, data_1 or data_2 have many
                              duplicates you will not get good
                              matches.
                        'many-to-one'
                              Every record in data_1 can match at most
                              one record from data_2, but more than
                              one record from data_1 can match to the
                              same record in data_2. This is good for
                              when data_2 is a lookup table and data_1
                              is messy, such as geocoding or matching
                              against golden records.
                        'many-to-many'
                              Every record in data_1 can match
                              multiple records in data_2 and vice
                              versa. This is like a SQL inner join.

        Examples:
            >>> links = matcher.join(data_1, data_2, threshold=0.5)
            >>> list(links)
            [
                ((1, 2), 0.790),
                ((4, 5), 0.720),
                ((10, 11), 0.899)
            ]
        """

        assert constraint in {"one-to-one", "many-to-one", "many-to-many"}, (
            "%s is an invalid constraint option. Valid options include "
            "one-to-one, many-to-one, or many-to-many" % constraint
        )

        pairs = self.pairs(data_1, data_2)
        pair_scores = self.score(pairs)

        links: Links
        if constraint == "one-to-one":
            links = self.one_to_one(pair_scores, threshold)
        elif constraint == "many-to-one":
            links = self.many_to_one(pair_scores, threshold)
        else:
            links = pair_scores[pair_scores["score"] > threshold]

        links_evaluated: Links = list(links)  # type: ignore[assignment]
        _cleanup_scores(pair_scores)
        return links_evaluated

    def one_to_one(self, scores: Scores, threshold: float = 0.0) -> TupleLinks:
        """From the similarity scores of pairs of records, decide which
        pairs refer to the same entity.

        Every record in data_1 can match at most one record from
        data_2 and every record from data_2 can match at most one
        record from data_1. See
        https://en.wikipedia.org/wiki/Injective_function.

        This method is good for when both data_1 and data_2 are from
        different sources and you are interested in matching across
        the sources. If, individually, data_1 or data_2 have many duplicates
        you will not get good matches.

        Yields pairs of record ids with a confidence score as a float
        between 0 and 1. The record_ids within the pair should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        Args:
            scores: a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ with a dtype of `[('pairs', id_type, 2),
                    ('score', 'f4')]` where dtype is either a str
                    or int, and score is a number between 0 and
                    1. The 'pairs' column contains pairs of ids of
                    the records compared and the 'score' column
                    should contains the similarity score for that
                    pair of records.

            threshold: Number between 0 and 1. We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision


        Examples:
            >>> pairs = matcher.pairs(data)
            >>> scores = matcher.scores(pairs, threshold=0.5)
            >>> links = matcher.one_to_one(scores)
            >>> list(links)
            [
                ((1, 2), 0.790),
                ((4, 5), 0.720),
                ((10, 11), 0.899)
            ]
        """
        if threshold:
            scores = scores[scores["score"] > threshold]

        logger.debug("matching done, begin clustering")

        yield from clustering.greedyMatching(scores)

    def many_to_one(self, scores: Scores, threshold: float = 0.0) -> ArrayLinks:
        """
        From the similarity scores of pairs of records, decide which
        pairs refer to the same entity.

        Every record in data_1 can match at most one record from
        data_2, but more than one record from data_1 can match to the same
        record in data_2. See
        https://en.wikipedia.org/wiki/Surjective_function

        This method is good for when data_2 is a lookup table and data_1
        is messy, such as geocoding or matching against golden records.

        Yields pairs of record ids with a confidence score as a float
        between 0 and 1. The record_ids within the pair should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        Args:
            scores: a numpy `structured array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ with a dtype of `[('pairs', id_type, 2),
                    ('score', 'f4')]` where dtype is either a str
                    or int, and score is a number between 0 and
                    1. The 'pairs' column contains pairs of ids of
                    the records compared and the 'score' column
                    should contains the similarity score for that
                    pair of records.

            threshold: Number between 0 and 1. We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision

        Examples:
            >>> pairs = matcher.pairs(data)
            >>> scores = matcher.scores(pairs, threshold=0.5)
            >>> links = matcher.many_to_one(scores)
            >>> print(list(links))
            [
                ((1, 2), 0.790),
                ((4, 5), 0.720),
                ((7, 2), 0.623),
                ((10, 11), 0.899)
             ]
        """

        logger.debug("matching done, begin clustering")

        yield from clustering.pair_gazette_matching(scores, threshold, 1)


class GazetteerMatching(Matching):
    def __init__(
        self, num_cores: int | None, in_memory: bool = False, **kwargs
    ) -> None:
        super().__init__(num_cores, in_memory, **kwargs)

        self.db: PathLike
        if self.in_memory:
            self.db = ":memory:"
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.db = self.temp_dir.name + "/blocks.db"

        self.indexed_data: (
            MutableMapping[int, RecordDict] | MutableMapping[str, RecordDict]
        )
        self.indexed_data = {}  # type: ignore[assignment]

    def _close(self) -> None:
        if not self.in_memory:
            self.temp_dir.cleanup()

    def __del__(self) -> None:
        self._close()

    @overload
    def index(self, data: DataInt) -> None: ...

    @overload
    def index(self, data: DataStr) -> None: ...

    def index(self, data):  # pragma: no cover
        """
        Add records to the index of records to match against. If a record in
        `canonical_data` has the same key as a previously indexed record, the
        old record will be replaced.

        Args:
            data: a dictionary of records where the keys
                  are record_ids and the values are
                  dictionaries with the keys being
                  field_names
        """

        self.fingerprinter.index_all(data)

        id_type = core.sqlite_id_type(data)

        con = sqlite3.connect(self.db)

        # Set journal mode to WAL.
        con.execute("pragma journal_mode=wal")

        con.execute(
            f"""CREATE TABLE IF NOT EXISTS indexed_records
                       (block_key text,
                        record_id {id_type},
                        UNIQUE(block_key, record_id))"""
        )

        con.executemany(
            "REPLACE INTO indexed_records VALUES (?, ?)",
            self.fingerprinter(data.items(), target=True),
        )

        con.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS
                       indexed_records_block_key_idx
                       ON indexed_records
                       (block_key, record_id)"""
        )
        con.execute("""ANALYZE""")

        con.commit()
        con.close()

        self.indexed_data.update(data)

    @overload
    def unindex(self, data: DataInt) -> None:  # pragma: no cover
        ...

    @overload
    def unindex(self, data: DataStr) -> None:  # pragma: no cover
        ...

    def unindex(self, data):  # pragma: no cover
        """
        Remove records from the index of records to match against.

        Args:
            data: a dictionary of records where the keys
                  are record_ids and the values are
                  dictionaries with the keys being
                  field_names
        """

        for field in self.fingerprinter.index_fields:
            self.fingerprinter.unindex(
                {record[field] for record in data.values()}, field
            )

        con = sqlite3.connect(self.db)
        con.executemany(
            """DELETE FROM indexed_records
                           WHERE record_id = ?""",
            ((k,) for k in data.keys()),
        )

        con.commit()
        con.close()

        for k in data:
            del self.indexed_data[k]

    @overload
    def blocks(self, data: DataInt) -> BlocksInt: ...

    @overload
    def blocks(self, data: DataStr) -> BlocksStr: ...

    def blocks(self, data):
        """
        Yield groups of pairs of records that share fingerprints.

        Each group contains one record from data_1 paired with the records
        from the indexed records that data_1 shares a fingerprint with.

        Each pair within and among blocks will occur at most once. If
        you override this method, you need to take care to ensure that
        this remains true, as downstream methods, particularly
        :func:`many_to_n`, assumes that every pair of records is compared no
        more than once.

        Args:
            data: Dictionary of records, where the keys are record_ids
                  and the values are dictionaries with the keys being
                  field names

        Examples:
            >>> blocks = matcher.pairs(data)
            >>> print(list(blocks)
            [
                [
                    (
                        (1, {"name": "Pat", "address": "123 Main"}),
                        (8, {"name": "Pat", "address": "123 Main"}),
                    ),
                    (
                        (1, {"name": "Pat", "address": "123 Main"}),
                        (9, {"name": "Sam", "address": "123 Main"}),
                    ),
                ],
                [
                    (
                        (2, {"name": "Sam", "address": "2600 State"}),
                        (5, {"name": "Pam", "address": "2600 Stat"}),
                    ),
                    (
                        (2, {"name": "Sam", "address": "123 State"}),
                        (7, {"name": "Sammy", "address": "123 Main"}),
                    ),
                ],
            ]
        """

        id_type = core.sqlite_id_type(data)

        con = sqlite3.connect(self.db, check_same_thread=False)

        con.execute("BEGIN")

        con.execute(
            f"CREATE TEMPORARY TABLE blocking_map (block_key text, record_id {id_type})"
        )
        con.executemany(
            "INSERT INTO blocking_map VALUES (?, ?)", self.fingerprinter(data.items())
        )

        pairs = con.execute(
            """SELECT DISTINCT a.record_id, b.record_id
                               FROM blocking_map a
                               INNER JOIN indexed_records b
                               USING (block_key)
                               ORDER BY a.record_id"""
        )

        pair_blocks: (
            Iterable[tuple[int, Iterable[tuple[int, int]]]]
            | Iterable[tuple[str, Iterable[tuple[str, str]]]]
        )

        pair_blocks = itertools.groupby(pairs, lambda x: x[0])

        for _, pair_block in pair_blocks:
            yield [
                (
                    (a_record_id, data[a_record_id]),
                    (b_record_id, self.indexed_data[b_record_id]),
                )
                for a_record_id, b_record_id in pair_block
            ]

        pairs.close()
        con.execute("ROLLBACK")
        con.close()

    def score(self, blocks: Blocks) -> Generator[Scores]:
        """
        Scores groups of pairs of records. Yields structured numpy arrays
        representing pairs of records in the group and the associated
        probability that the pair is a match.

        Args:
            blocks: Iterator of blocks of records
        """

        matches = core.scoreGazette(
            blocks, self.data_model.distances, self.classifier, self.num_cores
        )

        return matches

    def many_to_n(
        self,
        score_blocks: Iterable[Scores],
        threshold: float = 0.0,
        n_matches: int = 1,
    ) -> ArrayLinks:
        """
        For each group of scored pairs, yield the highest scoring N pairs

        Args:
            score_blocks: Iterator of numpy `structured arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_,
                          each with a dtype of `[('pairs', id_type, 2),
                          ('score', 'f4')]` where dtype is either a str
                          or int, and score is a number between 0 and
                          1. The 'pairs' column contains pairs of ids of
                          the records compared and the 'score' column
                          should contains the similarity score for that
                          pair of records.

            threshold: Number between 0 and 1. We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision

            n_matches: How many top scoring pairs to select per group
        """

        yield from clustering.gazetteMatching(score_blocks, threshold, n_matches)

    @overload
    def search(
        self,
        data: DataInt,
        threshold: float = 0.0,
        n_matches: int = 1,
        generator: bool = False,
    ) -> LookupResultsInt:  # pragma: no cover
        ...

    @overload
    def search(
        self,
        data: DataStr,
        threshold: float = 0.0,
        n_matches: int = 1,
        generator: bool = False,
    ) -> LookupResultsStr:  # pragma: no cover
        ...

    def search(
        self,
        data,
        threshold=0.0,
        n_matches=1,
        generator=False,
    ):  # pragma: no cover
        """
        Identifies pairs of records that could refer to the same entity,
        returns tuples containing tuples of possible matches, with a
        confidence score for each match. The record_ids within each
        tuple should refer to potential matches from a messy data
        record to canonical records. The confidence score is the
        estimated probability that the records refer to the same
        entity.

        Args:
            data: a dictionary of records from a messy
                  dataset, where the keys are record_ids and
                  the values are dictionaries with the keys
                  being field names.

            threshold: a number between 0 and 1. We will consider
                       records as potential duplicates if the predicted
                       probability of being a duplicate is
                       above the threshold.

                       Lowering the number will increase
                       recall, raising it will increase
                       precision
            n_matches: the maximum number of possible matches from
                       canonical_data to return for each record in
                       data. If set to `None` all possible
                       matches above the threshold will be
                       returned.
            generator: when `True`, match will generate a sequence of
                       possible matches, instead of a list.

        Examples:
            >>> matches = gazetteer.search(data, threshold=0.5, n_matches=2)
            >>> print(matches)
            [
                (((1, 6), 0.72), ((1, 8), 0.6)),
                (((2, 7), 0.72),),
                (((3, 6), 0.72), ((3, 8), 0.65)),
                (((4, 6), 0.96), ((4, 5), 0.63)),
            ]
        """
        blocks = self.blocks(data)
        pair_scores = self.score(blocks)
        search_results = self.many_to_n(pair_scores, threshold, n_matches)

        results = self._format_search_results(data, search_results)

        if generator:
            return results
        else:
            return list(results)

    @overload
    def _format_search_results(
        self, search_d: DataInt, results: ArrayLinks
    ) -> LookupResultsInt: ...

    @overload
    def _format_search_results(
        self, search_d: DataStr, results: ArrayLinks
    ) -> LookupResultsStr: ...

    def _format_search_results(self, search_d, results):
        seen: set[RecordID] = set()

        for result in results:
            a: RecordID | None = None
            b: RecordID
            score: float
            prepared_result: list[tuple[RecordID, float]] = []
            for (a, b), score in result:
                prepared_result.append((b, score))

            assert a is not None

            yield a, tuple(prepared_result)
            seen.add(a)

        for k in search_d.keys() - seen:
            yield k, ()


class StaticMatching(Matching):
    """
    Class for initializing a dedupe object from a settings file.
    """

    def __init__(
        self,
        settings_file: BinaryIO,
        num_cores: int | None = None,
        in_memory: bool = False,
        **kwargs,
    ) -> None:  # pragma: no cover
        """
        Args:
            settings_file: A file object containing settings
                           info produced from the
                           :func:`~dedupe.api.ActiveMatching.write_settings` method.

            num_cores: The number of cpus to use for parallel
                       processing, defaults to the number of cpus
                       available on the machine. If set to 0, then
                       multiprocessing will be disabled.

            in_memory: If True, :meth:`dedupe.Dedupe.pairs` will generate
                       pairs in RAM with the sqlite3 ':memory:' option
                       rather than writing to disk. May be faster if
                       sufficient memory is available.

        .. warning::

            If using multiprocessing on Windows or Mac OS X, then
            you must protect calls to the Dedupe methods with a
            `if __name__ == '__main__'` in your main module, see
            https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
        """
        super().__init__(num_cores, in_memory, **kwargs)

        try:
            self.data_model = pickle.load(settings_file)
            self.classifier = pickle.load(settings_file)
            self.predicates = pickle.load(settings_file)
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe."
            )
        except ModuleNotFoundError as exc:
            if "No module named 'rlr'" in str(exc):
                raise SettingsFileLoadingException(
                    "This settings file was created with a previous "
                    "version of dedupe that used the 'rlr' library. "
                    "To continue to use this settings file, you need "
                    "install that library: `pip install rlr`"
                )
            else:
                raise SettingsFileLoadingException(
                    "Something has gone wrong with loading the settings file. "
                    "Try deleting the file"
                ) from exc
        except:  # noqa: E722
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file"
            )

        logger.info("Predicate set:")
        for predicate in self.predicates:
            logger.info(predicate)

        self._fingerprinter = blocking.Fingerprinter(self.predicates)


class ActiveMatching(Matching):
    """
    Class for training a matcher.
    """

    active_learner: labeler.DisagreementLearner | None
    training_pairs: TrainingData

    def __init__(
        self,
        variable_definition: Collection[Variable],
        num_cores: int | None = None,
        in_memory: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            variable_definition: A list of Variable objects describing
                                 the variables will be used for
                                 training a model. See :ref:`variable_definitions`

            num_cores: The number of cpus to use for parallel
                       processing. If set to `None`, uses all cpus
                       available on the machine. If set to 0, then
                       multiprocessing will be disabled.

            in_memory: If True, :meth:`dedupe.Dedupe.pairs` will generate
                       pairs in RAM with the sqlite3 ':memory:' option
                       rather than writing to disk. May be faster if
                       sufficient memory is available.

        .. warning::

            If using multiprocessing on Windows or Mac OS X, then
            you must protect calls to the Dedupe methods with a
            `if __name__ == '__main__'` in your main module, see
            https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods

        """
        super().__init__(num_cores, in_memory, **kwargs)

        self.data_model = datamodel.DataModel(variable_definition)
        self.training_pairs = {"distinct": [], "match": []}
        self.classifier = sklearn.model_selection.GridSearchCV(
            estimator=sklearn.linear_model.LogisticRegression(),
            param_grid={"C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]},
            scoring="f1",
            n_jobs=-1,
        )
        self.active_learner = None

        # Interactive labeling session. These attributes are only populated
        # once a session file is supplied; without one the behavior of
        # `uncertain_pairs` and `mark_pairs` is unchanged.
        self.session: session_io.LabelingSession | None = None
        self._session_mode = "dedupe"
        self._session_data: list[Mapping] | None = None
        self._resolver: session_io.RecordResolver | None = None
        self._training_index: dict[str, str] | None = None
        self._promoted: set[str] = set()

    def _field_schema(self) -> list[str]:
        return [variable.field for variable in self.data_model.field_variables]

    def _set_session_input(
        self, mode: Literal["dedupe", "link"], datasets: list[Mapping]
    ) -> None:
        """Remember the data a session should be bound to.

        This is deliberately separate from attaching a session so that a
        training file can be read first without being recorded as manual
        labels. The record index is built lazily, only once a session is
        actually used, so callers that never pass a session file pay nothing.
        """
        self._session_mode = mode
        self._session_data = datasets
        self._resolver = None

    def _resolver_or_none(self) -> session_io.RecordResolver | None:
        if self._resolver is None and self._session_data is not None:
            self._resolver = session_io.RecordResolver(
                self._session_data, self._field_schema()
            )
        return self._resolver

    def _reset_session(self) -> None:
        self.session = None
        self._resolver = None
        self._training_index = None
        self._promoted = set()

    # -- session plumbing --------------------------------------------

    def _key_for_records(self, records: tuple) -> str:
        resolver = self._resolver_or_none()
        if resolver is not None:
            return resolver.key_for_pair(records, self._session_mode)
        return session_io.pair_content_key(records, self._session_mode)

    def _ids_for_records(self, records: tuple) -> tuple | None:
        resolver = self._resolver_or_none()
        if resolver is None:
            return None
        return resolver.ids_for_pair(records, self._session_mode)

    def _records_for_event(self, event: Mapping) -> tuple | None:
        ids = event.get("ids")
        resolver = self._resolver_or_none()
        if ids and resolver is not None:
            records = resolver.records_for_ids(ids, self._session_mode)
            if records is not None:
                return records
        stored = event.get("records")
        if stored is not None:
            decoded = session_io.decode_records(stored)
            if decoded is not None:
                return decoded
        return None

    def _attach_session(
        self, session_file: SessionFile | None
    ) -> session_io.LabelingSession | None:
        """Open ``session_file`` (if needed) and replay its completed labels."""
        if session_file is None:
            return self.session

        if self.session is not None and session_io.source_matches(
            self.session.source, session_file
        ):
            return self.session

        fingerprint = None
        if self._session_data is not None:
            fingerprint = session_io.data_fingerprint(
                self._session_data, self._field_schema()
            )

        session = session_io.open_session(
            session_file, self._session_mode, fingerprint
        )
        self.session = session
        self._promoted = set()
        self._rebuild_training_index()

        restored = self._replay_session(session)
        if self.active_learner is not None and restored:
            self._refit_active_learner()

        if session.dirty:
            session.save()
        return session

    def _rebuild_training_index(self) -> None:
        # The index is only needed while a session is attached. Skipping it
        # otherwise keeps `prepare_training` cheap for large datasets.
        if self.session is None:
            self._training_index = None
            return

        index: dict[str, str] = {}
        for label in ("match", "distinct"):
            for pair in self.training_pairs[label]:
                key = self._key_for_records(tuple(pair))
                previous = index.get(key)
                if previous is not None and previous != label:
                    index[key] = "unsure"
                else:
                    index[key] = label
        self._training_index = index

    def _replay_session(
        self, session: session_io.LabelingSession
    ) -> int:
        """Apply the session's latest label for each pair to the training set.

        Returns the number of events that were applied.
        """
        applied = 0
        for event in session.resolved():
            records = self._records_for_event(event)
            if records is None:
                warnings.warn(
                    "A stored label could not be matched to a record pair in the "
                    "current input data and was skipped: "
                    f"{event.get('pair')!r}.",
                    session_io.SessionDataWarning,
                )
                continue

            # The session already warned when a contradictory label was
            # recorded among its own events. Here the only possible conflict is
            # with a label from a training file, so surface that too.
            if self._upsert_training(records, event["label"], warn_conflict=True):
                applied += 1

        return applied

    def _refit_active_learner(self) -> None:
        """Rebuild the active learner's history from ``self.training_pairs``."""
        if self.active_learner is None:
            return
        examples, y = flatten_training(self.training_pairs)
        try:
            self.active_learner.reset(examples, y)
        except dedupe.predicates.NoIndexError as e:
            warnings.warn(
                "The record\n"
                f"{e.failing_record}\n"
                "is not known to the active learner. Make sure all labeled "
                "pairs are in the data or training file of the "
                "`prepare_training()` method",
                UserWarning,
            )

    def _upsert_training(
        self, records: tuple, label: str, warn_conflict: bool
    ) -> bool:
        """Add or replace a labeled pair in the training set.

        Returns True when the training set changed.
        """
        key = self._key_for_records(records)
        index = self._training_index
        existing = index.get(key) if index is not None else None

        if existing == label:
            return False

        if existing is not None:
            self._remove_training_key(key)
            if warn_conflict and {existing, label} == {"match", "distinct"}:
                self._conflict_warning(records, existing, label)

        self._add_training_records(records, label)
        if index is not None:
            index[key] = label
        return True

    def _add_training_records(self, records: tuple, label: str) -> None:
        pair = records  # type: ignore[assignment]
        if label in ("match", "unsure"):
            self.training_pairs["match"].append(pair)
        if label in ("distinct", "unsure"):
            self.training_pairs["distinct"].append(pair)

    def _remove_training_key(self, key: str) -> None:
        for label in ("match", "distinct"):
            kept = [
                pair
                for pair in self.training_pairs[label]
                if self._key_for_records(tuple(pair)) != key
            ]
            self.training_pairs[label] = kept

    @staticmethod
    def _conflict_warning(records: tuple, previous: str, label: str) -> None:
        message = (
            f"Conflicting labels for record pair {records!r}: "
            f"{previous!r} was already recorded and is now {label!r}. "
            "The change is kept in the training session history."
        )
        logger.warning(message)
        warnings.warn(message, session_io.SessionConflictWarning)

    def _pending_examples(self) -> TrainingExamples:
        """Return a pair that was shown but not answered, if there is one.

        Pairs restored from disk are offered before any new pair, so a pair
        that was on screen when a previous process stopped is shown again.
        Within one process each pending pair is offered once; if the caller
        never labels it, it stays pending on disk for the next resume.
        """
        if self.session is None:
            return []

        for entry in list(self.session.pending):
            key = entry.get("pair")
            if key in self._promoted:
                continue
            records = self._records_for_event(entry)
            if records is None:
                warnings.warn(
                    "An unlabeled pair from the labeling session could not be "
                    "matched to records in the current input data and was "
                    f"dropped: {key!r}.",
                    session_io.SessionDataWarning,
                )
                self.session.remove_pending(key)
                self.session.save()
                continue
            self._promoted.add(key)
            return [records]  # type: ignore[list-item]
        return []

    def _pop_fresh_pair(self, session: session_io.LabelingSession) -> TrainingExample:
        """Pop a candidate we have not already recorded in the session."""
        if self.active_learner is None:
            raise IndexError("No more unlabeled examples to label")

        while True:
            pair = self.active_learner.pop()
            records = tuple(pair)
            key = self._key_for_records(records)

            if key in session.latest or session.find_pending(key) is not None:
                continue

            session.add_pending(key, self._ids_for_records(records), records)
            session.save()
            # Mark it as offered within this process so the next call draws a
            # new pair instead of re-showing the one just handed out. It stays
            # pending on disk until it is labeled.
            self._promoted.add(key)
            return pair

    def cleanup_training(self) -> None:  # pragma: no cover
        """
        Clean up data we used for training. Free up memory.
        """
        del self.training_pairs
        del self.active_learner
        self.session = None
        self._resolver = None
        self._session_data = None
        self._training_index = None

    def _read_training(self, training_file: TextIO) -> None:
        """
        Read training from previously built training data file object

        Args:
            training_file: file object containing the training data
        """

        logger.info("reading training from file")
        training_pairs = serializer.read_training(training_file)
        self._mark_pairs_direct(training_pairs)
        self._rebuild_training_index()

    def train(
        self, recall: float = 1.00, index_predicates: bool = True
    ) -> None:  # pragma: no cover
        """
        Learn final pairwise classifier and fingerprinting rules. Requires that
        adequate training data has been already been provided.

        Args:
            recall: The proportion of true dupe pairs in our
                    training data that that the learned fingerprinting
                    rules must cover. If we lower the recall, there will
                    be pairs of true dupes that we will never
                    directly compare.

                    recall should be a float between 0.0 and 1.0.

            index_predicates: Should dedupe consider predicates
                              that rely upon indexing the
                              data. Index predicates can be slower
                              and take substantial memory. Without
                              index predicates, you may get lower
                              recall when true-dupes are not blocked
                              together.
        """
        assert (
            self.active_learner is not None
        ), "Please initialize with the prepare_training method"

        examples, y = flatten_training(self.training_pairs)
        self.classifier.fit(self.data_model.distances(examples), y)

        self.predicates = self.active_learner.learn_predicates(recall, index_predicates)
        self._fingerprinter = blocking.Fingerprinter(self.predicates)
        self.fingerprinter.reset_indices()

    def write_training(self, file_obj: TextIO) -> None:  # pragma: no cover
        """
        Write a JSON file that contains labeled examples

        Args:
            file_obj: file object to write training data to

        Examples:
            >>> with open('training.json', 'w') as f:
            >>>     matcher.write_training(f)
        """
        serializer.write_training(self.training_pairs, file_obj)

    def write_settings(self, file_obj: BinaryIO) -> None:  # pragma: no cover
        """
        Write a settings file containing the
        data model and predicates to a file object

        Args:
            file_obj: file object to write settings data into

        Examples:
            >>> with open('learned_settings', 'wb') as f:
            >>>     matcher.write_settings(f)
        """

        pickle.dump(self.data_model, file_obj)
        pickle.dump(self.classifier, file_obj)
        pickle.dump(self.predicates, file_obj)

    def uncertain_pairs(
        self, session_file: SessionFile | None = None
    ) -> TrainingExamples:
        """
         Returns a list of pairs of records from the sample of record pairs
         tuples that Dedupe is most curious to have labeled.

         This method is mainly useful for building a user interface for training
         a matching model.

        Args:
            session_file: Optional path or writable text file object used to
                remember the labeling session. When supplied, the pairs that
                are shown to the user are recorded before they are returned, so
                an interrupted labeling session can be resumed with
                :meth:`~resume_session` or by passing the same file to
                :meth:`~mark_pairs`. Pairs that were already shown but not
                answered are returned first.

        Examples:
            >>> pair = matcher.uncertain_pairs()
            >>> print(pair)
            [({'name' : 'Georgie Porgie'}, {'name' : 'Georgette Porgette'})]

            >>> pair = matcher.uncertain_pairs(session_file='labeling.json')
        """
        assert (
            self.active_learner is not None
        ), "Please initialize with the prepare_training method"

        if session_file is None:
            return [self.active_learner.pop()]

        session = self._attach_session(session_file)
        assert session is not None

        pending = self._pending_examples()
        if pending:
            return pending

        return [self._pop_fresh_pair(session)]

    def resume_session(self, session_file: SessionFile) -> int:
        """
        Resume a previously saved labeling session.

        Pairs that were already labeled are loaded into the training data, so
        they are available to :meth:`~train`, and pairs that were shown but not
        yet labeled become available again through :meth:`~uncertain_pairs`.

        Args:
            session_file: Path or writable text file object of a session
                previously created by :meth:`~uncertain_pairs`.

        Returns:
            The number of labeled record pairs restored from the session.

        Raises:
            :class:`~dedupe.session.SessionMismatchError`: if the session was
                created for a different kind of matcher or against different
                input data.
            :class:`~dedupe.session.SessionFileLoadingException`: if the file
                is not a valid labeling session.

        Examples:
            >>> matcher.prepare_training(data)
            >>> restored = matcher.resume_session('labeling.json')
            >>> for pair in matcher.uncertain_pairs(session_file='labeling.json'):
            >>>     ...
        """
        assert (
            self.active_learner is not None
        ), "Please initialize with the prepare_training method"

        session = self._attach_session(session_file)
        assert session is not None
        return len(session.resolved())

    def mark_pairs(
        self,
        labeled_pairs: TrainingData,
        session_file: SessionFile | None = None,
    ) -> None:
        """
        Add users labeled pairs of records to training data and update the
        matching model

        This method is useful for building a user interface for training a
        matching model or for adding training data from an existing source.

        Args:
            labeled_pairs: A dictionary with two keys, `match` and `distinct`
                           the values are lists that can contain pairs of
                           records

            session_file: Optional path or writable text file object of a
                          labeling session. When supplied, the labels are
                          recorded in the session so that they survive an
                          interruption and can be applied again on resume.
                          Repeated submissions are idempotent; relabeling is
                          recorded, and a contradictory label raises a
                          :class:`~dedupe.session.SessionConflictWarning`
                          instead of silently replacing the earlier label.

        Examples:
            >>> labeled_examples = {
            >>>     "match": [],
            >>>     "distinct": [
            >>>         (
            >>>             {"name": "Georgie Porgie"},
            >>>             {"name": "Georgette Porgette"},
            >>>         )
            >>>     ],
            >>> }
            >>> matcher.mark_pairs(labeled_examples)

        .. note::
           `mark_pairs()` is primarily designed to be used with
           :func:`~uncertain_pairs` to incrementally build a training
           set.

           If you have existing training data, you should likely
           format the data into the right form and supply the training
           data to the :func:`~prepare_training` method with the
           `training_file` argument.

           If that is not possible or desirable, you can use
           `mark_pairs()` to train a linker with existing data.
           However, you must ensure that every record that
           appears in the `labeled_pairs` argument appears in either
           the data or training file supplied to the
           :func:`~prepare_training` method.
        """
        self._checkTrainingPairs(labeled_pairs)

        session = (
            self._attach_session(session_file) if session_file is not None else None
        )

        if session is None:
            self._mark_pairs_direct(labeled_pairs)
        else:
            self._mark_pairs_session(labeled_pairs, session)

        if session is not None and session.dirty:
            session.save()

    def _mark_pairs_direct(self, labeled_pairs: TrainingData) -> None:
        self.training_pairs["match"].extend(labeled_pairs["match"])
        self.training_pairs["distinct"].extend(labeled_pairs["distinct"])

        if self._training_index is not None:
            # Keep conflict detection accurate if a session is attached but
            # this call did not opt into it.
            self._rebuild_training_index()

        if self.active_learner:
            examples, y = flatten_training(labeled_pairs)

            try:
                self.active_learner.mark(examples, y)
            except dedupe.predicates.NoIndexError as e:
                raise UserWarning(
                    "The record\n"
                    f"{e.failing_record}\n"
                    "is not known to to the active learner. "
                    "Make sure all `labeled_pairs` "
                    "are in the data or training file "
                    "of the `prepare_training()` method"
                )

    def _mark_pairs_session(
        self, labeled_pairs: TrainingData, session: session_io.LabelingSession
    ) -> None:
        # A pair submitted under both `match` and `distinct` in the same call
        # means "unsure" (see `dedupe.convenience._mark_pair`). Treat it as a
        # single event rather than a spurious conflict.
        labels_by_key: dict[str, set[str]] = {}
        records_by_key: dict[str, tuple] = {}
        order: list[str] = []

        for label in ("match", "distinct"):
            for pair in labeled_pairs[label]:
                records = tuple(pair)
                key = self._key_for_records(records)
                if key not in labels_by_key:
                    labels_by_key[key] = set()
                    records_by_key[key] = records
                    order.append(key)
                labels_by_key[key].add(label)

        changed = False
        for key in order:
            records = records_by_key[key]
            labels = labels_by_key[key]
            label = "unsure" if len(labels) > 1 else next(iter(labels))

            known_to_session = key in session.latest
            outcome = session.record_label(
                key, self._ids_for_records(records), label, records
            )
            if outcome == "duplicate":
                continue

            # record_label warns when it contradicts the session's own
            # history. If the pair was only known from a training file, that
            # contradiction is surfaced here instead, so nothing is replaced
            # silently.
            if self._upsert_training(
                records, label, warn_conflict=not known_to_session
            ):
                changed = True

        if changed and self.active_learner is not None:
            self._refit_active_learner()

    def _checkTrainingPairs(self, labeled_pairs: TrainingData) -> None:
        try:
            labeled_pairs.items()
            labeled_pairs["match"]
            labeled_pairs["distinct"]
        except (AttributeError, KeyError):
            raise ValueError(
                "labeled_pairs must be a dictionary with keys " '"distinct" and "match"'
            )

        if labeled_pairs["match"]:
            pair = labeled_pairs["match"][0]
            self._checkRecordPair(pair)

        if labeled_pairs["distinct"]:
            pair = labeled_pairs["distinct"][0]
            self._checkRecordPair(pair)

        if not labeled_pairs["distinct"] and not labeled_pairs["match"]:
            warnings.warn("Didn't return any labeled record pairs")

    def _checkRecordPair(self, record_pair: TrainingExample) -> None:
        try:
            a, b = record_pair
        except ValueError:
            raise ValueError(
                "The elements of data_sample must be pairs " "of record_pairs"
            )
        try:
            record_pair[0].keys() and record_pair[1].keys()
        except AttributeError:
            raise ValueError(
                "A pair of record_pairs must be made up of two " "dictionaries "
            )

        self.data_model.check(record_pair[0])
        self.data_model.check(record_pair[1])


class StaticDedupe(StaticMatching, DedupeMatching):
    """
    Class for deduplication using saved settings. If you have already
    trained a :class:`Dedupe` object and saved the settings, you can
    load the saved settings with StaticDedupe.
    """


class Dedupe(ActiveMatching, DedupeMatching):
    """
    Class for active learning deduplication. Use deduplication when you have
    data that can contain multiple records that can all refer to the same
    entity.
    """

    def prepare_training(
        self,
        data: Data,
        training_file: TextIO | None = None,
        sample_size: int = 1500,
        blocked_proportion: float = 0.9,
    ) -> None:
        """
        Initialize the active learner with your data and, optionally,
        existing training data.

        Sets up the learner.

        Args:
            data: Dictionary of records, where the keys are
                  record_ids and the values are dictionaries
                  with the keys being field names
            training_file: file object containing training data
            sample_size: Size of the sample to draw
            blocked_proportion: The proportion of record pairs to be sampled from similar records, as opposed to randomly selected pairs.

        Examples:
            >>> matcher.prepare_training(data_d, 150000, .5)

            >>> with open('training_file.json') as f:
            >>>     matcher.prepare_training(data_d, training_file=f)
        """
        self._checkData(data)

        # Reset active learner
        self.active_learner = None
        self._reset_session()
        self._set_session_input("dedupe", [data])

        if training_file:
            self._read_training(training_file)

        # We need the active learner to know about all our
        # existing training data, so add them to data dictionary
        examples, y = flatten_training(self.training_pairs)

        self.active_learner = labeler.DedupeDisagreementLearner(
            self.data_model.predicates,
            self.data_model.distances,
            data,
            index_include=examples,
        )

        self.active_learner.mark(examples, y)

    def _checkData(self, data: Data) -> None:
        if len(data) == 0:
            raise ValueError("Dictionary of records is empty.")

        self.data_model.check(next(iter(data.values())))


class Link(ActiveMatching):
    """
    Mixin Class for Active Learning Record Linkage
    """

    def prepare_training(
        self,
        data_1: Data,
        data_2: Data,
        training_file: TextIO | None = None,
        sample_size: int = 1500,
        blocked_proportion: float = 0.9,
    ) -> None:
        """
        Initialize the active learner with your data and, optionally,
        existing training data.

        Args:
            data_1: Dictionary of records from first dataset, where the
                    keys are record_ids and the values are dictionaries
                    with the keys being field names
            data_2: Dictionary of records from second dataset, same
                    form as data_1
            training_file: file object containing training data

            sample_size: The size of the sample to draw.

            blocked_proportion: The proportion of record pairs to
                                be sampled from similar records,
                                as opposed to randomly selected
                                pairs.

        Examples:
            >>> matcher.prepare_training(data_1, data_2, 150000)

            or

            >>> with open('training_file.json') as f:
            >>>     matcher.prepare_training(data_1, data_2, training_file=f)
        """
        self._checkData(data_1, data_2)

        # Reset active learner
        self.active_learner = None
        self._reset_session()
        self._set_session_input("link", [data_1, data_2])

        if training_file:
            self._read_training(training_file)

        # We need the active learner to know about all our
        # existing training data, so add them to data dictionaries
        examples, y = flatten_training(self.training_pairs)

        self.active_learner = labeler.RecordLinkDisagreementLearner(
            self.data_model.predicates,
            self.data_model.distances,
            data_1,
            data_2,
            index_include=examples,
        )

        self.active_learner.mark(examples, y)

    def _checkData(self, data_1: Data, data_2: Data) -> None:
        if len(data_1) == 0:
            raise ValueError("Dictionary of records from first dataset is empty.")
        elif len(data_2) == 0:
            raise ValueError("Dictionary of records from second dataset is empty.")

        self.data_model.check(next(iter(data_1.values())))
        self.data_model.check(next(iter(data_2.values())))


class RecordLink(Link, RecordLinkMatching):
    """
    Class for active learning record linkage.

    Use RecordLinkMatching when you have two datasets that you want to
    join.
    """


class StaticRecordLink(StaticMatching, RecordLinkMatching):
    """
    Class for record linkage using saved settings. If you have already
    trained a RecordLink instance, you can load the saved settings with
    StaticRecordLink.
    """


class Gazetteer(Link, GazetteerMatching):
    """
    Class for active learning gazetteer matching.

    Gazetteer matching is for matching a messy data set against a
    'canonical dataset'.  This class is useful for such tasks as matching messy
    addresses against a clean list
    """


class StaticGazetteer(StaticMatching, GazetteerMatching):
    """
    Class for gazetter matching using saved settings.

    If you have already trained a :class:`Gazetteer` instance, you can
    load the saved settings with StaticGazetteer.
    """


class EmptyTrainingException(Exception):
    pass


class SettingsFileLoadingException(Exception):
    pass


def flatten_training(
    training_pairs: TrainingData,
) -> tuple[TrainingExamples, LabelsLike]:
    examples: TrainingExamples = []
    y = []

    for label in ("match", "distinct"):
        label = cast(Literal["match", "distinct"], label)

        pairs = training_pairs[label]
        examples.extend(pairs)
        encoded_y = 1 if label == "match" else 0
        y.extend([encoded_y] * len(pairs))

    return examples, numpy.array(y)


def _cleanup_scores(arr: Scores) -> None:
    try:
        mmap_file = arr.filename  # type: ignore
    except AttributeError:
        pass
    else:
        arr._mmap.close()  # type: ignore # Unmap file to prevent PermissionError when deleting temp file
        del arr
        if mmap_file:
            os.remove(mmap_file)
