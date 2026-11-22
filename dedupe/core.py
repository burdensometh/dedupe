#!/usr/bin/python
from __future__ import annotations

import collections
import functools
import itertools
import multiprocessing
import multiprocessing.dummy
import os
import queue
import tempfile
from typing import TYPE_CHECKING, overload

import numpy

from dedupe.backport import RLock
from dedupe.ids import maybe_encode_record_id, pair_needs_encoding

if TYPE_CHECKING:
    from typing import Any, Generator, Iterable, Iterator, Literal, Sequence, Union

    from dedupe._typing import (
        Block,
        Blocks,
        Classifier,
        ClosableJoinable,
        Data,
        FeaturizerFunction,
        MapLike,
        RecordID,
        RecordIDDType,
        RecordPairs,
        Scores,
    )
    _Queue = Union[multiprocessing.dummy.Queue, multiprocessing.Queue]


class BlockingError(Exception):
    pass


class ScoreDupes:
    def __init__(
        self,
        featurizer: FeaturizerFunction,
        classifier: Classifier,
        records_queue: _Queue,
        exception_queue: _Queue,
        score_file_path: str,
        dtype: numpy.dtype,
        id_dtype: RecordIDDType,
        encode_ids: bool,
        offset,
    ):
        self.featurizer = featurizer
        self.classifier = classifier
        self.records_queue = records_queue
        self.exception_queue = exception_queue
        self.score_file_path = score_file_path
        self.dtype = dtype
        self.id_dtype = id_dtype
        self.encode_ids = encode_ids
        self.offset = offset

    def __call__(self) -> None:
        while True:
            record_pairs: RecordPairs | None = self.records_queue.get()
            if record_pairs is None:
                break

            try:
                self.fieldDistance(record_pairs)
            except Exception as e:
                self.exception_queue.put(e)
                raise

    def fieldDistance(self, record_pairs: RecordPairs) -> None:
        record_ids, records = zip(*(zip(*record_pair) for record_pair in record_pairs))
        if not records:
            return

        features = self.featurizer(records)
        scores = self.classifier.predict_proba(features)[:, -1]

        mask = scores > 0
        if not mask.any():
            return
        scores = scores[mask]
        if self.encode_ids:
            # Heterogeneous or composite (source, original_id) ids cannot
            # share a numpy column in their raw form, so they are rendered
            # as stable, lossless text before entering the score array.
            record_id_array = numpy.array(
                [
                    [
                        maybe_encode_record_id(record_id, True)
                        for record_id in pair
                    ]
                    for pair, keep in zip(record_ids, mask)
                    if keep
                ],
                dtype=self.id_dtype,
            )
        else:
            record_id_array = numpy.array(record_ids)[mask]

        with self.offset.get_lock():
            fp: Scores
            fp = numpy.memmap(
                self.score_file_path,
                dtype=self.dtype,
                offset=self.offset.value,
                shape=(len(record_id_array),),
            )
            fp["pairs"] = record_id_array
            fp["score"] = scores
            fp.flush()

            self.offset.value += len(record_id_array) * self.dtype.itemsize


def scoreDuplicates(
    record_pairs: RecordPairs,
    featurizer: FeaturizerFunction,
    classifier: Classifier,
    num_cores: int = 1,
) -> Scores:
    if num_cores < 2:
        from multiprocessing.dummy import Process, Queue
    else:
        from .backport import Process, Queue  # type: ignore

    first, record_pairs = peek(record_pairs)
    if first is None:
        raise BlockingError(
            "No records have been blocked together. "
            "Is the data you are trying to match like "
            "the data you trained on? If so, try adding "
            "more training data."
        )

    record_pairs_queue: _Queue = Queue(2)
    exception_queue: _Queue = Queue()
    scored_pairs_file, score_file_path = tempfile.mkstemp()
    os.close(scored_pairs_file)

    # explicitly defining the lock from the "spawn context" seems to
    # be necessary for python 3.7 on mac os.
    offset = multiprocessing.Value("Q", 0, lock=RLock())

    first_id_pair = (first[0][0], first[1][0])
    id_dtype = sniff_id_type([first_id_pair])
    # Composite ids, and any batch mixing integer with string ids, need to
    # be rendered as text before entering the score array.
    encode_ids = pair_needs_encoding([first_id_pair])
    dtype = numpy.dtype([("pairs", id_dtype, 2), ("score", "f4")])

    n_map_processes = max(num_cores, 1)
    score_records = ScoreDupes(
        featurizer,
        classifier,
        record_pairs_queue,
        exception_queue,
        score_file_path,
        dtype,
        id_dtype,
        encode_ids,
        offset,
    )
    map_processes = [Process(target=score_records) for _ in range(n_map_processes)]

    for process in map_processes:
        process.start()

    fillQueue(record_pairs_queue, record_pairs, n_map_processes)

    for process in map_processes:
        process.join()

    try:
        exc = exception_queue.get_nowait()
    except queue.Empty:
        pass
    else:
        raise ChildProcessError from exc

    scored_pairs: Scores

    if offset.value:  # type: ignore
        scored_pairs = numpy.memmap(score_file_path, dtype=dtype)
    else:
        scored_pairs = numpy.array([], dtype=dtype)

    return scored_pairs


def fillQueue(
    queue: _Queue, iterable: Iterable[Any], stop_signals: int, chunk_size: int = 20000
) -> None:
    iterable = iter(iterable)

    while True:
        chunk = tuple(itertools.islice(iterable, chunk_size))
        if chunk:
            queue.put(chunk)
            del chunk

        else:
            # put poison pills in queue to tell scorers that they are
            # done
            for _ in range(stop_signals):
                queue.put(None)
            break


class ScoreGazette:
    def __init__(self, featurizer: FeaturizerFunction, classifier: Classifier):
        self.featurizer = featurizer
        self.classifier = classifier

    def __call__(self, block: Block) -> Scores:
        record_ids, records = zip(*(zip(*each) for each in block))

        features = self.featurizer(records)
        scores = self.classifier.predict_proba(features)[:, -1]

        id_dtype = sniff_id_type(record_ids)
        encode_ids = pair_needs_encoding(record_ids)
        if encode_ids:
            # Composite ids and mixed int/str batches are rendered as
            # stable, lossless text before entering the score array.
            ids = numpy.array(
                [
                    [maybe_encode_record_id(record_id, True) for record_id in pair]
                    for pair in record_ids
                ],
                dtype=id_dtype,
            )
        else:
            ids = numpy.array(record_ids, dtype=id_dtype)

        mask = scores > 0
        scores = scores[mask]
        ids = ids[mask]

        dtype = numpy.dtype([("pairs", id_dtype, 2), ("score", "f4")])
        scored_pairs: Scores = numpy.empty(shape=len(scores), dtype=dtype)
        scored_pairs["pairs"] = ids
        scored_pairs["score"] = scores

        return scored_pairs


def scoreGazette(
    record_pairs: Blocks,
    featurizer: FeaturizerFunction,
    classifier: Classifier,
    num_cores: int = 1,
) -> Generator[Scores]:
    first, record_pairs = peek(record_pairs)
    if first is None:
        return  # terminate iteration

    imap, pool = appropriate_imap(num_cores)

    score_records = ScoreGazette(featurizer, classifier)

    yield from imap(score_records, record_pairs)

    # The underlying processes in the pool should terminate when the
    # pool is garbage collected, but sometimes it takes a while
    # before GC, so do it explicitly here
    pool.close()
    pool.join()


class MockPool:
    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


def appropriate_imap(num_cores: int) -> tuple[MapLike, ClosableJoinable]:
    if num_cores < 2:
        imap: MapLike = map

        # in order to make it simpler to cleanup a pool of processes
        # always return something that we can close and join

        pool: ClosableJoinable = MockPool()
    else:
        from .backport import Pool

        pool = Pool(processes=num_cores)
        imap = functools.partial(pool.imap_unordered, chunksize=20000)

    return imap, pool


def peek(seq: Iterator[Any]) -> tuple[Any | None, Iterator[Any]]:
    try:
        first = next(seq)
    except TypeError as e:
        if "not an iterator" not in str(e):
            raise
        try:
            seq = iter(seq)
            first = next(seq)
        except StopIteration:
            return None, iter(seq)
    except StopIteration:
        return None, iter(seq)

    return first, itertools.chain([first], seq)


def isIndexed(data: Data, offset: int) -> bool:
    return all(i in data for i in range(offset, offset + len(data)))


def index(data: Data, offset: int = 0) -> Data:
    if isIndexed(data, offset):
        return data
    else:
        data = dict(zip(itertools.count(offset), data.values()))
        return data


def Enumerator(start: int = 0) -> collections.defaultdict[Any, int]:
    return collections.defaultdict(itertools.count(start).__next__, ())


@overload
def sniff_id_type(ids: Sequence[tuple[int, int]]) -> type[int]: ...


@overload
def sniff_id_type(
    ids: Sequence[tuple[object, object]]
) -> tuple[type[str], Literal[256]]: ...


def sniff_id_type(ids: Sequence[tuple[RecordID, RecordID]]) -> RecordIDDType:
    """Choose the numpy dtype for the record-id column.

    ``ids`` is a sequence of ``(id_a, id_b)`` id pairs.
    """
    id_a, id_b = ids[0]
    python_type = type(id_a)
    dtype: RecordIDDType
    if python_type is int:
        # A batch that mixes integers with strings, or carries composite
        # ids, is encoded as text even when the first id is an integer.
        if pair_needs_encoding([(id_a, id_b)]):
            dtype = (str, 256)
        else:
            int(id_a)  # make sure we can cast to int
            dtype = int
    elif python_type in (str, tuple):
        # Strings and composite (source, original_id) ids are encoded as
        # text before they enter the score array.
        dtype = (str, 256)
    else:
        raise ValueError("Invalid type for record id")

    return dtype


def sqlite_id_type(
    example_id: RecordID, encode_ids: bool
) -> Literal["text", "integer"]:
    """SQLite column type for the record id column.

    When ids are encoded (a heterogeneous or composite-keyed collection),
    every stored id is text, so the column must be ``text``. Otherwise the
    collection is homogeneous: plain integer ids use an ``integer`` column
    and plain string ids use ``text``.
    """
    if encode_ids or isinstance(example_id, (str, tuple)):
        return "text"
    if isinstance(example_id, bool) or not isinstance(example_id, int):
        raise ValueError("Invalid type for record id")
    return "integer"


def unique(seq: Iterable[Any]) -> list[Any]:
    """Return the unique elements of a collection even if those elements are
    unhashable and unsortable, like dicts and sets"""
    cleaned: list[Any] = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned
