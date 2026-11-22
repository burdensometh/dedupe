from dedupe.api import (  # noqa: F401
    Dedupe,
    Gazetteer,
    RecordLink,
    StaticDedupe,
    StaticGazetteer,
    StaticRecordLink,
)
from dedupe.canonical import CanonicalRecord, FieldRule  # noqa: F401
from dedupe.convenience import (  # noqa: F401
    canonicalize,
    console_label,
    training_data_dedupe,
    training_data_link,
)
from dedupe.serializer import read_training, write_training  # noqa: F401

__all__ = [
    "CanonicalRecord",
    "Dedupe",
    "FieldRule",
    "Gazetteer",
    "RecordLink",
    "StaticDedupe",
    "StaticGazetteer",
    "StaticRecordLink",
    "canonicalize",
    "console_label",
    "training_data_dedupe",
    "training_data_link",
    "read_training",
    "write_training",
]
