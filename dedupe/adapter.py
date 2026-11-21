"""
Persistable input field adapters.

Two data sources frequently disagree on what their fields are called
(``zip`` versus ``zipcode``, ``street`` versus ``address``) and on how
values are represented (zip codes with or without leading zeros,
upper- versus lower-case street names, numeric ids stored as integers on
one side and strings on the other).

An :class:`SourceAdapter` describes, declaratively, how the records of one
data source are mapped onto the *common* field names used by the data
model. The mapping is determined entirely at training time, is saved
inside the settings file, and is re-applied unchanged at matching time.
No rules are ever inferred from the data while matching: a renamed
source field or a value with an incompatible type raises an
:class:`AdaptationError` instead.

Adapted records carry exactly the common fields the data model expects
plus a copy of the original record (under :data:`ORIGINAL_RECORD_KEY`),
so the raw field values can always be recovered from a link result while
the record ids are left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence, Union, cast

if TYPE_CHECKING:
    from dedupe._typing import Data, RecordDict, RecordDictPair
    from dedupe.datamodel import DataModel

CaseOption = Literal["upper", "lower", "none"]
TargetType = Literal["str", "int", "float"]

#: Key under which the unmodified source record is stored on each
#: adapted record, so the original values remain reachable from the
#: output of blocking and matching.
ORIGINAL_RECORD_KEY = "_dedupe_original"

#: Values that count as missing on every adapter unless an explicit,
#: narrower set of null values is configured.
DEFAULT_NULL_VALUES: frozenset[Any] = frozenset({None, ""})

_STRING_TYPES = frozenset({"String", "ShortString", "Text", "Exact", "Categorical"})
_NUMERIC_TYPES = frozenset({"Price"})
# Types whose input semantics are outside the scope of the field
# adapter; they must be supplied directly on the common field.
_UNSUPPORTED_TYPES = frozenset({"Set", "LatLong"})


class AdaptationError(ValueError):
    """
    Raised when a source record cannot be converted with the configured
    adapter. This happens when a source field is missing, when a value
    does not have the configured type (for example an integer-typed
    field holding ``"abc"``), or when the adapter itself is invalid for
    the data model.
    """


@dataclass(frozen=True)
class FieldSpec:
    """
    Declarative conversion rule for a single common field, as found on
    one data source.

    Args:
        source_field: Name of the field on the raw source records.
        target_field: Name of the common field used by the data model.
        type: The single field type semantics applied in both stages.
            ``"str"`` always produces a (possibly padded, possibly
            case-folded) string, ``"int"`` an ``int`` and ``"float"``
            a ``float``. Values that cannot be converted are reported
            with :class:`AdaptationError`, never silently coerced.
        case: For ``type="str"``, the single case semantics applied in
            both stages. ``"none"`` leaves the case untouched.
        pad: For ``type="str"``, zero-pad digit strings to this width.
            This is how leading zeros are given one fixed, explicit
            representation on both sides, e.g. ``pad=5`` turns both
            ``"02138"`` and ``2138`` into ``"02138"``.
        null_values: The values treated as missing. Every configured
            null becomes the same single null marker, ``None``, in the
            adapted record, which is the only null semantics used by
            blocking and distance calculation.
    """

    source_field: str
    target_field: str
    type: TargetType = "str"
    case: CaseOption = "none"
    pad: int | None = None
    null_values: frozenset[Any] = field(default_factory=lambda: DEFAULT_NULL_VALUES)

    def __post_init__(self) -> None:
        if not isinstance(self.source_field, str) or not self.source_field:
            raise AdaptationError(
                "source_field must be a non-empty string, " f"got {self.source_field!r}"
            )
        if not isinstance(self.target_field, str) or not self.target_field:
            raise AdaptationError(
                "target_field must be a non-empty string, " f"got {self.target_field!r}"
            )
        if self.type not in ("str", "int", "float"):
            raise AdaptationError(
                f"Unsupported field type {self.type!r}; "
                "expected one of 'str', 'int', 'float'"
            )
        if self.case not in ("upper", "lower", "none"):
            raise AdaptationError(
                f"Unsupported case option {self.case!r}; "
                "expected one of 'upper', 'lower', 'none'"
            )
        if self.type != "str":
            if self.case != "none":
                raise AdaptationError(
                    f"The case option only applies to string fields; "
                    f"{self.target_field!r} is configured as {self.type!r}"
                )
            if self.pad is not None:
                raise AdaptationError(
                    "The pad option only applies to string fields; "
                    f"{self.target_field!r} is configured as {self.type!r}"
                )
        if self.pad is not None and (
            not isinstance(self.pad, int) or isinstance(self.pad, bool) or self.pad <= 0
        ):
            raise AdaptationError(f"pad must be a positive integer, got {self.pad!r}")
        if not isinstance(self.null_values, frozenset):
            raise AdaptationError(
                "null_values must be a frozenset so the adapter is "
                "immutable and persistable"
            )

    def convert(self, value: Any) -> Any:
        """Apply one fixed null/case/type semantics to a raw value."""
        if value in self.null_values:
            return None

        try:
            converted: Any
            if self.type == "str":
                converted = self._convert_str(value)
            elif self.type == "int":
                converted = self._convert_int(value)
            else:
                converted = self._convert_float(value)
        except AdaptationError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise AdaptationError(
                f"Could not convert value {value!r} from source field "
                f"{self.source_field!r} to common field "
                f"{self.target_field!r} ({self.type}): {exc}"
            ) from exc

        # A value may be a configured null only after conversion (e.g.
        # whitespace strings), keep the single null marker semantics.
        if converted in self.null_values:
            return None
        return converted

    def _convert_str(self, value: Any) -> str:
        if isinstance(value, bool):
            raise AdaptationError(
                f"Refusing to treat boolean {value!r} as a string for "
                f"common field {self.target_field!r}"
            )
        if isinstance(value, int):
            # Integer input into a declared string field is the usual
            # reason leading zeros were lost upstream (e.g. zip codes
            # stored as 2138 instead of "02138"). The conversion is an
            # explicit, configured one, so padding can restore them.
            value = str(value)
        elif not isinstance(value, str):
            raise AdaptationError(
                f"Common field {self.target_field!r} is configured as a "
                f"string but the source value is {value!r} "
                f"({type(value).__name__}); configure the adapter with "
                "type='int' or type='float' if the source holds numbers"
            )

        if self.case == "upper":
            value = value.upper()
        elif self.case == "lower":
            value = value.lower()

        if self.pad is not None and value.isdigit():
            value = value.zfill(self.pad)

        return value

    def _convert_int(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdaptationError(
                f"Common field {self.target_field!r} is configured as an "
                f"integer but the source value is {value!r} "
                f"({type(value).__name__}); numeric strings are not "
                "silently parsed"
            )
        return value

    def _convert_float(self, value: Any) -> float | int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdaptationError(
                f"Common field {self.target_field!r} is configured as a "
                f"float but the source value is {value!r} "
                f"({type(value).__name__}); numeric strings are not "
                "silently parsed"
            )
        return value


@dataclass(frozen=True)
class SourceAdapter:
    """
    The complete, persistable field mapping for one data source.

    Args:
        fields: The :class:`FieldSpec` rules mapping this source's raw
            field names onto the common fields of the data model.
        name: Optional label used in error messages.
    """

    fields: tuple[FieldSpec, ...]
    name: str = "source"

    def __post_init__(self) -> None:
        if not self.fields:
            raise AdaptationError(f"Adapter {self.name!r} has no field mappings")
        source_fields = [spec.source_field for spec in self.fields]
        target_fields = [spec.target_field for spec in self.fields]
        if len(set(source_fields)) != len(source_fields):
            raise AdaptationError(
                f"Adapter {self.name!r} maps a source field more than "
                "once; each source field may feed only one common field"
            )
        if len(set(target_fields)) != len(target_fields):
            raise AdaptationError(
                f"Adapter {self.name!r} maps more than one source field "
                "onto the same common field"
            )
        object.__setattr__(self, "fields", tuple(self.fields))

    @property
    def target_fields(self) -> frozenset[str]:
        return frozenset(spec.target_field for spec in self.fields)

    @property
    def source_fields(self) -> frozenset[str]:
        return frozenset(spec.source_field for spec in self.fields)

    def transform(self, record: Mapping[str, Any]) -> RecordDict:
        """
        Convert one raw source record into a record on the common field
        names. The original record is preserved verbatim under
        :data:`ORIGINAL_RECORD_KEY`; record ids are handled by the
        caller and are never altered.

        Adapting a record that was already adapted is a no-op: training
        files written by an adapted matcher hold adapted records and may
        be read back by the same matcher, so applying the rules twice
        must neither fail nor mutate the stored original.
        """
        if ORIGINAL_RECORD_KEY in record:
            return dict(record)

        adapted: dict[str, Any] = {}
        for spec in self.fields:
            if spec.source_field not in record:
                raise AdaptationError(
                    f"Source field {spec.source_field!r} required by "
                    f"adapter {self.name!r} is missing from record "
                    f"{record!r}. A source field was renamed or the "
                    "adapter configuration no longer matches the data; "
                    "matching rules are not inferred at match time."
                )
            adapted[spec.target_field] = spec.convert(record[spec.source_field])

        adapted[ORIGINAL_RECORD_KEY] = dict(record)
        return adapted

    def transform_data(self, data: Data) -> Data:
        """Convert every record of a ``{record_id: record}`` mapping."""
        transformed = {
            record_id: self.transform(record) for record_id, record in data.items()
        }
        return cast("Data", transformed)


def original_record(adapted_record: Mapping[str, Any]) -> RecordDict:
    """
    Recover the raw, un-adapted source record stored on an adapted
    record. Raises if the record was not produced by an adapter.
    """
    try:
        return adapted_record[ORIGINAL_RECORD_KEY]
    except (KeyError, TypeError):
        raise AdaptationError(
            "This record carries no original source record; it was not "
            "produced by a SourceAdapter"
        )


def validate_adapters(
    adapters: Sequence[SourceAdapter], data_model: DataModel
) -> tuple[SourceAdapter, ...]:
    """
    Check that adapters fully and consistently describe the data model
    *before* any data is read. This is the only place field mapping
    rules are established: nothing is inferred while matching.
    """
    adapters = tuple(adapters)

    if not adapters:
        raise AdaptationError("At least one SourceAdapter is required")

    model_fields = {
        variable.field: variable.type for variable in data_model.field_variables
    }

    seen_semantics: dict[str, FieldSpec] = {}
    for adapter in adapters:
        if adapter.target_fields != frozenset(model_fields):
            missing = sorted(frozenset(model_fields) - adapter.target_fields)
            extra = sorted(adapter.target_fields - frozenset(model_fields))
            details = []
            if missing:
                details.append(f"missing mappings for {missing}")
            if extra:
                details.append(f"mappings for unknown fields {extra}")
            raise AdaptationError(
                f"Adapter {adapter.name!r} does not line up with the "
                "data model: " + "; ".join(details)
            )

        for spec in adapter.fields:
            variable_type = model_fields[spec.target_field]
            _check_type_compatibility(spec, variable_type)

            previous = seen_semantics.get(spec.target_field)
            if previous is not None and not _same_semantics(previous, spec):
                raise AdaptationError(
                    "All sources must apply the same null, case and "
                    f"type semantics to common field {spec.target_field!r}; "
                    f"{previous!r} disagrees with {spec!r}"
                )
            seen_semantics.setdefault(spec.target_field, spec)

    return adapters


def _same_semantics(left: FieldSpec, right: FieldSpec) -> bool:
    return (
        left.type == right.type
        and left.case == right.case
        and left.pad == right.pad
        and left.null_values == right.null_values
    )


def _check_type_compatibility(spec: FieldSpec, variable_type: str) -> None:
    if variable_type in _UNSUPPORTED_TYPES:
        raise AdaptationError(
            f"Common field {spec.target_field!r} has variable type "
            f"{variable_type!r}, which the field adapter does not "
            "support. Provide that field directly under its common name."
        )
    if variable_type in _STRING_TYPES and spec.type != "str":
        raise AdaptationError(
            f"Common field {spec.target_field!r} is a {variable_type} "
            f"variable and requires type='str', got {spec.type!r}"
        )
    if variable_type in _NUMERIC_TYPES and spec.type not in ("int", "float"):
        raise AdaptationError(
            f"Common field {spec.target_field!r} is a {variable_type} "
            "variable and requires type='int' or type='float', got "
            f"{spec.type!r}"
        )
    # Exists variables can sit on top of any field type.


def adapt_record_pairs(
    pairs: Iterable[RecordDictPair],
    adapters: Sequence[SourceAdapter] | None,
) -> list[RecordDictPair]:
    """
    Apply the per-source adapters to a sequence of record pairs. With a
    single adapter both sides of every pair use it; with two adapters
    the first applies to the left-hand source and the second to the
    right-hand source.
    """
    if not adapters:
        return list(pairs)
    adapted_pairs: list[RecordDictPair] = []
    for pair in pairs:
        record_1, record_2 = pair
        adapted_pairs.append(
            (
                adapters[0].transform(record_1),
                adapters[-1].transform(record_2),
            )
        )
    return adapted_pairs


Adapters = Union[SourceAdapter, Sequence[SourceAdapter], None]


def normalize_adapters(adapters: Adapters) -> tuple[SourceAdapter, ...] | None:
    """Accept one adapter or a sequence of them, as plain tuples."""
    if adapters is None:
        return None
    if isinstance(adapters, SourceAdapter):
        return (adapters,)
    adapters = tuple(adapters)
    if not adapters or not all(
        isinstance(adapter, SourceAdapter) for adapter in adapters
    ):
        raise AdaptationError(
            "adapters must be a SourceAdapter or a sequence of " "SourceAdapter objects"
        )
    return adapters
