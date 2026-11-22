from __future__ import annotations

import copyreg
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy

from dedupe._typing import FieldVariable
from dedupe.missing import (
    ABSENT,
    CONTRACT_VERSION,
    PAIR_FEATURE_WIDTH,
    MissingState,
    pair_state_index,
    value_state,
)
from dedupe.variables.interaction import InteractionType

if TYPE_CHECKING:
    from typing import Collection, Generator, Iterable, Sequence

    from dedupe._typing import (
        Comparator,
        InteractionVariable,
        RecordDict,
        RecordDictPair,
        Variable,
    )
    from dedupe.predicates import Predicate

_STATE_COUNT = len(MissingState)

# _PAIR_FEATURE_INDEX[low, high] gives the one-hot slot for an unordered pair
# of states, where ``low`` and ``high`` are the MissingState values ordered so
# that low <= high.
_PAIR_FEATURE_INDEX = numpy.empty((_STATE_COUNT, _STATE_COUNT), dtype=numpy.intp)
for _a in range(_STATE_COUNT):
    for _b in range(_STATE_COUNT):
        _PAIR_FEATURE_INDEX[_a, _b] = pair_state_index(
            MissingState(_a), MissingState(_b)
        )


class DataModel:
    # ``version`` describes the on-disk layout of the comparison features.
    # Version 3 replaced the single "both present" indicator that older
    # models used for every declared-missing column with the richer
    # comparison-feature contract from :mod:`dedupe.missing`.
    version = CONTRACT_VERSION

    def __init__(self, variable_definitions: Collection[Variable]):
        for item in variable_definitions:
            if isinstance(item, Mapping):
                raise ValueError(
                    "It looks like you are trying to use a variable definition "
                    "composed of dictionaries. dedupe 3.0 uses variable objects "
                    'directly. So instead of [{"field": "name", "type": "String"}] '
                    'we now do [dedupe.variables.String("name")].'
                )

        variable_definitions = list(variable_definitions)
        if not variable_definitions:
            raise ValueError("The variable definitions cannot be empty")
        if not any(variable.predicates for variable in variable_definitions):
            raise ValueError(
                "At least one of the variable types needs to be a type"
                "other than 'Custom'. 'Custom' types have no associated"
                "blocking rules"
            )

        # This is a protocol check, not a class inheritance check
        self.field_variables: list[FieldVariable] = [
            variable
            for variable in variable_definitions
            if isinstance(variable, FieldVariable)
        ]

        # we need to keep track of ordering of variables because in
        # order to calculate derived fields like interaction and missing
        # data fields.
        columns: list[Variable] = []
        for variable in self.field_variables:
            if len(variable) == 1:
                columns.append(variable)
            elif len(variable) > 1:
                assert hasattr(variable, "higher_vars")
                columns.extend(variable.higher_vars)

        self._derived_start = len(columns)

        # i'm not really satisfied with how we are dealing with interactions
        # here. seems like there should be a cleaner path, but i don't see it
        # today
        interaction_vars = interactions(variable_definitions, self.field_variables)
        columns += interaction_vars

        self._interaction_indices = interaction_indices(columns)

        # Each entry is the column whose per-record state stands for a whole
        # declared-missing unit: a field declared as possibly missing, or an
        # interaction that contains one. Each such unit contributes a one-hot
        # block over the unordered pair states (see dedupe.missing) instead of
        # the single binary "both present" feature that older models used. This
        # keeps "not provided", "provided but empty", and "comparable" apart,
        # and keeps all three apart from a genuine comparison of zero.
        self._missing_field_indices = self._build_missing_units(interaction_vars)

        self._object_version = self.version
        self._legacy_missing_contract = False
        self._setup_contract_metadata()

        self._len = (
            len(columns) + len(self._missing_field_indices) * PAIR_FEATURE_WIDTH
        )

    def __len__(self) -> int:
        return self._len

    def _build_missing_units(
        self, interaction_vars: Sequence[InteractionVariable]
    ) -> list[int]:
        """Locate the representative column of each declared-missing unit.

        A unit is either a field variable declared as possibly missing or an
        interaction that contains such a field. For a multi-column field (e.g.
        ``Categorical`` or ``Exists``) the representative column is its first
        column; every column of the unit carries the same per-record state. For
        an interaction it is the interaction's own column, whose state is
        derived from its constituent columns.
        """
        units: list[int] = []
        start = 0
        for variable in self.field_variables:
            if getattr(variable, "has_missing", False):
                units.append(start)
            start += len(variable)

        current_column = self._derived_start
        for variable in interaction_vars:
            if getattr(variable, "has_missing", False):
                units.append(current_column)
            current_column += 1

        return units

    def _setup_contract_metadata(self) -> None:
        # The record fields that the variable definition declares as possibly
        # missing. A record may legitimately omit these keys entirely.
        self._declared_missing_fields = {
            var.field
            for var in self.field_variables
            if getattr(var, "has_missing", False)
        }

    # Changing this from a property to just a normal attribute causes
    # pickling problems, because we are removing static methods from
    # their class context. This could be fixed by defining comparators
    # outside of classes in fieldclasses
    @property
    def _field_comparators(
        self,
    ) -> Generator[tuple[str, Comparator, int, int]]:
        start = 0
        stop = 0
        for var in self.field_variables:
            stop = start + len(var)
            comparator = cast("Comparator", var.comparator)
            yield (var.field, comparator, start, stop)
            start = stop

    @property
    def predicates(self) -> set[Predicate]:
        predicates = set()
        for var in self.field_variables:
            for predicate in var.predicates:
                predicates.add(predicate)
        return predicates

    def distances(
        self, record_pairs: Sequence[RecordDictPair]
    ) -> numpy.typing.NDArray[numpy.float64]:
        num_records = len(record_pairs)

        distances = numpy.empty((num_records, len(self)), "f4")

        if self._legacy_missing_contract:
            return self._base_distances(record_pairs, distances)

        if not self._missing_field_indices:
            # No field was declared as possibly missing, so there is no
            # comparison-feature block to build. All fields are required to be
            # present (see ``check``), so the base computation is enough.
            return self._base_distances(record_pairs, distances)

        n_columns = self._derived_start + len(self._interaction_indices)
        # How each side of each pair presents every column. These drive the
        # interaction states and then the missing-data comparison features.
        state_1 = numpy.empty((num_records, n_columns), dtype=numpy.int8)
        state_2 = numpy.empty((num_records, n_columns), dtype=numpy.int8)

        for i, (record_1, record_2) in enumerate(record_pairs):
            for field, compare, start, stop in self._field_comparators:
                if field in self._declared_missing_fields:
                    value_1 = record_1[field] if field in record_1 else ABSENT
                    value_2 = record_2[field] if field in record_2 else ABSENT

                    side_1 = value_state(value_1)
                    side_2 = value_state(value_2)
                    state_1[i, start:stop] = numpy.int8(side_1)
                    state_2[i, start:stop] = numpy.int8(side_2)

                    if (
                        side_1 is MissingState.COMPARABLE
                        and side_2 is MissingState.COMPARABLE
                    ):
                        distances[i, start:stop] = compare(value_1, value_2)
                    elif hasattr(compare, "missing"):
                        # Comparators that model presence themselves (e.g.
                        # Exists) still get to see the missing values.
                        distances[i, start:stop] = compare(
                            None if value_1 is ABSENT else value_1,
                            None if value_2 is ABSENT else value_2,
                        )
                    else:
                        distances[i, start:stop] = numpy.nan
                else:
                    # Fields that were not declared as possibly missing keep
                    # the historical behavior.
                    state_1[i, start:stop] = numpy.int8(MissingState.COMPARABLE)
                    state_2[i, start:stop] = numpy.int8(MissingState.COMPARABLE)

                    value_1 = record_1[field]
                    value_2 = record_2[field]
                    if value_1 is not None and value_2 is not None:
                        distances[i, start:stop] = compare(value_1, value_2)
                    elif hasattr(compare, "missing"):
                        distances[i, start:stop] = compare(value_1, value_2)
                    else:
                        distances[i, start:stop] = numpy.nan

        # Interaction columns derive both their value and their state from
        # their constituent columns, so an interaction over a declared missing
        # field follows the same contract. Doing this after the loop keeps the
        # result independent of row order.
        current_column = self._derived_start
        for indices in self._interaction_indices:
            distances[:, current_column] = numpy.prod(distances[:, indices], axis=1)
            state_1[:, current_column] = numpy.min(state_1[:, indices], axis=1)
            state_2[:, current_column] = numpy.min(state_2[:, indices], axis=1)
            current_column += 1

        is_missing = numpy.isnan(distances[:, :n_columns])
        distances[:, :n_columns][is_missing] = 0

        self._add_missing_features(distances, state_1, state_2, n_columns)

        return distances

    def _add_missing_features(
        self,
        distances: numpy.typing.NDArray[numpy.float64],
        state_1: numpy.typing.NDArray[numpy.int8],
        state_2: numpy.typing.NDArray[numpy.int8],
        start_column: int,
    ) -> None:
        current_column = start_column
        rows = numpy.arange(len(distances))
        for column_index in self._missing_field_indices:
            side_1 = numpy.asarray(state_1[:, column_index], dtype=numpy.intp)
            side_2 = numpy.asarray(state_2[:, column_index], dtype=numpy.intp)
            low = numpy.minimum(side_1, side_2)
            high = numpy.maximum(side_1, side_2)

            block = distances[:, current_column : current_column + PAIR_FEATURE_WIDTH]
            block[:] = 0
            block[rows, _PAIR_FEATURE_INDEX[low, high]] = 1

            current_column += PAIR_FEATURE_WIDTH

    def _base_distances(
        self,
        record_pairs: Sequence[RecordDictPair],
        distances: numpy.typing.NDArray[numpy.float64],
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Fill ``distances`` using the unambiguous, non-contract path.

        This is the behavior used for every field that is not declared as
        possibly missing, and also by legacy models for every field. ``None``
        is the missing marker; a missing comparison is zeroed out and, if a
        field carries a missing indicator, a single "both present" feature is
        emitted for it.
        """
        for i, (record_1, record_2) in enumerate(record_pairs):
            for field, compare, start, stop in self._field_comparators:
                if record_1[field] is not None and record_2[field] is not None:
                    distances[i, start:stop] = compare(record_1[field], record_2[field])
                elif hasattr(compare, "missing"):
                    distances[i, start:stop] = compare(record_1[field], record_2[field])
                else:
                    distances[i, start:stop] = numpy.nan

        return self._base_add_derived_distances(distances)

    def _base_add_derived_distances(
        self, distances: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        current_column = self._derived_start

        for indices in self._interaction_indices:
            distances[:, current_column] = numpy.prod(distances[:, indices], axis=1)
            current_column += 1

        is_missing = numpy.isnan(distances[:, :current_column])

        distances[:, :current_column][is_missing] = 0

        if self._missing_field_indices:
            distances[:, current_column:] = (
                1 - is_missing[:, self._missing_field_indices]
            )

        return distances

    def check(self, record: RecordDict) -> None:
        for field, _, _, _ in self._field_comparators:
            if field in record:
                continue
            if (
                not self._legacy_missing_contract
                and field in self._declared_missing_fields
            ):
                # "Not provided" is a legitimate way to present a field that
                # the variable definition declared as possibly missing. Models
                # using the legacy contract still require the key.
                continue
            raise ValueError(
                "Records do not line up with data model. "
                "The field '%s' is in data_model but not "
                "in a record" % field
            )

    def __getstate__(self):
        d = self.__dict__
        d["object_version"] = self._object_version
        return d

    def __setstate__(self, d):
        version = d.pop("object_version", None)
        if version is None and "_variables" in d:
            d["_len"] = len(d.pop("_variables"))
            d["primary_variables"] = d.pop("primary_fields")
        elif version == 1:
            d["field_variables"] = d.pop("primary_variables")

        self.__dict__ = d

        # Models pickled before the comparison-feature contract existed keep
        # their original missing features, including the original dimensions.
        version = version if version is not None else 1
        self._object_version = version
        self._legacy_missing_contract = version < CONTRACT_VERSION
        self._setup_contract_metadata()


def interactions(
    variables: Iterable[Variable], primary_variables: Iterable[FieldVariable]
) -> list[InteractionVariable]:
    field_d = {field.name: field for field in primary_variables}

    interactions: list[InteractionVariable] = []
    for variable in variables:
        if isinstance(variable, InteractionType):
            variable.expandInteractions(field_d)
            interactions.extend(variable.higher_vars)
    return interactions


def interaction_indices(variables: list[Variable]) -> list[list[int]]:
    var_names = [var.name for var in variables]
    indices = []
    for var in variables:
        if hasattr(var, "interaction_fields"):
            interaction_indices = [var_names.index(f) for f in var.interaction_fields]
            indices.append(interaction_indices)
    return indices


def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))


copyreg.pickle(types.MethodType, reduce_method)
