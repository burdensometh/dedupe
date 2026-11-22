import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    MutableSequence,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    runtime_checkable,
)

import numpy
import numpy.typing

if TYPE_CHECKING:
    from dedupe.predicates import Predicate


RecordDict = Mapping[str, Any]
RecordID = Union[int, str]
RecordIDDType = Union[Type[int], Tuple[Type[str], Literal[256]]]
RecordIDPair = Union[Tuple[int, int], Tuple[str, str]]
RecordInt = Tuple[int, RecordDict]
RecordStr = Tuple[str, RecordDict]
Record = Union[RecordInt, RecordStr]
RecordPairInt = Tuple[RecordInt, RecordInt]
RecordPairStr = Tuple[RecordStr, RecordStr]
RecordPairs = Union[Iterator[RecordPairInt], Iterator[RecordPairStr]]
BlockInt = List[RecordPairInt]
BlockStr = List[RecordPairStr]
Block = Union[RecordPairInt, RecordPairStr]
BlocksInt = Iterator[BlockInt]
BlocksStr = Iterator[BlockStr]
Blocks = Union[BlocksInt, BlocksStr]
ClusterInt = Tuple[
    Tuple[int, ...], Union[numpy.typing.NDArray[numpy.float64], Tuple[float, ...]]
]
ClusterStr = Tuple[
    Tuple[str, ...], Union[numpy.typing.NDArray[numpy.float64], Tuple[float, ...]]
]
ClustersInt = Iterable[ClusterInt]
ClustersStr = Iterable[ClusterStr]
Clusters = Union[ClustersInt, ClustersStr]

DataInt = Mapping[int, RecordDict]
DataStr = Mapping[str, RecordDict]
Data = Union[DataInt, DataStr]

RecordDictPair = Tuple[RecordDict, RecordDict]
RecordDictPairs = List[RecordDictPair]
ArrayLinks = Iterable[numpy.ndarray]
TupleLinksInt = Iterable[Tuple[Tuple[int, int], float]]
TupleLinksStr = Iterable[Tuple[Tuple[str, str], float]]
TupleLinks = Union[TupleLinksInt, TupleLinksStr]
Links = Union[ArrayLinks, TupleLinks]
LookupResultsInt = Iterable[Tuple[int, Tuple[Tuple[int, float], ...]]]
LookupResultsStr = Iterable[Tuple[str, Tuple[Tuple[str, float], ...]]]
LookupResults = Union[LookupResultsInt, LookupResultsStr]
JoinConstraint = Literal["one-to-one", "many-to-one", "many-to-many"]
Comparator = Callable[[Any, Any], Union[Union[int, float], Sequence[Union[int, float]]]]
CustomComparator = Callable[[Any, Any], Union[int, float]]
Scores = Union[numpy.memmap, numpy.ndarray]
Labels = List[Literal[0, 1]]
LabelsLike = Iterable[Literal[0, 1]]
Cover = Dict["Predicate", FrozenSet[int]]
ComparisonCoverInt = Dict["Predicate", FrozenSet[Tuple[int, int]]]
ComparisonCoverStr = Dict["Predicate", FrozenSet[Tuple[str, str]]]
ComparisonCover = Union[ComparisonCoverInt, ComparisonCoverStr]
PredicateFunction = Callable[[Any], FrozenSet[str]]


class TrainingData(TypedDict):
    match: MutableSequence[RecordDictPair]
    distinct: MutableSequence[RecordDictPair]


class TrainingWeights(TypedDict, total=False):
    """Optional per-label weights for labeled training pairs.

    Each list must align, element by element, with the pairs stored
    under the same label in a :class:`TrainingData` dictionary. A
    ``None`` entry means the default weight of 1.0 for that pair.
    """

    match: MutableSequence[Optional[float]]
    distinct: MutableSequence[Optional[float]]


class WeightedTrainingData(TrainingData, total=False):
    """Training data with optional sample weights.

    The ``weights`` key is optional. When it is absent, every pair
    has the default weight of 1.0.
    """

    weights: TrainingWeights


# Optional per-example weights, aligned with a sequence of training pairs
WeightsLike = Optional[Iterable[float]]


# Takes pairs of records and generates a (n_samples X n_features) array
FeaturizerFunction = Callable[
    [Sequence[RecordDictPair]], numpy.typing.NDArray[numpy.float64]
]


class Classifier(Protocol):
    """Takes an array of pairwise distances and computes the likelihood they are a pair."""

    def fit(self, X: numpy.typing.NDArray[numpy.float64], y: LabelsLike) -> None: ...

    def predict_proba(
        self, X: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]: ...


class WeightedClassifier(Classifier, Protocol):
    """A classifier whose fit method can account for per-example weights."""

    def fit(
        self,
        X: numpy.typing.NDArray[numpy.float64],
        y: LabelsLike,
        sample_weight: WeightsLike = None,
    ) -> None: ...


class ClosableJoinable(Protocol):
    def close(self) -> None: ...

    def join(self) -> None: ...


class Variable(Protocol):
    name: str
    predicates: List["Predicate"]
    has_missing: bool

    def __len__(self) -> int: ...


@runtime_checkable
class FieldVariable(Variable, Protocol):
    field: str
    comparator: Comparator


class InteractionVariable(Variable, Protocol):
    interaction_fields: List[str]


MapLike = Callable[[Callable[[Any], Any], Iterable], Iterable]

PathLike = Union[str, os.PathLike]
