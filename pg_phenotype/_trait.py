"""The shared phenotype input: one column on the pedigree's rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pg_phenotype import _native
from pg_phenotype._errors import ValidationError
from pg_phenotype._input import floats

TraitKind = Literal["continuous", "binary", "ordinal", "categorical"]
#: The kinds the core reads, in its order (``TraitKind::ALL``).
KINDS: tuple[str, ...] = tuple(_native.trait_kinds())


def _raw(values: object) -> np.ndarray:
    to_numpy = getattr(values, "to_numpy", None)
    return np.asarray(to_numpy() if to_numpy is not None else values)


@dataclass(frozen=True, init=False)
class Trait:
    """One phenotype column aligned to the pedigree's input rows.

    Args:
        values: One value per pedigree row; NA, None or NaN where unknown.
            Strings are categorical labels, coded in sorted order; a pandas
            categorical is coded in its category order.
        kind: ``"continuous"``, ``"binary"``, ``"ordinal"`` or
            ``"categorical"``.  When omitted it is inferred: booleans and
            values all in {0, 1} are binary, an ordered pandas categorical
            ordinal, strings and other categoricals categorical, any
            non-integer number continuous; other integers need an explicit
            kind.

    Attributes:
        values: float64, NaN where unknown (category codes for strings).
        kind: The trait kind.
        levels: The category labels in code order, for string or
            categorical input.

    Raises:
        ValidationError: ``invalid_trait_kind`` for an unknown kind,
            ``ambiguous_trait_kind`` when integer values leave it open, and
            ``invalid_trait_value`` for a non-integer ordinal or
            categorical value.
    """

    values: np.ndarray
    kind: TraitKind
    levels: tuple[str, ...] | None

    def __init__(self, values: object, kind: TraitKind | None = None) -> None:
        if kind is not None and kind not in KINDS:
            raise ValidationError(
                "invalid_trait_kind", f"kind must be one of {', '.join(KINDS)}, got {kind!r}", kind=kind
            )
        dtype = getattr(values, "dtype", None)
        if isinstance(dtype, _pd_categorical_type()):
            array, levels = _from_categorical(values)
            kind = kind or ("ordinal" if getattr(dtype, "ordered", False) else "categorical")
        else:
            raw = _raw(values)
            if raw.dtype.kind in "USO" and any(isinstance(v, str) for v in raw):
                array, levels = _from_labels(raw)
                kind = kind or "categorical"
            else:
                array, levels = floats(values, "trait"), None
                kind = kind or _infer(array, raw.dtype)
        if kind in ("ordinal", "categorical") and levels is None:
            present = array[~np.isnan(array)]
            bad = np.flatnonzero(present != np.trunc(present))
            if bad.size:
                position = int(np.flatnonzero(~np.isnan(array))[bad[0]])
                raise ValidationError(
                    "invalid_trait_value",
                    f"trait[{position}] = {array[position]} is not an integer {kind} code",
                    field="trait",
                    position=position,
                    value=float(array[position]),
                )
        object.__setattr__(self, "values", array)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "levels", levels)


def _pd_categorical_type() -> tuple[type, ...]:
    """Pandas's categorical dtype, or nothing when pandas is not installed."""
    try:
        import pandas as pd
    except ImportError:
        return ()
    return (pd.CategoricalDtype,)


def _from_categorical(values: object) -> tuple[np.ndarray, tuple[str, ...]]:
    """A pandas categorical's codes in category order, NaN where missing."""
    cat: Any = getattr(values, "cat", values)
    codes = np.asarray(cat.codes, dtype=np.float64)
    return np.where(codes < 0, np.nan, codes), tuple(str(c) for c in cat.categories)


def _from_labels(raw: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """String labels coded in sorted order, NaN where missing."""
    labels = sorted({v for v in raw if isinstance(v, str)})
    code = {label: float(i) for i, label in enumerate(labels)}
    return np.array([code.get(v, np.nan) if isinstance(v, str) else np.nan for v in raw]), tuple(labels)


def _infer(array: np.ndarray, dtype: np.dtype) -> TraitKind:
    if dtype == np.bool_:
        return "binary"
    present = array[~np.isnan(array)]
    if np.isin(present, (0.0, 1.0)).all():
        return "binary"
    if (present != np.trunc(present)).any():
        return "continuous"
    raise ValidationError(
        "ambiguous_trait_kind",
        "integer trait values beyond {0, 1}: pass kind= continuous, ordinal or categorical",
        field="trait",
    )
