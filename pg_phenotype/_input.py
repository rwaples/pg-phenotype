"""Host-side coercion of the pedigree columns into the arrays the core reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pg_phenotype._errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

REQUIRED = ("id", "mother", "father")
OPTIONAL = ("twin", "sex")


class PedigreeArrays(NamedTuple):
    """The pedigree columns as contiguous int64 arrays, ``-1`` for missing."""

    id: np.ndarray
    mother: np.ndarray
    father: np.ndarray
    twin: np.ndarray | None
    sex: np.ndarray | None


def _column(frame: object, name: str) -> np.ndarray | None:
    """Column *name* of a polars/pandas frame or a mapping, or ``None``."""
    columns = getattr(frame, "columns", None)
    if columns is not None:
        if name not in list(columns):
            return None
        series = frame[name]  # ty: ignore[not-subscriptable]
        to_numpy = getattr(series, "to_numpy", None)
        return np.asarray(to_numpy() if to_numpy is not None else series)
    mapping: Mapping[str, object] = frame  # ty: ignore[invalid-assignment]
    return None if name not in mapping else np.asarray(mapping[name])


def _ids(values: np.ndarray, name: str, *, missing_ok: bool) -> np.ndarray:
    """*values* as int64, with NA/NaN as ``-1`` when *missing_ok*."""
    if values.ndim != 1:
        raise ValidationError(
            "invalid_shape", f"{name} must be one-dimensional", field=name, expected_ndim=1, actual_shape=values.shape
        )
    if values.dtype.kind in "iu":
        return values.astype(np.int64, copy=False)
    if values.dtype.kind == "f" or values.dtype == object:
        floats = np.asarray(values, dtype=np.float64)
        missing = np.isnan(floats)
        if missing.any() and not missing_ok:
            position = int(np.flatnonzero(missing)[0])
            raise ValidationError(
                "invalid_integer_value", f"{name}[{position}] is missing", field=name, position=position, value=None
            )
        whole = np.where(missing, -1.0, floats)
        bad = whole != np.trunc(whole)
        if bad.any():
            position = int(np.flatnonzero(bad)[0])
            raise ValidationError(
                "invalid_integer_value",
                f"{name}[{position}] = {floats[position]} is not an integer",
                field=name,
                position=position,
                value=float(floats[position]),
            )
        return whole.astype(np.int64)
    raise ValidationError(
        "invalid_integer_value", f"{name} has dtype {values.dtype}, not integer", field=name, position=0, value=None
    )


def pedigree_arrays(pedigree: object) -> PedigreeArrays:
    """The ``id``, ``mother``, ``father``, ``twin`` and ``sex`` arrays of *pedigree*.

    Args:
        pedigree: A polars or pandas frame, or a mapping of column name to array.

    Returns:
        The columns; ``twin`` and ``sex`` are ``None`` when absent.

    Raises:
        ValidationError: ``missing_field`` for an absent required column,
            ``invalid_integer_value`` for a non-integer or missing id.
    """

    def read(name: str) -> np.ndarray | None:
        values = _column(pedigree, name)
        if values is None:
            if name in REQUIRED:
                raise ValidationError("missing_field", f"pedigree has no {name!r} column", field=name)
            return None
        return np.ascontiguousarray(_ids(values, name, missing_ok=name != "id"))

    def required(name: str) -> np.ndarray:
        values = read(name)
        assert values is not None
        return values

    return PedigreeArrays(required("id"), required("mother"), required("father"), read("twin"), read("sex"))


def floats(values: object, name: str) -> np.ndarray:
    """*values* as a contiguous float64 array, NA/None as NaN."""
    to_numpy = getattr(values, "to_numpy", None)
    if to_numpy is not None:
        try:
            values = to_numpy(dtype=np.float64, na_value=np.nan)  # pandas
        except TypeError:
            values = to_numpy()  # polars: nulls become NaN for floats, None for objects
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValidationError(
            "invalid_shape", f"{name} must be one-dimensional", field=name, expected_ndim=1, actual_shape=array.shape
        )
    if array.dtype == object:
        array = np.array([np.nan if v is None else v for v in array], dtype=np.float64)
    return np.ascontiguousarray(array, dtype=np.float64)
