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


_TWO_POW_63 = 2.0**63


def _bad_integer(name: str, position: int, value: object, why: str) -> ValidationError:
    return ValidationError(
        "invalid_integer_value", f"{name}[{position}] = {value} {why}", field=name, position=position, value=value
    )


def as_int64(floats: np.ndarray, known: np.ndarray, name: str, *, what: str = "an integer") -> np.ndarray:
    """The *known* entries of *floats* as int64, ``-1`` elsewhere.

    A known entry must be a whole number in ``[-2^63, 2^63)``, the range the R
    binding accepts too.
    """
    whole = np.where(known, floats, -1.0)
    ok = (whole == np.trunc(whole)) & (whole >= -_TWO_POW_63) & (whole < _TWO_POW_63)
    bad = np.flatnonzero(~ok)
    if bad.size:
        position = int(bad[0])
        raise _bad_integer(name, position, float(floats[position]), f"is not {what}")
    return whole.astype(np.int64)


def as_float64(values: np.ndarray, name: str) -> np.ndarray:
    """*values* (float or object) as float64, None as NaN; a structured error otherwise."""
    if values.dtype != object:
        return np.asarray(values, dtype=np.float64)
    out = np.empty(len(values), dtype=np.float64)
    for position, v in enumerate(values):
        try:
            out[position] = np.nan if v is None else float(v)
        except (TypeError, ValueError):
            raise _bad_integer(name, position, None, f"({v!r}) is not a number") from None
    return out


def _ids(values: np.ndarray, name: str, *, missing_ok: bool) -> np.ndarray:
    """*values* as int64, with NA/NaN as ``-1`` when *missing_ok*."""
    if values.ndim != 1:
        raise ValidationError(
            "invalid_shape", f"{name} must be one-dimensional", field=name, expected_ndim=1, actual_shape=values.shape
        )
    if values.dtype.kind == "i":
        return values.astype(np.int64, copy=False)
    if values.dtype.kind == "u":
        too_big = np.flatnonzero(values > np.iinfo(np.int64).max)
        if too_big.size:
            position = int(too_big[0])
            raise _bad_integer(name, position, float(values[position]), "is outside [-2^63, 2^63)")
        return values.astype(np.int64)
    if values.dtype.kind == "f" or values.dtype == object:
        floats = as_float64(values, name)
        missing = np.isnan(floats)
        if missing.any() and not missing_ok:
            position = int(np.flatnonzero(missing)[0])
            raise ValidationError(
                "invalid_integer_value", f"{name}[{position}] is missing", field=name, position=position, value=None
            )
        return as_int64(floats, ~missing, name)
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
