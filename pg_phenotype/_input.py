"""Host-side coercion of the pedigree columns into the arrays the core reads."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from pg_phenotype._errors import ParameterError, ValidationError

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


def _column(frame: object, name: str) -> object | None:
    """Column *name* of a polars/pandas frame or a mapping, or ``None``."""
    columns = getattr(frame, "columns", None)
    if columns is not None:
        return frame[name] if name in list(columns) else None  # ty: ignore[not-subscriptable]
    mapping: Mapping[str, object] = frame  # ty: ignore[invalid-assignment]
    return mapping.get(name)


def _array(values: object) -> np.ndarray:
    """*values* as an array, through ``to_numpy`` for a polars or pandas series."""
    to_numpy = getattr(values, "to_numpy", None)
    return np.asarray(to_numpy() if to_numpy is not None else values)


def _private(out: np.ndarray, source: object, raw: np.ndarray) -> np.ndarray:
    """*out*, read from *source* as *raw*, copied while the caller can still write it.

    Native calls read their inputs with the GIL released, so no other Python
    thread may write one mid-call.  A new array, and the read-only views that
    polars and copy-on-write pandas hand out, need no copy.
    """
    from_caller = raw is source or raw.base is not None
    if from_caller and out.flags.writeable and np.may_share_memory(out, raw):
        return out.copy()
    return out


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


def _is_missing(value: object) -> bool:
    """None, a float NaN, or pandas' ``NA``: what an object column holds for a missing entry."""
    if value is None or type(value).__name__ == "NAType":
        return True
    return isinstance(value, float | np.floating) and bool(np.isnan(value))


def _missing_id(name: str, position: int) -> ValidationError:
    return ValidationError(
        "invalid_integer_value", f"{name}[{position}] is missing", field=name, position=position, value=None
    )


def _object_ids(values: np.ndarray, name: str, *, missing_ok: bool) -> np.ndarray:
    """An object column as int64, ``-1`` for a missing entry.

    Integers are taken exactly, so an id past ``2^53`` keeps its value; a
    float must be whole.  Either must lie in ``[-2^63, 2^63)``.
    """
    out = np.empty(len(values), dtype=np.int64)
    for position, v in enumerate(values):
        if _is_missing(v):
            if not missing_ok:
                raise _missing_id(name, position)
            out[position] = -1
            continue
        if isinstance(v, int | np.integer):
            whole = int(v)
        elif isinstance(v, float | np.floating):
            if not float(v).is_integer():
                raise _bad_integer(name, position, float(v), "is not an integer")
            whole = int(v)
        else:
            raise _bad_integer(name, position, None, f"({v!r}) is not a number")
        if not -(2**63) <= whole < 2**63:
            raise _bad_integer(name, position, float(whole), "is outside [-2^63, 2^63)")
        out[position] = whole
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
    if values.dtype == object:
        return _object_ids(values, name, missing_ok=missing_ok)
    if values.dtype.kind == "f":
        missing = np.isnan(values)
        if missing.any() and not missing_ok:
            raise _missing_id(name, int(np.flatnonzero(missing)[0]))
        return as_int64(values.astype(np.float64, copy=False), ~missing, name)
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
        column = _column(pedigree, name)
        if column is None:
            if name in REQUIRED:
                raise ValidationError("missing_field", f"pedigree has no {name!r} column", field=name)
            return None
        raw = _array(column)
        return _private(np.ascontiguousarray(_ids(raw, name, missing_ok=name != "id")), column, raw)

    def required(name: str) -> np.ndarray:
        values = read(name)
        assert values is not None
        return values

    return PedigreeArrays(required("id"), required("mother"), required("father"), read("twin"), read("sex"))


def id_array(values: object, name: str) -> np.ndarray:
    """*values* as a contiguous int64 array of ids, none missing."""
    raw = _array(values)
    return _private(np.ascontiguousarray(_ids(raw, name, missing_ok=False)), values, raw)


def floats(values: object, name: str) -> np.ndarray:
    """*values* as a contiguous float64 array, NA/None as NaN.

    Raises:
        TypeError: *values* holds text, which numpy would parse.
    """
    raw = _array(values)
    if raw.dtype.kind in "USO" and any(isinstance(v, str | bytes) for v in raw.ravel()):
        raise TypeError(f"{name} must be numeric, not text")
    to_numpy = getattr(values, "to_numpy", None)
    if to_numpy is not None:
        # pandas; polars takes no arguments, and its nulls are already NaN
        # for floats, None for objects.
        with contextlib.suppress(TypeError):
            raw = np.asarray(to_numpy(dtype=np.float64, na_value=np.nan))
    if raw.ndim != 1:
        raise ValidationError(
            "invalid_shape", f"{name} must be one-dimensional", field=name, expected_ndim=1, actual_shape=raw.shape
        )
    if raw.dtype == object:
        return np.array([np.nan if _is_missing(v) else v for v in raw], dtype=np.float64)
    return _private(np.ascontiguousarray(raw, dtype=np.float64), values, raw)


def strata(stratum: object) -> tuple[np.ndarray, np.ndarray]:
    """Integer labels and a known mask from one value per row, NA/None/NaN unknown."""
    values = floats(stratum, "stratum")
    known = ~np.isnan(values)
    labels = as_int64(values, known, "stratum", what="an integer label")
    # Unknown rows carry label 0, which the core ignores under `known`.
    labels[~known] = 0
    return np.ascontiguousarray(labels), np.ascontiguousarray(known)


_INT64 = (-(2**63), 2**63 - 1)


def count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {value!r}")
    value = int(value)
    if not _INT64[0] <= value <= _INT64[1]:
        raise ParameterError(
            "parameter_out_of_range",
            f"{name} = {value} is outside the int64 range",
            name=name,
            value=float(value),
            domain="[-2^63, 2^63)",
        )
    return value
