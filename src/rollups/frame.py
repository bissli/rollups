"""DataFrame-native join, group-by, and construction.

Four functions taking and returning `pandas.DataFrame`, none of them
depending on `DataSet`. `join_dataframes` and `bucket_dataframe` mirror
`DataSet.join` and `DataSet.bucket`. `empty_dataframe` and
`dataframe_from_list` mirror the constructors of the same shape.

Notes
-----
- Every gap these functions introduce reads as null under `isna`. Which
  null it is follows the column's dtype: NaN, NaT, or pd.NA, and None
  where the column stays object. Read a gap with `isna`, never
  `is None`.
- An int column holding a gap widens to float, as it does anywhere else
  in pandas.
- See docs/dataframe-native.md for the contract differences against
  `DataSet.join` and `DataSet.bucket`.
"""


import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


JOIN_TYPES = ('inner', 'outer', 'left', 'right', 'cross')


# Stands in for a null while two key columns are matched, so that a None
# on one side matches the NaN on the other.
NULL_KEY = object()


# Notes:
# - Ops whose pandas form already answers NA on an all-null group, so
#   the vectorized path and the per-group path agree on an empty group.
# - sum needs min_count=1 or pandas totals an all-null group to 0.
# - Anything else - len, a lambda, a libb.stats helper - falls through
#   to the per-group path, which runs the callable itself.
FAST_OPS = {
    sum: ('sum', {'min_count': 1}),
    max: ('max', {}),
    min: ('min', {}),
    }


def _free_name(stem: str, taken: Sequence[str]) -> str:
    """Name built from `stem` that collides with nothing in `taken`.

    Parameters
    ----------
    stem : str
        Preferred name.
    taken : sequence of str
        Names already in use.

    Returns
    -------
    str
        `stem`, or `stem` wrapped in underscores until it is free.
    """
    name = stem
    while name in taken:
        name = f'_{name}_'
    return name


def join_dataframes(
    left: pd.DataFrame,
    lkey: str | Sequence[str] | None,
    right: pd.DataFrame,
    rkey: str | Sequence[str] | None,
    how: str = 'inner',
    *,
    lsuffix: str = '',
    rsuffix: str = '',
    lcols: Sequence[str] | None = None,
    rcols: Sequence[str] | None = None,
    first: bool = False,
    prefer: str = 'left',
) -> pd.DataFrame:
    """Join two frames on their key columns, coalescing shared columns.

    Parameters
    ----------
    left : pd.DataFrame
        Left frame.
    lkey : str or sequence of str or None
        Key column(s) of `left`. None only with ``how='cross'``.
    right : pd.DataFrame
        Right frame.
    rkey : str or sequence of str or None
        Key column(s) of `right`, positionally paired with `lkey`.
    how : str, default 'inner'
        One of 'inner', 'outer', 'left', 'right', 'cross'.
    lsuffix, rsuffix : str, default ''
        Suffix appended to every column of that side, key columns
        included.
    lcols, rcols : str or sequence of str or None, default None
        Columns that side contributes. None contributes all. A key
        column left out still drives the match but is dropped from the
        result.
    first : bool, default False
        Pair the nth row of a key group on one side with the nth row of
        the same key group on the other, instead of taking the cartesian
        product. This DROPS the surplus rows of the longer side.
    prefer : str, default 'left'
        Which side wins a column both sides carry: 'left' or 'right'.

    Returns
    -------
    pd.DataFrame
        Joined rows under a fresh RangeIndex.

    Raises
    ------
    ValueError
        If `how` or `prefer` is not one of its listed names, if the two
        key lists differ in length, if a key is None under any `how` but
        'cross' (or is given under 'cross'), if a named key or selected
        column is absent from its frame, if either frame repeats a
        column name, or if `first` is set under ``how='cross'``.

    Notes
    -----
    - A column both sides carry stays one column, taking the `prefer`
      side's value wherever that value is not null and the other side's
      value where it is. Where only one side carries a name, pandas
      would spell it `name_x`/`name_y`; here it is just `name`.
    - Null keys match each other, so all-null rows on both sides pair up.
    - Column order is `left`'s columns, then `right`'s columns that
      `left` does not already carry, each in its own frame's order.
    - Row order: 'inner', 'left', and 'outer' run in `left` row order,
      each row followed by its matches in `right` row order, and 'outer'
      then appends the right-only rows in `right` row order. 'right'
      runs in `right` row order.
    - `first` pairs by position within the key group, so 'inner' keeps
      ``min(nleft, nright)`` rows per key, 'left' keeps `nleft`, 'right'
      keeps `nright`, and 'outer' keeps ``max(nleft, nright)``. It logs
      one warning counting the rows it dropped.
    """
    if how not in JOIN_TYPES:
        raise ValueError(
            f'This join type is not supported {how}, '
            f'expected one of {", ".join(JOIN_TYPES)}')
    if prefer not in {'left', 'right'}:
        raise ValueError(f"prefer must be 'left' or 'right', got {prefer!r}")

    lkey = [lkey] if isinstance(lkey, str) else list(lkey or [])
    rkey = [rkey] if isinstance(rkey, str) else list(rkey or [])

    if how == 'cross':
        if lkey or rkey:
            raise ValueError(
                "how='cross' pairs every row with every row, "
                'so it takes no key columns')
        if first:
            raise ValueError(
                'first pairs rows within a key group, which '
                "how='cross' does not have")
    else:
        if not lkey or not rkey:
            raise ValueError(
                f'how={how!r} needs a key on both sides; '
                "pass how='cross' to pair every row with every row")
        if len(lkey) != len(rkey):
            raise ValueError(f'Key lists differ in length: {lkey} against {rkey}')

    lcols = [lcols] if isinstance(lcols, str) else lcols
    rcols = [rcols] if isinstance(rcols, str) else rcols

    for frame, keys, cols, side in ((left, lkey, lcols, 'left'),
                                    (right, rkey, rcols, 'right')):
        missing = [c for c in list(keys) + list(cols or []) if c not in frame.columns]
        if missing:
            raise ValueError(
                f'{side} frame has no column '
                f'{", ".join(map(repr, missing))}')

    lpick = [c for c in left.columns if lcols is None or c in set(lcols)]
    rpick = [c for c in right.columns if rcols is None or c in set(rcols)]
    # A suffix stringifies the label it lands on, so apply one only
    # where the caller asked for it: an integer or Timestamp column
    # label survives the default call unchanged.
    lout = [f'{c}{lsuffix}' for c in lpick] if lsuffix else list(lpick)
    rout = [f'{c}{rsuffix}' for c in rpick] if rsuffix else list(rpick)

    # Both the labels as they arrived and the labels the suffix leaves:
    # two distinct labels can stringify onto one name.
    for side, labels in (('left', list(left.columns)), ('left', lout),
                         ('right', list(right.columns)), ('right', rout)):
        repeated = sorted({c for c in labels if labels.count(c) > 1}, key=str)
        if repeated:
            raise ValueError(
                f'{side} frame carries '
                f'{", ".join(map(repr, repeated))} more than once; '
                'column names must be unique')

    # Notes:
    # - The merge needs a key column even where lcols/rcols dropped it,
    #   and needs the two sides' keys under distinct names even where
    #   they share one. Both go in under reserved names and come back
    #   out before the result is returned.
    # - A RangeIndex on each side makes the row positions below line up
    #   with the frames, whatever index the caller passed in.
    reserved = list(lout) + list(rout)
    taken = set(reserved)
    lpos = _free_name('__lpos', reserved)
    rpos = _free_name('__rpos', reserved + [lpos])
    seq = _free_name('__seq', reserved + [lpos, rpos])
    lmatch = [_free_name(f'__lmatch{i}', reserved) for i in range(len(lkey))]
    rmatch = [_free_name(f'__rmatch{i}', reserved) for i in range(len(rkey))]

    # A merge suffix lands on a name both sides carry, so it is the
    # SUFFIXED name that has to be free: a left column already called
    # v__l would collide with the v__l the merge makes out of v.
    shared = [c for c in lout if c in set(rout)]
    lsfx, rsfx = '__l', '__r'
    while any(f'{c}{lsfx}' in taken or f'{c}{rsfx}' in taken for c in shared):
        lsfx, rsfx = f'_{lsfx}_', f'_{rsfx}_'

    lwork = left.loc[:, lpick].set_axis(lout, axis=1).reset_index(drop=True)
    rwork = right.loc[:, rpick].set_axis(rout, axis=1).reset_index(drop=True)
    for name, col in zip(lmatch, lkey):
        lwork[name] = left[col].reset_index(drop=True)
    for name, col in zip(rmatch, rkey):
        rwork[name] = right[col].reset_index(drop=True)
    lwork[lpos] = np.arange(len(lwork))
    rwork[rpos] = np.arange(len(rwork))

    for lname, rname in zip(lmatch, rmatch):
        lk, rk = lwork[lname], rwork[rname]
        # Notes:
        # - pandas refuses outright to match an object key against a
        #   numeric one, and among objects a None misses the NaN that
        #   means the same thing. One sentinel for every null, then a
        #   shared factorization, reduces both sides to integer codes
        #   that always match and always sort.
        # - Two numeric keys skip this and match vectorized, NaN to NaN;
        #   so does any pair of equal dtype holding no null at all.
        if pd.api.types.is_numeric_dtype(lk) and pd.api.types.is_numeric_dtype(rk):
            continue
        if lk.dtype == rk.dtype and not (lk.hasnans or rk.hasnans):
            continue
        lvals = lk.to_numpy(dtype=object, na_value=NULL_KEY)
        rvals = rk.to_numpy(dtype=object, na_value=NULL_KEY)
        codes, _ = pd.factorize(np.concatenate([lvals, rvals]), use_na_sentinel=False)
        lwork[lname] = codes[:len(lvals)]
        rwork[rname] = codes[len(lvals):]

    if first:
        # Pairing the nth row of a key group with the nth row of the
        # other side is a merge on the key plus that ordinal.
        lwork[seq] = lwork.groupby(lmatch, dropna=False, sort=False).cumcount()
        rwork[seq] = rwork.groupby(rmatch, dropna=False, sort=False).cumcount()
        lmatch, rmatch = lmatch + [seq], rmatch + [seq]

    if how == 'cross':
        merged = lwork.merge(rwork, how='cross', suffixes=(lsfx, rsfx))
    else:
        merged = lwork.merge(
            rwork, how=how, left_on=lmatch, right_on=rmatch,
            suffixes=(lsfx, rsfx), sort=False)

    if first:
        lost = (len(lwork) - merged[lpos].nunique(),
                len(rwork) - merged[rpos].nunique())
        if any(lost):
            logger.warning(
                f'first=True dropped {lost[0]} left and {lost[1]} right rows '
                'that had no positional partner')

    # A name both sides carry arrives suffixed twice over; fold the pair
    # back into the single column the contract promises.
    for name in shared:
        win, lose = (f'{name}{lsfx}', f'{name}{rsfx}')
        if prefer == 'right':
            win, lose = lose, win
        keep, fill = merged[win], merged[lose]
        try:
            merged[name] = keep.where(keep.notna(), fill)
        except (TypeError, ValueError):
            # Series.where will not widen a categorical or a nullable
            # dtype, and will not take a value from another dtype
            # family, so those two coalesce through object instead.
            merged[name] = (keep.astype(object)
                            .where(keep.notna(), fill.astype(object))
                            .infer_objects())
        merged = merged.drop(columns=[f'{name}{lsfx}', f'{name}{rsfx}'])

    lead, trail = (rpos, lpos) if how == 'right' else (lpos, rpos)
    merged = merged.sort_values([lead, trail], na_position='last', kind='stable')

    ordered = lout + [c for c in rout if c not in set(lout)]
    return merged.loc[:, ordered].reset_index(drop=True)
