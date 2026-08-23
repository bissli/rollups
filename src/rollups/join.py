"""Operations relating a pair of datasets.

Each function here pairs the rows of two datasets, by key or by
position, and answers a new dataset or a set of row lists. None of them
mutates its inputs beyond the type conversion a read triggers.

Notes
-----
- See docs/joins.md for the join-type table and worked examples.
"""


import logging

from libb import OrderedSet
from libb import lazydict as attrdict

logger = logging.getLogger(__name__)


def join_datasets(adataset: 'DataSet', akey: tuple[str],
                  bdataset: 'DataSet', bkey: tuple[str],
                  jointype: str = 'inner', amod: str = '', bmod: str = '',
                  acol: list[str] | None = None,
                  bcol: list[str] | None = None, first: bool = False,
                  bfirst: bool = False, cls: type | None = None) -> 'DataSet':
    """Join two datasets on their key columns.

    Parameters
    ----------
    adataset : DataSet
        Left dataset.
    akey : str or tuple of str
        Key column(s) from the left dataset.
    bdataset : DataSet
        Right dataset.
    bkey : str or tuple of str
        Key column(s) from the right dataset.
    jointype : str, default 'inner'
        One of 'inner', 'outer', 'left', 'right'.
    amod, bmod : str, default ''
        Suffix appended to each side's column names.
    acol, bcol : list of str or None, default None
        Columns to carry over from each side. None takes all.
    first : bool, default False
        Pair rows sequentially instead of taking the cartesian
        product. This DROPS rows where a key repeats.
    bfirst : bool, default False
        Resolve an overlapping column from the right side first.
    cls : type or None, default None
        Class the result is built with. None takes `adataset`'s class,
        so a standalone caller needs none.

    Returns
    -------
    DataSet
        Joined rows.

    Raises
    ------
    ValueError
        If `jointype` is not one of the four supported names.

    Notes
    -----
    - A repeated key fans out into every pairing unless `first` is
      set; `first` logs a warning naming the key whose rows it drops.
    - An overlapping column takes the left value unless it is None,
      in which case the right fills in. `bfirst` reverses that.
    - A None key joins every row against every row.
    - See docs/joins.md for the join-type table and worked examples.
    - `cls` builds the result. It defaults to `adataset`'s class, so a
      standalone caller needs no class; `DataSet.join` passes the class
      it was reached through.
    """
    cls = cls or adataset.__class__
    adataset.ensure_types()
    bdataset.ensure_types()
    if acol is None:
        acol = adataset.cols
    if bcol is None:
        bcol = bdataset.cols

    if not isinstance(akey, tuple | list):
        akey = (akey,)
    if not isinstance(bkey, tuple | list):
        bkey = (bkey,)

    adict, bdict = {}, {}
    if None in akey:
        adict[(None,)] = adataset.container
    else:
        for row in adataset:
            thiskey = tuple(row[_] for _ in akey)
            if thiskey not in adict:
                adict[thiskey] = []
            adict[thiskey].append(row)

    if None in bkey:
        bdict[(None,)] = bdataset.container
    else:
        for row in bdataset:
            thiskey = tuple(row[_] for _ in bkey)
            if thiskey not in bdict:
                bdict[thiskey] = []
            bdict[thiskey].append(row)

    acols = OrderedSet((f'{k}{amod}', t) for k, t in adataset.columns if k in acol)
    bcols = OrderedSet((f'{k}{bmod}', t) for k, t in bdataset.columns if k in bcol)

    jcols_dict = dict(acols)

    for name, typ in bcols:
        if name not in jcols_dict:
            jcols_dict[name] = typ
        else:
            existing_typ = jcols_dict[name]
            if existing_typ is type(None):
                jcols_dict[name] = typ
            elif typ is not type(None) and bfirst:
                jcols_dict[name] = typ

    jcols = list(jcols_dict.items())

    if jointype == 'outer':
        jkeys = OrderedSet(adict) | OrderedSet(bdict)
    elif jointype == 'inner':
        jkeys = OrderedSet(adict) & OrderedSet(bdict)
    elif jointype == 'left':
        jkeys = OrderedSet(adict)
    elif jointype == 'right':
        jkeys = OrderedSet(bdict)
    else:
        raise ValueError(f'This join type is not supported {jointype}')

    def merge_rows(arow, brow):
        jrow = attrdict()
        _arow = {f'{c}{amod}': arow.get(c) for c in acol}
        _brow = {f'{c}{bmod}': brow.get(c) for c in bcol}
        for k in set(list(_arow.keys()) + list(_brow.keys())):
            _aval = _arow.get(k)
            _bval = _brow.get(k)
            jrow[k] = (_bval if _bval is not None else _aval) if bfirst else (_aval if _aval is not None else _bval)
        return jrow

    joined = cls(columns=jcols)
    for jkey in jkeys:
        arows = adict.get(jkey, [dict(list(zip(akey, len(akey) * [None])))])
        brows = bdict.get(jkey, [dict(list(zip(bkey, len(bkey) * [None])))])
        if not first:
            for arow in arows:
                for brow in brows:
                    joined.append(merge_rows(arow, brow))
        else:
            while arows or brows:
                arow = arows.pop() if arows else {}
                brow = brows.pop() if brows else {}
                joined.append(merge_rows(arow, brow))

                if jointype == 'inner':
                    if arows or brows:
                        logger.warning(f'Dropped rows for key {str(jkey)}')
                    break
                if jointype == 'left' and not arows:
                    if brows:
                        logger.warning(f'Dropped brows for key {str(jkey)}')
                    break
                if jointype == 'right' and not brows:
                    if arows:
                        logger.warning(f'Dropped arows for key {str(jkey)}')
                    break

    return joined


def diff_datasets(ds1, ds2, keycols, comparecols):
    """Compare two datasets and categorize rows by their differences.

    Parameters
    ----------
    ds1 : DataSet
        First dataset.
    ds2 : DataSet
        Second dataset.
    keycols : list of str
        Columns used to match rows between the datasets.
    comparecols : list of str
        Columns compared for differences.

    Returns
    -------
    tuple of list
        (same, diff, only_in_ds1, only_in_ds2), where:
        same holds complete ds1 rows whose key is in both and whose
        comparecols all match; diff holds rows with a matching key but
        differing values, each carrying the key columns plus comparecols
        where a differing column holds (ds1_value, ds2_value) and a
        matching column holds None; only_in_ds1 and only_in_ds2 hold
        complete rows whose key the other side lacks.

    Notes
    -----
    - Keys are walked in sorted order.
    - Where a key repeats within a dataset, the last row wins.
    - A diff row carries only the key and comparison columns; same and
      only rows keep every column from their source.
    """
    key_fn = lambda row: tuple(row[col] for col in keycols)
    ds1_map = {key_fn(row): row for row in ds1}
    ds2_map = {key_fn(row): row for row in ds2}

    same, diff, only_in_ds1 = [], [], []
    for key in sorted(ds1_map):
        ds1_row = ds1_map[key]
        ds2_row = ds2_map.pop(key, None)
        if ds2_row is not None:
            diff_row = attrdict(zip(keycols, key))
            for col in comparecols:
                if ds1_row[col] != ds2_row[col]:
                    diff_row[col] = (ds1_row[col], ds2_row[col])
                else:
                    diff_row[col] = None
            if any(diff_row[col] is not None for col in comparecols):
                diff.append(diff_row)
            else:
                same.append(ds1_row)
        else:
            only_in_ds1.append(ds1_row)
    only_in_ds2 = list(ds2_map.values())
    return same, diff, only_in_ds1, only_in_ds2


def meld_datasets(meldee, melders, melder_ids, columns, inplace=True):
    """Combine columns from several aligned datasets by row position.

    Parameters
    ----------
    meldee : DataSet
        Base dataset to add columns to.
    melders : list of DataSet
        Datasets to take columns from.
    melder_ids : list of str
        Prefix per melder, used to name the merged columns.
    columns : list of list of str
        Columns to take from each melder, positionally matched.
    inplace : bool, default True
        If True, modify `meldee`; if False, work on a deep copy.

    Returns
    -------
    DataSet
        The melded dataset.

    Raises
    ------
    ValueError
        If a melder's row count differs from the meldee's.

    Notes
    -----
    - Rows are matched by position, not by key, so every dataset has to
      hold the same rows in the same order.
    - A merged column is named `{prefix}_{col}`.
    """
    result = meldee if inplace else meldee.deepcopy()

    for dataset, prefix, cols in zip(melders, melder_ids, columns):
        if len(result) != len(dataset):
            raise ValueError(f'Length mismatch: {len(dataset)} != {len(result)}')

        for col in cols:
            new_col_name = f'{prefix}_{col}'
            col_type = dataset.colmap.get(col, type(None))
            values = [source_row.get(col) for source_row in dataset]
            result.add_column(new_col_name, col_type, values=values)

    return result
