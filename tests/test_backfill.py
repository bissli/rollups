import pytest
from opendate import Date
from rollups import DataSet


@pytest.fixture
def basic_dataset():
    """Dataset with interior None gaps."""
    return DataSet([
        {'val': 1},
        {'val': None},
        {'val': None},
        {'val': 4},
    ])


@pytest.fixture
def multi_col_dataset():
    """Dataset with multiple columns."""
    return DataSet([
        {'id': 1, 'val': 10, 'name': 'a'},
        {'id': 2, 'val': None, 'name': 'b'},
        {'id': 3, 'val': 30, 'name': 'c'},
    ])


def test_backfill_basic(basic_dataset):
    """Verify each interior None takes the value that precedes it.

    Mutation: fill from the following non-None value (pandas bfill)
        instead of the preceding one.
    Oracle: hand-computed [1, 1, 1, 4]; a next-value fill gives
        [1, 4, 4, 4].
    """
    basic_dataset.backfill('val')
    assert list(basic_dataset.unwind('val')) == [1, 1, 1, 4]


def test_backfill_leading_nones():
    """Verify a leading run of Nones takes the first later value.

    Mutation: drop the back-pass, leaving a plain forward fill.
    Oracle: hand-computed [3, 3, 3]; forward fill alone gives
        [None, None, 3].
    """
    ds = DataSet([
        {'val': None},
        {'val': None},
        {'val': 3},
    ])
    ds.backfill('val')
    assert list(ds.unwind('val')) == [3, 3, 3]


def test_backfill_trailing_nones():
    """Verify a trailing run of Nones carries the last value.

    Mutation: fill only gaps that sit between two values, skipping any
        run with no value after it.
    Oracle: hand-computed [1, 1, 1]; an interior-only fill leaves
        [1, None, None].
    """
    ds = DataSet([
        {'val': 1},
        {'val': None},
        {'val': None},
    ])
    ds.backfill('val')
    assert list(ds.unwind('val')) == [1, 1, 1]


def test_backfill_no_nones_writes_target_column():
    """Verify a gap-free column is still copied to the target column.

    Mutation: an early return when the source column holds no None, so
        the target column is never written.
    Oracle: hand-computed ['val', 'copy'] membership and [1, 2, 3] in
        both columns; the early return leaves 'copy' absent.
    """
    ds = DataSet([
        {'val': 1},
        {'val': 2},
        {'val': 3},
    ])
    ds.backfill('val', 'copy')
    assert ds.cols == ['copy', 'val']
    assert list(ds.unwind('val')) == [1, 2, 3]
    assert list(ds.unwind('copy')) == [1, 2, 3]


def test_backfill_all_nones():
    """Verify an all-None column comes back unchanged.

    Mutation: seed the fill with next(v for v in values if v is not
        None), which raises StopIteration when every value is None.
    Oracle: hand-computed [None, None, None].
    """
    ds = DataSet([
        {'val': None},
        {'val': None},
        {'val': None},
    ])
    ds.backfill('val')
    assert list(ds.unwind('val')) == [None, None, None]


def test_backfill_mixed_none_values():
    """Verify the carried value updates at every later non-None value.

    Mutation: carry the first non-None value throughout, ignoring later
        values.
    Oracle: hand-computed [1, 1, 3, 3, 3]; a first-value-only carry
        gives [1, 1, 3, 1, 1].
    """
    ds = DataSet([
        {'val': 1},
        {'val': None},
        {'val': 3},
        {'val': None},
        {'val': None},
    ])
    ds.backfill('val')
    assert list(ds.unwind('val')) == [1, 1, 3, 3, 3]


def test_backfill_new_colname():
    """Verify the new column lands at the source position, source kept.

    Mutation: drop index=colix, appending the new column after 'b'.
    Oracle: hand-computed cols ['a', 'val_filled', 'val', 'b'] and an
        untouched source [1, None, 3].
    """
    ds = DataSet([
        {'a': 9, 'val': 1, 'b': 'x'},
        {'a': 9, 'val': None, 'b': 'y'},
        {'a': 9, 'val': 3, 'b': 'z'},
    ])
    ds.backfill('val', 'val_filled')
    assert ds.cols == ['a', 'val_filled', 'val', 'b']
    assert list(ds.unwind('val')) == [1, None, 3]
    assert list(ds.unwind('val_filled')) == [1, 1, 3]


def test_backfill_in_place():
    """Verify an omitted new_colname writes back over the source column.

    Mutation: drop the `or colname` fallback, so the fill lands in a
        column named None and 'val' keeps its gaps.
    Oracle: hand-computed [2, 2, 2] in 'val', with 'val' the only
        column.
    """
    ds = DataSet([
        {'val': None},
        {'val': 2},
        {'val': None},
    ])
    ds.backfill('val')
    assert ds.cols == ['val']
    assert list(ds.unwind('val')) == [2, 2, 2]


def test_backfill_preserves_other_columns(multi_col_dataset):
    """Verify only the named column is read and written.

    Mutation: unwind self.cols[0] rather than colname, filling 'val'
        from the 'id' column.
    Oracle: hand-computed val [10, 10, 30], not id's [1, 2, 3]; id and
        name unchanged.
    """
    multi_col_dataset.backfill('val')
    assert list(multi_col_dataset.unwind('id')) == [1, 2, 3]
    assert list(multi_col_dataset.unwind('name')) == ['a', 'b', 'c']
    assert list(multi_col_dataset.unwind('val')) == [10, 10, 30]


def test_backfill_preserves_column_order():
    """Verify an in-place backfill keeps the column at its position.

    Mutation: drop index=colix, so the rewritten column is appended.
    Oracle: hand-computed ['a', 'b', 'c']; the append gives
        ['a', 'c', 'b'].
    """
    ds = DataSet([
        {'a': 1, 'b': None, 'c': 3},
        {'a': 4, 'b': 5, 'c': 6},
    ])
    assert ds.cols == ['a', 'b', 'c']
    ds.backfill('b')
    assert ds.cols == ['a', 'b', 'c']
    assert list(ds.unwind('b')) == [5, 5]


def test_backfill_empty_dataset():
    """Verify a rowless dataset still gains the target column.

    Mutation: an early return on an empty container, before the column
        is added.
    Oracle: hand-computed cols ['a', 'val_filled', 'val', 'b'] and an
        empty value list.
    """
    ds = DataSet([], cols=['a', 'val', 'b'], typs=[int, float, str])
    ds.backfill('val', 'val_filled')
    assert ds.cols == ['a', 'val_filled', 'val', 'b']
    assert list(ds.unwind('val_filled')) == []


def test_backfill_single_row():
    """Verify a one-row column is copied as it stands.

    Mutation: an early return for a column of fewer than two rows, so
        the target column is never written.
    Oracle: hand-computed cols ['copy', 'val'] with [42] in both, and
        [None] for a lone gap.
    """
    ds = DataSet([{'val': 42}])
    ds.backfill('val', 'copy')
    assert ds.cols == ['copy', 'val']
    assert list(ds.unwind('copy')) == [42]

    gap = DataSet([{'val': None}])
    gap.backfill('val')
    assert list(gap.unwind('val')) == [None]


def test_backfill_float_type():
    """Verify the filled column keeps the source column's own type.

    Mutation: read coltyp from self.columns[0] rather than the source
        column, retyping 'val' as str.
    Oracle: hand-computed floats [1.5, 1.5, 3.7]; the wrong type reads
        back as ['1.5', '1.5', '3.7'].
    """
    ds = DataSet([
        {'label': 'a', 'val': 1.5},
        {'label': 'b', 'val': None},
        {'label': 'c', 'val': 3.7},
    ])
    ds.backfill('val')
    assert list(ds.unwind('val')) == [1.5, 1.5, 3.7]
    assert ds.columns[ds.cols.index('val')][1] is float


def test_backfill_string_type():
    """Verify a numeric-looking string column is not numified.

    Mutation: read coltyp from self.columns[colix - 1], retyping the
        string column as the neighboring float column.
    Oracle: hand-computed ['007', '007', 'x9']; the float type reads
        '007' back as 7.0.
    """
    ds = DataSet([
        {'num': 1.0, 'code': '007'},
        {'num': 2.0, 'code': None},
        {'num': 3.0, 'code': 'x9'},
    ])
    ds.backfill('code')
    assert list(ds.unwind('code')) == ['007', '007', 'x9']
    assert ds.columns[ds.cols.index('code')][1] is str


def test_backfill_date_type():
    """Verify Date values fill both backward at the head and forward.

    Mutation: drop the back-pass, leaving the leading None unfilled.
    Oracle: hand-computed [d1, d1, d1, d2, d2]; forward fill alone
        gives [None, d1, d1, d2, d2].
    """
    d1 = Date.parse('2025-1-1')
    d2 = Date.parse('2025-6-1')
    ds = DataSet([
        {'val': None},
        {'val': d1},
        {'val': None},
        {'val': d2},
        {'val': None},
    ])
    ds.backfill('val')
    assert list(ds.unwind('val')) == [d1, d1, d1, d2, d2]


def test_backfill_falsy_values_preserved():
    """Verify 0 counts as a value to carry, not as a gap.

    Mutation: test the value for truthiness (`if not v`) rather than
        `v is None`.
    Oracle: hand-computed [0, 0, 0]; the truthiness test gives
        [None, None, None].
    """
    ds = DataSet([
        {'val': 0},
        {'val': None},
        {'val': None},
    ])
    ds.backfill('val')
    assert list(ds.unwind('val')) == [0, 0, 0]
