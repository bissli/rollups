"""Comprehensive tests for DataSet diff (difference calculation) operations.

Diff Operation Overview
-----------------------
The diff method adds a new column containing differences between consecutive
values in a numeric column. The index parameter controls the differencing
direction and reference point.

Differencing Modes
------------------

1. Forward Difference (index=0, default):
   - First value: None (no previous value to compare)
   - Subsequent values: current minus previous
   - Example: [10, 20, 30] -> [None, 10, 10]

2. Backward Difference (index=-1 or index=last):
   - Last value: None (no next value to compare)
   - Previous values: current minus next
   - Example: [10, 20, 30] -> [-10, -10, None]

3. Middle Index Difference (index=1, 2, etc.):
   - Value at index: None (pivot point)
   - Before index: current minus next
   - After index: current minus previous
   - Example with index=1: [10, 20, 30, 40] -> [-10, None, 10, 10]

Visual Example
--------------
Dataset: x = [1, 2, 3, 1, 2, 3] (6 rows)

Forward diff (index=0):
    Row 0: None (no previous)
    Row 1: 2-1 = 1
    Row 2: 3-2 = 1
    Row 3: 1-3 = -2
    Row 4: 2-1 = 1
    Row 5: 3-2 = 1
    Result: [None, 1, 1, -2, 1, 1]

Backward diff (index=-1):
    Row 0: 1-2 = -1
    Row 1: 2-3 = -1
    Row 2: 3-1 = 2
    Row 3: 1-2 = -1
    Row 4: 2-3 = -1
    Row 5: None (no next)
    Result: [-1, -1, 2, -1, -1, None]

Middle index diff (index=2):
    Row 0: 1-2 = -1 (before pivot: current-next)
    Row 1: 2-3 = -1 (before pivot: current-next)
    Row 2: None (pivot)
    Row 3: 1-3 = -2 (after pivot: current-previous)
    Row 4: 2-1 = 1 (after pivot: current-previous)
    Row 5: 3-2 = 1 (after pivot: current-previous)
    Result: [-1, -1, None, -2, 1, 1]

Type Support
------------
- Supported types: int, float
- Unsupported types: str, Date, DateTime (will raise AssertionError)
- None values: Handled by libb.safe_diff()
- Result column: Same type as input column

Common Use Cases
----------------
- Time series analysis: Calculate period-over-period changes
- Rate of change: Compute movement between successive readings
- Sequential data: Identify trends and reversals
- Data validation: Detect anomalies in sequential measurements
"""
import math

import pytest
from opendate import Date, DateTime, Time
from rollups import DataSet, diff_datasets

# --- Fixtures ---


@pytest.fixture
def basic_diff_dataset():
    """Basic dataset for diff tests."""
    ds = DataSet([
        {'x': 1, 'y': 8., 'z': 'a'},
        {'x': 2, 'y': 21.8, 'z': 'a'},
        {'x': 3, 'y': 3.2, 'z': 'a'},
        {'x': 1, 'y': 0.1, 'z': 'b'},
        {'x': 2, 'y': 22., 'z': 'b'},
        {'x': 3, 'y': 3., 'z': 'b'}
    ])
    return ds


@pytest.fixture
def float_dataset():
    """Float dataset for diff tests."""
    ds = DataSet([
        {'val': 1.5},
        {'val': 2.5},
        {'val': 4.0},
        {'val': 3.0},
    ])
    ds.columns = [('val', float)]
    return ds


# --- Basic Diff Tests (Forward, Backward, Middle) ---

def test_diff_forward_basic(basic_diff_dataset):
    """Forward diff is current minus previous, first row None.

    Mutation: values[i-1] - values[i] at core.py.
    Oracle: hand-computed [None, 1, 1, -2, 1, 1] over x = 1,2,3,1,2,3.
    """
    basic_diff_dataset.diff('x', 'x_fwd')
    assert list(basic_diff_dataset.unwind('x_fwd')) == [None, 1, 1, -2, 1, 1]


def test_diff_backward_basic(basic_diff_dataset):
    """Backward diff (index=-1) is current minus next, last row None.

    Mutation: values[i] - values[i-1] in the index=-1 branch at
        core.py, which wraps row 0 onto the final value.
    Oracle: hand-computed [-1, -1, 2, -1, -1, None] over x = 1,2,3,1,2,3.
    """
    basic_diff_dataset.diff('x', 'x_bwd', index=-1)
    assert list(basic_diff_dataset.unwind('x_bwd')) == [-1, -1, 2, -1, -1, None]


def test_diff_middle_index(basic_diff_dataset):
    """Pivot at index=2 differences toward the pivot from both sides.

    Mutation: values[i] - values[i-1] before the pivot at
        core.py, dropping the current-minus-next rule.
    Oracle: hand-computed [-1, -1, None, -2, 1, 1] over x = 1,2,3,1,2,3.
    """
    basic_diff_dataset.diff('x', 'x_idx2', index=2)
    assert list(basic_diff_dataset.unwind('x_idx2')) == [-1, -1, None, -2, 1, 1]


def test_diff_index_zero_explicit():
    """An explicit index=0 gives the same forward diff as the default.

    Mutation: values[i] - values[i-2] at core.py, a widened
        differencing window.
    Oracle: hand-computed [None, 5, -2] over val = 5, 10, 8.
    """
    ds = DataSet([
        {'val': 5},
        {'val': 10},
        {'val': 8},
    ])
    ds.diff('val', 'val_diff', index=0)
    assert list(ds.unwind('val_diff')) == [None, 5, -2]


# --- Type Preservation Tests ---

@pytest.mark.parametrize(('type_cls', 'values', 'expected_diff'), [
    (int, [10, 20, 25], [None, 10, 5]),
    (float, [1.5, 2.5, 4.0], [None, 1.0, 1.5]),
])
def test_diff_type_preservation(type_cls, values, expected_diff):
    """The new column carries the source column's declared type.

    Mutation: a hardcoded float in place of coltyp at
        core.py.
    Oracle: the declared source type, plus hand-computed differences.
    """
    ds = DataSet([{'val': v} for v in values])
    ds.columns = [('val', type_cls)]
    ds.diff('val', 'val_diff')

    assert ds.colmap['val_diff'] == type_cls
    result = list(ds.unwind('val_diff'))
    assert result[0] is None
    assert result[1:] == pytest.approx(expected_diff[1:])


def test_diff_float_column(float_dataset):
    """Forward diff keeps the sign of a falling float step.

    Mutation: abs() around the difference at core.py.
    Oracle: hand-computed [None, 1.0, 1.5, -1.0] over 1.5, 2.5, 4.0, 3.0.
    """
    float_dataset.diff('val', 'val_diff')
    result = list(float_dataset.unwind('val_diff'))
    assert result[0] is None
    assert result[1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(1.5)
    assert result[3] == pytest.approx(-1.0)


def test_diff_float_precision():
    """Differences are exact IEEE-754 subtractions, never rounded.

    Mutation: round(value, 6) applied to the difference at
        core.py.
    Oracle: hand-computed 2.222 - 1.111 = 1.111 exactly, while
        3.333 - 2.222 = 1.1110000000000002.
    """
    ds = DataSet([
        {'val': 1.111},
        {'val': 2.222},
        {'val': 3.333},
    ])
    ds.columns = [('val', float)]
    ds.diff('val', 'val_diff')
    result = list(ds.unwind('val_diff'))
    assert result[0] is None
    assert result[1] == 1.111
    assert result[2] == 1.1110000000000002


# --- Index Validation Tests ---

def test_diff_index_out_of_bounds():
    """The row count, not the last index, is the first rejected index.

    Mutation: index > len(values) in place of index > last_idx at
        core.py.
    Oracle: the boundary pair - index=3 on 3 rows raises, index=2 is the
        backward diff [-1, -1, None].
    """
    ds = DataSet([
        {'val': 1},
        {'val': 2},
        {'val': 3},
    ])
    with pytest.raises(
        ValueError,
        match='index 3 out of range for dataset with 3 rows'):
        ds.diff('val', 'val_diff', index=3)
    with pytest.raises(
        ValueError,
        match='index 100 out of range for dataset with 3 rows'):
        ds.diff('val', 'val_diff', index=100)

    ds.diff('val', 'val_diff', index=2)
    assert list(ds.unwind('val_diff')) == [-1, -1, None]


def test_diff_negative_index_not_minus_one():
    """Only -1 is an accepted negative index; -2 is rejected.

    Mutation: index < -2 in place of index < -1 at core.py,
        which lets -2 fall through and silently wrap onto the last row.
    Oracle: the boundary -2, one step past the only legal negative.
    """
    ds = DataSet([
        {'val': 10},
        {'val': 20},
        {'val': 30},
        {'val': 40},
        {'val': 50},
    ])
    with pytest.raises(
        ValueError,
        match='index -2 out of range for dataset with 5 rows'):
        ds.diff('val', 'val_diff', index=-2)


def test_diff_index_equals_last():
    """A pivot on the last row differences backward, last row None.

    Mutation: values[i] - values[i-1] at core.py, which wraps
        row 0 onto the final value.
    Oracle: hand-computed [-15, -5, None] over val = 10, 25, 30.
    """
    ds = DataSet([
        {'val': 10},
        {'val': 25},
        {'val': 30},
    ])
    ds.diff('val', 'val_diff', index=2)
    assert list(ds.unwind('val_diff')) == [-15, -5, None]


# --- Small Dataset Tests (Parameterized) ---

@pytest.mark.parametrize(('direction', 'index', 'expected'), [
    ('forward', 0, [None]),
    ('backward', -1, [None]),
])
def test_diff_single_row(direction, index, expected):
    """A lone row has no neighbor, so its difference is None.

    Mutation: dropping the trailing diffs.append(None) at
        core.py, leaving the column one value short of the
        row count.
    Oracle: a one-row dataset has no pair to subtract, so [None].
    """
    ds = DataSet([{'val': 42}])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff', index=index)
    assert list(ds.unwind('val_diff')) == expected


@pytest.mark.parametrize(('direction', 'index', 'expected'), [
    ('forward', 0, [None, 15]),
    ('backward', -1, [-15, None]),
])
def test_diff_two_rows(direction, index, expected):
    """Forward and backward put the None at opposite ends.

    Mutation: swapping the index==0 and index==-1 branch bodies at
        core.py.
    Oracle: hand-computed [None, 15] and [-15, None] over val = 10, 25.
    """
    ds = DataSet([
        {'val': 10},
        {'val': 25},
    ])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff', index=index)
    assert list(ds.unwind('val_diff')) == expected


@pytest.mark.parametrize(('direction', 'index', 'expected'), [
    ('forward', 0, [None, 5, -3]),
    ('backward', -1, [-5, 3, None]),
    ('middle', 1, [-5, None, -3]),
])
def test_diff_three_rows(direction, index, expected):
    """All three pivots over one non-monotonic three-row series.

    Mutation: values[i] - values[i-1] before the pivot at
        core.py, which reads values[-1] for row 0.
    Oracle: hand-computed per direction over val = 5, 10, 7.
    """
    ds = DataSet([
        {'val': 5},
        {'val': 10},
        {'val': 7},
    ])
    ds.diff('val', 'val_diff', index=index)
    assert list(ds.unwind('val_diff')) == expected


# --- Empty Dataset Tests ---

def test_diff_empty_dataset():
    """An empty dataset still gains the typed difference column.

    Mutation: a bare return in the empty guard at core.py,
        skipping add_column.
    Oracle: the declared int type on an added column holding no values.
    """
    ds = DataSet([], columns=[('val', int)])
    ds.diff('val', 'val_diff')
    assert len(ds) == 0
    assert 'val_diff' in ds.cols
    assert ds.colmap['val_diff'] == int
    assert list(ds.unwind('val_diff')) == []


# --- None Value Handling Tests ---

def test_diff_with_none_values():
    """A None on either side of a pair makes that difference None.

    Mutation: coalescing None to 0 in place of libb.safe_diff at
        core.py, which would give [None, -10, 30, 10].
    Oracle: hand-computed [None, None, None, 10] over 10, None, 30, 40.
    """
    ds = DataSet([
        {'val': 10},
        {'val': None},
        {'val': 30},
        {'val': 40},
    ])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff')
    assert list(ds.unwind('val_diff')) == [None, None, None, 10]


# --- Column Preservation Tests ---

def test_diff_preserves_other_columns():
    """Differencing writes only the new column, leaving the rest alone.

    Mutation: add_column(colname, ...) in place of new_colname at
        core.py, overwriting the source column.
    Oracle: hand-computed [None, 10, 15] beside the untouched inputs.
    """
    ds = DataSet([
        {'id': 1, 'val': 10},
        {'id': 2, 'val': 20},
        {'id': 3, 'val': 35},
    ])
    ds.diff('val', 'val_diff')
    assert list(ds.unwind('id')) == [1, 2, 3]
    assert list(ds.unwind('val')) == [10, 20, 35]
    assert list(ds.unwind('val_diff')) == [None, 10, 15]


def test_diff_column_order_preserved():
    """The difference column is appended, never inserted.

    Mutation: add_column(..., index=0) at core.py.
    Oracle: the pre-diff column list with the new name at the tail.
    """
    ds = DataSet([
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 4, 'b': 5, 'c': 6},
    ])
    original_cols = ds.cols.copy()
    ds.diff('b', 'b_diff')

    assert ds.cols == original_cols + ['b_diff']


# --- Middle Index Position Tests ---

def test_diff_index_middle():
    """A middle pivot holds None and both sides difference toward it.

    Mutation: dropping the i == index branch at core.py,
        so the pivot row would carry 30 - 20 = 10 instead of None.
    Oracle: hand-computed [-10, -10, None, 20, 10] over 10, 20, 30, 50, 60.
    """
    ds = DataSet([
        {'val': 10},
        {'val': 20},
        {'val': 30},
        {'val': 50},
        {'val': 60},
    ])
    ds.diff('val', 'val_diff', index=2)
    assert list(ds.unwind('val_diff')) == [-10, -10, None, 20, 10]


def test_diff_index_second_position():
    """Rows after the pivot difference against the previous row.

    Mutation: values[i] - values[i+1] after the pivot at
        core.py, which would give -15 instead of 15 on row 2.
    Oracle: hand-computed [-10, None, 15, 5] over 10, 20, 35, 40.
    """
    ds = DataSet([
        {'val': 10},
        {'val': 20},
        {'val': 35},
        {'val': 40},
    ])
    ds.diff('val', 'val_diff', index=1)
    assert list(ds.unwind('val_diff')) == [-10, None, 15, 5]


def test_diff_index_second_to_last():
    """The pivot sits on the requested row, not its neighbor.

    Mutation: elif i == index - 1 at core.py, an off-by-one
        that parks the None on row 2.
    Oracle: hand-computed [-5, -5, -5, None, 5] over 5, 10, 15, 20, 25.
    """
    ds = DataSet([
        {'val': 5},
        {'val': 10},
        {'val': 15},
        {'val': 20},
        {'val': 25},
    ])
    ds.diff('val', 'val_diff', index=3)
    assert list(ds.unwind('val_diff')) == [-5, -5, -5, None, 5]


# --- Value Pattern Tests ---

@pytest.mark.parametrize(('test_id', 'values', 'expected'), [
    ('negative', [100, 80, 60, 50], [None, -20, -20, -10]),
    ('all_same', [100, 100, 100, 100], [None, 0, 0, 0]),
    ('mixed', [10, -5, 15, -20], [None, -15, 20, -35]),
    ('all_negative', [-10, -20, -5, -15], [None, -10, 15, -10]),
    ('zero', [10, 0, -5, 0], [None, -10, -5, 5]),
    ('alternating', [100, 10, 100, 10, 100], [None, -90, 90, -90, 90]),
])
def test_diff_value_patterns(test_id, values, expected):
    """Sign and magnitude survive falling, mixed, and negative series.

    Mutation: abs() around the difference at core.py, or the
        operands reversed.
    Oracle: hand-computed differences per pattern.
    """
    ds = DataSet([{'val': v} for v in values])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff')
    assert list(ds.unwind('val_diff')) == expected


def test_diff_forward_matches_pairwise_oracle():
    """Forward diff matches a pairwise re-implementation, row for row.

    Mutation: an off-by-one in the forward range at core.py,
        such as range(2, len(values)) or values[i] - values[i-2].
    Oracle: a differential re-implementation, zip(values, values[1:]).
    """
    values = [3, 3, -7, 12, 0, 5, -5, 100, 99, 1]
    ds = DataSet([{'val': v} for v in values])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff')

    expected = [None] + [b - a for a, b in zip(values, values[1:])]
    assert list(ds.unwind('val_diff')) == expected


def test_diff_large_numbers():
    """Large integer steps are differenced without a scale change.

    Mutation: a dropped or added scale factor at core.py,
        such as dividing the difference by 1000.
    Oracle: hand-computed 1_500_000 and 500_000.
    """
    ds = DataSet([
        {'val': 1_000_000},
        {'val': 2_500_000},
        {'val': 3_000_000},
    ])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff')
    assert list(ds.unwind('val_diff')) == [None, 1_500_000, 500_000]


# --- Unsupported Type Tests ---

@pytest.mark.parametrize(('type_cls', 'values'), [
    (str, ['a', 'b', 'c']),
    (Date, [Date(2024, 1, 1), Date(2024, 1, 2)]),
])
def test_diff_assertion_error_for_unsupported_types(type_cls, values):
    """Non-numeric columns are refused before any differencing.

    Mutation: widening the type set at core.py, e.g.
        coltyp is not None.
    Oracle: the AssertionError naming int and float.
    """
    ds = DataSet([{'val': v} for v in values])
    ds.columns = [('val', type_cls)]
    with pytest.raises(AssertionError, match='only supports int and float'):
        ds.diff('val', 'val_diff')


# --- Multiple Operations Tests ---

def test_diff_multiple_operations_same_dataset():
    """Each call honors its own index on the same source column.

    Mutation: the index argument ignored at core.py, e.g.
        index >= -1, so val_diff2 would repeat the forward val_diff1.
    Oracle: hand-computed forward [None, 10, 10, 10] and backward
        [-10, -10, -10, None] over the same 10, 20, 30, 40.
    """
    ds = DataSet([
        {'val': 10},
        {'val': 20},
        {'val': 30},
        {'val': 40},
    ])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff1')
    ds.diff('val', 'val_diff2', index=-1)

    assert 'val_diff1' in ds.cols
    assert 'val_diff2' in ds.cols
    assert list(ds.unwind('val_diff1')) == [None, 10, 10, 10]
    assert list(ds.unwind('val_diff2')) == [-10, -10, -10, None]


def test_diff_with_existing_computed_columns():
    """Differencing leaves an earlier computed column intact.

    Mutation: add_column(colname, ...) in place of new_colname at
        core.py, which would overwrite 'val' and leave
        'doubled' stale against it.
    Oracle: hand-computed [20, 40, 60] beside [None, 10, 10].
    """
    ds = DataSet([
        {'val': 10},
        {'val': 20},
        {'val': 30},
    ])
    ds.columns = [('val', int)]
    ds.add_column('doubled', int, value=lambda r: r['val'] * 2)
    ds.diff('val', 'val_diff')

    assert 'doubled' in ds.cols
    assert 'val_diff' in ds.cols
    assert list(ds.unwind('doubled')) == [20, 40, 60]
    assert list(ds.unwind('val_diff')) == [None, 10, 10]


def test_diff_in_place_replaces_source_column():
    """Differencing a column into itself replaces it, adding no column.

    Mutation: reading values after add_column at core.py,
        which would difference the already-overwritten column.
    Oracle: hand-computed [None, 10, 15] from the pre-diff 10, 20, 35.
    """
    ds = DataSet([
        {'value': 10},
        {'value': 20},
        {'value': 35},
    ])
    ds.columns = [('value', int)]
    ds.diff('value', 'value')

    assert ds.cols == ['value']
    assert list(ds.unwind('value')) == [None, 10, 15]


def test_diff_index_last_minus_one():
    """index=-1 reaches the backward branch, not the pivot branch.

    Mutation: elif index == last_idx at core.py, dropping -1
        from the branch, which sends -1 to the pivot loop and yields
        [-30, 10, 15, 5].
    Oracle: hand-computed [-10, -15, -5, None] over 10, 20, 35, 40.
    """
    ds = DataSet([
        {'val': 10},
        {'val': 20},
        {'val': 35},
        {'val': 40},
    ])
    ds.diff('val', 'val_diff', index=-1)
    assert list(ds.unwind('val_diff')) == [-10, -15, -5, None]


# --- diff_datasets Function Tests - Basic ---

def test_diff_datasets_identical():
    """Identical datasets put every row in same and none in diff.

    Mutation: == in place of != at join.py, which would flag
        every matching column as a change.
    Oracle: hand-partitioned - two shared keys, no differing values.
    """
    ds1 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 100},
        {'id': 2, 'name': 'Bob', 'value': 200},
    ])
    ds2 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 100},
        {'id': 2, 'name': 'Bob', 'value': 200},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(
        ds1, ds2, ['id'], ['name', 'value'])

    assert len(same) == 2
    assert len(diff) == 0
    assert len(only_in_ds1) == 0
    assert len(only_in_ds2) == 0


def test_diff_datasets_completely_different():
    """Disjoint keys land wholly in the two only_in lists.

    Mutation: only_in_ds2 built from the keys ds1 also holds at
        join.py, which would leave it empty here.
    Oracle: hand-partitioned - ids 1,2 against ids 3,4.
    """
    ds1 = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
    ])
    ds2 = DataSet([
        {'id': 3, 'value': 300},
        {'id': 4, 'value': 400},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == 0
    assert len(diff) == 0
    assert len(only_in_ds1) == 2
    assert len(only_in_ds2) == 2
    assert only_in_ds1[0]['id'] == 1
    assert only_in_ds1[1]['id'] == 2
    assert only_in_ds2[0]['id'] == 3
    assert only_in_ds2[1]['id'] == 4


def test_diff_datasets_some_differences():
    """A diff row pairs ds1 then ds2, and Nones the matching columns.

    Mutation: (ds2_row[col], ds1_row[col]) tuple order at
        join.py.
    Oracle: hand-computed (100, 150) and ('Bob', 'Robert') with the
        unchanged column None.
    """
    ds1 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 100},
        {'id': 2, 'name': 'Bob', 'value': 200},
    ])
    ds2 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 150},
        {'id': 2, 'name': 'Robert', 'value': 200},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(
        ds1, ds2, ['id'], ['name', 'value'])

    assert len(same) == 0
    assert len(diff) == 2
    assert len(only_in_ds1) == 0
    assert len(only_in_ds2) == 0

    diff_row_1 = next(r for r in diff if r['id'] == 1)
    assert diff_row_1['name'] is None
    assert diff_row_1['value'] == (100, 150)

    diff_row_2 = next(r for r in diff if r['id'] == 2)
    assert diff_row_2['name'] == ('Bob', 'Robert')
    assert diff_row_2['value'] is None


def test_diff_datasets_partial_overlap():
    """A matched key is consumed, so it cannot also be only_in_ds2.

    Mutation: ds2_map.get(key, None) in place of pop at
        join.py, which would leave ids 2 and 3 in only_in_ds2.
    Oracle: hand-partitioned - same 3, diff 2, only 1, only 4.
    """
    ds1 = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
        {'id': 3, 'value': 300},
    ])
    ds2 = DataSet([
        {'id': 2, 'value': 250},
        {'id': 3, 'value': 300},
        {'id': 4, 'value': 400},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == 1
    assert len(diff) == 1
    assert len(only_in_ds1) == 1
    assert len(only_in_ds2) == 1

    assert same[0]['id'] == 3
    assert diff[0]['id'] == 2
    assert diff[0]['value'] == (200, 250)
    assert only_in_ds1[0]['id'] == 1
    assert only_in_ds2[0]['id'] == 4


# --- diff_datasets Function Tests - Keys ---

def test_diff_datasets_multiple_key_columns():
    """Every key column takes part in matching, not just the first.

    Mutation: tuple(row[keycols[0]]) at join.py, a dropped
        term that collapses both US rows onto one key.
    Oracle: hand-partitioned - CA differs, TX only in ds1, FL only in ds2.
    """
    ds1 = DataSet([
        {'country': 'US', 'state': 'CA', 'population': 39_000_000},
        {'country': 'US', 'state': 'TX', 'population': 29_000_000},
    ])
    ds2 = DataSet([
        {'country': 'US', 'state': 'CA', 'population': 39_500_000},
        {'country': 'US', 'state': 'FL', 'population': 21_000_000},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(
        ds1, ds2, ['country', 'state'], ['population']
    )

    assert len(same) == 0
    assert len(diff) == 1
    assert len(only_in_ds1) == 1
    assert len(only_in_ds2) == 1

    diff_row = diff[0]
    assert diff_row['country'] == 'US'
    assert diff_row['state'] == 'CA'
    assert diff_row['population'] == (39_000_000, 39_500_000)
    assert only_in_ds1[0]['state'] == 'TX'
    assert only_in_ds2[0]['state'] == 'FL'


def test_diff_datasets_key_sorting():
    """Results follow sorted key order, not ds1 row order.

    Mutation: for key in ds1_map at join.py, dropping the
        sort, which would give same[0] id 3 from ds1's own order.
    Oracle: hand-sorted key order 1, 2, 3 against the input order 3, 1, 2.
    """
    ds1 = DataSet([
        {'id': 3, 'value': 300},
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
    ])
    ds2 = DataSet([
        {'id': 2, 'value': 250},
        {'id': 3, 'value': 300},
        {'id': 1, 'value': 100},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == 2
    assert len(diff) == 1
    assert same[0]['id'] == 1
    assert same[1]['id'] == 3
    assert diff[0]['id'] == 2


# --- diff_datasets Function Tests - Empty Datasets (Parameterized) ---

@pytest.mark.parametrize(
    (
        'ds1_data',
        'ds2_data',
        'expected_same',
        'expected_diff',
        'expected_only1',
        'expected_only2'
    ),
    [
        # Both empty
        ([], [], 0, 0, 0, 0),
        # Empty ds1
        ([], [{'id': 1, 'value': 100}], 0, 0, 0, 1),
        # Empty ds2
        ([{'id': 1, 'value': 100}], [], 0, 0, 1, 0),
    ])
def test_diff_datasets_empty_scenarios(
    ds1_data,
    ds2_data,
    expected_same,
    expected_diff,
    expected_only1,
    expected_only2):
    """An empty side leaves the other side's rows in its only_in list.

    Mutation: an early return of four empty lists when either dataset is
        empty, ahead of join.py.
    Oracle: hand-counted per scenario - the non-empty side keeps its row.
    """
    ds1 = DataSet(ds1_data)
    ds2 = DataSet(ds2_data)

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == expected_same
    assert len(diff) == expected_diff
    assert len(only_in_ds1) == expected_only1
    assert len(only_in_ds2) == expected_only2


def test_diff_datasets_single_row_each():
    """A diff row carries the value pair, not the ds2 row.

    Mutation: diff.append(ds2_row) at join.py, which would
        put a bare 150 under 'value'.
    Oracle: hand-computed (100, 150).
    """
    ds1 = DataSet([{'id': 1, 'value': 100}])
    ds2 = DataSet([{'id': 1, 'value': 150}])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == 0
    assert len(diff) == 1
    assert len(only_in_ds1) == 0
    assert len(only_in_ds2) == 0
    assert diff[0]['value'] == (100, 150)


# --- diff_datasets Function Tests - None Values ---

def test_diff_datasets_with_none_values():
    """A None on one side is a change, and keeps its position.

    Mutation: skipping a column when either side is None at
        join.py, which would call both rows the same.
    Oracle: hand-computed (None, 100) and (200, None).
    """
    ds1 = DataSet([
        {'id': 1, 'value': None},
        {'id': 2, 'value': 200},
    ])
    ds2 = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': None},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == 0
    assert len(diff) == 2

    diff_row_1 = next(r for r in diff if r['id'] == 1)
    assert diff_row_1['value'] == (None, 100)

    diff_row_2 = next(r for r in diff if r['id'] == 2)
    assert diff_row_2['value'] == (200, None)


def test_diff_datasets_same_none_values():
    """None on both sides is a match, not a change.

    Mutation: flagging any None as a change at join.py, e.g.
        ds1_row[col] != ds2_row[col] or ds1_row[col] is None.
    Oracle: None == None, so the row belongs in same.
    """
    ds1 = DataSet([
        {'id': 1, 'value': None},
    ])
    ds2 = DataSet([
        {'id': 1, 'value': None},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == 1
    assert len(diff) == 0


# --- diff_datasets Function Tests - Multiple Columns ---

def test_diff_datasets_multiple_compare_columns():
    """One changed column out of three is enough to make a diff row.

    Mutation: all() in place of any() at join.py, which would
        send id 1 to same because its name is unchanged.
    Oracle: hand-partitioned - id 1 differs on age and salary only,
        id 2 matches throughout.
    """
    ds1 = DataSet([
        {'id': 1, 'name': 'Alice', 'age': 30, 'salary': 50000},
        {'id': 2, 'name': 'Bob', 'age': 25, 'salary': 45000},
    ])
    ds2 = DataSet([
        {'id': 1, 'name': 'Alice', 'age': 31, 'salary': 55000},
        {'id': 2, 'name': 'Bob', 'age': 25, 'salary': 45000},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(
        ds1, ds2, ['id'], ['name', 'age', 'salary']
    )

    assert len(same) == 1
    assert len(diff) == 1

    same_row = same[0]
    assert same_row['id'] == 2
    assert same_row['name'] == 'Bob'

    diff_row = diff[0]
    assert diff_row['id'] == 1
    assert diff_row['name'] is None
    assert diff_row['age'] == (30, 31)
    assert diff_row['salary'] == (50000, 55000)


def test_diff_datasets_only_includes_specified_columns():
    """A diff row holds the key and compare columns and nothing else.

    Mutation: seeding diff_row from the whole ds1 row at
        join.py, which would carry 'extra' through.
    Oracle: the requested column set - id, name, value - with 'extra'
        present in both sources but absent from the diff row.
    """
    ds1 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 100, 'extra': 'data1'},
    ])
    ds2 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 150, 'extra': 'data2'},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(
        ds1, ds2, ['id'], ['name', 'value'])

    assert len(diff) == 1
    diff_row = diff[0]
    assert 'id' in diff_row
    assert 'name' in diff_row
    assert 'value' in diff_row
    assert 'extra' not in diff_row
    assert diff_row['value'] == (100, 150)


def test_diff_datasets_all_columns_differ():
    """Every differing column is recorded, not just the first.

    Mutation: a break after the first change at join.py,
        leaving col2 and col3 None.
    Oracle: hand-computed pairs for all three columns.
    """
    ds1 = DataSet([
        {'id': 1, 'col1': 'a', 'col2': 10, 'col3': 100.0},
    ])
    ds2 = DataSet([
        {'id': 1, 'col1': 'b', 'col2': 20, 'col3': 200.0},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(
        ds1, ds2, ['id'], ['col1', 'col2', 'col3']
    )

    assert len(diff) == 1
    diff_row = diff[0]
    assert diff_row['col1'] == ('a', 'b')
    assert diff_row['col2'] == (10, 20)
    assert diff_row['col3'] == (100.0, 200.0)


# --- diff_datasets Function Tests - Type Comparisons ---

def test_diff_datasets_numeric_types():
    """Comparison is by value, so 100 and 100.0 are not a change.

    Mutation: is not in place of != at join.py, which would
        call the int 100 and the float 100.0 a change.
    Oracle: hand-computed 100 == 100.0 puts id 1 in same, while
        10.5 != 15.7 puts id 2 in diff.
    """
    ds1 = DataSet([
        {'id': 1, 'int_val': 100, 'float_val': 10.5},
        {'id': 2, 'int_val': 100, 'float_val': 10.5},
    ])
    ds2 = DataSet([
        {'id': 1, 'int_val': 100.0, 'float_val': 10.5},
        {'id': 2, 'int_val': 150, 'float_val': 15.7},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(
        ds1, ds2, ['id'], ['int_val', 'float_val']
    )

    assert len(same) == 1
    assert same[0]['id'] == 1
    assert len(diff) == 1
    diff_row = diff[0]
    assert diff_row['id'] == 2
    assert diff_row['int_val'] == (100, 150.0)
    assert diff_row['float_val'] == (10.5, 15.7)


def test_diff_datasets_string_comparison():
    """String columns compare for inequality, not for order.

    Mutation: ds1_row[col] < ds2_row[col] at join.py, which
        would call 'pending' vs 'active' a match.
    Oracle: hand-computed ('pending', 'active') in diff, with the equal
        'inactive' pair in same.
    """
    ds1 = DataSet([
        {'id': 1, 'status': 'pending'},
        {'id': 2, 'status': 'inactive'},
    ])
    ds2 = DataSet([
        {'id': 1, 'status': 'active'},
        {'id': 2, 'status': 'inactive'},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['status'])

    assert len(same) == 1
    assert same[0]['id'] == 2
    assert len(diff) == 1
    assert diff[0]['status'] == ('pending', 'active')


def test_diff_datasets_date_comparison():
    """Dates compare to the day, not to the month.

    Mutation: comparing (year, month) at join.py, which would
        call Jan 1 and Jan 15 a match.
    Oracle: hand-computed pair (Date(2024, 1, 1), Date(2024, 1, 15)),
        with the equal Feb 1 pair in same.
    """
    ds1 = DataSet([
        {'id': 1, 'date': Date(2024, 1, 1)},
        {'id': 2, 'date': Date(2024, 2, 1)},
    ])
    ds2 = DataSet([
        {'id': 1, 'date': Date(2024, 1, 15)},
        {'id': 2, 'date': Date(2024, 2, 1)},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['date'])

    assert len(same) == 1
    assert len(diff) == 1
    assert diff[0]['date'] == (Date(2024, 1, 1), Date(2024, 1, 15))


# --- diff_datasets Function Tests - Row Preservation ---

def test_diff_datasets_preserves_full_rows_in_same():
    """A same row is the whole ds1 row, not just the compared columns.

    Mutation: same.append(ds2_row) at join.py, which would
        carry ds2's 'extra' value.
    Oracle: 'data' from ds1 where ds2 holds 'different'.
    """
    ds1 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 100, 'extra': 'data'},
    ])
    ds2 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 100, 'extra': 'different'},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(
        ds1, ds2, ['id'], ['name', 'value'])

    assert len(same) == 1
    same_row = same[0]
    assert same_row['id'] == 1
    assert same_row['name'] == 'Alice'
    assert same_row['value'] == 100
    assert same_row['extra'] == 'data'


def test_diff_datasets_preserves_full_rows_in_only_lists():
    """An only_in row is the whole source row, not a key projection.

    Mutation: appending attrdict(zip(keycols, key)) at
        join.py and 3005, dropping the uncompared columns.
    Oracle: 'data1' and 'data2', neither a key nor a compare column.
    """
    ds1 = DataSet([
        {'id': 1, 'name': 'Alice', 'value': 100, 'extra': 'data1'},
    ])
    ds2 = DataSet([
        {'id': 2, 'name': 'Bob', 'value': 200, 'extra': 'data2'},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(only_in_ds1) == 1
    assert only_in_ds1[0]['extra'] == 'data1'

    assert len(only_in_ds2) == 1
    assert only_in_ds2[0]['extra'] == 'data2'


# --- diff_datasets Function Tests - Large Datasets ---

def test_diff_datasets_large_dataset():
    """Rows are matched by key, so only the changed keys reach diff.

    Mutation: matching ds1 row i against ds2 row i+1 at
        join.py, which would flag every key as changed.
    Oracle: hand-derived - ds2 adds 5 to the odd ids only, so diff holds
        exactly the odd ids and id 3 pairs (30, 35).
    """
    ds1 = DataSet([{'id': i, 'value': i * 10} for i in range(100)])
    ds2 = DataSet([
        {'id': i, 'value': i * 10 + (5 if i % 2 else 0)}
        for i in range(100)])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == 50
    assert len(diff) == 50
    assert len(only_in_ds1) == 0
    assert len(only_in_ds2) == 0
    assert {r['id'] for r in diff} == {i for i in range(100) if i % 2}
    assert next(r for r in diff if r['id'] == 3)['value'] == (30, 35)


def test_diff_datasets_duplicate_keys_uses_last():
    """A repeated key keeps the last row, not the first.

    Mutation: setdefault in place of the dict comprehension at
        join.py, which would compare 100 instead of 999.
    Oracle: hand-computed (999, 200) from the second id 1 row.
    """
    ds1 = DataSet([
        {'id': 1, 'value': 100},
        {'id': 1, 'value': 999},
    ])
    ds2 = DataSet([
        {'id': 1, 'value': 200},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(diff) == 1
    assert diff[0]['value'] == (999, 200)


# --- Edge Case Tests ---

def test_diff_with_datetime_values():
    """A DateTime column is refused before any differencing.

    Mutation: widening the type set at core.py to admit any
        subtractable type.
    Oracle: the AssertionError naming int and float.
    """
    ds = DataSet([
        {'val': DateTime(2024, 1, 1, 10, 30)},
        {'val': DateTime(2024, 1, 2, 10, 30)},
    ])
    ds.columns = [('val', DateTime)]
    with pytest.raises(AssertionError, match='only supports int and float'):
        ds.diff('val', 'val_diff')


def test_diff_with_time_values():
    """A Time column is refused before any differencing.

    Mutation: widening the type set at core.py to admit any
        subtractable type.
    Oracle: the AssertionError naming int and float.
    """
    ds = DataSet([
        {'val': Time(10, 30, 0)},
        {'val': Time(11, 45, 0)},
    ])
    ds.columns = [('val', Time)]
    with pytest.raises(AssertionError, match='only supports int and float'):
        ds.diff('val', 'val_diff')


def test_diff_very_large_dataset():
    """Every row of a thousand keeps its own differencing window.

    Mutation: values[i] - values[i-2] at core.py, which would
        give 4i - 4 instead of 2i - 1.
    Oracle: hand-derived i^2 - (i-1)^2 = 2i - 1 over val = i^2.
    """
    ds = DataSet([{'val': i * i} for i in range(1000)])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff')

    result = list(ds.unwind('val_diff'))
    assert result == [None] + [2 * i - 1 for i in range(1, 1000)]


def test_diff_with_all_none_values():
    """An all-None column differences to all None.

    Mutation: coalescing None to 0 in place of libb.safe_diff at
        core.py, which would give [None, 0, 0].
    Oracle: hand-computed [None, None, None].
    """
    ds = DataSet([
        {'val': None},
        {'val': None},
        {'val': None},
    ])
    ds.columns = [('val', int)]
    ds.diff('val', 'val_diff')
    assert list(ds.unwind('val_diff')) == [None, None, None]


def test_diff_with_infinity_values():
    """Infinities keep their sign through the subtraction.

    Mutation: values[i-1] - values[i] at core.py, which would
        turn both -inf results into +inf.
    Oracle: IEEE-754 - 100 - inf = -inf and -inf - 100 = -inf.
    """
    ds = DataSet([
        {'val': float('inf')},
        {'val': 100.0},
        {'val': float('-inf')},
    ])
    ds.columns = [('val', float)]
    ds.diff('val', 'val_diff')
    result = list(ds.unwind('val_diff'))
    assert result[0] is None
    assert result[1] == float('-inf')
    assert result[2] == float('-inf')


def test_diff_with_nan_values():
    """NaN propagates through a difference instead of becoming None.

    Mutation: a NaN guard added beside the None guard behind
        core.py, returning None for a NaN operand.
    Oracle: IEEE-754 - both the pair spanning NaN give NaN, so only the
        first row is None.
    """
    ds = DataSet([
        {'val': 10.0},
        {'val': float('nan')},
        {'val': 30.0},
    ])
    ds.columns = [('val', float)]
    ds.diff('val', 'val_diff')
    result = list(ds.unwind('val_diff'))
    assert result[0] is None
    assert math.isnan(result[1])
    assert math.isnan(result[2])


def test_diff_datasets_with_datetime_key():
    """DateTime keys match on equality, not on object identity.

    Mutation: tuple(id(row[col]) ...) at join.py, so two
        equal timestamps built as separate objects stop matching.
    Oracle: hand-partitioned - Jan 1 shared, Jan 2 only in ds1, Jan 3
        only in ds2.
    """
    ds1 = DataSet([
        {'dt': DateTime(2024, 1, 1, 10, 0), 'value': 100},
        {'dt': DateTime(2024, 1, 2, 10, 0), 'value': 200},
    ])
    ds2 = DataSet([
        {'dt': DateTime(2024, 1, 1, 10, 0), 'value': 150},
        {'dt': DateTime(2024, 1, 3, 10, 0), 'value': 300},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['dt'], ['value'])

    assert len(same) == 0
    assert len(diff) == 1
    assert diff[0]['dt'] == DateTime(2024, 1, 1, 10, 0)
    assert diff[0]['value'] == (100, 150)
    assert only_in_ds1[0]['dt'] == DateTime(2024, 1, 2, 10, 0)
    assert only_in_ds2[0]['dt'] == DateTime(2024, 1, 3, 10, 0)


def test_diff_datasets_with_date_key():
    """Date keys match, and the unmatched dates split by source.

    Mutation: only_in_ds1 and only_in_ds2 swapped at join.py.
    Oracle: hand-partitioned - Jan 1 shared and equal, Jan 2 only in ds1,
        Jan 3 only in ds2.
    """
    ds1 = DataSet([
        {'date': Date(2024, 1, 1), 'value': 100},
        {'date': Date(2024, 1, 2), 'value': 200},
    ])
    ds2 = DataSet([
        {'date': Date(2024, 1, 1), 'value': 100},
        {'date': Date(2024, 1, 3), 'value': 300},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['date'], ['value'])

    assert len(same) == 1
    assert len(diff) == 0
    assert same[0]['date'] == Date(2024, 1, 1)
    assert only_in_ds1[0]['date'] == Date(2024, 1, 2)
    assert only_in_ds2[0]['date'] == Date(2024, 1, 3)


def test_diff_datasets_with_time_key():
    """Time keys match to the hour they name.

    Mutation: keying on the hour taken modulo 2 at join.py,
        which would match 10:00 with 12:00.
    Oracle: hand-partitioned - 10:00 shared, 11:00 only in ds1, 12:00
        only in ds2.
    """
    ds1 = DataSet([
        {'time': Time(10, 0, 0), 'value': 100},
        {'time': Time(11, 0, 0), 'value': 200},
    ])
    ds2 = DataSet([
        {'time': Time(10, 0, 0), 'value': 150},
        {'time': Time(12, 0, 0), 'value': 300},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['time'], ['value'])

    assert len(same) == 0
    assert len(diff) == 1
    assert diff[0]['time'] == Time(10, 0, 0)
    assert diff[0]['value'] == (100, 150)
    assert only_in_ds1[0]['time'] == Time(11, 0, 0)
    assert only_in_ds2[0]['time'] == Time(12, 0, 0)


def test_diff_datasets_with_boolean_values():
    """A boolean change is recorded in ds1-then-ds2 order.

    Mutation: (ds2_row[col], ds1_row[col]) tuple order at
        join.py, giving (False, True).
    Oracle: hand-computed (True, False), with the False pair in same.
    """
    ds1 = DataSet([
        {'id': 1, 'active': True},
        {'id': 2, 'active': False},
    ])
    ds2 = DataSet([
        {'id': 1, 'active': False},
        {'id': 2, 'active': False},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['active'])

    assert len(same) == 1
    assert len(diff) == 1
    assert diff[0]['active'] == (True, False)


def test_diff_datasets_with_empty_string_values():
    """An empty string is a value, not a missing one.

    Mutation: a truthiness guard at join.py, e.g.
        ds1_row[col] and ds2_row[col] and ds1_row[col] != ds2_row[col],
        which would call both rows the same.
    Oracle: hand-computed ('', 'Alice') and ('Bob', '').
    """
    ds1 = DataSet([
        {'id': 1, 'name': ''},
        {'id': 2, 'name': 'Bob'},
    ])
    ds2 = DataSet([
        {'id': 1, 'name': 'Alice'},
        {'id': 2, 'name': ''},
    ])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['name'])

    assert len(same) == 0
    assert len(diff) == 2
    assert diff[0]['name'] == ('', 'Alice')
    assert diff[1]['name'] == ('Bob', '')


def test_diff_with_very_small_float_differences():
    """A difference far below the input scale is kept, not rounded away.

    Mutation: round(value, 9) applied to the difference at
        core.py, which would give 0.0.
    Oracle: IEEE-754 - 1.0000000002 - 1.0000000001 is 1.00000008e-10.
    """
    ds = DataSet([
        {'val': 1.0000000001},
        {'val': 1.0000000002},
        {'val': 1.0000000003},
    ])
    ds.columns = [('val', float)]
    ds.diff('val', 'val_diff')
    result = list(ds.unwind('val_diff'))
    assert result[0] is None
    assert result[1] == pytest.approx(1e-10, rel=1e-6)
    assert result[2] == pytest.approx(1e-10, rel=1e-6)


def test_diff_column_overwrites_existing():
    """Differencing into an existing column replaces it in place.

    Mutation: skipping add_column when the name is already a column at
        core.py, which would leave the 999 placeholders.
    Oracle: hand-computed [None, 10, 10] over the 999 placeholders, in a
        single column named val_diff.
    """
    ds = DataSet([
        {'val': 10, 'val_diff': 999},
        {'val': 20, 'val_diff': 999},
        {'val': 30, 'val_diff': 999},
    ])
    ds.columns = [('val', int), ('val_diff', int)]

    ds.diff('val', 'val_diff')

    assert ds.cols.count('val_diff') == 1
    result = list(ds.unwind('val_diff'))
    assert result == [None, 10, 10]


def test_diff_datasets_none_in_key_column():
    """A None key reaches the sort and raises, rather than passing.

    Mutation: for key in ds1_map at join.py, dropping the
        sort, which would silently accept the None key.
    Oracle: sorted() over (None,) and (1,) raises TypeError.
    """
    ds1 = DataSet([
        {'id': 1, 'value': 100},
        {'id': None, 'value': 200},
    ])
    ds2 = DataSet([
        {'id': 1, 'value': 100},
        {'id': None, 'value': 250},
    ])

    with pytest.raises(TypeError):
        diff_datasets(ds1, ds2, ['id'], ['value'])


def test_diff_preserves_row_order_after_operation():
    """Differencing follows row order, and does not sort the rows.

    Mutation: sorting values before differencing at core.py,
        which would give [None, 10, 10] and reorder the rows.
    Oracle: hand-computed [None, -20, 10] over the unsorted 30, 10, 20.
    """
    ds = DataSet([
        {'id': 'c', 'val': 30},
        {'id': 'a', 'val': 10},
        {'id': 'b', 'val': 20},
    ])
    ds.diff('val', 'val_diff')

    assert list(ds.unwind('id')) == ['c', 'a', 'b']
    assert list(ds.unwind('val')) == [30, 10, 20]
    assert list(ds.unwind('val_diff')) == [None, -20, 10]


def test_diff_datasets_very_large_comparison():
    """Every key differing by one is reported, in ds1-then-ds2 order.

    Mutation: (ds2_row[col], ds1_row[col]) tuple order at
        join.py, giving (1, 0) on the first row.
    Oracle: hand-derived - ds2 adds 1 to every value, so 500 diff rows
        and a first pair of (0, 1).
    """
    ds1 = DataSet([{'id': i, 'value': i} for i in range(500)])
    ds2 = DataSet([{'id': i, 'value': i + 1} for i in range(500)])

    same, diff, only_in_ds1, only_in_ds2 = diff_datasets(ds1, ds2, ['id'], ['value'])

    assert len(same) == 0
    assert len(diff) == 500
    assert len(only_in_ds1) == 0
    assert len(only_in_ds2) == 0
    assert diff[0]['id'] == 0
    assert diff[0]['value'] == (0, 1)


# --- Index Boundary on a One-Row Dataset ---

def test_diff_index_one_rejected_on_single_row():
    """On one row the only legal pivots are 0 and -1, so 1 is refused.

    Mutation: index != +1 in place of index != -1 at core.py,
        which waves index=1 through on a one-row dataset and silently
        returns [None] instead of raising.
    Oracle: the boundary pair - last_idx is 0 on one row, so index=1 is
        one past the end and index=0 is the legal forward diff [None].
    """
    ds = DataSet([{'val': 7}])
    ds.columns = [('val', int)]

    with pytest.raises(ValueError) as excinfo:
        ds.diff('val', 'val_diff', index=1)
    assert str(excinfo.value) == 'index 1 out of range for dataset with 1 rows'
    assert 'val_diff' not in ds.cols

    ds.diff('val', 'val_diff', index=0)
    assert list(ds.unwind('val_diff')) == [None]


# --- Exact Assertion Messages ---

def test_diff_unsupported_type_message_is_exact():
    """diff() refuses a non-numeric column with the exact message.

    Mutation: the assert message rewritten at core.py to
        'XXonly supports int and float typesXX', which a substring
        match still accepts.
    Oracle: the message spelled out in full and compared end to end.
    """
    ds = DataSet([{'val': 'a'}, {'val': 'b'}])
    ds.columns = [('val', str)]

    with pytest.raises(AssertionError) as excinfo:
        ds.diff('val', 'val_diff')
    assert str(excinfo.value) == 'only supports int and float types'


def test_pct_change_unsupported_type_message_is_exact():
    """pct_change() refuses a non-numeric column with the exact message.

    Mutation: the assert message rewritten at core.py to
        'XXonly supports int and float typesXX' or shouted in upper
        case, neither of which a substring match rejects.
    Oracle: the message spelled out in full and compared end to end.
    """
    ds = DataSet([{'label': 'a'}, {'label': 'b'}])
    ds.columns = [('label', str)]

    with pytest.raises(AssertionError) as excinfo:
        ds.pct_change('label', 'label_chg')
    assert str(excinfo.value) == 'only supports int and float types'
    assert 'label_chg' not in ds.cols


# --- pct_change Column Typing ---

def test_pct_change_types_the_named_column_not_the_first_one():
    """pct_change() reads the type of the column it was given.

    Mutation: x != colname in place of x == colname at
        core.py, which picks the first OTHER column - here
        the str label column, whose type the assert then refuses.
    Oracle: hand-computed [None, 0.5, -0.5] over 100.0, 150.0, 75.0,
        with float carried through from the priced column.
    """
    ds = DataSet([
        {'label': 'a', 'price': 100.0},
        {'label': 'b', 'price': 150.0},
        {'label': 'c', 'price': 75.0},
    ])
    ds.columns = [('label', str), ('price', float)]

    ds.pct_change('price', 'price_chg')

    assert list(ds.unwind('price_chg')) == [None, 0.5, -0.5]
    assert ds.colmap['price_chg'] is float


def test_pct_change_new_column_is_typed_float():
    """The percent change column is declared, not left untyped.

    Mutation: add_column(new_colname, None, ...) in place of coltyp at
        core.py, leaving the new column with no type, so a
        later conversion has nothing to convert to.
    Oracle: float, the declared type of the source column, against a
        hand-computed [None, 0.25, -0.2].
    """
    ds = DataSet([
        {'price': 4.0},
        {'price': 5.0},
        {'price': 4.0},
    ])
    ds.columns = [('price', float)]

    ds.pct_change('price', 'price_chg')

    assert ds.colmap['price_chg'] is float
    assert list(ds.unwind('price_chg')) == [None, 0.25, -0.2]


def test_pct_change_declares_a_float_column_that_survives_reconstruction():
    """Verify the change column is typed float, so a ratio of ints keeps.

    Mutation: declaring the new column with the source int type, which
        truncates 0.5 to 0 once the dataset is rebuilt from its rows.
    Oracle: hand-computed [None, 0.5, -0.2] off a dataset rebuilt from
        the rows and columns, where an int column would give [None, 0, 0].
    """
    ds = DataSet([{'n': 10}, {'n': 15}, {'n': 12}])
    ds.pct_change('n', 'chg')

    assert ds.colmap['chg'] is float

    rebuilt = DataSet([dict(r) for r in ds], columns=ds.columns)
    assert [r['chg'] for r in rebuilt] == [None, 0.5, -0.2]


if __name__ == '__main__':
    pytest.main([__file__])
