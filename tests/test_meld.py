"""Comprehensive tests for DataSet meld_datasets operations.

Meld Operation Overview
-----------------------
The meld_datasets function combines columns from multiple aligned datasets by
index position. It's useful for adding related data from different sources that
share the same row order.

Core Functionality
------------------
Given a base dataset (meldee) and one or more source datasets (melders), the
function copies specified columns from each melder into the meldee, prefixing
the column names to avoid conflicts.

Parameters
----------
- meldee: Base dataset to add columns to
- melders: List of source datasets (must have same row count as meldee)
- melder_ids: List of prefix strings for naming merged columns
- columns: List of column lists specifying which columns to take from each melder
- inplace: If True, modifies meldee; if False, returns new dataset

Column Naming
-------------
Columns from melders are added with format: {prefix}_{column_name}
Example: With prefix 'one' and column 'amount', result is 'one_amount'

Key Differences from join()
----------------------------
- meld_datasets: Fast, position-based, assumes aligned rows
- join(): Slower, key-based matching, handles unaligned data
"""
import datetime
import math
import zoneinfo

import pytest
from opendate import UTC, Date, DateTime, Time
from rollups import DataSet, meld_datasets

# --- Fixtures ---


@pytest.fixture
def basic_base():
    """Basic base dataset for meld tests."""
    return DataSet([
        {'id': 1, 'name': 'Alice'},
        {'id': 2, 'name': 'Bob'},
    ])


@pytest.fixture
def basic_melder():
    """Basic melder dataset for meld tests."""
    return DataSet([
        {'score': 95, 'grade': 'A'},
        {'score': 87, 'grade': 'B'},
    ])


@pytest.fixture
def multi_melder_setup():
    """Setup for multiple melder tests."""
    base = DataSet([{'id': 1}, {'id': 2}])
    daily = DataSet([{'amount': 100}, {'amount': 200}])
    monthly = DataSet([{'amount': 500}, {'amount': 800}])
    return base, daily, monthly


# --- Basic Meld Tests ---

def test_meld_basic_single_melder(basic_base, basic_melder):
    """Verify one melder's column lands prefixed and row-aligned.

    Mutation: reading the melder rows in reverse, or naming the new
        column `{col}_{prefix}` instead of `{prefix}_{col}`.
    Oracle: hand-paired scores, 95 to row 0 and 87 to row 1, under
        'test_score'.
    """
    result = meld_datasets(
        basic_base,
        [basic_melder],
        ['test'],
        [['score']],
        inplace=False)

    assert len(result) == 2
    assert 'test_score' in result.cols
    assert result[0]['id'] == 1
    assert result[0]['test_score'] == 95
    assert result[1]['test_score'] == 87


def test_meld_multiple_melders(multi_melder_setup):
    """Verify each melder's values land under its own prefix.

    Mutation: zipping melders against a reversed melder_ids, which swaps
        the two prefixes over identically named 'amount' columns.
    Oracle: hand-paired daily 100/200 against monthly 500/800.
    """
    base, daily, monthly = multi_melder_setup

    result = meld_datasets(
        base,
        [daily, monthly],
        ['day', 'month'],
        [['amount'], ['amount']],
        inplace=False
    )

    assert 'day_amount' in result.cols
    assert 'month_amount' in result.cols
    assert result[0]['day_amount'] == 100
    assert result[0]['month_amount'] == 500
    assert result[1]['day_amount'] == 200
    assert result[1]['month_amount'] == 800


def test_meld_multiple_columns_per_melder():
    """Verify every requested column of one melder is copied.

    Mutation: iterating `cols[:1]`, so only the first requested column of
        each melder is copied.
    Oracle: hand-listed a/10/100.0 and b/20/200.0 across both rows.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([
        {'col1': 'a', 'col2': 10, 'col3': 100.0},
        {'col1': 'b', 'col2': 20, 'col3': 200.0},
    ])

    result = meld_datasets(
        base,
        [melder],
        ['prefix'],
        [['col1', 'col2', 'col3']],
        inplace=False
    )

    assert 'prefix_col1' in result.cols
    assert 'prefix_col2' in result.cols
    assert 'prefix_col3' in result.cols
    assert result[0]['prefix_col1'] == 'a'
    assert result[0]['prefix_col2'] == 10
    assert result[0]['prefix_col3'] == 100.0
    assert result[1]['prefix_col1'] == 'b'
    assert result[1]['prefix_col2'] == 20
    assert result[1]['prefix_col3'] == 200.0


def test_meld_three_melders():
    """Verify each melder is paired with its own column list.

    Mutation: zipping `columns` out of step with `melders`, which every
        melder here exposes because each carries a different column name
        and a mispaired lookup yields None.
    Oracle: hand-paired v1/v2/v3 values from their own melders.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    m1 = DataSet([{'v1': 10}, {'v1': 20}])
    m2 = DataSet([{'v2': 100}, {'v2': 200}])
    m3 = DataSet([{'v3': 1000}, {'v3': 2000}])

    result = meld_datasets(
        base,
        [m1, m2, m3],
        ['a', 'b', 'c'],
        [['v1'], ['v2'], ['v3']],
        inplace=False
    )

    assert result[0]['a_v1'] == 10
    assert result[0]['b_v2'] == 100
    assert result[0]['c_v3'] == 1000
    assert result[1]['a_v1'] == 20
    assert result[1]['b_v2'] == 200
    assert result[1]['c_v3'] == 2000


# --- Inplace Parameter Tests (Parameterized) ---

@pytest.mark.parametrize('inplace', [True, False])
def test_meld_inplace_parameter(inplace):
    """Verify inplace picks between writing to the meldee and a copy.

    Mutation: deepcopy() swapped for copy(), which shares row objects, so
        a non-inplace meld still writes the value into the base's rows.
    Oracle: the base rows themselves, keyed 'test_value' only when
        inplace is True.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([{'value': 100}, {'value': 200}])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=inplace)

    assert 'test_value' in result.cols
    assert result[0]['test_value'] == 100

    if inplace:
        assert result is base
        assert 'test_value' in base.cols
        assert base[0]['test_value'] == 100
        assert base[1]['test_value'] == 200
    else:
        assert result is not base
        assert 'test_value' not in base.cols
        assert 'test_value' not in base[0]
        assert 'test_value' not in base[1]


# --- Length Mismatch Error Tests ---

@pytest.mark.parametrize(('base_len', 'melder_len'), [
    (2, 3),
    (3, 2),
    (1, 5),
    (5, 1),
])
def test_meld_length_mismatch_raises_error(base_len, melder_len):
    """Verify a melder of the wrong length raises, counts in order.

    Mutation: the two counts swapped in the message, which every case
        here exposes because melder and meldee lengths differ.
    Oracle: the raised text, melder count first and meldee count second.
    """
    base = DataSet([{'id': i} for i in range(base_len)])
    melder = DataSet([{'value': i} for i in range(melder_len)])

    expected = f'Length mismatch: {melder_len} != {base_len}'
    with pytest.raises(ValueError, match=expected):
        meld_datasets(base, [melder], ['test'], [['value']])


def test_meld_first_melder_longer_raises_error():
    """Verify a bad first melder raises even though the second fits.

    Mutation: checking only the last melder's length, which passes here
        because the second melder does match the meldee.
    Oracle: the raised '3 != 2' pair, naming the first melder's count.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    m1 = DataSet([{'v': 10}, {'v': 20}, {'v': 30}])
    m2 = DataSet([{'v': 100}, {'v': 200}])

    with pytest.raises(ValueError, match='Length mismatch: 3 != 2'):
        meld_datasets(base, [m1, m2], ['m1', 'm2'], [['v'], ['v']])


def test_meld_second_melder_shorter_raises_error():
    """Verify the length guard runs for every melder, not just the first.

    Mutation: hoisting the guard out of the loop so only melders[0] is
        checked, which leaves add_column to raise its own 'values length
        1 must match dataset length 2' instead.
    Oracle: the raised text, which must be the '1 != 2' length mismatch.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    m1 = DataSet([{'v': 10}, {'v': 20}])
    m2 = DataSet([{'v': 100}])

    with pytest.raises(ValueError, match='Length mismatch: 1 != 2'):
        meld_datasets(base, [m1, m2], ['m1', 'm2'], [['v'], ['v']])


# --- Empty and Single Row Tests ---

def test_meld_empty_datasets():
    """Verify an empty meld still declares the melded column.

    Mutation: an `if not dataset: continue` short circuit that skips
        column creation when the melder holds no rows.
    Oracle: the resulting column list and its NoneType default type.
    """
    base = DataSet([])
    melder = DataSet([])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert len(result) == 0
    assert result.cols == ['test_value']
    assert result.colmap['test_value'] is type(None)


# --- Column Preservation Tests ---

def test_meld_preserves_original_columns(basic_melder):
    """Verify the meldee's own columns survive, ahead of the melded one.

    Mutation: inserting the melded column at index 0, or rebuilding the
        result from the melder so the meldee's columns are lost.
    Oracle: the exact column order ['id', 'name', 'age', 'test_score']
        plus the meldee's own values in both rows.
    """
    base = DataSet([
        {'id': 1, 'name': 'Alice', 'age': 30},
        {'id': 2, 'name': 'Bob', 'age': 25},
    ])

    result = meld_datasets(base, [basic_melder], ['test'], [['score']], inplace=False)

    assert result.cols == ['id', 'name', 'age', 'test_score']
    assert result[0]['name'] == 'Alice'
    assert result[0]['age'] == 30
    assert result[1]['name'] == 'Bob'
    assert result[1]['age'] == 25


def test_meld_selective_columns():
    """Verify only the requested melder columns are copied.

    Mutation: iterating the melder's own `dataset.cols` instead of the
        requested `cols`, which would drag col2 and col4 along.
    Oracle: the exact column list ['id', 'test_col1', 'test_col3'].
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([
        {'col1': 'a', 'col2': 'b', 'col3': 'c', 'col4': 'd'},
        {'col1': 'e', 'col2': 'f', 'col3': 'g', 'col4': 'h'},
    ])

    result = meld_datasets(
        base,
        [melder],
        ['test'],
        [['col1', 'col3']],
        inplace=False
    )

    assert result.cols == ['id', 'test_col1', 'test_col3']
    assert result[0]['test_col1'] == 'a'
    assert result[0]['test_col3'] == 'c'
    assert result[1]['test_col1'] == 'e'
    assert result[1]['test_col3'] == 'g'


# --- None Value Tests ---

def test_meld_with_none_values():
    """Verify a None in the middle keeps the rows below it in place.

    Mutation: dropping None from the value list, which shifts 300 up a
        row (and trips add_column's own length check).
    Oracle: hand-paired 100/None/300 against rows 0/1/2.
    """
    base = DataSet([{'id': 1}, {'id': 2}, {'id': 3}])
    melder = DataSet([
        {'value': 100},
        {'value': None},
        {'value': 300},
    ])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert result[0]['test_value'] == 100
    assert result[1]['test_value'] is None
    assert result[2]['test_value'] == 300


def test_meld_missing_column_in_melder():
    """Verify a column the melder lacks becomes a None column.

    Mutation: `source_row[col]` in place of `.get(col)`, or
        `dataset.colmap[col]` in place of its NoneType default - both
        raise KeyError here.
    Oracle: None in every row plus the NoneType column type.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([{'value': 100}, {'value': 200}])

    result = meld_datasets(base, [melder], ['test'], [['missing_col']], inplace=False)

    assert 'test_missing_col' in result.cols
    assert result.colmap['test_missing_col'] is type(None)
    assert result[0]['test_missing_col'] is None
    assert result[1]['test_missing_col'] is None


# --- Prefix Tests ---

@pytest.mark.parametrize(('prefix', 'expected_col'), [
    ('test', 'test_value'),
    ('', '_value'),
    ('my_prefix', 'my_prefix_value'),
    ('long_prefix_name', 'long_prefix_name_value'),
])
def test_meld_prefix_variations(prefix, expected_col):
    """Verify the name is always `{prefix}_{col}`, empty prefix included.

    Mutation: dropping the separator when the prefix is empty, i.e.
        `col if not prefix else f'{prefix}_{col}'`.
    Oracle: the expected name per prefix, '_value' for the empty one.
    """
    base = DataSet([{'id': 1}])
    melder = DataSet([{'value': 100}])

    result = meld_datasets(base, [melder], [prefix], [['value']], inplace=False)

    assert len(result) == 1
    assert expected_col in result.cols
    assert result[0][expected_col] == 100


def test_meld_column_with_underscore():
    """Verify a melder column carrying an underscore keeps its full name.

    Mutation: naming from the melder column's last underscore segment,
        which would yield 'test_value'.
    Oracle: the hand-written name 'test_my_value'.
    """
    base = DataSet([{'id': 1}])
    melder = DataSet([{'my_value': 100}])

    result = meld_datasets(base, [melder], ['test'], [['my_value']], inplace=False)

    assert 'test_my_value' in result.cols
    assert result[0]['test_my_value'] == 100


# --- Data Type Tests ---

def test_meld_different_data_types():
    """Verify str, int, float and bool values cross unchanged.

    Mutation: coercing melded values through str(), which every row here
        exposes on the numeric and boolean columns.
    Oracle: hand-listed values per type in both rows, with `is True` /
        `is False` identity on the boolean.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([
        {'str_val': 'hello', 'int_val': 42, 'float_val': math.pi, 'bool_val': True},
        {'str_val': 'world', 'int_val': 99, 'float_val': math.e, 'bool_val': False},
    ])

    result = meld_datasets(
        base,
        [melder],
        ['test'],
        [['str_val', 'int_val', 'float_val', 'bool_val']],
        inplace=False
    )

    assert result[0]['test_str_val'] == 'hello'
    assert result[0]['test_int_val'] == 42
    assert result[0]['test_float_val'] == math.pi
    assert result[0]['test_bool_val'] is True
    assert result[1]['test_str_val'] == 'world'
    assert result[1]['test_int_val'] == 99
    assert result[1]['test_float_val'] == math.e
    assert result[1]['test_bool_val'] is False


def test_meld_preserves_column_types():
    """Verify the melder's declared type follows its column across.

    Mutation: `col_type = type(None)` in place of the melder's own type,
        or reading `dataset.container` and so skipping the melder's lazy
        conversion, which would carry the raw '10.5' string over.
    Oracle: the melded column typed float and holding 10.5, not '10.5'.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    base.columns = [('id', int)]

    melder = DataSet([{'value': '10.5'}, {'value': '20.5'}])
    melder.columns = [('value', float)]

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert result.colmap['id'] is int
    assert result.colmap['test_value'] is float
    assert result[0]['test_value'] == 10.5
    assert result[1]['test_value'] == 20.5


# --- Column Overwriting Tests ---

def test_meld_overwrites_existing_columns():
    """Verify a melded column replaces one the meldee already carries.

    Mutation: skipping a column already present, e.g. `if new_col_name in
        result.cols: continue`, which would leave 999/888 standing.
    Oracle: the melder's 100/200 in place of the meldee's own values,
        with two columns and no duplicate.
    """
    base = DataSet([
        {'id': 1, 'test_value': 999},
        {'id': 2, 'test_value': 888},
    ])
    melder = DataSet([
        {'value': 100},
        {'value': 200},
    ])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert result[0]['test_value'] == 100
    assert result[1]['test_value'] == 200
    assert len(result.cols) == 2


def test_meld_empty_column_list():
    """Verify an empty column list copies nothing from the melder.

    Mutation: `for col in (cols or dataset.cols)`, a plausible "empty
        means all" default, which would add 'test_value'.
    Oracle: the meldee's single 'id' column standing alone.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([{'value': 100}, {'value': 200}])

    result = meld_datasets(base, [melder], ['test'], [[]], inplace=False)

    assert 'test_value' not in result.cols
    assert len(result.cols) == 1


# --- Sequential Melding Tests ---

def test_meld_sequential_melding():
    """Verify a second meld builds on the first without touching it.

    Mutation: deepcopy() swapped for copy(), whose shared rows let the
        second meld write 'second_v2' back into the first result.
    Oracle: the intermediate result's own rows and column list after the
        second meld.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    m1 = DataSet([{'v1': 10}, {'v1': 20}])
    m2 = DataSet([{'v2': 100}, {'v2': 200}])

    first = meld_datasets(base, [m1], ['first'], [['v1']], inplace=False)
    second = meld_datasets(first, [m2], ['second'], [['v2']], inplace=False)

    assert second.cols == ['id', 'first_v1', 'second_v2']
    assert second[0]['first_v1'] == 10
    assert second[0]['second_v2'] == 100
    assert second[1]['first_v1'] == 20
    assert second[1]['second_v2'] == 200
    assert first.cols == ['id', 'first_v1']
    assert 'second_v2' not in first[0]


def test_meld_modifies_inplace_then_continues():
    """Verify melding defaults to inplace and writes into the base rows.

    Mutation: the `inplace` default flipped to False, so the caller's own
        dataset never gains the column and add_column's lambda raises.
    Oracle: the doubled values 200/400, computed from the base's own rows
        after the meld.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([{'value': 100}, {'value': 200}])

    meld_datasets(base, [melder], ['test'], [['value']])

    assert 'test_value' in base.cols
    base.add_column('computed', int, value=lambda r: r['test_value'] * 2)
    assert base[0]['computed'] == 200
    assert base[1]['computed'] == 400


# --- Real World Scenario Tests ---

def test_meld_prefix_keeps_colliding_source_columns_apart():
    """Verify prefixing keeps colliding source columns apart.

    Mutation: melding a column under its own name, so each melder's
        'label' clobbers the meldee's and 'one_amount' never doubles
        up into 'one_one_amount'.
    Oracle: the three spellings of the label, one per source.
    """
    daily = DataSet([
        {'label': 'Group-A', 'day_amount': 1000},
        {'label': 'Group-B', 'day_amount': -500},
    ])
    monthly = DataSet([
        {'label': 'group-a', 'one_amount': 5000},
        {'label': 'group-b', 'one_amount': -2000},
    ])
    yearly = DataSet([
        {'label': 'GROUP-A', 'two_amount': 50000},
        {'label': 'GROUP-B', 'two_amount': -10000},
    ])

    result = meld_datasets(
        daily,
        [monthly, yearly],
        ['one', 'two'],
        [['one_amount', 'label'], ['two_amount', 'label']],
        inplace=False
    )

    assert result[0]['label'] == 'Group-A'
    assert result[0]['day_amount'] == 1000
    assert result[0]['one_one_amount'] == 5000
    assert result[0]['one_label'] == 'group-a'
    assert result[0]['two_two_amount'] == 50000
    assert result[0]['two_label'] == 'GROUP-A'
    assert result[1]['label'] == 'Group-B'
    assert result[1]['one_one_amount'] == -2000
    assert result[1]['two_two_amount'] == -10000


# --- Large Dataset Tests ---

@pytest.mark.parametrize('num_rows', [10, 100, 500])
def test_meld_various_row_counts(num_rows):
    """Verify positional alignment holds across the whole dataset.

    Mutation: the value list shifted by one row, e.g. `values[1:] +
        [None]`, which the first, middle and last rows all expose.
    Oracle: value == index * 10, hand-computed at three positions.
    """
    base = DataSet([{'id': i} for i in range(num_rows)])
    melder = DataSet([{'value': i * 10} for i in range(num_rows)])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert len(result) == num_rows
    assert result[0]['test_value'] == 0
    assert result[num_rows // 2]['test_value'] == (num_rows // 2) * 10
    assert result[num_rows - 1]['test_value'] == (num_rows - 1) * 10


# --- Edge Case Tests ---

@pytest.mark.parametrize(('col', 'typ', 'val1', 'val2'), [
    ('date', Date, Date(2024, 1, 1), Date(2024, 6, 15)),
    ('dt', DateTime, DateTime(2024, 1, 1, 10, 30, tzinfo=UTC),
     DateTime(2024, 6, 15, 14, 45, tzinfo=UTC)),
    ('time', Time, Time(10, 30, 0, tzinfo=UTC), Time(14, 45, 0, tzinfo=UTC)),
])
def test_meld_with_temporal_types(col, typ, val1, val2):
    """Verify a temporal column crosses with its declared type intact.

    Mutation: `col_type = type(None)` in place of the melder's declared
        type, which leaves the melded column untyped.
    Oracle: the melder's own declared type plus the two hand-built
        values.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([{col: val1}, {col: val2}])
    melder.columns = [(col, typ)]

    result = meld_datasets(base, [melder], ['test'], [[col]], inplace=False)

    assert result.colmap[f'test_{col}'] is typ
    assert result[0][f'test_{col}'] == val1
    assert result[1][f'test_{col}'] == val2


def test_meld_all_none_values():
    """Verify an all-None melder column still crosses as a column.

    Mutation: dropping None from the value list, which leaves nothing to
        write and trips add_column's length check.
    Oracle: None in both rows plus the NoneType column type an all-None
        melder column carries.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([{'value': None}, {'value': None}])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert result.colmap['test_value'] is type(None)
    assert result[0]['test_value'] is None
    assert result[1]['test_value'] is None


def test_meld_with_infinity_values():
    """Verify infinities cross unchanged, sign included.

    Mutation: a non-finite sanitizer on the melded values, e.g.
        `v if math.isfinite(v) else None`.
    Oracle: inf at row 0 and -inf at row 1, compared against freshly
        built floats.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([
        {'value': float('inf')},
        {'value': float('-inf')},
    ])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert result[0]['test_value'] == float('inf')
    assert result[1]['test_value'] == float('-inf')


def test_meld_with_nan_values():
    """Verify NaN crosses as NaN rather than being nulled or zeroed.

    Mutation: a NaN-to-None (or NaN-to-0.0) sanitizer on the melded
        values.
    Oracle: math.isnan at row 0, with 100.0 at row 1 proving the column
        was not blanked wholesale.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([
        {'value': float('nan')},
        {'value': 100.0},
    ])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert math.isnan(result[0]['test_value'])
    assert result[1]['test_value'] == 100.0


def test_meld_many_melders():
    """Verify every melder in a long list is consumed.

    Mutation: the melder list truncated, e.g. `zip(melders[:5], ...)`,
        which drops the tail prefixes.
    Oracle: 11 columns for one base column plus ten melders, and
        hand-computed 100/200 from the tenth melder.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melders = [DataSet([{'v': (i + 1) * 10}, {'v': (i + 1) * 20}]) for i in range(10)]
    prefixes = [f'prefix_{i}' for i in range(10)]
    columns = [['v'] for _ in range(10)]

    result = meld_datasets(base, melders, prefixes, columns, inplace=False)

    assert len(result.cols) == 11
    assert result[0]['prefix_0_v'] == 10
    assert result[1]['prefix_0_v'] == 20
    assert result[0]['prefix_9_v'] == 100
    assert result[1]['prefix_9_v'] == 200


def test_meld_with_empty_string_values():
    """Verify an empty string crosses as '' and is not read as missing.

    Mutation: `source_row.get(col) or None`, which collapses the falsy ''
        to None.
    Oracle: '' at row 0 against 'hello' at row 1.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([{'value': ''}, {'value': 'hello'}])

    result = meld_datasets(base, [melder], ['test'], [['value']], inplace=False)

    assert result[0]['test_value'] == ''
    assert result[1]['test_value'] == 'hello'


def test_meld_column_order():
    """Verify melded columns are appended in melder order.

    Mutation: `add_column(..., index=0)`, or walking the melders in
        reverse, either of which reorders the result.
    Oracle: the exact list ['id', 'first_v1', 'second_v2'].
    """
    base = DataSet([{'id': 1}])
    m1 = DataSet([{'v1': 10}])
    m2 = DataSet([{'v2': 20}])

    result = meld_datasets(
        base,
        [m1, m2],
        ['first', 'second'],
        [['v1'], ['v2']],
        inplace=False
    )

    assert result.cols == ['id', 'first_v1', 'second_v2']


def test_meld_preserves_row_values():
    """Verify values align by row position, not by sort or reversal.

    Mutation: reading the melder rows reversed or sorted, which the three
        distinct values expose at rows 0 and 2.
    Oracle: hand-paired 100/200/300 against rows a/b/c.
    """
    base = DataSet([
        {'id': 'a', 'value': 1},
        {'id': 'b', 'value': 2},
        {'id': 'c', 'value': 3},
    ])
    melder = DataSet([
        {'extra': 100},
        {'extra': 200},
        {'extra': 300},
    ])

    result = meld_datasets(base, [melder], ['test'], [['extra']], inplace=False)

    for i, expected in enumerate([100, 200, 300]):
        assert result[i]['test_extra'] == expected


def test_meld_same_column_name_different_melders():
    """Verify two melders sharing a column name stay apart.

    Mutation: naming the melded column from `col` alone, so the second
        melder's 'value' overwrites the first melder's.
    Oracle: hand-paired 10/100 and 20/200 under the two prefixes.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    m1 = DataSet([{'value': 10}, {'value': 20}])
    m2 = DataSet([{'value': 100}, {'value': 200}])

    result = meld_datasets(
        base,
        [m1, m2],
        ['first', 'second'],
        [['value'], ['value']],
        inplace=False
    )

    assert result[0]['first_value'] == 10
    assert result[0]['second_value'] == 100
    assert result[1]['first_value'] == 20
    assert result[1]['second_value'] == 200


def test_meld_with_boolean_values():
    """Verify a leading False crosses as False, typed bool.

    Mutation: `source_row.get(col) or None`, which collapses the falsy
        False in row 0 to None.
    Oracle: `is False` / `is True` identity plus the bool column type
        guessed from the False exemplar row.
    """
    base = DataSet([{'id': 1}, {'id': 2}])
    melder = DataSet([{'flag': False}, {'flag': True}])

    result = meld_datasets(base, [melder], ['test'], [['flag']], inplace=False)

    assert result.colmap['test_flag'] is bool
    assert result[0]['test_flag'] is False
    assert result[1]['test_flag'] is True


def test_meld_out_of_place_keeps_a_driver_timezone():
    """Verify the out-of-place meld does not strip a datetime's offset.

    Both sides are covered: the meldee's own column, which crosses in
    the `deepcopy()` meld works on, and the melder's, which is read out
    of a dataset that never gets copied.

    Mutation: an opendate that leaves a driver's zoneinfo.ZoneInfo on
        the value, so the `deepcopy()` meld starts from rebuilds it from
        pendulum's `.tz`, which answers None for a tzinfo pendulum does
        not own.
    Oracle: the -4 hour offset the source value reports, read off both
        columns of the result, against the source instant.
    """
    source = datetime.datetime(2026, 8, 28, 8, 1, 27,
                               tzinfo=zoneinfo.ZoneInfo('America/New_York'))
    base = DataSet([{'c': source}], columns=[('c', DateTime)])
    melder = DataSet([{'c': source}], columns=[('c', DateTime)])

    result = meld_datasets(base, [melder], ['test'], [['c']], inplace=False)

    for name in ('c', 'test_c'):
        assert result[0][name].utcoffset() == datetime.timedelta(hours=-4)
        assert result[0][name] == source


if __name__ == '__main__':
    pytest.main([__file__])
