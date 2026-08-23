import logging
import math

import pytest
from opendate import Date, DateTime, Time
from rollups import DataSet

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Basic dataset for shift tests."""
    return DataSet([
        {'x': 1, 'y': 8., 'z': 'a'},
        {'x': 2, 'y': 21.8, 'z': 'a'},
        {'x': 3, 'y': 3.2, 'z': 'a'},
        {'x': 1, 'y': 0.1, 'z': 'b'},
        {'x': 2, 'y': 22., 'z': 'b'},
        {'x': 3, 'y': 3., 'z': 'b'}
    ])


@pytest.fixture
def five_row_dataset():
    """Simple five-row dataset for shift tests."""
    return DataSet([
        {'val': 1},
        {'val': 2},
        {'val': 3},
        {'val': 4},
        {'val': 5},
    ])


@pytest.fixture
def three_row_dataset():
    """Simple three-row dataset for shift tests."""
    return DataSet([
        {'val': 1},
        {'val': 2},
        {'val': 3},
    ])


# --- Basic Shift Tests ---

def test_shift_forward_basic(basic_dataset):
    """Verify a one-period forward shift pads the head, not the tail.

    Mutation: colval[:-periods] + [None] * periods, padding after the
        values instead of before them.
    Oracle: hand-computed - x = [1, 2, 3, 1, 2, 3] one place forward is
        [None, 1, 2, 3, 1, 2].
    """
    basic_dataset.shift('x', 1)
    assert list(basic_dataset.unwind('x')) == [None, 1, 2, 3, 1, 2]


def test_shift_forward_multiple_periods(five_row_dataset):
    """Verify a three-period shift pads three slots, not one.

    Mutation: [None] in place of [None] * periods, padding a single
        slot however many periods were asked for.
    Oracle: hand-computed [None, None, None, 1, 2] from [1, 2, 3, 4, 5].
    """
    five_row_dataset.shift('val', 3)
    assert list(five_row_dataset.unwind('val')) == [None, None, None, 1, 2]


def test_shift_backward(five_row_dataset):
    """Verify a negative shift pulls values up and pads the tail.

    Mutation: [None] * -periods + colval[-periods:], padding the head
        instead of the tail on the backward branch.
    Oracle: hand-computed [3, 4, 5, None, None] from [1, 2, 3, 4, 5].
    """
    five_row_dataset.shift('val', -2)
    assert list(five_row_dataset.unwind('val')) == [3, 4, 5, None, None]


def test_shift_backward_single_period(three_row_dataset):
    """Verify -1 shifts backward rather than counting as no shift.

    Mutation: elif periods < -1, sending -1 to the zero-period branch.
    Oracle: hand-computed [2, 3, None] from [1, 2, 3].
    """
    three_row_dataset.shift('val', -1)
    assert list(three_row_dataset.unwind('val')) == [2, 3, None]


def test_shift_zero_periods(three_row_dataset, caplog):
    """Verify a zero shift warns and leaves every value in place.

    Mutation: dropping the logger.warning from the zero-period branch.
    Oracle: the values read before the call, plus the logged text.
    """
    original_vals = list(three_row_dataset.unwind('val'))
    with caplog.at_level(logging.WARNING):
        three_row_dataset.shift('val', 0)
    assert 'Shifting by 0' in caplog.text
    assert list(three_row_dataset.unwind('val')) == original_vals


# --- Column Name Tests ---

def test_shift_with_new_column_name(three_row_dataset):
    """Verify new_colname takes the shift and the source is untouched.

    Mutation: new_colname = colname, writing the shift back over the
        source column and never creating the named one.
    Oracle: hand-computed - val stays [1, 2, 3] while val_shifted holds
        [None, 1, 2].
    """
    three_row_dataset.shift('val', 1, 'val_shifted')
    assert 'val' in three_row_dataset.cols
    assert 'val_shifted' in three_row_dataset.cols
    assert list(three_row_dataset.unwind('val')) == [1, 2, 3]
    assert list(three_row_dataset.unwind('val_shifted')) == [None, 1, 2]


def test_shift_in_place_replaces_column(three_row_dataset):
    """Verify a shift with no new_colname replaces the column itself.

    Mutation: defaulting new_colname to a derived name such as
        colname + '_shift', leaving the source unshifted beside it.
    Oracle: hand-computed [None, 1, 2] in val, and a one-column schema.
    """
    three_row_dataset.shift('val', 1)
    assert list(three_row_dataset.unwind('val')) == [None, 1, 2]
    assert three_row_dataset.cols == ['val']


def test_shift_preserves_other_columns():
    """Verify shifting one column moves no other column's values.

    Mutation: shifting whole rows rather than the named column, so id
        and tag slide down with val.
    Oracle: hand-computed - id and tag keep their original order while
        val_shifted holds [None, 10.0, 20.0].
    """
    ds = DataSet([
        {'id': 1, 'val': 10.0, 'tag': 'a'},
        {'id': 2, 'val': 20.0, 'tag': 'b'},
        {'id': 3, 'val': 30.0, 'tag': 'c'},
    ])
    ds.columns = [('id', int), ('val', float), ('tag', str)]
    ds.shift('val', 1, 'val_shifted')
    assert list(ds.unwind('id')) == [1, 2, 3]
    assert list(ds.unwind('tag')) == ['a', 'b', 'c']
    assert list(ds.unwind('val')) == [10.0, 20.0, 30.0]
    assert list(ds.unwind('val_shifted')) == [None, 10.0, 20.0]


# --- Boundary Condition Tests (Parameterized) ---

@pytest.mark.parametrize(('periods', 'expected'), [
    (2, [None, None, 1]),
    (3, [None, None, None]),
    (5, [None, None, None]),
    (-2, [3, None, None]),
    (-3, [None, None, None]),
    (-5, [None, None, None]),
])
def test_shift_boundary_periods(periods, expected):
    """Verify the blank-everything guard trips at the row count exactly.

    Mutation: guarding on abs(periods) >= len(colval) - 1, which blanks
        the column a period early and loses the value that survives at
        two periods.
    Oracle: hand-computed - three rows shifted two places keep one
        value, shifted three or more keep none, and the row count never
        moves off three.
    """
    ds = DataSet([
        {'val': 1},
        {'val': 2},
        {'val': 3},
    ])
    ds.shift('val', periods)
    result = list(ds.unwind('val'))
    assert len(result) == 3
    assert result == expected


@pytest.mark.parametrize('periods', [1, -1])
def test_shift_single_row(periods):
    """Verify a one-row column blanks whichever way it is shifted.

    Mutation: leaving colval alone when abs(periods) >= the row count,
        so the lone value never moves off the end.
    Oracle: hand-computed - 42 is the only value, and neither side has
        a value to move into its place.
    """
    ds = DataSet([{'val': 42}])
    ds.shift('val', periods)
    assert list(ds.unwind('val')) == [None]


# --- Type Preservation Tests (Parameterized) ---

@pytest.mark.parametrize(('values', 'typ', 'expected_shifted'), [
    ([1.5, 2.5, 3.5], float, [None, 1.5, 2.5]),
    (['a', 'b', 'c'], str, [None, 'a', 'b']),
    ([100, 200, 300], int, [None, 100, 200]),
    ([True, False, True], bool, [None, True, False]),
])
def test_shift_type_preservation(values, typ, expected_shifted):
    """Verify the new column takes the type of the column named.

    Mutation: reading coltyp from column 0 rather than from the column
        matching colname, which hands the str label column's type to
        every shift.
    Oracle: the declared type of val, plus hand-computed shifted values.
    """
    ds = DataSet([{'label': 'x', 'val': v} for v in values])
    ds.columns = [('label', str), ('val', typ)]
    ds.shift('val', 1, 'shifted')
    assert ds.colmap['shifted'] == typ
    assert list(ds.unwind('shifted')) == expected_shifted


def test_shift_date_column():
    """Verify a Date column keeps its type and values across a shift.

    Mutation: coltyp read as columns[colix][0], the column name, so the
        shifted column stops being a Date column.
    Oracle: the declared Date type, and hand-computed dates one row
        down.
    """
    ds = DataSet([
        {'date': Date(2024, 1, 1)},
        {'date': Date(2024, 1, 2)},
        {'date': Date(2024, 1, 3)},
    ])
    ds.columns = [('date', Date)]
    ds.shift('date', 1, 'date_shifted')
    assert ds.colmap['date_shifted'] == Date
    assert list(ds.unwind('date_shifted')) == [
        None, Date(2024, 1, 1), Date(2024, 1, 2)]


def test_shift_datetime_column():
    """Verify a DateTime column keeps its type and time of day.

    Mutation: coltyp hard-coded to str, so the shifted column is no
        longer a DateTime column.
    Oracle: the declared DateTime type, and hand-computed timestamps
        one row down, times of day included.
    """
    ds = DataSet([
        {'dt': DateTime(2024, 1, 1, 10, 30)},
        {'dt': DateTime(2024, 1, 2, 11, 45)},
        {'dt': DateTime(2024, 1, 3, 14, 15)},
    ])
    ds.columns = [('dt', DateTime)]
    ds.shift('dt', 1, 'dt_shifted')
    assert ds.colmap['dt_shifted'] == DateTime
    assert list(ds.unwind('dt_shifted')) == [
        None, DateTime(2024, 1, 1, 10, 30), DateTime(2024, 1, 2, 11, 45)]


# --- None Value Handling Tests ---

def test_shift_with_none_values():
    """Verify None values shift like any other and keep their slots.

    Mutation: dropping None values before shifting, which compacts the
        column and lands the survivors in the wrong rows.
    Oracle: hand-computed [None, 1, None, 3, None] from
        [1, None, 3, None, 5].
    """
    ds = DataSet([
        {'val': 1},
        {'val': None},
        {'val': 3},
        {'val': None},
        {'val': 5},
    ])
    ds.shift('val', 1)
    assert list(ds.unwind('val')) == [None, 1, None, 3, None]


# --- Empty Dataset Tests ---

def test_shift_empty_dataset():
    """Verify shifting an empty column pads nothing at all.

    Mutation: blanking with [None] * periods rather than
        [None] * len(colval), which invents a row the dataset lacks.
    Oracle: the row count stays 0 and the column reads empty.
    """
    ds = DataSet([], columns=[('val', int)])
    ds.shift('val', 1)
    assert len(ds) == 0
    assert list(ds.unwind('val')) == []
    assert ds.cols == ['val']


# --- Edge Case Tests ---

def test_shift_time_type():
    """Verify a Time column keeps its type and values across a shift.

    Mutation: coltyp hard-coded to str, so the shifted column is no
        longer a Time column.
    Oracle: the declared Time type, and hand-computed times one row
        down.
    """
    ds = DataSet([
        {'time': Time(10, 30, 0)},
        {'time': Time(11, 45, 0)},
        {'time': Time(14, 15, 0)},
    ])
    ds.columns = [('time', Time)]
    ds.shift('time', 1, 'time_shifted')
    assert ds.colmap['time_shifted'] == Time
    assert list(ds.unwind('time_shifted')) == [
        None, Time(10, 30, 0), Time(11, 45, 0)]


def test_shift_with_infinity_values():
    """Verify infinities move down a row and the last value drops off.

    Mutation: rotating with colval[-periods:] + colval[:-periods],
        which wraps pi round to the front instead of dropping it.
    Oracle: hand-computed [None, inf, -inf]; pi falls off the end.
    """
    ds = DataSet([
        {'val': float('inf')},
        {'val': float('-inf')},
        {'val': math.pi},
    ])
    ds.columns = [('val', float)]
    ds.shift('val', 1, 'shifted')
    assert list(ds.unwind('shifted')) == [
        None, float('inf'), float('-inf')]


def test_shift_with_nan_values():
    """Verify a NaN shifts as a value and is not read as missing.

    Mutation: normalizing NaN to None on write, val if val == val else
        None, which quietly turns the shifted NaN into a blank.
    Oracle: math.isnan on the shifted slot, and 2.0 one row further
        down.
    """
    ds = DataSet([
        {'val': float('nan')},
        {'val': 2.0},
        {'val': 3.0},
    ])
    ds.columns = [('val', float)]
    ds.shift('val', 1, 'shifted')
    shifted = list(ds.unwind('shifted'))
    assert shifted[0] is None
    assert math.isnan(shifted[1])
    assert shifted[2] == 2.0


def test_shift_very_large_dataset():
    """Verify a 100-period shift over 10000 rows lands where computed.

    Mutation: slicing colval[periods:] rather than colval[:-periods],
        taking the tail of the column instead of the head.
    Oracle: hand-computed - row 100 holds 0 and row 9999 holds
        9999 - 100 = 9899.
    """
    ds = DataSet([{'val': i} for i in range(10000)])
    ds.columns = [('val', int)]
    ds.shift('val', 100, 'shifted')

    shifted = list(ds.unwind('shifted'))
    assert len(shifted) == 10000
    assert shifted[:100] == [None] * 100
    assert shifted[100] == 0
    assert shifted[9999] == 9899


def test_shift_multiple_consecutive():
    """Verify chained shifts compound and undo each other.

    Mutation: dropping None values before shifting, so shift1's leading
        None vanishes and shift2 comes back one row out.
    Oracle: hand-computed - two one-period shifts equal one two-period
        shift, and shifting back two restores the head values.
    """
    ds = DataSet([{'val': i} for i in range(1, 6)])
    ds.shift('val', 1, 'shift1')
    ds.shift('shift1', 1, 'shift2')
    ds.shift('shift2', -2, 'roundtrip')

    assert list(ds.unwind('val')) == [1, 2, 3, 4, 5]
    assert list(ds.unwind('shift1')) == [None, 1, 2, 3, 4]
    assert list(ds.unwind('shift2')) == [None, None, 1, 2, 3]
    assert list(ds.unwind('roundtrip')) == [1, 2, 3, None, None]


def test_shift_all_none_values():
    """Verify an all-None shift blanks the column it is written to.

    Mutation: skipping None values when writing the shifted column, so
        whatever sat in the target column survives the shift.
    Oracle: hand-computed - keep starts [1, 2, 3] and must read all
        None once the all-None column is shifted onto it.
    """
    ds = DataSet([
        {'val': None, 'keep': 1},
        {'val': None, 'keep': 2},
        {'val': None, 'keep': 3},
    ])
    ds.shift('val', 1)
    assert list(ds.unwind('val')) == [None, None, None]
    ds.shift('val', 1, 'keep')
    assert list(ds.unwind('keep')) == [None, None, None]


def test_shift_mixed_none_values():
    """Verify a backward shift carries None values up with the rest.

    Mutation: dropping None values before shifting, so the backward
        shift starts from 2 and the source's own Nones disappear.
    Oracle: hand-computed [2, None, 4, None, None] from
        [None, 2, None, 4, None].
    """
    ds = DataSet([
        {'val': None},
        {'val': 2},
        {'val': None},
        {'val': 4},
        {'val': None},
    ])
    ds.shift('val', -1, 'shifted')
    assert list(ds.unwind('shifted')) == [2, None, 4, None, None]


def test_shift_backward_with_new_column():
    """Verify a backward shift honors new_colname and spares the source.

    Mutation: new_colname = colname on the backward branch, pulling the
        values up in place instead of into the named column.
    Oracle: hand-computed - val stays [1, 2, 3, 4, 5] while backward
        holds [3, 4, 5, None, None].
    """
    ds = DataSet([
        {'val': 1},
        {'val': 2},
        {'val': 3},
        {'val': 4},
        {'val': 5},
    ])
    ds.shift('val', -2, 'backward')
    assert list(ds.unwind('val')) == [1, 2, 3, 4, 5]
    assert list(ds.unwind('backward')) == [3, 4, 5, None, None]


def test_shift_two_rows():
    """Verify a two-row column keeps one value when shifted one place.

    Mutation: guarding on periods >= len(colval) - 1, which blanks a
        two-row column that still has a value to spare.
    Oracle: hand-computed [None, 1] from [1, 2].
    """
    ds = DataSet([
        {'val': 1},
        {'val': 2},
    ])
    ds.shift('val', 1)
    assert list(ds.unwind('val')) == [None, 1]


def test_shift_preserves_column_order():
    """Verify the shifted column is appended after the existing ones.

    Mutation: passing index=0 to add_column, which puts b_shifted in
        front of a.
    Oracle: hand-computed column order ['a', 'b', 'c', 'b_shifted'].
    """
    ds = DataSet([
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 4, 'b': 5, 'c': 6},
    ])
    ds.shift('b', 1, 'b_shifted')
    assert ds.cols == ['a', 'b', 'c', 'b_shifted']
    assert list(ds.unwind('a')) == [1, 4]
    assert list(ds.unwind('b')) == [2, 5]
    assert list(ds.unwind('c')) == [3, 6]
    assert list(ds.unwind('b_shifted')) == [None, 2]


def test_shift_empty_string_values():
    """Verify '' is a value to shift, not a stand-in for missing.

    Mutation: filling the vacated slot with '' rather than None, which
        makes the head indistinguishable from a real empty string.
    Oracle: hand-computed [None, '', 'a'] - slot 0 is the vacated None,
        slot 1 the empty string that was in row 0.
    """
    ds = DataSet([
        {'val': ''},
        {'val': 'a'},
        {'val': ''},
    ])
    ds.shift('val', 1, 'shifted')
    assert list(ds.unwind('shifted')) == [None, '', 'a']


def test_shift_with_summary_row():
    """Verify a shift leaves a pending summary row out of the shift.

    Mutation: new_colname = colname, writing the shift back over val,
        which drops 300 from the total the summary computes afterward.
    Oracle: hand-computed - val totals 100 + 200 + 300 = 600, and
        val_shifted was never declared for the summary, so it reads
        None.
    """
    ds = DataSet([
        {'id': 1, 'val': 100},
        {'id': 2, 'val': 200},
        {'id': 3, 'val': 300},
    ])
    ds.columns = [('id', int), ('val', int)]
    ds.add_summary_row()
    ds.shift('val', 1, 'val_shifted')

    assert len(ds) == 3
    assert list(ds.unwind('val_shifted')) == [None, 100, 200]
    assert ds.summary['id'] == 'Total'
    assert ds.summary['val'] == 600
    assert ds.summary['val_shifted'] is None


# --- Default Argument Tests ---

def test_shift_default_periods_is_one(five_row_dataset):
    """Verify shift() with no periods moves values exactly one row.

    Mutation: the periods default changed from 1 to any other count,
        so an unqualified shift lands the values on the wrong rows.
    Oracle: hand-computed [None, 1, 2, 3, 4] from [1, 2, 3, 4, 5], the
        same result an explicit periods=1 must give.
    """
    five_row_dataset.shift('val')
    assert list(five_row_dataset.unwind('val')) == [None, 1, 2, 3, 4]


def test_shift_zero_warning_text_and_scope(three_row_dataset, caplog):
    """Verify only a zero shift warns, with the documented message.

    Mutation: the warning text rewritten, or the warning moved out of
        the zero-period branch so every shift warns.
    Oracle: the message named in the shift contract, and a one-period
        shift that must log nothing.
    """
    with caplog.at_level(logging.WARNING, logger='rollups.core'):
        three_row_dataset.shift('val', 1)
        assert [r.getMessage() for r in caplog.records] == []
        three_row_dataset.shift('val', 0)
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert warned[0].getMessage() == 'Shifting by 0, results will be unshifted'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
