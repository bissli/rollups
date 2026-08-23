import datetime
import logging
import math

import pytest
from opendate import Date, DateTime, Time
from rollups import DataSet

# --- Basic add_column Tests ---


def test_add_column_no_values():
    """Verify a new column with no value lands as None in every row.

    Mutation: dropping the `row[name] = row.get(name)` fallback at
        core.py, so a valueless column is declared but never
        written into the rows.
    Oracle: hand-computed [None, None] plus the declared type int, read
        back through the rows rather than through colmap alone.
    """
    ds = DataSet([
        {'a': 1, 'b': 2},
        {'a': 3, 'b': 4},
    ])
    ds.add_column('c', int)
    assert ds.cols == ['a', 'b', 'c']
    assert ds.colmap['c'] is int
    assert list(ds.unwind('c')) == [None, None]


def test_add_column_constant_value():
    """Verify a non-callable value is written to every row.

    Mutation: writing the constant to self.container[0] only, or
        inverting the `callable(value)` test at core.py so
        the string is called.
    Oracle: hand-computed three copies of 'constant'.
    """
    ds = DataSet([
        {'a': 1},
        {'a': 2},
        {'a': 3},
    ])
    ds.add_column('b', str, value='constant')
    assert list(ds.unwind('b')) == ['constant', 'constant', 'constant']


def test_add_column_callable_value():
    """Verify a callable value is evaluated once per row.

    Mutation: `value_fn(self.container[0])` in place of
        `value_fn(row)` at core.py, which broadcasts row 0.
    Oracle: hand-computed [11, 22, 33]; the broadcast defect gives
        [11, 11, 11].
    """
    ds = DataSet([
        {'a': 1, 'b': 10},
        {'a': 2, 'b': 20},
        {'a': 3, 'b': 30},
    ])
    ds.add_column('sum', int, value=lambda row: row['a'] + row['b'])
    assert list(ds.unwind('sum')) == [11, 22, 33]


def test_add_column_with_values_list():
    """Verify a values list is matched to rows by position.

    Mutation: `zip(self.container, reversed(values))` at
        core.py, or zipping values against the column list.
    Oracle: hand-computed [10, 20, 30] against a reversed [30, 20, 10].
    """
    ds = DataSet([
        {'a': 1},
        {'a': 2},
        {'a': 3},
    ])
    ds.add_column('b', int, values=[10, 20, 30])
    assert list(ds.unwind('b')) == [10, 20, 30]


def test_add_column_replaces_existing():
    """Verify re-adding a column records the new type, not the old one.

    Mutation: appending `(name, old_type)` in place of `(name, typ)` at
        core.py, keeping str after an int re-declaration.
    Oracle: hand-computed colmap {'a': int, 'b': int} and values [2, 4]
        parsed from the '2' and '4' strings.
    """
    ds = DataSet([
        {'a': 1, 'b': '2'},
        {'a': 3, 'b': '4'},
    ])
    ds.columns = [('a', int), ('b', str)]
    assert ds.colmap['b'] is str
    ds.add_column('b', int, value=lambda row: int(row['b']))
    assert ds.columns == [('a', int), ('b', int)]
    assert list(ds.unwind('b')) == [2, 4]


def test_add_column_empty_dataset():
    """Verify add_column on an empty dataset only extends the schema.

    Mutation: hoisting the value write out of the per-row loop (e.g.
        `self.container[0][name] = value`), which raises IndexError
        with no rows, or a length check that rejects an empty list.
    Oracle: hand-computed schema ['a', 'b', 'c'] against a row count
        that must stay 0.
    """
    ds = DataSet([], columns=[('a', int)])
    ds.add_column('b', str, value='test')
    ds.add_column('c', int, values=[])
    assert ds.cols == ['a', 'b', 'c']
    assert len(ds) == 0


def test_add_column_multiple_times():
    """Verify successive adds append in call order.

    Mutation: `self._columns.insert(0, ...)` in place of `append` at
        core.py.
    Oracle: hand-computed order ['a', 'b', 'c', 'd'] and the full row
        {'a': 1, 'b': 2, 'c': 3, 'd': 4}.
    """
    ds = DataSet([
        {'a': 1},
    ])
    ds.add_column('b', int, value=2)
    ds.add_column('c', int, value=3)
    ds.add_column('d', int, value=4)
    assert ds.cols == ['a', 'b', 'c', 'd']
    assert ds[0] == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


# --- Parameterized Type Tests ---

@pytest.mark.parametrize(('col_type', 'test_value'), [
    (int, 42),
    (float, math.pi),
    (str, 'hello'),
    (Date, Date(2024, 1, 15)),
    (DateTime, DateTime(2024, 1, 15, 10, 30)),
    (Time, Time(10, 30, 0)),
    (bool, True),
])
def test_add_column_preserves_type(col_type, test_value):
    """Verify the declared type is recorded, whatever the value is.

    Mutation: recording `(name, type(value))` in place of
        `(name, typ)` at core.py, which reads the type
        off the data and so records NoneType for a valueless column.
    Oracle: the caller's own type argument, checked on a column given a
        value and on one left empty.
    """
    ds = DataSet([{'a': 1}])
    ds.add_column('b', col_type, value=test_value)
    ds.add_column('c', col_type)
    assert ds.colmap['b'] is col_type
    assert ds.colmap['c'] is col_type
    assert ds[0]['b'] == test_value
    assert isinstance(ds[0]['b'], col_type)
    assert ds[0]['c'] is None


def test_add_column_bool_type_values():
    """Verify a values list wins over a value, bools included.

    Mutation: testing `value` before `values` at core.py and
        1460, so the constant overwrites the positional list.
    Oracle: hand-computed [True, False] against the [True, True] the
        losing branch would write.
    """
    ds = DataSet([{'a': 1}, {'a': 2}])
    ds.add_column('b', bool, value=True, values=[True, False])
    assert ds.colmap['b'] is bool
    assert ds[0]['b'] is True
    assert ds[1]['b'] is False


# --- Index Position Tests ---

@pytest.mark.parametrize(('index', 'expected_cols'), [
    (0, ['c', 'a', 'b']),
    (1, ['a', 'c', 'b']),
    (2, ['a', 'b', 'c']),
])
def test_add_column_with_index_position(index, expected_cols):
    """Verify index places the column at that position exactly.

    Mutation: an off-by-one in `self._columns.insert(index, ...)` at
        core.py, or ignoring index and appending.
    Oracle: hand-computed order per index, the head and middle cases
        straddling the append behavior.
    """
    ds = DataSet([
        {'a': 1, 'b': 2},
    ])
    ds.columns = [('a', int), ('b', int)]
    ds.add_column('c', int, index=index, value=3)
    assert ds.cols == expected_cols


def test_add_column_insert_existing_at_new_index():
    """Verify an existing column moves to the given index, value kept.

    Mutation: dropping `del self._columns[existing_index]` at
        core.py, leaving the columns setter to keep the last
        copy and so the original order.
    Oracle: hand-computed order ['c', 'a', 'b'] and the surviving value
        3.
    """
    ds = DataSet([
        {'a': 1, 'b': 2, 'c': 3},
    ])
    ds.columns = [('a', int), ('b', int), ('c', int)]
    assert ds.cols == ['a', 'b', 'c']

    ds.add_column('c', int, index=0)
    assert ds.cols == ['c', 'a', 'b']
    assert ds[0]['c'] == 3


# --- Parameterized Values Length Error Tests ---

@pytest.mark.parametrize(('values', 'dataset_len', 'error_match'), [
    ([10, 20], 4, 'values length 2 must match dataset length 4'),
    ([10, 20, 30, 40, 50], 2, 'values length 5 must match dataset length 2'),
    ([], 2, 'values length 0 must match dataset length 2'),
    ([99], 0, 'values length 1 must match dataset length 0'),
])
def test_add_column_values_length_mismatch(values, dataset_len, error_match):
    """Verify a values list of the wrong length is rejected.

    Mutation: `<` or `>` in place of `!=` at core.py - the
        short cases survive a `>` test and the long case survives a `<`
        test.
    Oracle: hand-computed message naming both lengths, with the empty
        dataset case sitting on the zero boundary.
    """
    ds = DataSet([{'a': i} for i in range(dataset_len)])
    with pytest.raises(ValueError, match=error_match):
        ds.add_column('b', int, values=values)


# --- Callable Value Tests ---

def test_add_column_callable_depends_on_multiple_columns():
    """Verify the callable receives the live row as an lazydict.

    Mutation: passing a plain dict or a copy of the row to the callable
        at core.py, which breaks attribute access.
    Oracle: hand-computed (10 + 5) / 2 = 7.5 and (20 + 4) / 3 = 8.0.
    """
    ds = DataSet([
        {'a': 10, 'b': 5, 'c': 2},
        {'a': 20, 'b': 4, 'c': 3},
    ])
    ds.add_column('result', float, value=lambda row: (row['a'] + row.b) / row.c)
    assert list(ds.unwind('result')) == [7.5, 8.0]


def test_add_column_callable_returns_none():
    """Verify a callable returning None still writes the key.

    Mutation: guarding the write with `if val is not None` at
        core.py, which leaves the odd rows without the key.
    Oracle: hand-computed [None, 2, None] over a % 2 test.
    """
    ds = DataSet([
        {'a': 1},
        {'a': 2},
        {'a': 3},
    ])
    ds.add_column('b', int, value=lambda row: row['a'] if row['a'] % 2 == 0 else None)
    assert list(ds.unwind('b')) == [None, 2, None]


def test_add_column_callable_not_called_when_not_callable():
    """Verify a non-callable value is written as is, falsy ones too.

    Mutation: `elif value:` in place of `elif value is not None:` at
        core.py, which drops a falsy constant into the
        row.get() branch and writes None.
    Oracle: hand-computed [0, 0] and [False, False] against the
        [None, None] the truthiness test would give.
    """
    ds = DataSet([
        {'a': 1},
        {'a': 2},
    ])
    ds.add_column('b', str, value='static')
    ds.add_column('c', int, value=0)
    ds.add_column('d', bool, value=False)
    assert list(ds.unwind('b')) == ['static', 'static']
    assert list(ds.unwind('c')) == [0, 0]
    assert list(ds.unwind('d')) == [False, False]


# --- Existing Column and Data Preservation Tests ---

def test_add_column_preserves_existing_columns():
    """Verify a mid-list insert keeps the other columns and their data.

    Mutation: `self._columns[index] = (name, typ)` in place of
        `insert(index, ...)` at core.py, which drops the
        column already sitting at that position.
    Oracle: hand-computed schema [('a', int), ('c', int), ('b', int)]
        and full rows still carrying their original a and b values.
    """
    ds = DataSet([
        {'a': 1, 'b': 2},
        {'a': 3, 'b': 4},
    ])
    ds.columns = [('a', int), ('b', int)]
    ds.add_column('c', int, index=1, value=99)
    assert ds.columns == [('a', int), ('c', int), ('b', int)]
    assert ds[0] == {'a': 1, 'b': 2, 'c': 99}
    assert ds[1] == {'a': 3, 'b': 4, 'c': 99}


def test_add_column_with_none_in_values():
    """Verify a None inside a values list is written, not skipped.

    Mutation: `if val is not None` guarding the write at
        core.py, which would leave row 1 without the key and
        shift nothing else.
    Oracle: hand-computed [10, None, 30].
    """
    ds = DataSet([
        {'a': 1},
        {'a': 2},
        {'a': 3},
    ])
    ds.add_column('b', int, values=[10, None, 30])
    assert list(ds.unwind('b')) == [10, None, 30]


def test_add_column_no_value_no_values():
    """Verify a valueless add keeps each row's own existing value.

    Mutation: `row[name] = None` in place of `row[name] = row.get(name)`
        at core.py, or reading the value off
        self.container[0] for every row.
    Oracle: hand-computed [10, 20, None], the last row having no such
        key to keep.
    """
    ds = DataSet([
        {'a': 1, 'c': 10},
        {'a': 2, 'c': 20},
        {'a': 3},
    ])
    ds.add_column('c', int)
    assert list(ds.unwind('c')) == [10, 20, None]


def test_add_column_type_change_converts_convertible_only():
    """Verify a type change converts what it can and keeps what it cannot.

    Mutation: dropping `self._types_converted = False` at
        core.py, so '42' never becomes 42; or `return result`
        in place of `return result if result is not None else val` in
        _convert_value, so the unparsable 'old' becomes None.
    Oracle: hand-computed ['old', 42] - one value past int() and one
        not - from the same str-to-int re-declaration.
    """
    ds = DataSet([
        {'a': 1, 'b': 'old'},
        {'a': 2, 'b': '42'},
    ])
    ds.columns = [('a', int), ('b', str)]

    ds.add_column('b', int)

    assert ds[0]['b'] == 'old'
    assert ds[1]['b'] == 42


# --- Type Conversion and Cache Tests ---

def test_add_column_callable_with_type_conversion():
    """Verify the callable supplies the value when the column exists.

    Mutation: skipping the value branch for an existing column at
        core.py, leaving plain type conversion of the
        original string to fill the column.
    Oracle: hand-computed dates one day past the source strings, which
        conversion alone cannot produce.
    """
    ds = DataSet([
        {'a': 1, 'date_str': '2024-01-15'},
        {'a': 2, 'date_str': '2024-02-20'},
    ])
    ds.columns = [('a', int), ('date_str', str)]

    ds.add_column('date_str', Date,
                  value=lambda x: Date.parse(x['date_str']).add(days=1))

    assert isinstance(ds[0]['date_str'], Date)
    assert ds[0]['date_str'] == Date(2024, 1, 16)
    assert ds[1]['date_str'] == Date(2024, 2, 21)


def test_automatic_date_type_conversion():
    """Verify a bare type change re-arms conversion of stored strings.

    Mutation: dropping `self._types_converted = False` at
        core.py, leaving the values as the '2024-01-15'
        strings they arrived as.
    Oracle: hand-computed Date(2024, 1, 15) and Date(2024, 2, 20).
    """
    ds = DataSet([
        {'a': 1, 'date_str': '2024-01-15'},
        {'a': 2, 'date_str': '2024-02-20'},
    ])
    ds.columns = [('a', int), ('date_str', str)]

    ds.add_column('date_str', Date)

    assert isinstance(ds[0]['date_str'], Date)
    assert ds[0]['date_str'] == Date(2024, 1, 15)
    assert isinstance(ds[1]['date_str'], Date)
    assert ds[1]['date_str'] == Date(2024, 2, 20)


def test_add_column_callable_with_date_business_method():
    """Verify pending conversions run before the callable sees a row.

    Mutation: dropping @ensure_types_converted from add_column at
        core.py, so the callable meets a raw datetime.date
        with no .business(); or broadcasting row 0 to every row.
    Oracle: hand-computed 2024-01-12 and 2024-03-14, one business day
        back from a Monday and a Friday.
    """
    ds = DataSet([
        {'id': 1, 'as_of_date': datetime.date(2024, 1, 15)},
        {'id': 2, 'as_of_date': datetime.date(2024, 3, 15)},
    ])
    ds.columns = [('id', int), ('as_of_date', Date)]

    assert isinstance(ds.container[0]['as_of_date'], datetime.date)
    assert not isinstance(ds.container[0]['as_of_date'], Date)

    ds.add_column('prev_date', Date,
                  value=lambda x: x.as_of_date.business().subtract(days=1))

    assert isinstance(ds[0]['as_of_date'], Date)
    assert isinstance(ds[0]['prev_date'], Date)
    assert ds[0]['prev_date'] == Date(2024, 1, 12)
    assert ds[1]['prev_date'] == Date(2024, 3, 14)


def test_add_column_type_change_invalidates_cache():
    """Verify changing a column's type clears the converted flag.

    Mutation: dropping the `old_type != typ` arm at core.py,
        which leaves the flag True and the strings unparsed.
    Oracle: the flag read either side of the call, plus Date(2024, 1,
        15) proving the later read really re-converted.
    """
    ds = DataSet([
        {'a': 1, 'date_str': '2024-01-15'},
        {'a': 2, 'date_str': '2024-02-20'},
    ])
    ds.columns = [('a', int), ('date_str', str)]

    _ = ds[0]
    assert ds._types_converted is True

    ds.add_column('date_str', Date)
    assert ds._types_converted is False

    assert isinstance(ds[0]['date_str'], Date)
    assert ds[0]['date_str'] == Date(2024, 1, 15)


def test_add_column_same_type_preserves_cache():
    """Verify re-adding a column at its own type keeps the cache warm.

    Mutation: dropping the `old_type != typ` test at
        core.py, so any re-add invalidates and every row is
        walked again.
    Oracle: the flag, still True, against the callable's own output
        ['100_modified', '200_modified'].
    """
    ds = DataSet([
        {'a': 1, 'b': '100'},
        {'a': 2, 'b': '200'},
    ])
    ds.columns = [('a', int), ('b', str)]

    _ = ds[0]
    assert ds._types_converted is True

    ds.add_column('b', str, value=lambda x: x['b'] + '_modified')
    assert ds._types_converted is True

    assert ds[0]['b'] == '100_modified'
    assert ds[1]['b'] == '200_modified'


def test_add_column_new_column_preserves_cache():
    """Verify adding a brand new column keeps the cache warm.

    Mutation: dropping the `old_type is not None` guard at
        core.py, so a new column invalidates the cache that
        the decorator just filled.
    Oracle: the flag, still True, plus hand-computed sums 3 and 7 off
        the already converted ints.
    """
    ds = DataSet([
        {'a': '1', 'b': '2'},
        {'a': '3', 'b': '4'},
    ])
    ds.columns = [('a', int), ('b', int)]

    _ = ds[0]
    assert ds._types_converted is True
    assert isinstance(ds[0]['a'], int)

    ds.add_column('c', int, value=lambda x: x['a'] + x['b'])
    assert ds._types_converted is True

    assert ds[0]['c'] == 3
    assert ds[1]['c'] == 7


def test_add_column_callable_multiple_date_operations():
    """Verify a second callable reads the column the first one wrote.

    Mutation: writing the callable's result to a copy of the row at
        core.py, so x.month_end is missing on the next call.
    Oracle: hand-computed business month ends 2024-03-28 and
        2024-06-28, giving day counts 13 and -2 from the source dates.
    """
    ds = DataSet([
        {'label': 'A', 'date': datetime.date(2024, 3, 15)},
        {'label': 'B', 'date': datetime.date(2024, 6, 30)},
    ])
    ds.columns = [('label', str), ('date', Date)]

    ds.add_column('month_end', Date, value=lambda x: x.date.business().end_of('month'))
    ds.add_column('days_to_eom', int, value=lambda x: (x.month_end - x.date).days)

    assert list(ds.unwind('month_end')) == [Date(2024, 3, 28), Date(2024, 6, 28)]
    assert list(ds.unwind('days_to_eom')) == [13, -2]


# --- Edge Case Tests ---

def test_add_column_negative_index():
    """Verify a negative index inserts the way list.insert does.

    Mutation: `insert(abs(index), ...)` or an append fallback for a
        negative index at core.py.
    Oracle: hand-computed ['a', 'b', 'd', 'c'] - index -1 lands before
        the last column, not after it.
    """
    ds = DataSet([
        {'a': 1, 'b': 2, 'c': 3},
    ])
    ds.columns = [('a', int), ('b', int), ('c', int)]

    ds.add_column('d', int, index=-1, value=4)

    assert ds.cols == ['a', 'b', 'd', 'c']
    assert ds[0]['d'] == 4


def test_add_column_index_larger_than_columns():
    """Verify an out-of-range index appends rather than raising.

    Mutation: a bounds check raising IndexError, or clamping the index
        to 0, at core.py.
    Oracle: hand-computed ['a', 'b', 'c'] from an index of 100.
    """
    ds = DataSet([
        {'a': 1, 'b': 2},
    ])
    ds.columns = [('a', int), ('b', int)]

    ds.add_column('c', int, index=100, value=3)

    assert ds.cols == ['a', 'b', 'c']
    assert ds[0]['c'] == 3


def test_add_column_callable_raises_exception():
    """Verify an error raised inside the callable reaches the caller.

    Mutation: wrapping the per-row loop at core.py in
        contextlib.suppress(Exception), which would swallow the divide
        by zero and leave the column half filled.
    Oracle: the ZeroDivisionError itself, raised on the second row.
    """
    ds = DataSet([
        {'a': 1},
        {'a': 0},
        {'a': 3},
    ])

    with pytest.raises(ZeroDivisionError):
        ds.add_column('b', int, value=lambda row: 10 / row['a'])


def test_add_column_with_generator_values():
    """Verify values must be sized: a generator is rejected.

    Mutation: `values = list(values)` before the length check at
        core.py, which silently accepts an unsized iterable
        and so cannot report a length mismatch for one.
    Oracle: the TypeError from len() on a generator, against the
        [10, 20, 30] a materializing implementation would write.
    """
    ds = DataSet([
        {'a': 1},
        {'a': 2},
        {'a': 3},
    ])

    with pytest.raises(TypeError):
        ds.add_column('b', int, values=(x * 10 for x in range(1, 4)))


def test_add_column_special_characters_in_name():
    """Verify a column name is used verbatim, however awkward.

    Mutation: normalizing the name at core.py (a strip,
        a lower, or a space-to-underscore swap), which would file the
        row value under a key the caller never asked for.
    Oracle: the caller's own strings, a 1000-char name and one holding
        spaces and punctuation, looked up unchanged.
    """
    ds = DataSet([{'a': 1}])
    long_name = 'x' * 1000
    special_name = 'col with spaces & symbols!'

    ds.add_column(long_name, int, value=42)
    ds.add_column(special_name, int, value=7)

    assert ds.cols == ['a', long_name, special_name]
    assert ds[0][long_name] == 42
    assert ds[0][special_name] == 7


def test_add_column_callable_accesses_new_column():
    """Verify a later callable sees a column an earlier call added.

    Mutation: writing the value into a copy of the row rather than the
        container row at core.py, so 'b' is missing when the
        second callable runs.
    Oracle: hand-computed [10, 20] then [11, 21], the second built off
        the first.
    """
    ds = DataSet([
        {'a': 1},
        {'a': 2},
    ])

    ds.add_column('b', int, value=lambda row: row['a'] * 10)
    ds.add_column('c', int, value=lambda row: row['b'] + 1)

    assert list(ds.unwind('b')) == [10, 20]
    assert list(ds.unwind('c')) == [11, 21]


def test_add_column_to_single_row_dataset():
    """Verify a one-value list matches a one-row dataset.

    Mutation: an off-by-one in the length test at core.py
        (`len(values) - 1 != len(self.container)`), which would reject
        this correct call.
    Oracle: hand-computed 42 and 100 in the single row, with no
        ValueError raised.
    """
    ds = DataSet([{'a': 1}])
    ds.add_column('b', int, value=42)
    ds.add_column('c', int, values=[100])
    assert ds[0] == {'a': 1, 'b': 42, 'c': 100}


def test_add_column_preserves_summary():
    """Verify add_column leaves a declared summary row untouched.

    Mutation: resetting self._summary_args (to () or to a fresh
        add_summary_row()) inside add_column, which would re-derive the
        totaled columns and pick up the new numeric column.
    Oracle: hand-computed total 300 over 'value' alone, with the new
        column 'c' still None in the summary despite holding 10 twice.
    """
    ds = DataSet([
        {'a': 1, 'value': 100},
        {'a': 2, 'value': 200},
    ])
    ds.columns = [('a', int), ('value', int)]
    ds.add_summary_row(label='Total', columns=['value'])

    _ = ds.summary

    ds.add_column('c', int, value=10)

    assert ds._summary_args == (0, 'Total', ['value'], None)
    summary = ds.calc_summary_row(*ds._summary_args)
    assert summary['a'] == 'Total'
    assert summary['value'] == 300
    assert summary['c'] is None


# --- remove_column / rename_column Tests ---


def test_rename_column_onto_existing_keeps_source_type():
    """Verify a rename onto a live column carries the source type.

    Mutation: skipping the pre-emptive remove_column(rename) at
        core.py - either by inverting its `rename in
        self.colmap` guard or by passing None instead of the name - so
        the stale target entry survives and the columns setter keeps
        its type in place of the source column's.
    Oracle: hand-derived schema {'label': int} - the target entry is
        dropped first, so 'label' can only carry 'code' type int; the
        skipped removal leaves str.
    """
    ds = DataSet([
        {'code': 1, 'label': 'x'},
        {'code': 2, 'label': 'y'},
    ])
    ds.columns = [('code', int), ('label', str)]

    ds.rename_column('code', 'label')

    assert ds.cols == ['label']
    assert ds.colmap == {'label': int}
    assert ds.container == [{'label': 1}, {'label': 2}]


def test_rename_column_onto_existing_drops_stale_target_values():
    """Verify a displaced column is cleared from untouched rows.

    Mutation: skipping the pre-emptive remove_column(rename) at
        core.py, which leaves the old target value in any
        row the rename loop does not touch.
    Oracle: hand-derived rows [{'label': 1}, {}] - row two has no
        'code' key, so nothing may be written under 'label' there.
    """
    ds = DataSet([{'code': 1, 'label': 'x'}])
    ds.columns = [('code', int), ('label', str)]
    ds.append({'label': 'y'})

    ds.rename_column('code', 'label')

    assert ds.container == [{'label': 1}, {}]


def test_rename_column_same_name_warns_naming_the_column(caplog):
    """Verify a self-rename is a no-op that names the column it skips.

    Mutation: logger.warning(None) in place of the formatted message
        at core.py, so the warning no longer says which
        column was skipped.
    Oracle: the literal 'value' read back out of the captured record,
        against the mutant's bare 'None'.
    """
    ds = DataSet([
        {'id': 1, 'value': 10},
        {'id': 2, 'value': 20},
    ])
    ds.columns = [('id', int), ('value', int)]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='rollups.core'):
        ds.rename_column('value', 'value')

    records = [r for r in caplog.records
               if r.name == 'rollups.core'
               and r.levelno == logging.WARNING]
    assert len(records) == 1
    assert 'value' in records[0].getMessage()
    assert ds.cols == ['id', 'value']
    assert ds.container == [{'id': 1, 'value': 10}, {'id': 2, 'value': 20}]


def test_remove_column_missing_logs_the_missing_name(caplog):
    """Verify removing an absent column names it in the debug log.

    Mutation: logger.debug(None) in place of the formatted message at
        core.py, dropping the column name from the log of
        the no-op branch.
    Oracle: the literal 'ghost' read back out of the captured record,
        against the mutant's bare 'None'.
    """
    ds = DataSet([
        {'id': 1, 'value': 10},
        {'id': 2, 'value': 20},
    ])
    ds.columns = [('id', int), ('value', int)]

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='rollups.core'):
        ds.remove_column('ghost')

    records = [r for r in caplog.records
               if r.name == 'rollups.core'
               and r.levelno == logging.DEBUG]
    assert len(records) == 1
    assert 'ghost' in records[0].getMessage()
    assert ds.cols == ['id', 'value']
    assert ds.container == [{'id': 1, 'value': 10}, {'id': 2, 'value': 20}]


if __name__ == '__main__':
    pytest.main([__file__])


def test_add_column_fills_a_column_named_after_a_dict_method():
    """Verify a new column named 'items' fills with None, not a method.

    Mutation: reading the existing value through the row's own get(),
        which answers a missing key with the dict attribute of that
        name, so the row gains a bound method as data.
    Oracle: hand-computed - no row carries 'items', so every row holds
        None under it.
    """
    ds = DataSet([{'a': 1}, {'a': 2}])

    ds.add_column('items', str)

    assert [row['items'] for row in ds] == [None, None]
