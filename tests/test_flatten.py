"""Comprehensive tests for DataSet flatten (unpivot) operations.

Flatten Operation Overview
--------------------------
The flatten method transforms a wide-format dataset into long-format by unpivoting
specified columns. This is useful for data visualization, aggregation, and analysis
where metrics need to be in a single column.

Example:
    Wide format:
        | category | sales | profit |
        | A        | 100   | 20     |
        | B        | 200   | 40     |

    After flatten(['category'], ['sales', 'profit']):
        | category | key    | val |
        | A        | sales  | 100 |
        | A        | profit | 20  |
        | B        | sales  | 200 |
        | B        | profit | 40  |
"""
import pytest
from opendate import Date, DateTime, Time
from rollups import DataSet

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Basic dataset for flatten tests."""
    ds = DataSet([
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 2, 'b': 3, 'c': 3},
        {'a': 3, 'b': 4, 'c': 3}
    ])
    return ds


@pytest.fixture
def category_metrics_dataset():
    """Dataset with category and metrics for flatten tests."""
    ds = DataSet([
        {'category': 'A', 'sales': 100, 'profit': 20},
        {'category': 'B', 'sales': 200, 'profit': 40},
    ])
    return ds


@pytest.fixture
def typed_dataset():
    """Dataset with typed columns for flatten tests."""
    ds = DataSet([
        {'date': Date(2024, 1, 1), 'amount': 100.0, 'count': 5},
        {'date': Date(2024, 1, 2), 'amount': 200.0, 'count': 10},
    ])
    ds.columns = [('date', Date), ('amount', float), ('count', int)]
    return ds


# --- Basic Flatten Tests ---

def test_flatten_with_kept_columns(basic_dataset):
    """Verify each output row pairs its own source cell with the kept value.

    Mutation: val taking row[kept[0]] rather than row[k] in the kept
        branch of flatten.
    Oracle: hand-computed (a, key, val) triples off the fixture rows.
    """
    result = basic_dataset.flatten(['a'], ['b'])

    assert result.cols == ['a', 'key', 'val']
    assert [(r.a, r.key, r.val) for r in result] == [
        (1, 'b', 2),
        (2, 'b', 3),
        (3, 'b', 4),
        ]


def test_flatten_without_kept_columns(basic_dataset):
    """Verify an empty kept list drops every source column but key/val.

    Mutation: dropping the `if c in kept` filter when flatten builds the
        output schema, leaking a, b and c into the result.
    Oracle: hand-computed column list and (key, val) pairs.
    """
    result = basic_dataset.flatten([], ['b'])

    assert result.cols == ['key', 'val']
    assert [(r.key, r.val) for r in result] == [('b', 2), ('b', 3), ('b', 4)]


def test_flatten_multiple_flattened_columns(basic_dataset):
    """Verify the no-kept branch cycles columns within each source row.

    Mutation: swapping the row and column loops in the no-kept branch,
        which emits every 'b' row before every 'c' row.
    Oracle: hand-computed (key, val) pairs in row-major order.
    """
    result = basic_dataset.flatten([], ['b', 'c'], key='xx')

    assert result.cols == ['xx', 'val']
    assert [(r.xx, r.val) for r in result] == [
        ('b', 2), ('c', 3),
        ('b', 3), ('c', 3),
        ('b', 4), ('c', 3),
        ]


# --- Custom Key/Value Name Tests (Parameterized) ---

@pytest.mark.parametrize(('key_name', 'val_name', 'expected_cols'), [
    ('column_name', 'val', ['column_name', 'val']),
    ('key', 'amount', ['key', 'amount']),
    ('metric', 'value', ['metric', 'value']),
])
def test_flatten_custom_column_names(key_name, val_name, expected_cols):
    """Verify the key and val arguments name and fill the output columns.

    Mutation: hardcoding 'key'/'val' in place of the arguments, or
        writing the column name into val and the cell into key.
    Oracle: hand-computed column list plus the single (b, 2) cell.
    """
    ds = DataSet([
        {'a': 1, 'b': 2, 'c': 3}
    ])

    result = ds.flatten([], ['b'], key=key_name, val=val_name)

    assert result.cols == expected_cols
    assert getattr(result[0], key_name) == 'b'
    assert getattr(result[0], val_name) == 2


# --- Kept Column Tests ---

def test_flatten_single_kept_column_as_string(category_metrics_dataset):
    """Verify a bare string kept argument behaves as a one-element list.

    Mutation: dropping the `isinstance(kept, list | tuple)` wrap, which
        makes attrgetter iterate the letters of 'category'.
    Oracle: differential against the same call with kept as a list.
    """
    result = category_metrics_dataset.flatten('category', ['sales', 'profit'])
    listed = category_metrics_dataset.flatten(['category'], ['sales', 'profit'])

    assert result.cols == ['category', 'key', 'val']
    assert result == listed


def test_flatten_single_kept_column_as_list(category_metrics_dataset):
    """Verify one kept column repeats across each row's flattened cells.

    Mutation: val taking row[flattened[0]], so the profit rows carry
        the sales cell.
    Oracle: hand-computed (category, key, val) triples for both rows.
    """
    result = category_metrics_dataset.flatten(['category'], ['sales', 'profit'])

    assert [(r.category, r.key, r.val) for r in result] == [
        ('A', 'sales', 100),
        ('A', 'profit', 20),
        ('B', 'sales', 200),
        ('B', 'profit', 40),
        ]


def test_flatten_multiple_kept_columns():
    """Verify two kept columns both land on every flattened row.

    Mutation: attrgetter(kept[0]) in place of attrgetter(*kept), which
        drops quarter from the emitted rows.
    Oracle: hand-computed 6-row expansion of the 3x2 grid.
    """
    ds = DataSet([
        {'region': 'North', 'quarter': 'Q1', 'metric_a': 100, 'metric_b': 60},
        {'region': 'North', 'quarter': 'Q2', 'metric_a': 150, 'metric_b': 80},
        {'region': 'South', 'quarter': 'Q1', 'metric_a': 200, 'metric_b': 120},
    ])

    result = ds.flatten(['region', 'quarter'], ['metric_a', 'metric_b'])

    assert result.cols == ['region', 'quarter', 'key', 'val']
    assert [(r.region, r.quarter, r.key, r.val) for r in result] == [
        ('North', 'Q1', 'metric_a', 100),
        ('North', 'Q1', 'metric_b', 60),
        ('North', 'Q2', 'metric_a', 150),
        ('North', 'Q2', 'metric_b', 80),
        ('South', 'Q1', 'metric_a', 200),
        ('South', 'Q1', 'metric_b', 120),
        ]


def test_flatten_three_kept_columns():
    """Verify a three-column kept tuple is zipped back in the right order.

    Mutation: zip(keeps, kept) in place of zip(kept, keeps), or slicing
        the attrgetter tuple short, which misplaces month and day.
    Oracle: hand-computed 4-row expansion of the 2x2 grid.
    """
    ds = DataSet([
        {'year': 2024, 'month': 1, 'day': 1, 'temp': 20, 'pressure': 1013},
        {'year': 2024, 'month': 1, 'day': 2, 'temp': 22, 'pressure': 1015},
    ])

    result = ds.flatten(['year', 'month', 'day'], ['temp', 'pressure'])

    assert result.cols == ['year', 'month', 'day', 'key', 'val']
    assert [(r.year, r.month, r.day, r.key, r.val) for r in result] == [
        (2024, 1, 1, 'temp', 20),
        (2024, 1, 1, 'pressure', 1013),
        (2024, 1, 2, 'temp', 22),
        (2024, 1, 2, 'pressure', 1015),
        ]


# --- Type Preservation Tests ---

def test_flatten_preserves_kept_column_types(typed_dataset):
    """Verify a kept column carries its declared type into the result.

    Mutation: retyping the kept columns as str while the output schema
        is built, so date no longer declares Date.
    Oracle: hand-computed schema, checked as an ordered (name, type) list.
    """
    result = typed_dataset.flatten(['date'], ['amount', 'count'])

    assert result.columns == [('date', Date), ('key', str), ('val', float)]
    assert all(isinstance(r.date, Date) for r in result)
    assert [r.date for r in result] == [
        Date(2024, 1, 1),
        Date(2024, 1, 1),
        Date(2024, 1, 2),
        Date(2024, 1, 2),
        ]


def test_flatten_datetime_kept_column():
    """Verify a DateTime kept value repeats unchanged on each of its rows.

    Mutation: val taking row[kept[0]], which writes the timestamp into
        val in place of the metric.
    Oracle: hand-computed (timestamp, key, val) 4-row expansion.
    """
    ds = DataSet([
        {'timestamp': DateTime(2024, 1, 1, 10, 0), 'temp': 20.5, 'humidity': 60},
        {'timestamp': DateTime(2024, 1, 1, 11, 0), 'temp': 21.0, 'humidity': 58},
    ])
    ds.columns = [('timestamp', DateTime), ('temp', float), ('humidity', int)]

    result = ds.flatten(['timestamp'], ['temp', 'humidity'])

    assert result.colmap['timestamp'] == DateTime
    assert [(r.timestamp, r.key, r.val) for r in result] == [
        (DateTime(2024, 1, 1, 10, 0), 'temp', 20.5),
        (DateTime(2024, 1, 1, 10, 0), 'humidity', 60),
        (DateTime(2024, 1, 1, 11, 0), 'temp', 21.0),
        (DateTime(2024, 1, 1, 11, 0), 'humidity', 58),
        ]


def test_flatten_numeric_values():
    """Verify the generated key/val columns are typed str and float.

    Mutation: declaring val as int (or key as the source column's type),
        which a downstream numeric formatter would then truncate.
    Oracle: hand-computed schema plus the two source cells.
    """
    ds = DataSet([
        {'id': 1, 'int_val': 10, 'float_val': 10.5},
    ])
    ds.columns = [('id', int), ('int_val', int), ('float_val', float)]

    result = ds.flatten(['id'], ['int_val', 'float_val'])

    assert result.columns == [('id', int), ('key', str), ('val', float)]
    assert [(r.key, r.val) for r in result] == [('int_val', 10), ('float_val', 10.5)]


def test_flatten_string_kept_column():
    """Verify a str kept column survives beside the generated columns.

    Mutation: filtering the output schema on `flattened` rather than
        `kept`, which replaces name with math and science.
    Oracle: hand-computed schema plus the 4-row expansion.
    """
    ds = DataSet([
        {'name': 'A', 'math': 85, 'science': 90},
        {'name': 'B', 'math': 75, 'science': 80},
    ])
    ds.columns = [('name', str), ('math', int), ('science', int)]

    result = ds.flatten(['name'], ['math', 'science'])

    assert result.columns == [('name', str), ('key', str), ('val', float)]
    assert [(r.name, r.key, r.val) for r in result] == [
        ('A', 'math', 85),
        ('A', 'science', 90),
        ('B', 'math', 75),
        ('B', 'science', 80),
        ]


# --- Empty and Single Row Tests ---

def test_flatten_empty_dataset():
    """Verify the output schema comes from the columns, not from the rows.

    Mutation: deriving the kept columns from the first row, which leaves
        an empty dataset with only key and val.
    Oracle: hand-computed schema for a dataset with no rows to inspect.
    """
    ds = DataSet([], columns=[('a', int), ('b', float), ('c', float)])

    result = ds.flatten(['a'], ['b', 'c'])

    assert len(result) == 0
    assert result.columns == [('a', int), ('key', str), ('val', float)]


def test_flatten_single_row():
    """Verify one row expands to one output row per flattened column.

    Mutation: sorting `flattened` before the inner loop, which emits
        count ahead of rate.
    Oracle: hand-computed (name, key, val) pairs in call order.
    """
    ds = DataSet([
        {'name': 'Item1', 'rate': 50.0, 'count': 10}
    ])

    result = ds.flatten(['name'], ['rate', 'count'])

    assert [(r.name, r.key, r.val) for r in result] == [
        ('Item1', 'rate', 50.0),
        ('Item1', 'count', 10),
        ]


def test_flatten_single_flattened_column():
    """Verify one flattened column names itself in key on every row.

    Mutation: writing the kept column's name into key instead of the
        flattened column's name.
    Oracle: hand-computed (category, key, val) pairs.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'B', 'value': 200},
    ])

    result = ds.flatten(['category'], ['value'])

    assert [(r.category, r.key, r.val) for r in result] == [
        ('A', 'value', 100),
        ('B', 'value', 200),
        ]


# --- None Value Tests ---

def test_flatten_with_none_values():
    """Verify a None cell still emits its row rather than being skipped.

    Mutation: guarding the append with `if row[k] is not None`, which
        drops the two None rows.
    Oracle: hand-computed 4-row expansion holding None in two slots.
    """
    ds = DataSet([
        {'category': 'A', 'metric1': None, 'metric2': 100},
        {'category': 'B', 'metric1': 50, 'metric2': None},
    ])

    result = ds.flatten(['category'], ['metric1', 'metric2'])

    assert [(r.category, r.key, r.val) for r in result] == [
        ('A', 'metric1', None),
        ('A', 'metric2', 100),
        ('B', 'metric1', 50),
        ('B', 'metric2', None),
        ]


# --- Ordering Tests ---

def test_flatten_maintains_grouping_order():
    """Verify rows keep source order, so split groups are not merged.

    Mutation: sorting the container by the kept columns before groupby,
        which pulls the trailing A rows up beside the leading ones.
    Oracle: hand-computed group sequence for deliberately interleaved
        input.
    """
    ds = DataSet([
        {'group': 'A', 'x': 1, 'y': 2},
        {'group': 'B', 'x': 5, 'y': 6},
        {'group': 'A', 'x': 3, 'y': 4},
    ])

    result = ds.flatten(['group'], ['x', 'y'])

    assert [r.group for r in result] == ['A', 'A', 'B', 'B', 'A', 'A']
    assert [r.val for r in result] == [1, 2, 5, 6, 3, 4]


def test_flatten_column_order():
    """Verify kept columns follow source order, not the argument order.

    Mutation: building the output schema by iterating `kept`, which
        would emit m before z.
    Oracle: hand-computed column list against a reversed kept argument.
    """
    ds = DataSet([
        {'z': 1, 'a': 10, 'm': 20},
    ])

    result = ds.flatten(['m', 'z'], ['a'])

    assert result.cols == ['z', 'm', 'key', 'val']
    assert [(r.z, r.m, r.key, r.val) for r in result] == [(1, 20, 'a', 10)]


def test_flatten_creates_cartesian_product():
    """Verify every kept row pairs with every flattened column exactly once.

    Mutation: val taking row[flattened[0]], which gives each id three
        rows all holding its 'a' cell.
    Oracle: hand-computed (key, val) sets per id.
    """
    ds = DataSet([
        {'id': 1, 'a': 10, 'b': 20, 'c': 30},
        {'id': 2, 'a': 40, 'b': 50, 'c': 60},
    ])

    result = ds.flatten(['id'], ['a', 'b', 'c'])

    assert len(result) == 6
    assert {(r.key, r.val) for r in result if r.id == 1} == {
        ('a', 10), ('b', 20), ('c', 30)}
    assert {(r.key, r.val) for r in result if r.id == 2} == {
        ('a', 40), ('b', 50), ('c', 60)}


# --- Edge Case Tests ---

def test_flatten_with_time_kept_column():
    """Verify a Time kept column keeps its type and value on every row.

    Mutation: retyping the kept column as str when the output schema is
        built, so Time.instance conversion rewrites the values.
    Oracle: hand-computed (time, key, val) 4-row expansion.
    """
    ds = DataSet([
        {'time': Time(10, 0, 0), 'metric_a': 100, 'metric_b': 200},
        {'time': Time(11, 0, 0), 'metric_a': 150, 'metric_b': 250},
    ])
    ds.columns = [('time', Time), ('metric_a', int), ('metric_b', int)]

    result = ds.flatten(['time'], ['metric_a', 'metric_b'])

    assert result.colmap['time'] == Time
    assert [(r.time, r.key, r.val) for r in result] == [
        (Time(10, 0, 0), 'metric_a', 100),
        (Time(10, 0, 0), 'metric_b', 200),
        (Time(11, 0, 0), 'metric_a', 150),
        (Time(11, 0, 0), 'metric_b', 250),
        ]


def test_flatten_with_boolean_values():
    """Verify a bool cell reaches val unchanged, not coerced to 1.0/0.0.

    Mutation: appending with validate=True, which runs the float column
        type over the value and turns True into 1.0.
    Oracle: identity checks against the True/False singletons.
    """
    ds = DataSet([
        {'id': 1, 'flag_a': True, 'flag_b': False},
        {'id': 2, 'flag_a': False, 'flag_b': True},
    ])
    ds.columns = [('id', int), ('flag_a', bool), ('flag_b', bool)]

    result = ds.flatten(['id'], ['flag_a', 'flag_b'])

    assert [(r.id, r.key) for r in result] == [
        (1, 'flag_a'), (1, 'flag_b'), (2, 'flag_a'), (2, 'flag_b')]
    assert result[0].val is True
    assert result[1].val is False
    assert result[2].val is False
    assert result[3].val is True


def test_flatten_with_all_none_flattened():
    """Verify an all-None source row still emits its full set of rows.

    Mutation: skipping a row whose flattened cells are all None, which
        empties the result.
    Oracle: hand-computed category sequence over the 2x2 expansion.
    """
    ds = DataSet([
        {'category': 'A', 'metric1': None, 'metric2': None},
        {'category': 'B', 'metric1': None, 'metric2': None},
    ])

    result = ds.flatten(['category'], ['metric1', 'metric2'])

    assert [r.category for r in result] == ['A', 'A', 'B', 'B']
    assert [r.key for r in result] == ['metric1', 'metric2', 'metric1', 'metric2']
    assert all(r.val is None for r in result)


def test_flatten_with_empty_string_values():
    """Verify an empty-string cell is emitted rather than treated as absent.

    Mutation: guarding the append with `if row[k]`, which drops the two
        empty-string rows.
    Oracle: hand-computed (id, key, val) 4-row expansion.
    """
    ds = DataSet([
        {'id': 1, 'name': '', 'desc': 'test'},
        {'id': 2, 'name': 'foo', 'desc': ''},
    ])

    result = ds.flatten(['id'], ['name', 'desc'])

    assert [(r.id, r.key, r.val) for r in result] == [
        (1, 'name', ''),
        (1, 'desc', 'test'),
        (2, 'name', 'foo'),
        (2, 'desc', ''),
        ]


def test_flatten_large_dataset():
    """Verify every one of 300 rows carries the cell its key names.

    Mutation: val taking row[kept[0]], so every row holds its own id
        rather than the cell its key names.
    Oracle: differential re-computation of each cell from id and key.
    """
    ds = DataSet([{'id': i, 'a': i * 10, 'b': i * 20, 'c': i * 30} for i in range(100)])
    ds.columns = [('id', int), ('a', int), ('b', int), ('c', int)]

    result = ds.flatten(['id'], ['a', 'b', 'c'])

    scale = {'a': 10, 'b': 20, 'c': 30}
    assert len(result) == 300
    assert len({r.id for r in result}) == 100
    assert all(r.val == r.id * scale[r.key] for r in result)


def test_flatten_with_negative_values():
    """Verify a negative cell keeps its sign and its column pairing.

    Mutation: val taking abs(row[k]), or profit and loss swapping cells.
    Oracle: hand-computed (id, key, val) 4-row expansion.
    """
    ds = DataSet([
        {'id': 1, 'profit': -100, 'loss': -50},
        {'id': 2, 'profit': 200, 'loss': -30},
    ])

    result = ds.flatten(['id'], ['profit', 'loss'])

    assert [(r.id, r.key, r.val) for r in result] == [
        (1, 'profit', -100),
        (1, 'loss', -50),
        (2, 'profit', 200),
        (2, 'loss', -30),
        ]


def test_flatten_with_float_infinity():
    """Verify infinities pass through val without being clamped or dropped.

    Mutation: guarding the append with math.isfinite, or coercing val
        through a numeric parse that maps inf to None.
    Oracle: hand-computed (key, val) pairs against float('inf').
    """
    ds = DataSet([
        {'id': 1, 'pos_inf': float('inf'), 'neg_inf': float('-inf')},
    ])
    ds.columns = [('id', int), ('pos_inf', float), ('neg_inf', float)]

    result = ds.flatten(['id'], ['pos_inf', 'neg_inf'])

    assert [(r.key, r.val) for r in result] == [
        ('pos_inf', float('inf')),
        ('neg_inf', float('-inf')),
        ]


def test_flatten_with_mixed_type_values():
    """Verify val holds each source cell as it was, despite its float type.

    Mutation: coercing val to float on the way in, which rewrites the
        int cell 100 as 100.0.
    Oracle: hand-computed cells plus a type check on the int and str.
    """
    ds = DataSet([
        {'id': 1, 'int_col': 100, 'float_col': 99.5, 'str_col': 'test'},
    ])

    result = ds.flatten(['id'], ['int_col', 'float_col', 'str_col'])

    assert [(r.key, r.val) for r in result] == [
        ('int_col', 100),
        ('float_col', 99.5),
        ('str_col', 'test'),
        ]
    assert isinstance(result[0].val, int)
    assert isinstance(result[2].val, str)


def test_flatten_many_flattened_columns():
    """Verify a 20-column flatten pairs each key with its own cell.

    Mutation: val taking row[flattened[0]], which gives every row the
        col_0 cell.
    Oracle: differential re-computation of each cell from its key index.
    """
    columns = {f'col_{i}': i * 10 for i in range(20)}
    columns['id'] = 1
    ds = DataSet([columns])

    col_names = [f'col_{i}' for i in range(20)]
    result = ds.flatten(['id'], col_names)

    assert len(result) == 20
    assert {(r.key, r.val) for r in result} == {
        (f'col_{i}', i * 10) for i in range(20)}


def test_flatten_preserves_none_in_kept_column():
    """Verify a None kept value groups and is written out like any other.

    Mutation: skipping a group whose kept value is None, which drops
        the first row.
    Oracle: hand-computed (category, key, val) pairs.
    """
    ds = DataSet([
        {'category': None, 'value': 100},
        {'category': 'B', 'value': 200},
    ])

    result = ds.flatten(['category'], ['value'])

    assert [(r.category, r.key, r.val) for r in result] == [
        (None, 'value', 100),
        ('B', 'value', 200),
        ]


def test_flatten_single_row_single_column():
    """Verify the 1x1 boundary emits exactly one row.

    Mutation: an off-by-one over `flattened` (flattened[1:] or
        flattened[:-1]), which empties the result at this boundary.
    Oracle: hand-computed single (id, key, val) triple.
    """
    ds = DataSet([{'id': 1, 'value': 42}])

    result = ds.flatten(['id'], ['value'])

    assert [(r.id, r.key, r.val) for r in result] == [(1, 'value', 42)]


def test_flatten_with_date_in_flattened_column():
    """Verify a Date cell reaches val intact under the float column type.

    Mutation: val taking row[flattened[0]], so the end row carries the
        start date.
    Oracle: hand-computed (key, val) pairs against the source Dates.
    """
    ds = DataSet([
        {'id': 1, 'start': Date(2024, 1, 1), 'end': Date(2024, 12, 31)},
    ])
    ds.columns = [('id', int), ('start', Date), ('end', Date)]

    result = ds.flatten(['id'], ['start', 'end'])

    assert [(r.key, r.val) for r in result] == [
        ('start', Date(2024, 1, 1)),
        ('end', Date(2024, 12, 31)),
        ]


def test_flatten_with_datetime_in_flattened_column():
    """Verify a DateTime cell keeps its time-of-day in val.

    Mutation: reversing the flattened loop, which swaps the created and
        updated rows.
    Oracle: hand-computed (key, val) pairs against the source DateTimes.
    """
    ds = DataSet([
        {'id': 1,
         'created': DateTime(2024, 1, 1, 10, 0),
         'updated': DateTime(2024, 6, 15, 14, 30)},
    ])
    ds.columns = [('id', int), ('created', DateTime), ('updated', DateTime)]

    result = ds.flatten(['id'], ['created', 'updated'])

    assert [(r.key, r.val) for r in result] == [
        ('created', DateTime(2024, 1, 1, 10, 0)),
        ('updated', DateTime(2024, 6, 15, 14, 30)),
        ]


def test_flatten_duplicate_kept_column_values():
    """Verify adjacent rows sharing a kept value are not merged into one.

    Mutation: emitting the group key once per groupby group rather than
        once per row, which collapses the two A rows.
    Oracle: hand-computed (category, val) pairs in source order.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 200},
        {'category': 'B', 'value': 300},
    ])

    result = ds.flatten(['category'], ['value'])

    assert [(r.category, r.val) for r in result] == [
        ('A', 100),
        ('A', 200),
        ('B', 300),
        ]


def test_flatten_key_val_collision_with_kept():
    """Verify the generated key/val win when a kept column shares the name.

    Mutation: merging the kept values over the generated pair rather
        than under it, which resurrects 'original_key'.
    Oracle: hand-computed row for a source whose columns are key and val.
    """
    ds = DataSet([
        {'key': 'original_key', 'val': 'original_val', 'metric': 100},
    ])

    result = ds.flatten(['key', 'val'], ['metric'])

    assert result.cols == ['key', 'val']
    assert [(r.key, r.val) for r in result] == [('metric', 100)]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class _Sub(DataSet):
    """Subclass carrying a marker a plain DataSet cannot fake."""

    marker = 'sub'


def test_flatten_returns_the_receivers_class():
    """Verify flatten builds its result through self.__class__.

    Mutation: the result built as a bare `DataSet(columns=...)`, so a
    subclass loses its identity on every reverse-pivot.
    Oracle: the exact type, plus the row count a 1-row, 2-column
    flatten must produce.
    """
    ds = _Sub([{'k': 'a', 'x': 1.0, 'y': 2.0}])
    ds.columns = [('k', str), ('x', float), ('y', float)]

    flat = ds.flatten('k', ['x', 'y'])
    assert type(flat) is _Sub
    assert len(flat) == 2
