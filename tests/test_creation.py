"""Tests for DataSet creation and type conversion."""
import datetime
import io
import itertools
import json
import logging
import math
import pathlib
import tempfile
from decimal import Decimal

import pandas as pd
import pytest
from opendate import UTC, Date, DateTime, Time
from rollups import DataSet, guess_dataframe_dataset_columns

from libb import attrdict, lazydict

# --- basic creation ---


def test_creation_from_empty_list():
    """Verify creation from an empty list yields an empty DataSet.

    Mutation: dropping the `if rows else []` guard in guess_columns, so
        rows[exemplar] raises IndexError on a zero-row container.
    Oracle: hand-listed empty cols/typs for a container with no rows.
    """
    ds = DataSet([])
    assert len(ds) == 0
    assert ds.cols == []
    assert ds.typs == []


def test_creation_from_dicts():
    """Verify inference pairs each column with its own value's type.

    Mutation: guess_columns zipping cols with typs out of order (e.g.
        zip(sorted(cols), typs)), or typing every column from one value.
    Oracle: hand-typed name -> str, age -> int, score -> float, in the
        first row's own key order.
    """
    ds = DataSet([
        {'name': 'A', 'age': 30, 'score': 95.5},
        {'name': 'B', 'age': 25, 'score': 87.3}
    ])
    assert len(ds) == 2
    assert ds.cols == ['name', 'age', 'score']
    assert ds.colmap == {'name': str, 'age': int, 'score': float}


def test_creation_with_explicit_columns():
    """Verify declared column types are applied to the stored values.

    Mutation: dropping @ensure_types_converted from __getitem__, so the
        int 2 is handed back unconverted for a float column.
    Oracle: isinstance against float - 2 == 2.0 hides the defect.
    """
    ds = DataSet(
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}],
        columns=[('a', int), ('b', float)]
    )
    assert ds.colmap == {'a': int, 'b': float}
    assert isinstance(ds[0]['b'], float)
    assert ds[0]['b'] == 2.0


def test_creation_with_cols_and_typs():
    """Verify cols/typs drive conversion when `columns` is not given.

    Mutation: the constructor calling guess_columns even when cols and
        typs are supplied, typing both columns str.
    Oracle: hand-converted 1 and 2.5 from the strings handed in.
    """
    ds = DataSet(
        [{'x': '1', 'y': '2.5'}],
        cols=['x', 'y'],
        typs=[int, float]
    )
    assert ds[0]['x'] == 1
    assert ds[0]['y'] == 2.5
    assert isinstance(ds[0]['x'], int)
    assert isinstance(ds[0]['y'], float)


def test_creation_from_attrdicts():
    """Verify rows are stored as the attribute-access dict the container
    uses, and one already of that type is kept by reference.

    Mutation: the constructor storing dict(row), so attribute access
        raises, or rebuilding a row that is already a lazydict.
    Oracle: attribute reads on each row, and identity of a lazydict row
        handed to the constructor.
    """
    rows = [attrdict(name='A', age=30), attrdict(name='B', age=25)]
    ds = DataSet(rows)

    assert len(ds) == 2
    assert ds[0].name == 'A'
    assert ds[1].age == 25

    native = lazydict(name='C', age=40)

    assert DataSet([native]).container[0] is native


def test_row_callable_is_a_computed_column():
    """Verify a callable row value resolves on attribute access only.

    A row is a `libb.lazydict`, so storing a callable gives the row a
    computed column. Subscript and `.get` hand back the callable
    itself, which is what lets conversion and the writers see it.

    Mutation: building rows as `libb.attrdict` rather than
        `libb.lazydict`, so the callable is returned instead of called.
    Oracle: hand-computed 1 + 2 against the three access styles.
    """
    ds = DataSet([{'a': 1, 'b': 2, 'total': lambda row: row.a + row.b}])
    row = ds[0]

    assert row.total == 3
    assert callable(row['total'])
    assert callable(row.get('total'))

    row['a'] = 10

    assert row.total == 12


@pytest.mark.parametrize('declared', [str, float, int, Date])
def test_row_callable_survives_type_conversion(declared):
    """Verify conversion leaves a computed column callable, whatever
    type the column declares.

    Mutation: dropping the `callable(val)` guard from `_convert_value`
        or from the per-row loop. `str` is the one target type whose
        constructor accepts anything, so that column alone would freeze
        `str(function)` - the repr and address - into the data, while
        every other declared type happened to fail and leave it be.
    Oracle: the callable is still callable after conversion, and still
        computes 1 + 2.
    """
    ds = DataSet([{'a': 1, 'b': 2, 'total': lambda row: row.a + row.b}])
    ds.columns = [('a', int), ('b', int), ('total', declared)]
    ds.ensure_types()

    assert callable(ds.container[0]['total'])
    assert ds[0].total == 3


def test_row_callable_survives_a_validated_append():
    """Verify the validate=True path also leaves a computed column be.

    That path calls `_convert_value` directly rather than going through
    the per-row loop, so it needs the same guard.

    Mutation: guarding only the per-row loop and not `_convert_value`.
    Oracle: hand-computed 4 + 5 read off the appended row.
    """
    ds = DataSet([], columns=[('a', int), ('b', int), ('total', str)])
    ds.append({'a': 4, 'b': 5, 'total': lambda row: row.a + row.b},
              validate=True)

    assert ds[0].total == 9


def test_creation_from_dataset():
    """Verify DataSet(other) copies the row list but shares the rows.

    Mutation: `self.container = container.container` in place of
        list(container.container), so appending grows the original too.
    Oracle: the original's length and row identity after the copy is
        appended to.
    """
    original = DataSet([{'a': 1, 'b': 2}])
    copied = DataSet(original)

    assert copied.cols == original.cols
    assert copied[0] is original[0]

    copied.append(attrdict({'a': 3, 'b': 4}))

    assert len(copied) == 2
    assert len(original) == 1


def test_creation_preserves_column_order():
    """Verify columns come from the exemplar row, in its own key order.

    Mutation: guess_columns reading sorted(keys), or the union of every
        row's keys, instead of the exemplar row's keys.
    Oracle: a first row keyed out of alphabetical order, and a first row
        whose keys are a strict subset of a later row's.
    """
    ds = DataSet([
        {'c': 3, 'a': 1, 'b': 2},
        {'c': 6, 'a': 4, 'b': 5}
    ])
    assert ds.cols == ['c', 'a', 'b']

    sparse = DataSet([{'c': 3}, {'a': 1, 'b': 2}])
    assert sparse.cols == ['c']


def test_creation_with_generator():
    """Verify a generator container is materialized before it is read.

    Mutation: the constructor indexing the container (container[:] or
        container[exemplar]) before listing it, which raises TypeError
        on a generator.
    Oracle: hand-computed y = 2i at the last row (8 at i = 4).
    """
    gen = ({'x': i, 'y': i * 2} for i in range(5))
    ds = DataSet(gen, columns=[('x', int), ('y', int)])

    assert len(ds) == 5
    assert ds[0]['x'] == 0
    assert ds[4]['y'] == 8


def test_creation_with_pagination_info():
    """Verify pagination fields survive creation and pages rounds up.

    Mutation: floor in place of math.ceil in pages, or `self.total =
        total` dropping the fallback to the row count.
    Oracle: hand-computed ceil(103 / 5) = 21, and a 10-row container
        with no total given.
    """
    ds = DataSet(
        [{'x': i} for i in range(10)],
        page=2,
        per_page=5,
        total=103
    )

    assert ds.page == 2
    assert ds.per_page == 5
    assert ds.total == 103
    assert ds.pages == 21

    untotaled = DataSet([{'x': i} for i in range(10)], per_page=5)

    assert untotaled.total == 10
    assert untotaled.pages == 2


def test_creation_with_mixed_types():
    """Verify a column holding int and float types float either way.

    Mutation: flipping the promotion in guess_columns to demote on
        `typ is float and row_typ is int`.
    Oracle: hand-typed float for both orderings of the same two values.
    """
    int_first = DataSet([
        {'num': 1, 'text': 'a'},
        {'num': 2.5, 'text': 'b'}
    ])
    float_first = DataSet([
        {'num': 2.5, 'text': 'a'},
        {'num': 1, 'text': 'b'}
    ])

    assert int_first.colmap['num'] == float
    assert float_first.colmap['num'] == float
    assert int_first.colmap['text'] == str


def test_creation_with_duplicate_column_names():
    """Verify a repeated column name keeps its last type and position.

    Mutation: dropping `del seen[col]` from the columns setter, so the
        repeat keeps the first position and reads ['a', 'b'].
    Oracle: hand-listed ['b', 'a'] with 'a' typed float.
    """
    ds = DataSet(
        [{'a': 1, 'b': 2}],
        columns=[('a', int), ('b', int), ('a', float)]
    )

    assert len(ds.cols) == 2
    assert ds.cols == ['b', 'a']
    assert ds.colmap['a'] == float
    assert ds.colmap['b'] == int
    assert ds.cols.count('a') == 1


def test_creation_with_conflicting_cols_and_columns():
    """Verify `columns` wins over cols/typs when both are given.

    Mutation: the constructor testing cols/typs before `columns`, so
        the values stay strings.
    Oracle: hand-converted 1 and 2.0 against the str types in typs.
    """
    ds = DataSet(
        [{'a': '1', 'b': '2'}],
        cols=['a', 'b'],
        typs=[str, str],
        columns=[('a', int), ('b', float)]
    )

    assert ds.colmap == {'a': int, 'b': float}
    assert ds[0]['a'] == 1
    assert isinstance(ds[0]['b'], float)


# --- factory methods ---


def test_creation_from_empty():
    """Verify from_empty gives one row of per-type empty values.

    Mutation: bool, Date or Time falling into the str or int branch, or
        the int branch returning None.
    Oracle: hand-listed defaults - '' for str, 0 for int, 0.0 for
        float, None for every other type.
    """
    columns = [
        ('name', str),
        ('age', int),
        ('score', float),
        ('active', bool),
        ('date', Date),
        ('time', Time),
        ]
    ds = DataSet.from_empty(columns)

    assert len(ds) == 1
    assert ds[0]['name'] == ''
    assert ds[0]['age'] == 0
    assert ds[0]['score'] == 0.0
    assert ds[0]['active'] is None
    assert ds[0]['date'] is None
    assert ds[0]['time'] is None


def test_creation_from_list():
    """Verify from_list keys each tuple by column, in order.

    Mutation: zip(row, cols) in place of zip(cols, row) in from_list,
        keying every row by its own values.
    Oracle: hand-built first row {'id': 1, 'name': 'A', 'score': 95.5}.
    """
    rows = [(1, 'A', 95.5), (2, 'B', 87.3)]
    cols = ['id', 'name', 'score']
    typs = [int, str, float]

    ds = DataSet.from_list(rows, cols, typs)

    assert len(ds) == 2
    assert dict(ds[0]) == {'id': 1, 'name': 'A', 'score': 95.5}
    assert dict(ds[1]) == {'id': 2, 'name': 'B', 'score': 87.3}
    assert ds.colmap == {'id': int, 'name': str, 'score': float}


def test_creation_from_list_with_type_conversion():
    """Verify from_list converts each value to its own column's type.

    Mutation: _convert_value numifying every value, so the str column
        holds 2 rather than '2'.
    Oracle: hand-converted 1, '2' and 3.5 from three identical-looking
        numeric strings.
    """
    rows = [('1', '2', '3.5')]
    cols = ['a', 'b', 'c']
    typs = [int, str, float]

    ds = DataSet.from_list(rows, cols, typs)

    assert ds[0]['a'] == 1
    assert ds[0]['b'] == '2'
    assert isinstance(ds[0]['b'], str)
    assert ds[0]['c'] == 3.5


def test_creation_from_list_empty_rows():
    """Verify from_list keeps the declared schema with no rows.

    Mutation: the constructor falling back to guess_columns when the
        container is empty (`if columns and self.container`), losing
        the declared cols and typs.
    Oracle: the cols and typs handed to from_list.
    """
    ds = DataSet.from_list([], ['a', 'b'], [int, str])

    assert len(ds) == 0
    assert ds.cols == ['a', 'b']
    assert ds.typs == [int, str]


# --- dataframe creation ---


def test_creation_from_dataframe():
    """Verify from_dataframe maps dtypes to types and keeps frame order.

    Mutation: guess_dataframe_dataset_columns sending 'integer' to
        float, or listing use_cols sorted rather than in frame order.
    Oracle: hand-typed dtype map and the frame's own first record.
    """
    df = pd.DataFrame({
        'name': ['A', 'B'],
        'age': [30, 25],
        'score': [95.5, 87.3]
    })

    ds = DataSet.from_dataframe(df)

    assert len(ds) == 2
    assert ds.cols == ['name', 'age', 'score']
    assert ds.colmap == {'name': str, 'age': int, 'score': float}
    assert dict(ds[0]) == {'name': 'A', 'age': 30, 'score': 95.5}
    assert isinstance(ds[1]['age'], int)


def test_creation_from_dataframe_with_dates():
    """Verify a datetime64 column types and converts to DateTime.

    Mutation: dataset_type_map sending 'datetime64' to Date, dropping
        the clock time from every value.
    Oracle: hand-listed timestamps in UTC, one per row.
    """
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'value': [1, 2]
    })

    ds = DataSet.from_dataframe(df)

    assert len(ds) == 2
    assert ds.colmap['date'] == DateTime
    assert ds[0]['date'] == DateTime(2024, 1, 1, tzinfo=UTC)
    assert ds[1]['date'] == DateTime(2024, 1, 2, tzinfo=UTC)


def test_creation_from_dataframe_with_nans():
    """Verify from_dataframe reads NaN as None and leaves the rest.

    Mutation: dropping the np.isnan sweep, so NaN survives into the
        row, or widening it to blank every value.
    Oracle: hand-listed [1.0, None, 3.0] and [1.5, 2.5, None].
    """
    df = pd.DataFrame({
        'a': [1, None, 3],
        'b': [1.5, 2.5, None]
    })

    ds = DataSet.from_dataframe(df)

    assert ds[0]['a'] == 1.0
    assert ds[1]['a'] is None
    assert ds[2]['a'] == 3.0
    assert ds[0]['b'] == 1.5
    assert ds[2]['b'] is None


def test_creation_from_dataframe_with_col_filter():
    """Verify `cols` drops the other columns from schema and rows.

    Mutation: inverting the `if col in cols` test, or dropping the
        `df = df[list(_columns)]` reslice so the rows keep the excluded
        column while the schema does not.
    Oracle: hand-listed kept columns and the first row's own keys.
    """
    df = pd.DataFrame({
        'a': [1, 2],
        'b': [3, 4],
        'c': [5, 6]
    })

    ds = DataSet.from_dataframe(df, cols=['a', 'c'])

    assert ds.cols == ['a', 'c']
    assert dict(ds[0]) == {'a': 1, 'c': 5}


def test_creation_from_dataframe_with_explicit_columns():
    """Verify `columns` overrides the type guessed from the dtype.

    Mutation: from_dataframe ignoring `columns` (or its `if col in
        _columns` guard), leaving the guessed float and int.
    Oracle: hand-converted 1 (int) from 1.0 and 3.0 (float) from 3.
    """
    df = pd.DataFrame({
        'a': [1.0, 2.0],
        'b': [3, 4]
    })

    ds = DataSet.from_dataframe(df, columns=[('a', int), ('b', float)])

    assert ds.colmap == {'a': int, 'b': float}
    assert isinstance(ds[0]['a'], int)
    assert ds[0]['b'] == 3.0
    assert isinstance(ds[0]['b'], float)


# --- json creation ---


def test_creation_from_json_raw():
    """Verify from_json(raw=True) returns one DataSet, dates parsed.

    Mutation: json.loads without cls=JSONDecoderISODate, leaving
        '2014-10-01' a string, or raw=True returning the (ds, other)
        pair.
    Oracle: hand-computed Date(2014, 10, 1) and 2.0.
    """
    json_data = '[{"x": 2.0, "d": "2014-10-01"}]'
    ds = DataSet.from_json(json_data, raw=True)

    assert len(ds) == 1
    assert ds[0]['x'] == 2.0
    assert ds.colmap['d'] == Date
    assert ds[0]['d'] == Date(2014, 10, 1)


def test_creation_from_json_with_types():
    """Verify a declared `types` list wins over the parsed type.

    Mutation: from_json passing typs=None, so x keeps the parsed float,
        or the handle_date map dropped, typing 'd' as None.
    Oracle: isinstance int on 2.0, which compares equal either way.
    """
    json_data = '{"data": [{"x": 2.0, "d": "2014-10-01"}], "types": ["int", "date"]}'
    ds, other = DataSet.from_json(json_data, raw=False)

    assert len(ds) == 1
    assert ds.colmap['x'] == int
    assert ds.colmap['d'] == datetime.date
    assert ds[0]['x'] == 2
    assert isinstance(ds[0]['x'], int)
    assert ds[0]['d'] == Date(2014, 10, 1)


def test_creation_from_json_with_additional_metadata():
    """Verify from_json splits data/order/types from the other keys.

    Mutation: narrowing the exclusion set to {'data'}, so `other`
        carries order and types, or `order` ignored for column order.
    Oracle: hand-listed remaining keys, and cols read off `order`.
    """
    json_data = ('{"data": [{"x": 1, "y": 2}], "order": ["y", "x"], '
                 '"types": ["int", "int"], "metadata": "test", "count": 42}')
    ds, other = DataSet.from_json(json_data, raw=False)

    assert len(ds) == 1
    assert ds.cols == ['y', 'x']
    assert other == {'metadata': 'test', 'count': 42}


def test_creation_from_malformed_json():
    """Verify truncated json raises rather than yielding partial rows.

    Mutation: from_json guarding json.loads with a try/except that
        returns an empty DataSet.
    Oracle: the decoder's own JSONDecodeError on unterminated input.
    """
    malformed_json = '{"data": [{"x": 1}'
    with pytest.raises(json.JSONDecodeError):
        DataSet.from_json(malformed_json, raw=True)


# --- csv creation ---


def test_creation_from_csv_basic():
    """Verify read types columns from header suffixes and parses rows.

    Mutation: the TYPES suffix map rebound (':i' to str), or _parse
        returning the raw string for a numeric column.
    Oracle: hand-typed header map and hand-parsed rows.
    """
    csv_data = 'name:s,age:i,score:f\nA,30,95.5\nB,25,87.3\n'
    ds = DataSet.read(io.StringIO(csv_data))

    assert len(ds) == 2
    assert ds.cols == ['name', 'age', 'score']
    assert ds.colmap == {'name': str, 'age': int, 'score': float}
    assert dict(ds[0]) == {'name': 'A', 'age': 30, 'score': 95.5}
    assert dict(ds[1]) == {'name': 'B', 'age': 25, 'score': 87.3}


@pytest.mark.parametrize(
    ('type_suffix', 'expected_type', 'test_value', 'expected_value'),
    [
        (':s', str, 'hello', 'hello'),
        (':i', int, '42', 42),
        (':f', float, '3.14', 3.14),  # noqa
        (':d', Date, '2024-01-01', Date(2024, 1, 1)),
        (':b', bool, 'true', True),
    ])
def test_creation_from_csv_type_suffixes(
    type_suffix,
    expected_type,
    test_value,
    expected_value):
    """Verify each header type suffix picks its own parser.

    Mutation: a TYPES entry rebound to another type, or _parse's date
        branch slicing s[:8] rather than s[:10].
    Oracle: hand-parsed value per suffix, typed and compared.
    """
    csv_data = f'col{type_suffix}\n{test_value}\n'
    ds = DataSet.read(io.StringIO(csv_data))

    assert ds.colmap['col'] == expected_type
    assert ds[0]['col'] == expected_value


def test_creation_from_csv_with_booleans():
    """Verify a bool column reads both word forms of true and false.

    Mutation: dropping 'yes'/'no' from _parse's truthy/falsy sets, so
        an unrecognized word falls through to False.
    Oracle: hand-listed True/False per row, compared by identity.
    """
    csv_data = 'active:b,count:i\ntrue,1\nfalse,2\nyes,3\nno,4\n'
    ds = DataSet.read(io.StringIO(csv_data))

    assert ds.colmap['active'] == bool
    assert ds[0]['active'] is True
    assert ds[1]['active'] is False
    assert ds[2]['active'] is True
    assert ds[3]['active'] is False


def test_creation_from_csv_with_empty_values():
    """Verify an empty csv field reads as None, whatever the type.

    Mutation: dropping `if s == '': return None` from _parse, so a str
        column holds '' rather than None.
    Oracle: hand-listed None across the blank row, with the rows either
        side still parsed.
    """
    csv_data = 'a:i,b:f,c:s\n1,2.5,foo\n,,\n3,4.5,bar\n'
    ds = DataSet.read(io.StringIO(csv_data))

    assert ds[0]['c'] == 'foo'
    assert ds[1]['a'] is None
    assert ds[1]['b'] is None
    assert ds[1]['c'] is None
    assert ds[2]['a'] == 3


def test_creation_from_csv_skip_rows():
    """Verify `skips` drops that many lines before the header.

    Mutation: an off-by-one in `range(skips)`, so the header is read
        from the wrong line.
    Oracle: hand-counted two junk lines above the header.
    """
    csv_data = 'Header Line\nAnother Header\nname:s,value:i\nA,1\nB,2\n'
    ds = DataSet.read(io.StringIO(csv_data), skips=2)

    assert len(ds) == 2
    assert ds.cols == ['name', 'value']
    assert ds.colmap == {'name': str, 'value': int}
    assert ds[0]['name'] == 'A'
    assert ds[1]['value'] == 2


def test_creation_from_csv_leaves_the_header_alone_by_default():
    """Verify read renames no header of its own accord.

    Mutation: read rewriting header names by a built-in prefix rule
        instead of leaving that to the caller.
    Oracle: the header line as written, read back verbatim, including
        the name that a prefix rule would have caught.
    """
    csv_data = 'label,amount,amount total,cost\nx,1,2,3\n'
    ds = DataSet.read(io.StringIO(csv_data))

    assert ds.cols == ['label', 'amount', 'amount total', 'cost']


def test_creation_from_csv_rename_fields_separates_repeated_headers():
    """Verify `rename_fields` recovers both halves of a repeated header.

    Mutation: never calling the hook, or calling it once per data row
        instead of once on the header, either of which lets the repeat
        collapse onto the last value.
    Oracle: hand-numbered names, and the two values 1 and 2 that a
        collapsed column could not both hold.
    """
    csv_data = 'label,amount,amount\nx,1,2\n'
    counter = itertools.count(1)
    ds = DataSet.read(
        io.StringIO(csv_data),
        rename_fields=lambda fields: [
            f'amount{next(counter)}' if f == 'amount' else f
            for f in fields])

    assert ds.cols == ['label', 'amount1', 'amount2']
    assert ds[0]['amount1'] == '1'
    assert ds[0]['amount2'] == '2'


def test_creation_from_csv_auto_skip_empty_rows():
    """Verify a blank csv line is skipped rather than read as a row.

    Mutation: dropping `if not row: continue`, so each blank line
        becomes a row with no keys.
    Oracle: hand-counted two data rows among four lines.
    """
    csv_data = 'name:s,value:i\nA,1\n\nB,2\n\n'
    ds = DataSet.read(io.StringIO(csv_data))

    assert len(ds) == 2
    assert ds[0]['name'] == 'A'
    assert ds[1]['name'] == 'B'


def test_creation_from_csv_file():
    """Verify read opens a path and records it on the dataset.

    Mutation: inverting the `isinstance(file_or_name, str)` test, or
        never setting ds.filename.
    Oracle: the temp file's own path and the two rows written to it.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('name:s,age:i\nA,30\nB,25\n')
        filename = f.name

    try:
        ds = DataSet.read(filename)
        assert len(ds) == 2
        assert ds[0]['name'] == 'A'
        assert ds[1]['age'] == 25
        assert ds.filename == filename
    finally:
        pathlib.Path(filename).unlink()


def test_creation_from_csv_with_percentage():
    """Verify a float column reads percent and accounting-negative forms.

    Mutation: _parse's float branch calling float(s) directly instead
        of libb.parse, or scaling a percentage by 1/100.
    Oracle: hand-parsed 5.5 (not 0.055), 10.0, and -3.5 from '(3.5)'.
    """
    csv_data = 'rate:f\n5.5%\n10%\n(3.5)\n'
    ds = DataSet.read(io.StringIO(csv_data))

    assert ds[0]['rate'] == 5.5
    assert ds[1]['rate'] == 10.0
    assert ds[2]['rate'] == -3.5


def test_creation_from_csv_with_unicode():
    """Verify non-ASCII csv text survives reading intact.

    Mutation: _parse's str branch coercing to ASCII, or
        _csv_reader_wrapper swallowing a row it cannot decode.
    Oracle: the exact code points written into the buffer, and the
        neighboring int field proving the row was kept.
    """
    csv_data = ('name:s,value:i\n'
                '\u65e5\u672c\u8a9e,100\n'
                'Emoji\U0001f600,200\n')
    ds = DataSet.read(io.StringIO(csv_data))

    assert len(ds) == 2
    assert ds[0]['name'] == '\u65e5\u672c\u8a9e'
    assert ds[1]['name'] == 'Emoji\U0001f600'
    assert ds[1]['value'] == 200


# --- type inference ---


def test_creation_type_inference_date_vs_datetime():
    """Verify inference keeps Date, DateTime and Time distinct.

    Mutation: smart_type testing isinstance(val, datetime.date) rather
        than val.__class__, which types a DateTime column as Date.
    Oracle: hand-typed one column per class.
    """
    ds = DataSet([
        {'date': Date(2024, 1, 1),
         'datetime': DateTime(2024, 1, 1, 12, 30, 0),
         'time': Time(10, 30, 0)},
        {'date': Date(2024, 1, 2),
         'datetime': DateTime(2024, 1, 2, 13, 45, 30),
         'time': Time(14, 45, 30)}
    ])

    assert ds.colmap == {'date': Date, 'datetime': DateTime, 'time': Time}


def test_creation_type_inference_with_none():
    """Verify inference scans past None to the first typed value.

    Mutation: guess_columns stopping at the exemplar row (scan_end =
        scan_start + 1), typing a leading-None column NoneType.
    Oracle: hand-typed int from the first non-None value per column,
        against a column that is None throughout.
    """
    ds = DataSet([
        {'a': None, 'b': 1, 'c': None},
        {'a': 2, 'b': None, 'c': None}
    ])

    assert ds.colmap['a'] == int
    assert ds.colmap['b'] == int
    assert ds.colmap['c'] is type(None)


def test_creation_with_exemplar_parameter():
    """Verify `exemplar` picks the row that inference starts from.

    Mutation: guess_columns starting its scan at row 0 regardless of
        `exemplar`, typing both columns str.
    Oracle: hand-typed row 1 (int, float) against row 0's strings.
    """
    ds = DataSet(
        [
            {'a': 'wrong', 'b': 'types'},
            {'a': 123, 'b': 45.6}
        ],
        exemplar=1
    )

    assert ds.colmap == {'a': int, 'b': float}
    assert ds[1]['a'] == 123


# --- type conversion ---


def test_creation_type_conversion_string_to_number():
    """Verify numeric strings convert through libb.numify.

    Mutation: _convert_value calling typ(val) in place of libb.numify,
        so the grouped digits fail to parse and the string survives.
    Oracle: hand-computed 1234 from '1,234' and 45.6 from '45.6'.
    """
    ds = DataSet(
        [{'a': '1,234', 'b': '45.6'}],
        columns=[('a', int), ('b', float)]
    )

    assert ds[0]['a'] == 1234
    assert ds[0]['b'] == 45.6
    assert isinstance(ds[0]['a'], int)
    assert isinstance(ds[0]['b'], float)


def test_creation_type_conversion_preserves_none():
    """Verify None survives conversion, lazily and on validated append.

    Mutation: dropping _convert_value's None guard, so str(None) writes
        the string 'None' into a str column on the append path.
    Oracle: identity against None for an int and a str column.
    """
    ds = DataSet(
        [{'a': None, 'b': None}],
        columns=[('a', int), ('b', str)]
    )

    assert ds[0]['a'] is None
    assert ds[0]['b'] is None

    ds.append(attrdict({'a': None, 'b': None}), validate=True)

    assert ds[1]['a'] is None
    assert ds[1]['b'] is None


def test_creation_with_check_types_false():
    """Verify check_types=False leaves the values as handed in.

    Mutation: _ensure_types_converted dropping its self._check_types
        term, converting on first read anyway.
    Oracle: the raw strings handed to the constructor.
    """
    ds = DataSet(
        [{'a': '123', 'b': '45.6'}],
        columns=[('a', int), ('b', float)],
        check_types=False
    )

    assert ds[0]['a'] == '123'
    assert ds[0]['b'] == '45.6'
    assert isinstance(ds[0]['a'], str)


def test_creation_handles_date_string_conversion():
    """Verify Date and DateTime columns parse their own string form.

    Mutation: _convert_value routing a DateTime column through
        _cached_date_parse (or a Date column through the datetime
        parser), losing the clock time or the class.
    Oracle: hand-parsed 2024-01-01 and 2024-01-01 12:30 UTC.
    """
    ds = DataSet(
        [{'date': '2024-01-01', 'datetime': '2024-01-01 12:30:00'}],
        columns=[('date', Date), ('datetime', DateTime)]
    )

    assert isinstance(ds[0]['date'], Date)
    assert isinstance(ds[0]['datetime'], DateTime)
    assert ds[0]['date'] == Date(2024, 1, 1)
    assert ds[0]['datetime'] == DateTime(2024, 1, 1, 12, 30, 0, tzinfo=UTC)


# --- _types_converted flag ---


def test_types_converted_flag_starts_false():
    """Verify conversion is deferred, not run in the constructor.

    Mutation: the constructor calling convert_container_types eagerly.
    Oracle: the flag plus the raw string still sitting in the container.
    """
    ds = DataSet(
        [{'a': '123', 'b': '45.6'}],
        columns=[('a', int), ('b', float)]
    )

    assert ds._types_converted is False
    assert ds.container[0]['a'] == '123'


@pytest.mark.parametrize('trigger_method', [
    'index_access',
    'iteration',
    'unwind',
])
def test_types_converted_flag_triggers(trigger_method):
    """Verify each read path converts types before handing rows back.

    Mutation: dropping @ensure_types_converted from __getitem__ or
        __iter__ (unwind reads through __iter__).
    Oracle: the flag and the converted int, read straight off the
        container after each access path.
    """
    ds = DataSet(
        [{'a': '123'}, {'a': '456'}],
        columns=[('a', int)]
    )
    assert ds._types_converted is False

    if trigger_method == 'index_access':
        _ = ds[0]
    elif trigger_method == 'iteration':
        for _ in ds:
            pass
    elif trigger_method == 'unwind':
        list(ds.unwind('a'))

    assert ds._types_converted is True
    assert ds.container[0]['a'] == 123


def test_types_converted_with_check_types_false():
    """Verify check_types=False defers conversion to an explicit call.

    Mutation: _ensure_types_converted ignoring _check_types, or
        convert_container_types not setting the flag.
    Oracle: the raw string before the call, hand-converted 123 after.
    """
    ds = DataSet(
        [{'a': '123', 'b': '45.6'}],
        columns=[('a', int), ('b', float)],
        check_types=False
    )

    assert ds._types_converted is False
    _ = ds[0]
    assert ds._types_converted is False
    assert ds[0]['a'] == '123'

    ds.convert_container_types()

    assert ds._types_converted is True
    assert ds[0]['a'] == 123
    assert isinstance(ds[0]['a'], int)


# --- missing and sparse columns ---


def test_creation_with_missing_columns_in_some_rows():
    """Verify a declared column missing from a row fills with None.

    Mutation: dropping the `if name not in row: row[name] = None` fill,
        so the conversion loop raises KeyError on the next line.
    Oracle: hand-filled rows, compared whole so a stray key also fails.
    """
    ds = DataSet(
        [
            {'id': 1, 'name': 'A', 'value': 100},
            {'id': 2, 'name': 'B'},
            {'id': 3, 'value': 150},
        ],
        columns=[('id', int), ('name', str), ('value', int)]
    )

    assert len(ds) == 3
    assert dict(ds[0]) == {'id': 1, 'name': 'A', 'value': 100}
    assert dict(ds[1]) == {'id': 2, 'name': 'B', 'value': None}
    assert dict(ds[2]) == {'id': 3, 'name': None, 'value': 150}


def test_creation_with_column_missing_from_all_rows():
    """Verify a column absent from every row still appears, as None.

    Mutation: the fill loop walking the row's own keys rather than
        self.columns, so a column no row carries never appears.
    Oracle: hand-listed schema against rows that carry two of its four
        columns.
    """
    ds = DataSet(
        [
            {'id': 1, 'name': 'A'},
            {'id': 2, 'name': 'B'},
            {'id': 3, 'name': 'C'},
        ],
        columns=[('id', int), ('name', str), ('value', float), ('status', str)]
    )

    assert len(ds) == 3
    assert ds.cols == ['id', 'name', 'value', 'status']
    for row in ds:
        assert row['value'] is None
        assert row['status'] is None


def test_creation_with_sparse_data():
    """Verify rows with disjoint keys each fill the declared columns.

    Mutation: the fill writing None over a value already present
        (`row[name] = None` outside the `if name not in row` guard).
    Oracle: hand-filled rows for every subset of the three columns.
    """
    ds = DataSet(
        [
            {'a': 1},
            {'b': 2},
            {'c': 3},
            {'a': 4, 'b': 5},
            {'b': 6, 'c': 7},
            {'a': 8, 'c': 9},
        ],
        columns=[('a', int), ('b', int), ('c', int)]
    )

    assert len(ds) == 6
    assert dict(ds[0]) == {'a': 1, 'b': None, 'c': None}
    assert dict(ds[1]) == {'a': None, 'b': 2, 'c': None}
    assert dict(ds[2]) == {'a': None, 'b': None, 'c': 3}
    assert dict(ds[3]) == {'a': 4, 'b': 5, 'c': None}
    assert dict(ds[4]) == {'a': None, 'b': 6, 'c': 7}
    assert dict(ds[5]) == {'a': 8, 'b': None, 'c': 9}


def test_convert_container_types_fills_missing_columns():
    """Verify an explicit convert_container_types fills and converts.

    Mutation: the fill loop lifted out of the per-row loop, so only the
        first row gains its missing column.
    Oracle: hand-filled rows with each string converted to int.
    """
    ds = DataSet(
        [
            {'id': '1', 'value': '100'},
            {'id': '2'},
            {'value': '300'},
        ],
        columns=[('id', int), ('value', int)]
    )

    ds.convert_container_types()

    assert dict(ds[0]) == {'id': 1, 'value': 100}
    assert dict(ds[1]) == {'id': 2, 'value': None}
    assert dict(ds[2]) == {'id': None, 'value': 300}


def test_creation_missing_columns_preserves_type_conversion():
    """Verify a filled None does not stop the other values converting.

    Mutation: the fill running after the conversion loop, so a filled
        column is typed but its neighbors on that row are not, or the
        `if val is None: continue` guard removed so Date/int parsing
        raises on the fill.
    Oracle: hand-converted values per row beside each filled None.
    """
    ds = DataSet(
        [
            {'id': '1', 'name': 'A', 'score': '95.5'},
            {'id': '2', 'score': '87.3'},
            {'name': 'C', 'score': '92.1'},
        ],
        columns=[('id', int), ('name', str), ('score', float)]
    )

    assert dict(ds[0]) == {'id': 1, 'name': 'A', 'score': 95.5}
    assert dict(ds[1]) == {'id': 2, 'name': None, 'score': 87.3}
    assert dict(ds[2]) == {'id': None, 'name': 'C', 'score': 92.1}


def test_creation_missing_columns_with_dates():
    """Verify a Date column converts around its own missing values.

    Mutation: _convert_value routing a Date column through the datetime
        parser, so the value is a DateTime, or the fill writing '' and
        the date parser raising on it.
    Oracle: hand-parsed dates either side of a row with none.
    """
    ds = DataSet(
        [
            {'id': 1, 'date': '2024-01-01', 'value': 100},
            {'id': 2, 'value': 200},
            {'id': 3, 'date': '2024-01-03'},
        ],
        columns=[('id', int), ('date', Date), ('value', int)]
    )

    assert ds[0]['date'] == Date(2024, 1, 1)
    assert isinstance(ds[0]['date'], Date)
    assert ds[1]['date'] is None
    assert ds[2]['date'] == Date(2024, 1, 3)
    assert ds[0]['value'] == 100
    assert ds[1]['value'] == 200
    assert ds[2]['value'] is None


def test_creation_missing_columns_with_operations():
    """Verify filter_data and add_column read the filled columns.

    Mutation: filter_data keeping the rows its predicate rejects, or
        add_column writing one row's value to every row.
    Oracle: hand-counted two rows with a non-None id, and 100 * 2.
    """
    ds = DataSet(
        [
            {'id': 1, 'value': 100},
            {'id': 2},
            {'value': 300},
        ],
        columns=[('id', int), ('value', int)]
    )

    _ = ds[0]

    ds.filter_data(lambda r: r['id'] is not None)
    assert len(ds) == 2

    ds.add_column('doubled', int, value=lambda r: r['value'] * 2 if r['value'] else 0)
    assert ds[0]['doubled'] == 200
    assert ds[1]['doubled'] == 0


def test_creation_empty_rows_with_columns():
    """Verify a row with no keys at all still gains every column.

    Mutation: the fill loop skipping a row that is empty (`if not row:
        continue`), leaving it without the declared columns.
    Oracle: key presence in all three rows, against the one value that
        was supplied.
    """
    ds = DataSet(
        [
            {},
            {'a': 1},
            {},
        ],
        columns=[('a', int), ('b', str), ('c', float)]
    )

    assert len(ds) == 3
    for row in ds:
        assert 'a' in row
        assert 'b' in row
        assert 'c' in row
    assert ds[0]['a'] is None
    assert ds[1]['a'] == 1
    assert ds[2]['a'] is None


# --- stdlib to opendate conversion ---


@pytest.mark.parametrize(
    ('stdlib_type', 'custom_type', 'create_value', 'expected_custom_type'),
    [
        (datetime.date, Date, lambda: datetime.date(2024, 1, 15), Date),
        (
            datetime.datetime,
            DateTime,
            lambda: datetime.datetime(2024, 1, 15, 10, 30, 0),
            DateTime),
        (datetime.time, Time, lambda: datetime.time(9, 30, 0), Time),
    ])
def test_convert_stdlib_to_custom_type(
    stdlib_type,
    custom_type,
    create_value,
    expected_custom_type):
    """Verify a stdlib-typed column is upgraded to the opendate class.

    Mutation: dropping the date/datetime/time fast paths from
        convert_container_types, so `isinstance(val, typ)` short-
        circuits a stdlib-typed column and the stdlib object survives.
    Oracle: the class of the stored value before and after the call.
    """
    ds = DataSet(
        [
            {'id': 1, 'value': create_value()},
            {'id': 2, 'value': create_value()},
        ],
        columns=[('id', int), ('value', stdlib_type)]
    )

    assert isinstance(ds.container[0]['value'], stdlib_type)
    ds.convert_container_types()
    assert isinstance(ds.container[0]['value'], expected_custom_type)


def test_convert_time_from_datetime_time():
    """Verify a Time column keeps the clock reading of a stdlib time.

    Mutation: the Time fast path writing DateTime.instance(val), copied
        from the branch above it.
    Oracle: hand-listed h:m:s in UTC, including the 23:59:59 boundary.
    """
    ds = DataSet(
        [
            {'id': 1, 'start_time': datetime.time(9, 30, 0)},
            {'id': 2, 'start_time': datetime.time(14, 15, 30)},
            {'id': 3, 'start_time': datetime.time(23, 59, 59)},
        ],
        columns=[('id', int), ('start_time', Time)]
    )

    assert isinstance(ds[0]['start_time'], Time)
    assert isinstance(ds[1]['start_time'], Time)
    assert isinstance(ds[2]['start_time'], Time)
    assert ds[0]['start_time'] == Time(9, 30, 0, tzinfo=UTC)
    assert ds[1]['start_time'] == Time(14, 15, 30, tzinfo=UTC)
    assert ds[2]['start_time'] == Time(23, 59, 59, tzinfo=UTC)


def test_convert_time_from_datetime_datetime():
    """Verify a Time column drops the date part of a datetime.

    Mutation: dispatching on the value rather than the declared type
        (`isinstance(val, datetime.datetime) -> DateTime.instance`), so
        a Time column lands a DateTime.
    Oracle: hand-computed 09:30 and 14:15:30 UTC from timestamps on two
        different days.
    """
    ds = DataSet(
        [
            {'id': 1, 'timestamp': datetime.datetime(2024, 1, 1, 9, 30, 0)},
            {'id': 2, 'timestamp': datetime.datetime(2024, 1, 2, 14, 15, 30)},
        ],
        columns=[('id', int), ('timestamp', Time)]
    )

    assert isinstance(ds[0]['timestamp'], Time)
    assert isinstance(ds[1]['timestamp'], Time)
    assert ds[0]['timestamp'] == Time(9, 30, 0, tzinfo=UTC)
    assert ds[1]['timestamp'] == Time(14, 15, 30, tzinfo=UTC)


def test_convert_mixed_stdlib_and_custom_date_types():
    """Verify a column of stdlib and opendate dates converges on Date.

    Mutation: the Date fast path writing DateTime.instance(val), copied
        from the DateTime branch, leaving two of three rows non-Date.
    Oracle: hand-listed dates, one per row, across a column that
        arrived half stdlib and half opendate.
    """
    ds = DataSet(
        [
            {'id': 1, 'date': datetime.date(2024, 1, 15)},
            {'id': 2, 'date': Date(2024, 1, 16)},
            {'id': 3, 'date': datetime.date(2024, 1, 17)},
        ],
        columns=[('id', int), ('date', Date)]
    )

    ds.convert_container_types()

    assert all(isinstance(row['date'], Date) for row in ds.container)
    assert ds.container[0]['date'] == Date(2024, 1, 15)
    assert ds.container[1]['date'] == Date(2024, 1, 16)
    assert ds.container[2]['date'] == Date(2024, 1, 17)


def test_convert_datetime_from_string_vs_stdlib_object():
    """Verify a DateTime column takes both a string and a stdlib value.

    Mutation: _convert_value's string branch calling _cached_date_parse,
        so the parsed row loses its clock time.
    Oracle: hand-parsed 10:30 UTC from text against 14:45 UTC from the
        stdlib object.
    """
    ds = DataSet(
        [
            {'id': 1, 'timestamp': '2024-01-15 10:30:00'},
            {'id': 2, 'timestamp': datetime.datetime(2024, 1, 16, 14, 45, 0)},
        ],
        columns=[('id', int), ('timestamp', DateTime)]
    )

    ds.convert_container_types()

    assert isinstance(ds.container[0]['timestamp'], DateTime)
    assert isinstance(ds.container[1]['timestamp'], DateTime)
    assert ds.container[0]['timestamp'] == DateTime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
    assert ds.container[1]['timestamp'] == DateTime(2024, 1, 16, 14, 45, 0, tzinfo=UTC)


def test_convert_datetime_from_stdlib_with_business_method_call():
    """Verify a converted Date carries the opendate business calendar.

    Mutation: the Date fast path writing DateTime.instance(val), or the
        stdlib date left alone so .business() raises AttributeError.
    Oracle: hand-computed previous business day - Monday 2024-03-11 ->
        Friday 2024-03-08, Tuesday 2024-03-12 -> Monday 2024-03-11.
    """
    ds = DataSet(
        [
            {'id': 1, 'as_of_date': datetime.date(2024, 3, 11)},
            {'id': 2, 'as_of_date': datetime.date(2024, 3, 12)},
        ],
        columns=[('id', int), ('as_of_date', Date)]
    )

    ds.convert_container_types()

    prev_date_1 = ds.container[0]['as_of_date'].business().subtract(days=1)
    prev_date_2 = ds.container[1]['as_of_date'].business().subtract(days=1)

    assert isinstance(prev_date_1, Date)
    assert prev_date_1 == Date(2024, 3, 8)
    assert prev_date_2 == Date(2024, 3, 11)


# --- edge cases ---


def test_creation_with_circular_reference_in_row():
    """Verify a self-referencing row is stored as is, not walked.

    Mutation: the constructor rebuilding rows (lazydict(row) on every
        row, or a deepcopy), which breaks the self-reference or the
        row's identity.
    Oracle: identity of the row object and of the cycle it holds.
    """
    row = lazydict(a=1)
    row['self'] = row

    ds = DataSet([row])

    assert len(ds) == 1
    assert ds[0]['a'] == 1
    assert ds.container[0] is row
    assert ds[0]['self'] is ds[0]


# --- array conversion ---


def test_to_array_selects_and_orders_columns():
    """Verify to_array takes the named columns in the order given.

    Mutation: ignoring the columns argument and using self.cols.
    Oracle: hand-computed - requesting ['b', 'a'] must invert the pairs
        the full-width call produces.
    """
    ds = DataSet([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])

    assert ds.to_array().tolist() == [[1, 2], [3, 4]]
    assert ds.to_array(columns=['b', 'a']).tolist() == [[2, 1], [4, 3]]


def test_to_array_numpy_type_overrides_column_type():
    """Verify numpy_type wins over the first column's declared type.

    Mutation: dropping the numpy_type argument and always deriving the
        dtype from the first column.
    Oracle: an int column asked for float must come back float64.
    """
    ds = DataSet([{'a': 1}, {'a': 2}])

    assert ds.to_array().dtype.kind == 'i'
    assert ds.to_array(numpy_type=float).dtype.kind == 'f'


# --- csv header and value parsing ---


def test_csv_bool_reads_short_and_uppercase_truthy_forms():
    """Verify 1, t, T, TRUE and Yes all read True in a bool column.

    Mutation: dropping '1' or 't' from _parse's truthy set, or matching
        it case-sensitively, so the value falls through to the
        unrecognized-value branch and reads False.
    Oracle: hand-listed True for all five spellings.
    """
    ds = DataSet.read(io.StringIO('flag:b\n1\nt\nT\nTRUE\nYes\n'))

    assert len(ds) == 5
    assert all(row['flag'] is True for row in ds)


def test_csv_bool_falsy_forms_match_rather_than_default(caplog):
    """Verify 0, f, F, FALSE and No are matched, not defaulted to False.

    Mutation: dropping '0', 'f', 'false' or 'no' from _parse's falsy
        set - the value still reads False, but only by falling through
        to the unrecognized-value branch.
    Oracle: an empty warning log proves the falsy branch ran; the
        hand-listed False values prove the result.
    """
    with caplog.at_level(logging.WARNING, logger='rollups.io'):
        ds = DataSet.read(io.StringIO('flag:b\n0\nf\nF\nFALSE\nNo\n'))

    warnings = [rec.getMessage() for rec in caplog.records
                if rec.name == 'rollups.io'
                and rec.levelno >= logging.WARNING]

    assert len(ds) == 5
    assert all(row['flag'] is False for row in ds)
    assert warnings == []


def test_csv_unrecognized_bool_warns_and_reads_false(caplog):
    """Verify an unknown bool word reads False and is named in the log.

    Mutation: the fall-through returning True rather than False, or the
        warning losing the value that provoked it.
    Oracle: hand-computed False, against the literal warning text
        naming 'maybe'.
    """
    with caplog.at_level(logging.WARNING, logger='rollups.io'):
        ds = DataSet.read(io.StringIO('flag:b\nmaybe\n'))

    warnings = [rec.getMessage() for rec in caplog.records
                if rec.name == 'rollups.io'
                and rec.levelno >= logging.WARNING]

    assert ds[0]['flag'] is False
    assert warnings == ['Unexpected bool value "maybe", treating as False']


def test_csv_date_column_ignores_a_trailing_time():
    """Verify a :d field reads the leading YYYY-MM-DD and drops the rest.

    Mutation: widening _parse's s[:10] slice, so strptime chokes on the
        leftover separator instead of parsing the date.
    Oracle: hand-computed Date(2024, 3, 15) from a full timestamp.
    """
    ds = DataSet.read(io.StringIO('when:d\n2024-03-15 13:45:00\n'))

    assert ds.colmap['when'] == Date
    assert ds[0]['when'] == Date(2024, 3, 15)


def test_csv_float_reads_ieee_special_values():
    """Verify inf, -inf and nan reach float() when libb.parse declines.

    Mutation: the float branch's fallback dropping the raw string, so a
        value libb.parse cannot read comes back None.
    Oracle: hand-computed math.inf and -math.inf, and math.isnan for
        the value that compares equal to nothing.
    """
    ds = DataSet.read(io.StringIO('x:f\ninf\n-inf\nnan\n'))

    assert ds[0]['x'] == math.inf
    assert ds[1]['x'] == -math.inf
    assert math.isnan(ds[2]['x'])


def test_csv_int_reads_accounting_and_grouped_forms():
    """Verify an :i field parses (5) as -5 and "1,234" as 1234.

    Mutation: the int branch calling int(s) directly instead of
        libb.parse, or inverting its `parsed is not None` test - either
        way both values come back None.
    Oracle: hand-computed -5 for the parenthesized negative and 1234
        for the thousands-separated value.
    """
    ds = DataSet.read(io.StringIO('n:i\n(5)\n"1,234"\n'))

    assert ds[0]['n'] == -5
    assert ds[1]['n'] == 1234


def test_csv_reader_keywords_reach_csv_reader():
    """Verify read forwards its keywords, so delimiter='|' splits fields.

    Mutation: dropping **kw from the csv.reader call, which falls back
        to a comma and reads the whole line as one field.
    Oracle: hand-split header and row against the pipe delimiter.
    """
    ds = DataSet.read(io.StringIO('name:s|age:i\nA|30\n'), delimiter='|')

    assert ds.cols == ['name', 'age']
    assert dict(ds[0]) == {'name': 'A', 'age': 30}


def test_csv_skips_logs_the_number_of_rows_dropped(caplog):
    """Verify the skip count reaches the debug log, not just the reader.

    Mutation: the debug call losing its message, leaving no record of
        how many lines were dropped ahead of the header.
    Oracle: the hand-counted two skipped rows, named in the record.
    """
    csv_data = 'Junk\nMore junk\nname:s\nA\n'

    with caplog.at_level(logging.DEBUG, logger='rollups.io'):
        ds = DataSet.read(io.StringIO(csv_data), skips=2)

    messages = [rec.getMessage() for rec in caplog.records]

    assert ds.cols == ['name']
    assert 'Skipped 2 rows in csv' in messages


def test_csv_type_suffix_splits_from_the_right():
    """Verify a colon inside a column name survives the suffix split.

    Mutation: splitting from the left, or letting the split run past
        one field - either way the header raises instead of yielding
        a name and a type.
    Oracle: hand-split 'ratio:pct:f' into the name 'ratio:pct' and the
        float suffix.
    """
    ds = DataSet.read(io.StringIO('ratio:pct:f\n1.5\n'))

    assert ds.cols == ['ratio:pct']
    assert ds.colmap == {'ratio:pct': float}
    assert ds[0]['ratio:pct'] == 1.5


def test_csv_field_without_a_suffix_reads_as_str():
    """Verify a bare header field keeps its name and types as str.

    Mutation: defaulting the type to None rather than str, or naming
        the column None rather than the header text.
    Oracle: hand-typed {'name': str, 'age': int}, with '  A  ' stripped
        to 'A' proving the str parser ran.
    """
    ds = DataSet.read(io.StringIO('name,age:i\n  A  ,30\n'))

    assert ds.cols == ['name', 'age']
    assert ds.colmap == {'name': str, 'age': int}
    assert dict(ds[0]) == {'name': 'A', 'age': 30}


def test_csv_header_only_keeps_the_declared_columns():
    """Verify a file with no data rows still carries its header schema.

    Mutation: dropping the columns argument from the final DataSet
        call, leaving the schema to be guessed from rows that do not
        exist.
    Oracle: hand-typed {'name': str, 'age': int} against zero rows.
    """
    ds = DataSet.read(io.StringIO('name:s,age:i\n'))

    assert len(ds) == 0
    assert ds.cols == ['name', 'age']
    assert ds.colmap == {'name': str, 'age': int}


# --- dataframe column guessing ---


def test_guess_columns_maps_every_object_dtype_it_names():
    """Verify decimal, date, datetime, time, bytes, boolean and mixed.

    Mutation: any key of dataset_type_map rebound, so the lookup misses
        and the column types None instead of its mapped type.
    Oracle: hand-written pandas dtype name to DataSet type pairs, one
        object column per name.
    """
    day = datetime.date(2024, 1, 1)
    stamp = datetime.datetime(2024, 1, 1, 9, 30)
    clock = datetime.time(9, 30)
    df = pd.DataFrame({
        'dec': pd.Series([Decimal('1.5'), Decimal('2.5')], dtype=object),
        'day': pd.Series([day, day], dtype=object),
        'stamp': pd.Series([stamp, stamp], dtype=object),
        'clock': pd.Series([clock, clock], dtype=object),
        'blob': pd.Series([b'x', b'y'], dtype=object),
        'flag': pd.Series([True, False], dtype=object),
        'mixed': pd.Series(['a', 1.5], dtype=object),
        })

    assert guess_dataframe_dataset_columns(df) == [
        ('dec', float),
        ('day', Date),
        ('stamp', DateTime),
        ('clock', Time),
        ('blob', bytes),
        ('flag', bool),
        ('mixed', object),
        ]


def test_guess_columns_ignores_nulls_when_inferring_a_type():
    """Verify a null among the values does not blur the column's type.

    Mutation: infer_dtype called with skipna off, which reports an
        object column holding a None as mixed - typing it object, or
        losing it altogether where the name has no entry in the map.
    Oracle: hand-typed str and int for columns whose non-null values
        are all strings and all integers.
    """
    df = pd.DataFrame({
        'name': pd.Series(['a', None, 'b'], dtype=object),
        'num': pd.Series([1, None, 2], dtype=object),
        })

    assert guess_dataframe_dataset_columns(df) == [('name', str), ('num', int)]


def test_read_parses_scientific_notation_as_the_full_magnitude():
    """Verify a float or int column reads 1e3 as 1000, not libb.parse's 13.

    Mutation: preferring libb.parse over float()/int(), which strips the
        exponent and reads '1e3' as 13 and '2.5E2' as 2.52.
    Oracle: hand-computed 1000.0/250.0 for a float column, 1000/250 for
        an int column.
    """
    floats = list(DataSet.read(io.StringIO('v:f\n1e3\n2.5E2\n')).unwind('v'))
    ints = list(DataSet.read(io.StringIO('v:i\n1e3\n2.5E2\n')).unwind('v'))

    assert floats == [1000.0, 250.0]
    assert ints == [1000, 250]


def test_read_survives_an_int_cell_too_large_to_convert():
    """Verify an overflowing int cell costs its own value, not the file.

    int(float('1e400')) sees inf and raises OverflowError, which the
    surrounding except pair did not name, so one cell aborted the read.

    Mutation: dropping OverflowError from the int branch's except clause
        in io.read_csv_rows, which makes DataSet.read raise instead of
        returning rows.
    Oracle: the later row, whose values are hand-known and can only be
        reached if the reader got past the overflowing cell.
    """
    rows = DataSet.read(io.StringIO('a:i,b:i\n1e400,7\n5,8\n'))

    assert len(rows) == 2
    assert (rows[1]['a'], rows[1]['b']) == (5, 8)


def test_from_list_rejects_a_cols_typs_length_mismatch():
    """Verify a short typs list raises rather than dropping a column.

    Mutation: dropping the length check, so zip() truncates to the
        shorter list and column 'b' vanishes with no error.
    Oracle: a ValueError, against the silent ['a'] the zip would give.
    """
    with pytest.raises(ValueError, match='cols and typs length mismatch'):
        DataSet.from_list([(1, 2)], ['a', 'b'], [int])


def test_guess_columns_types_an_unmapped_dtype_as_object():
    """Verify a dtype with no entry in the map comes back object, not None.

    Mutation: the map lookup defaulting to None, so a mixed-integer or
        all-None column returns an untyped (None) column.
    Oracle: hand-typed object for a mixed-integer column and an all-None
        column.
    """
    df = pd.DataFrame({
        'mixed_int': pd.Series(['a', 1], dtype=object),
        'empty': pd.Series([None, None], dtype=object),
        })

    assert guess_dataframe_dataset_columns(df) == [
        ('mixed_int', object),
        ('empty', object),
        ]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# --- from_csv Tests ---


def test_from_csv_reads_like_read():
    """Verify from_csv parses a csv the same way read does.

    Mutation: from_csv returning cls(file_or_name) or an empty DataSet
        rather than delegating to read.
    Oracle: hand-written csv text with a typed header, compared against
        read on the identical source.
    """
    text = 'a:i,b\n1,x\n2,y\n'
    by_csv = [dict(r) for r in DataSet.from_csv(io.StringIO(text))]
    by_read = [dict(r) for r in DataSet.read(io.StringIO(text))]

    assert by_csv == [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]
    assert by_csv == by_read


def test_from_csv_forwards_keyword_arguments():
    """Verify from_csv passes **kw through to read.

    Mutation: dropping **kw from the delegated call, which would leave
        the junk first line in place and read it as the header.
    Oracle: a source whose real header is on line two - skips=1 must
        yield columns a and b, not a column named after the junk line.
    """
    text = 'junk line\na:i,b\n1,x\n'

    ds = DataSet.from_csv(io.StringIO(text), skips=1)

    assert ds.cols == ['a', 'b']
    assert [dict(r) for r in ds] == [{'a': 1, 'b': 'x'}]
