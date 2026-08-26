import datetime

import pytest
from opendate import Date, DateTime, Time
from rollups import DataSet


def test_json_raw():
    """Verify json(raw=True) emits a bare array with ISO dates.

    Mutation: swapping the `if raw:` branches at io.py, or
        dropping cls=JSONEncoderISODate from the dumps call below it.
    Oracle: hand-written json text for one Date and one float column.
    """
    d = [{'d': Date(2014, 10, 1), 'x': 2.}]
    assert DataSet(d).json(raw=True) == '[{"d": "2014-10-01", "x": 2.0}]'


def test_json_with_metadata():
    """Verify json(raw=False) wraps the rows in order, types and data.

    Mutation: str(typ) in place of typ.__name__ at io.py,
        or unpacking `types, order` from the zip on the line above.
    Oracle: hand-written json text naming both columns and both types.
    """
    d = [{'d': Date(2014, 10, 1), 'x': 2.}]
    expect = ('{"order": ["d", "x"], "types": ["Date", "float"], '
              '"data": [{"d": "2014-10-01", "x": 2.0}]}')
    assert DataSet(d).json(raw=False) == expect


def test_json_column_subset():
    """Verify `columns` selects columns but leaves dataset order intact.

    Mutation: building cols by walking the `columns` argument rather
        than self.columns at io.py, which would emit the
        caller's order; or an empty `columns` selecting nothing.
    Oracle: hand-written json text for the reversed two-column request.
    """
    ds = DataSet([{'a': 1, 'b': 'two', 'c': 3.0}])
    expect = ('{"order": ["a", "c"], "types": ["int", "float"], '
              '"data": [{"a": 1, "c": 3.0}]}')
    assert ds.json(columns=['c', 'a']) == expect
    assert ds.json(columns=[]) == ds.json()


def test_json_format_value_receives_column_type():
    """Verify format_value is called with (row, column, column type).

    Mutation: passing the column name where the type belongs at
        io.py, or dropping the third argument.
    Oracle: a formatter echoing type name and value, hand-computed.
    """
    def fmt(row, col, typ):
        return f'{typ.__name__}:{row.get(col)}'

    d = [{'d': Date(2014, 10, 1), 'x': 2.}]
    result = DataSet(d).json(raw=True, format_value=fmt)
    assert result == '[{"d": "Date:2014-10-01", "x": "float:2.0"}]'


def test_json_extra_keys():
    """Verify **kw joins the metadata object and is dropped when raw.

    Mutation: dropping **kw from the dict at io.py, or
        folding it into the raw branch's bare array.
    Oracle: hand-written json text with and without the extra keys.
    """
    d = [{'x': 2.}]
    expect = ('{"order": ["x"], "types": ["float"], '
              '"data": [{"x": 2.0}], "status": "ok", "page": 1}')
    assert DataSet(d).json(raw=False, status='ok', page=1) == expect
    assert DataSet(d).json(raw=True, status='ok', page=1) == '[{"x": 2.0}]'


def test_json_converts_declared_types_first():
    """Verify json() serializes declared types, not the stored strings.

    Mutation: dropping @ensure_types_converted from json() at
        io.py, which leaves '5' quoted in the output.
    Oracle: hand-written json text where n is the bare number 5.
    """
    ds = DataSet([{'n': '5', 'd': '2014-10-01'}], typs=[int, Date])
    assert ds.json(raw=True) == '[{"n": 5, "d": "2014-10-01"}]'


def test_json_writes_null_for_a_column_the_row_does_not_carry():
    """Verify a declared column absent from a row serializes as null.

    Mutation: reading the row through its own get(), which answers a
        missing key with the dict attribute of that name, so a column
        named after one reaches the encoder as a bound method.
    Oracle: hand-written json text; the encoder refuses a bound method
        outright, so the wrong reader cannot even produce output.
    """
    ds = DataSet([{'a': 1}], columns=[('a', int), ('items', str)])

    assert ds.json(raw=True) == '[{"a": 1, "items": null}]'


def test_json_round_trips_without_naming_a_shape():
    """Verify from_json(ds.json()) answers a DataSet, the obvious round trip.

    The defaults disagreed - json wrote the object shape, from_json read
    the bare array - so this raised ValueError, and passing raw=False to
    settle it answered a tuple rather than a DataSet.

    Mutation: from_json defaulting raw to True or False again, or its
        shape test reading a bare array as the object shape.
    Oracle: the source dataset's own rows and columns.
    """
    ds = DataSet([{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}],
                 columns=[('a', int), ('b', str)])

    back = DataSet.from_json(ds.json())

    assert isinstance(back, DataSet)
    assert back.columns == [('a', int), ('b', str)]
    assert [dict(r) for r in back.container] == [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]


def test_json_round_trip_restores_every_declared_type():
    """Verify each declared type survives json() and from_json().

    to_json writes typ.__name__, and locate() cannot undo that for six of
    them: it answers None for 'date' and for the three opendate names,
    and the MODULE of that name for 'datetime' and 'time'.

    Mutation: dropping any entry from from_json's name-to-type map, which
        types that column None and loses the values with it.
    Oracle: the declared type itself, and the value as written.
    """
    declared = [
        ('i', int, 5),
        ('f', float, 1.5),
        ('s', str, 'x'),
        ('d', Date, Date(2024, 1, 15)),
        ('dt', DateTime, DateTime.instance(datetime.datetime(2024, 1, 15, 10, 30))),
        ('t', Time, Time.instance(datetime.time(10, 30))),
        ('sd', datetime.date, datetime.date(2024, 1, 15)),
        ('sdt', datetime.datetime, datetime.datetime(2024, 1, 15, 10, 30)),
        ('st', datetime.time, datetime.time(10, 30)),
        ]
    for name, typ, val in declared:
        ds = DataSet([{name: val}], columns=[(name, typ)])
        ds.ensure_types()
        written = ds[0][name]

        back = DataSet.from_json(ds.json())

        assert back.colmap[name] == typ, f'{name} lost its declared type'
        assert back[0][name] == written, f'{name} changed value'
        assert type(back[0][name]) is type(written), f'{name} changed class'


def test_from_json_reads_a_type_name_that_names_no_type_as_untyped():
    """Verify a type name resolving to a non-type leaves the column untyped.

    locate() answers the MODULE for a type whose __name__ matches one -
    array.array is the live example - and a module reaching isinstance
    raises TypeError on the first read of the dataset.

    Mutation: dropping the isinstance(typ, type) guard in from_json, which
        stores the module as the column type.
    Oracle: the payload's own value, readable at all only if nothing put
        a module where a type belongs.
    """
    ds = DataSet.from_json('{"order": ["a"], "types": ["array"], "data": [{"a": "x"}]}')

    assert ds.colmap['a'] is None
    assert ds[0]['a'] == 'x'


def test_from_json_leaves_a_str_column_alone():
    """Verify a declared str column comes back as the strings it held.

    The decoder ran dateutil.parser.parse over every string value in
    every column, so '5' and 'March 2024' became datetimes filled in from
    TODAY - the same payload decoding differently on different days.

    Mutation: from_json reading a typed payload through
        JSONDecoderISODate again, which converts before the declared type
        is consulted.
    Oracle: the input strings themselves; '20241002' is the sharp case,
        being a date in ISO basic format.
    """
    held = ['2024-01-15', '5', 'March 2024', 'hello', '20241002']
    ds = DataSet([{'s': v} for v in held], columns=[('s', str)])

    back = DataSet.from_json(ds.json())

    assert [r['s'] for r in back.container] == held


def test_from_json_reads_a_bare_array_and_still_guesses_dates():
    """Verify a payload declaring no types still reads an ISO date as one.

    Guessing is the fallback where nothing was declared, so dropping it
    would leave a bare array's dates as strings.

    Mutation: from_json parsing every payload plainly, so 'd' stays a str
        column, or guessing on the typed payload too.
    Oracle: hand-computed Date(2014, 10, 1) against a hand-written array.
    """
    ds = DataSet.from_json('[{"x": 2.0, "d": "2014-10-01"}]')

    assert isinstance(ds, DataSet)
    assert ds.colmap['d'] == Date
    assert ds[0]['d'] == Date(2014, 10, 1)
    assert ds[0]['x'] == 2.0


def test_from_json_keeps_the_explicit_raw_forms():
    """Verify naming raw= answers what it always did, tuple included.

    Every caller that names raw= predates the detecting default, so both
    explicit forms have to keep their return shape.

    Mutation: the detecting branch swallowing raw=False, which drops the
        extra keys, or raw=True reading the object shape.
    Oracle: hand-written payloads and the hand-listed leftover keys.
    """
    payload = ('{"data": [{"x": 1}], "order": ["x"], "types": ["int"], '
               '"metadata": "test"}')

    ds, other = DataSet.from_json(payload, raw=False)
    assert isinstance(ds, DataSet)
    assert other == {'metadata': 'test'}

    bare = DataSet.from_json('[{"x": 1}]', raw=True)
    assert isinstance(bare, DataSet)


def test_json_writes_a_time_column():
    """Verify a time column serializes rather than raising.

    libb's encoder answered date and datetime only, so any dataset
    carrying a time column raised TypeError on .json(), on write_json,
    and on anything else reaching that encoder.

    Mutation: dropping datetime.time from the encoder's isinstance
        tuple.
    Oracle: hand-written json text for both declared time types.
    """
    ds = DataSet([{'t': Time(10, 30)}], columns=[('t', Time)])
    assert ds.json(raw=True) == '[{"t": "10:30:00+00:00"}]'

    ds = DataSet([{'t': datetime.time(10, 30)}], columns=[('t', datetime.time)])
    assert ds.json(raw=True) == '[{"t": "10:30:00+00:00"}]'


if __name__ == '__main__':
    pytest.main([__file__])
