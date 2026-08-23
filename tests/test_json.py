import pytest
from opendate import Date
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
    ds = DataSet([{'a': 1}], columns=[('a', int), ('items', str)],
                 check_types=False)

    assert ds.json(raw=True) == '[{"a": 1, "items": null}]'


if __name__ == '__main__':
    pytest.main([__file__])
