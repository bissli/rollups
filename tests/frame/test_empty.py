"""Coverage for frame.empty_dataframe."""
from rollups import empty_dataframe


def test_empty_dataframe_gives_one_row_of_type_defaults():
    """Verify each declared type contributes its own empty value.

    Mutation: one default used for every column (say '' or None
    throughout), so an int column no longer starts at 0.
    Oracle: hand-written defaults per type -- 0, 0.0, '', and None for
    any type without a declared default, bool included.
    """
    df = empty_dataframe([('n', int), ('x', float), ('s', str), ('b', bool)])

    assert len(df) == 1
    assert list(df.columns) == ['n', 'x', 's', 'b']
    assert df['n'][0] == 0
    assert df['x'][0] == 0.0
    assert df['s'][0] == ''
    assert df['b'][0] is None


def test_empty_dataframe_keeps_declared_column_order():
    """Verify column order follows the argument, not sorted names.

    Mutation: building from a dict comprehension that sorts keys, so
    a caller indexing by position reads the wrong column.
    Oracle: a column list deliberately out of alphabetical order.
    """
    df = empty_dataframe([('z', int), ('a', int), ('m', int)])

    assert list(df.columns) == ['z', 'a', 'm']
