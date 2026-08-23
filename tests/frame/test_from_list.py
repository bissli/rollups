"""Coverage for frame.dataframe_from_list."""
import pytest
from rollups import dataframe_from_list


def test_dataframe_from_list_pairs_each_tuple_position_with_its_column():
    """Verify row tuples are read positionally against cols.

    Mutation: zipping rows against sorted(cols), so a column list out
    of alphabetical order transposes every value.
    Oracle: hand-written rows under deliberately unsorted column names.
    """
    df = dataframe_from_list(
        [('a', 1), ('b', 2)], ['z', 'n'], [str, int])

    assert list(df.columns) == ['z', 'n']
    assert list(df['z']) == ['a', 'b']
    assert list(df['n']) == [1, 2]


def test_dataframe_from_list_applies_the_declared_type_per_column():
    """Verify each column is coerced to the type declared for it.

    Mutation: the coercion loop dropped, leaving the strings handed in
    as strings so arithmetic on the column fails.
    Oracle: hand-converted ints and floats from string input.
    """
    df = dataframe_from_list(
        [('1', '2.5'), ('3', '4.5')], ['n', 'x'], [int, float])

    assert list(df['n']) == [1, 3]
    assert list(df['x']) == [2.5, 4.5]
    assert df['n'].sum() == 4


def test_dataframe_from_list_leaves_none_alone():
    """Verify a None survives coercion rather than becoming 0 or ''.

    Mutation: the `if v is not None` guard dropped, so int(None) raises
    or a None silently becomes a zero.
    Oracle: notna on the gap, against a populated neighbor.
    """
    df = dataframe_from_list([(1,), (None,)], ['n'], [int])

    assert df['n'].notna().tolist() == [True, False]


def test_dataframe_from_list_with_no_rows_keeps_the_columns():
    """Verify an empty row list still yields the declared columns.

    Mutation: dropping the columns argument from the DataFrame call, so
        an empty frame comes back with no columns at all and the
        coercion loop cannot find them.
    Oracle: the hand-written column list, against zero rows.
    """
    df = dataframe_from_list([], ['a', 'b', 'c'], [int, str, float])

    assert list(df.columns) == ['a', 'b', 'c']
    assert len(df) == 0


def test_dataframe_from_list_mismatch_names_the_two_lists():
    """Verify the length-mismatch assertion says which lists disagree.

    Mutation: the assertion message rewritten to text that no longer
        names cols and typs, leaving the caller a bare AssertionError.
    Oracle: the literal message text 'cols and typs length mismatch',
        anchored so no extra text may creep in around it.
    """
    with pytest.raises(AssertionError, match='^cols and typs length mismatch$'):
        dataframe_from_list([(1, 2)], ['a', 'b'], [int])
