"""Tests for bucket_dataframe, the DataFrame-native group-by.

Standalone counterpart to `DataSet.bucket`, taking and returning a
`pandas.DataFrame`. The aggregation shapes it reads are the same ones
`DataSet.bucket` reads, documented in tests/test_bucket.py.

Notes
-----
- A gap reads as null under `isna`, and which null it is follows the
  column's dtype, so a result is compared through `tests.frame.helpers`
  rather than against None directly.
- The tests named `..._agrees_with_dataset_bucket` are differential:
  their oracle is `DataSet.bucket` over the same rows. Those named for
  a divergence pin a place the two deliberately disagree.
"""
import numpy as np
import pandas as pd
import pytest
from rollups import DataSet
from rollups.frame import bucket_dataframe
from tests.frame.helpers import dataset_rows, frame_rows, tokens


@pytest.fixture
def df_two_groups():
    """Three rows over two groups, key 1 holding two of them."""
    return pd.DataFrame({'key': [1, 1, 2], 'b': [2, 3, 4], 'c': [4, 5, 7]})


def test_bucket_dataframe_sums_a_bare_column_name(df_two_groups):
    """Verify a bare column name totals the group, skipping nulls.

    Mutation: defaulting to max, which returns 3 rather than 5 for the
        two-row group; or dropping the null skip, which makes a null of
        every total that meets one.
    Oracle: hand-computed b 2+3=5 and 4, c 4+5=9 and 7, over a frame
        whose second row carries a null b.
    """
    nulled = df_two_groups.copy()
    nulled.loc[1, 'b'] = None

    assert bucket_dataframe(df_two_groups, 'key', ['b', 'c']).to_dict('list') == {
        'key': [1, 2], 'b': [5, 4], 'c': [9, 7]}
    assert bucket_dataframe(nulled, 'key', ['b'])['b'].tolist() == [2, 4]


def test_bucket_dataframe_totals_an_all_null_group_to_null_not_zero():
    """Verify a group whose column is entirely null aggregates to null.

    Mutation: dropping min_count=1 from the vectorized sum, which totals
        an all-null group to 0 and so reports a real zero where there was
        no data at all.
    Oracle: hand-computed 30.0 for group a against a null for group c,
        whose single w value is missing.
    """
    df = pd.DataFrame({'k': ['a', 'a', 'c'], 'w': [10.0, 20.0, None]})

    frame = bucket_dataframe(df, 'k', ['w'])

    assert frame['w'].tolist()[0] == 30.0
    assert pd.isna(frame['w'].tolist()[1])


def test_bucket_dataframe_vectorized_and_per_group_paths_agree():
    """Verify the fast ops handle a null and an empty group as the
    callable they stand in for does.

    Mutation: the vectorized branch diverging on nulls - summing without
        min_count, or reading skipna the other way - which shows up only
        where a group is entirely null. The paths differ at float and
        int64 extremes by design, so this pins the null handling, not
        the arithmetic.
    Oracle: differential - sum, max and min taken through the fast path
        against lambdas wrapping the same builtins, which cannot reach it.
    """
    df = pd.DataFrame({'k': ['a', 'a', 'b', 'c'],
                       'v': [1.0, 2.0, 3.0, None]})

    fast = bucket_dataframe(df, 'k', [('v', sum, 'total'), ('v', max, 'hi'),
                                      ('v', min, 'lo')])
    # The lambdas are load-bearing: an op the fast path recognizes by
    # identity has to reach the per-group path for this to compare
    # anything, so do not let the formatter unwrap them.
    slow = bucket_dataframe(df, 'k', [
        ('v', lambda s: sum(s), 'total'),  # noqa: PLW0108
        ('v', lambda s: max(s), 'hi'),  # noqa: PLW0108
        ('v', lambda s: min(s), 'lo')])  # noqa: PLW0108

    pd.testing.assert_frame_equal(fast, slow, check_dtype=False)
    assert fast['total'].tolist()[:2] == [3.0, 3.0]
    assert pd.isna(fast['total'].tolist()[2])


def test_bucket_dataframe_orders_groups_by_key_with_the_null_last():
    """Verify groups come back sorted, whatever order the rows arrived in.

    Mutation: sort=False on the groupby, which emits the groups in
        first-appearance order [2, 1]; or dropna left at its True
        default, which throws the null-key group away entirely.
    Oracle: hand-computed key order [1, 2, None] over rows arriving
        2, 1, None, 2, 1, plus the sums 1 and 5 those keys carry.
    """
    df = pd.DataFrame({'key': [2, 1, None, 2, 1], 'b': [3, 3, 9, 2, -2]})

    frame = bucket_dataframe(df, 'key', ['b'])

    assert frame['key'].tolist()[:2] == [1.0, 2.0]
    assert pd.isna(frame['key'].tolist()[2])
    assert frame['b'].tolist() == [1, 5, 9]


def test_bucket_dataframe_empty_keycols_total_every_row_into_one():
    """Verify an empty key list gives one row and invents no column.

    Mutation: planting a constant grouping column in the frame, which
        leaks into the result or overwrites a caller column of the same
        name; or returning one row per input row.
    Oracle: hand-computed totals a 2+1+4=7 and b 1+3-1=3 in a single row
        of exactly two columns.
    """
    df = pd.DataFrame({'a': [2, 1, 4], 'b': [1, 3, -1]})

    frame = bucket_dataframe(df, [], ['a', 'b'])

    assert list(frame.columns) == ['a', 'b']
    assert frame.to_dict('list') == {'a': [7], 'b': [3]}


def test_bucket_dataframe_keyless_leaves_a_user_dummy_column_alone():
    """Verify a keyless total over a column named __dummy__ reads its values.

    Mutation: grouping on a planted __dummy__ column, which overwrites
        the caller's column with 1s and totals 2 instead of 11.
    Oracle: hand-computed 5+6=11, the total a column overwritten with 1s
        could not produce.
    """
    df = pd.DataFrame({'__dummy__': [5, 6]})

    assert bucket_dataframe(df, [], ['__dummy__'])['__dummy__'].tolist() == [11]


@pytest.mark.parametrize(('agg', 'expected'), [
    ('b', {'b': [5, 4]}),
    (('b',), {'b': [5, 4]}),
    (('b', max), {'b': [3, 4]}),
    (('b', min, 'lowest'), {'lowest': [2, 4]}),
    (('b', list), {'b': [[2, 3], [4]]}),
])
def test_bucket_dataframe_reads_each_aggregation_shape(df_two_groups, agg,
                                                       expected):
    """Verify every terse aggregation shape means what the grammar says.

    Mutation: taking the column from agg[1] rather than agg[0], raising
        on a one-item tuple, or reading a string third item as a filter
        rather than an alias - each mislabels or misreads one shape while
        the others still pass.
    Oracle: hand-computed values per shape over groups {2,3} and {4}.
    """
    frame = bucket_dataframe(df_two_groups, 'key', [agg])

    assert frame.drop(columns='key').to_dict('list') == expected


def test_bucket_dataframe_filter_narrows_the_group_before_the_op(df_two_groups):
    """Verify a callable third item selects the values the op then sees.

    Mutation: applying the op to the unfiltered group, which returns 7
        for key 2 rather than leaving it with nothing to aggregate.
    Oracle: hand-computed max of c excluding 7 - 5 for key 1, and no
        value at all for key 2.
    """
    fn = lambda group: group['c'].loc[group['c'] != 7]

    frame = bucket_dataframe(df_two_groups, 'key', [('c', max, fn)])

    assert frame['c'].tolist()[0] == 5
    assert pd.isna(frame['c'].tolist()[1])


def test_bucket_dataframe_four_item_tuple_filters_and_aliases(df_two_groups):
    """Verify a four-item tuple applies both the filter and the alias.

    Mutation: dropping the alias when a filter is present, so the two
        aggregations collide on their source column names.
    Oracle: hand-computed rows where b == 3 - filtered_b 3 and
        filtered_c 5 for key 1, nothing for key 2.
    """
    fn = lambda col: lambda group: group[col].loc[group['b'] == 3]

    frame = bucket_dataframe(df_two_groups, 'key', [
        ('b', sum, fn('b'), 'filtered_b'),
        ('c', sum, fn('c'), 'filtered_c'),
        ])

    assert frame.iloc[0][['filtered_b', 'filtered_c']].tolist() == [3, 5]
    assert frame.iloc[1][['filtered_b', 'filtered_c']].isna().all()


def test_bucket_dataframe_filter_sees_the_key_columns(df_two_groups):
    """Verify the group handed to a filter still carries its key columns.

    Mutation: routing the group through groupby.apply, which on pandas 3
        hides the grouping columns and makes the filter raise KeyError.
    Oracle: hand-computed 5 and 4 from a filter that reads the key column
        it is grouped on.
    """
    fn = lambda group: group['b'].loc[group['key'] >= 1]

    frame = bucket_dataframe(df_two_groups, 'key', [('b', sum, fn)])

    assert frame['b'].tolist() == [5, 4]


def test_bucket_dataframe_filter_may_return_a_plain_list(df_two_groups):
    """Verify a filter returning a list aggregates, and an empty one drops out.

    Mutation: flipping the empty-sequence guard, so a filter that matched
        nothing totals to 0 instead of leaving a gap, or one that matched
        something is thrown away.
    Oracle: hand-computed - b over 1 keeps 2+3=5 and 4; c over 5 keeps
        nothing for key 1 and 7 for key 2.
    """
    big_b = lambda group: [v for v in group['b'].tolist() if v > 1]
    big_c = lambda group: [v for v in group['c'].tolist() if v > 5]

    frame = bucket_dataframe(df_two_groups, 'key', [
        ('b', sum, big_b, 'kept_b'),
        ('c', sum, big_c, 'kept_c'),
        ])

    assert frame['kept_b'].tolist() == [5, 4]
    assert pd.isna(frame['kept_c'].tolist()[0])
    assert frame['kept_c'].tolist()[1] == 7


def test_bucket_dataframe_survives_a_filter_that_returns_a_scalar(df_two_groups):
    """Verify a filter handing back one bare value does not crash the call.

    Mutation: reading len() off whatever the filter returned without
        checking it has one, which raises an opaque TypeError out of the
        aggregation instead of letting the op decide.
    Oracle: hand-computed - the op cannot add a bare float, so the column
        comes back null for both groups rather than raising.
    """
    frame = df_two_groups.copy()

    result = bucket_dataframe(frame, 'key', [('c', sum, lambda g: g['c'].sum())])

    assert result['c'].isna().all()


def test_bucket_dataframe_gives_one_row_per_group_for_a_series_op(df_two_groups):
    """Verify an op returning several values fills one cell, not several rows.

    Mutation: letting groupby.apply reshape a Series result, which
        explodes the two-value group into two output rows and leaks a
        level_ index column beside them.
    Oracle: hand-computed 2 rows for 2 groups, the key-1 cell holding
        both of that group's c values.
    """
    frame = bucket_dataframe(df_two_groups, 'key', [('c', lambda s: s.tolist())])

    assert list(frame.columns) == ['key', 'c']
    assert frame['c'].tolist() == [[4, 5], [7]]


def test_bucket_dataframe_merges_collections_an_op_cannot_add():
    """Verify sum over set or list cells merges them rather than giving up.

    Mutation: dropping the collection fallback, so summing a set column
        raises inside the op and the whole column comes back null - the
        behavior DataSet.bucket does not have.
    Oracle: hand-computed union {1, 2, 3} and concatenation [1, 2, 3],
        against the null a bare sum() would leave.
    """
    df = pd.DataFrame({'k': ['a', 'a'],
                       's': [{1, 2}, {3}],
                       'l': [[1], [2, 3]]})

    frame = bucket_dataframe(df, 'k', [('s', sum), ('l', sum)])

    assert frame['s'].tolist() == [{1, 2, 3}]
    assert frame['l'].tolist() == [[1, 2, 3]]


def test_bucket_dataframe_repeated_alias_keeps_the_last_value_in_place(
        df_two_groups):
    """Verify a repeated un-aliased aggregation overwrites, as bucket does.

    Mutation: suffixing the two results into b_x and b_y and keeping
        both, or moving the survivor to the end of the column order.
    Oracle: differential against DataSet.bucket - one 'b' column still
        between key and c, holding the maxima [3, 4] rather than the
        sums [5, 4].
    """
    rows = df_two_groups.to_dict('records')
    aggs = [('b', sum), ('b', max), ('c', sum)]

    frame = bucket_dataframe(df_two_groups, 'key', aggs)
    bucketed = DataSet([dict(r) for r in rows]).bucket('key', list(aggs))

    assert list(frame.columns) == bucketed.cols == ['key', 'b', 'c']
    assert frame['b'].tolist() == [row['b'] for row in bucketed] == [3, 4]


def test_bucket_dataframe_no_aggregations_returns_the_sorted_distinct_keys():
    """Verify an empty aggregation list gives the distinct keys, sorted.

    Mutation: returning every row rather than the distinct keys; or
        keeping first-appearance order, which every other path sorts.
    Oracle: hand-computed keys [1, 2] and one column, over rows arriving
        2, 1, 2.
    """
    df = pd.DataFrame({'key': [2, 1, 2], 'b': [1, 2, 3]})

    frame = bucket_dataframe(df, 'key', [])

    assert list(frame.columns) == ['key']
    assert frame['key'].tolist() == [1, 2]


def test_bucket_dataframe_no_keys_and_no_aggregations_is_empty(df_two_groups):
    """Verify asking for neither keys nor aggregations returns nothing.

    Mutation: falling through to the grouping path, which returns one row
        per input row under a synthetic key.
    Oracle: hand-specified zero rows and zero columns.
    """
    assert bucket_dataframe(df_two_groups, [], []).shape == (0, 0)


def test_bucket_dataframe_rejects_a_repeated_column_name():
    """Verify a duplicated column raises rather than aggregating to null.

    Mutation: no guard at all, after which df[col] hands back a frame,
        the vectorized branch is skipped, sum iterates the column LABELS
        and raises, and the fallback turns that into a null for every
        group - a silently empty column where a total was asked for.
    Oracle: hand-specified message naming 'x', against the [None, None]
        the unguarded path returns where the sums are 6 and 8.
    """
    df = pd.DataFrame(np.array([[1, 1, 5], [2, 2, 6]]), columns=['g', 'x', 'x'])

    with pytest.raises(ValueError, match="Frame carries 'x' more than once"):
        bucket_dataframe(df, 'g', ['x'])


def test_bucket_dataframe_groups_on_two_key_columns():
    """Verify two keys split the groups and each total skips its nulls.

    Mutation: grouping on the first key alone, which merges both key2
        groups into one total of 16.
    Oracle: hand-computed a 2+4=6 for key2 2 and 10 for key2 3; b 3-1=2
        for key2 2 and nothing for key2 3.
    """
    df = pd.DataFrame({'key1': [1, 1, 1, 1], 'key2': [2, 2, 2, 3],
                       'a': [2, None, 4, 10], 'b': [None, 3, -1, None]})

    frame = bucket_dataframe(df, ['key1', 'key2'], ['a', 'b'])

    assert frame['key2'].tolist() == [2, 3]
    assert frame['a'].tolist() == [6.0, 10.0]
    assert frame['b'].tolist()[0] == 2.0
    assert pd.isna(frame['b'].tolist()[1])


def test_bucket_dataframe_leaves_the_input_untouched(df_two_groups):
    """Verify the frame handed in is neither sorted, reindexed, nor widened.

    Mutation: planting the grouping column in the caller's frame, or
        sorting it in place before grouping.
    Oracle: differential against a copy taken before the call, over both
        the keyed and the keyless path.
    """
    before = df_two_groups.copy()

    bucket_dataframe(df_two_groups, 'key', ['b'])
    bucket_dataframe(df_two_groups, [], ['b'])

    assert df_two_groups.equals(before)


@pytest.mark.parametrize(('call', 'message'), [
    ((('b', sum, lambda g: g['b'], 'x', 'extra'),), 'length 1, 2, 3, or 4'),
    (((),), 'length 1, 2, 3, or 4'),
    (('gone',), "Frame has no column 'gone'"),
    ((('key', sum),), "'key' would overwrite a key column"),
])
def test_bucket_dataframe_rejects_a_malformed_aggregation(df_two_groups, call,
                                                          message):
    """Verify each malformed aggregation raises rather than guessing.

    Mutation: ignoring the items past the fourth, which binds the alias
        to the wrong element; letting a missing column through to a bare
        pandas KeyError; or letting an aggregation land on a key column,
        which silently replaces the key with its own total.
    Oracle: hand-specified ValueError text per malformed aggregation.
    """
    with pytest.raises(ValueError, match=message):
        bucket_dataframe(df_two_groups, 'key', list(call))


def test_bucket_dataframe_leaves_an_all_null_group_empty_where_bucket_pads_it():
    """Verify a collecting op returns nothing for a group with no values.

    This is the one deliberate divergence from DataSet.bucket, whose
    default filter substitutes a one-item [None] so that max() has
    something to work on; see docs/dataframe-native.md.

    Mutation: either path converging on the other, which would change
        what a migrated call site reads for a group holding no data.
    Oracle: the v column of group b read off each result - [None] under
        bucket and its length 1, against a null under bucket_dataframe.
    """
    rows = [{'k': 'a', 'v': 1}, {'k': 'b', 'v': None}]

    bucketed = DataSet([dict(r) for r in rows]).bucket('k', [('v', list)])
    frame = bucket_dataframe(pd.DataFrame(rows), 'k', [('v', list)])

    assert [row['v'] for row in bucketed] == [[1], [None]]
    assert frame['v'].tolist()[0] == [1.0]
    assert pd.isna(frame['v'].tolist()[1])


@pytest.mark.parametrize('keys', ['k', ['k', 'g'], []])
@pytest.mark.parametrize('agg', [['v'], ['v', 'w'], [('v', max)], [('v', min)],
                                 [('v', sum, 'total')],
                                 [('v', sum), ('w', max, 'wmax')]])
def test_bucket_dataframe_agrees_with_dataset_bucket(keys, agg):
    """Verify the frame group-by returns DataSet.bucket's rows and columns.

    Mutation: any divergence in grouping, ordering, null handling, or
        column naming between the two paths - the frame path silently
        answering something the row path does not.
    Oracle: differential against DataSet.bucket over unsorted rows
        carrying a null key, a null in each aggregated column, and one
        group whose values are entirely null.
    """
    rows = [{'k': 'b', 'g': 1, 'v': 2, 'w': None},
            {'k': 'a', 'g': 2, 'v': None, 'w': 1.5},
            {'k': 'a', 'g': 2, 'v': 3, 'w': 2.5},
            {'k': None, 'g': 1, 'v': 1, 'w': None},
            {'k': 'a', 'g': 1, 'v': 4, 'w': 2.5}]

    bucketed = DataSet([dict(r) for r in rows]).bucket(keys, list(agg))
    frame = bucket_dataframe(pd.DataFrame(rows), keys, list(agg))

    assert list(frame.columns) == bucketed.cols
    assert tokens(frame_rows(frame, bucketed.cols)) == tokens(dataset_rows(bucketed))


if __name__ == '__main__':
    pytest.main([__file__])
