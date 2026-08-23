"""Tests for join_dataframes, the DataFrame-native join.

Standalone counterpart to `DataSet.join`, taking and returning a
`pandas.DataFrame`. The join semantics it shares with `DataSet.join`
are documented in tests/test_join.py.

Notes
-----
- A gap reads as null under `isna`, and which null it is follows the
  column's dtype, so a result is compared through `tests.frame.helpers`
  rather than against None directly.
- The tests named `..._agrees_with_dataset_join` are differential:
  their oracle is `DataSet.join` over the same rows. Those named for a
  divergence pin a place the two deliberately disagree.
"""
import logging

import numpy as np
import pandas as pd
import pytest
from rollups import DataSet
from rollups.frame import join_dataframes
from tests.frame.helpers import dataset_rows, frame_rows, tokens


@pytest.fixture
def df_keys_12_bc():
    """Two rows, unique keys 1 and 2."""
    return pd.DataFrame({'key': [1, 2], 'b': [1, 1], 'c': [1, 1]})


@pytest.fixture
def df_keys_12_de():
    """Two rows, unique keys 1 and 2, disjoint columns."""
    return pd.DataFrame({'key': [1, 2], 'd': [2, 2], 'e': [2, 2]})


@pytest.fixture
def df_dup_key1_de():
    """Two rows sharing key 1, so a join against it fans out."""
    return pd.DataFrame({'key': [1, 1], 'd': [3, 4], 'e': [3, 4]})


@pytest.mark.parametrize(('how', 'expected'), [
    ('inner', [(2, 'q', 20), (3, 'r', 30)]),
    ('left', [(1, 'p', None), (2, 'q', 20), (3, 'r', 30)]),
    ('right', [(2, 'q', 20), (3, 'r', 30), (4, None, 40)]),
    ('outer', [(1, 'p', None), (2, 'q', 20), (3, 'r', 30), (4, None, 40)]),
])
def test_join_dataframes_keeps_the_keys_each_type_promises(how, expected):
    """Verify each join type keeps its own side's keys and fills the gaps.

    Mutation: mapping a name to the wrong pandas how, e.g. 'left' to
        'outer', which lets key 4 through where a left join must drop it.
    Oracle: hand-computed rows per type over keys {1,2,3} against
        {2,3,4}, in the row order the contract fixes.
    """
    left = pd.DataFrame({'id': [1, 2, 3], 'x': ['p', 'q', 'r']})
    right = pd.DataFrame({'id': [2, 3, 4], 'y': [20, 30, 40]})

    frame = join_dataframes(left, 'id', right, 'id', how)

    assert list(frame.columns) == ['id', 'x', 'y']
    assert frame_rows(frame, ['id', 'x', 'y']) == expected


def test_join_dataframes_runs_in_left_row_order_not_sorted_key_order():
    """Verify row order follows the left frame, with right-only rows last.

    Mutation: returning pandas' merge order straight, which sorts an
        outer join by key and so brings left's two key-5 rows together.
        Or sorting on the right position first, which pulls the
        right-only key 9 to the front.
    Oracle: hand-computed [5, 1, 5, 9] - left's own order with its two
        key-5 rows still apart - against the [1, 5, 5, 9] a sort by key
        would give.
    """
    left = pd.DataFrame({'k': [5, 1, 5], 'a': ['p', 'q', 'r']})
    right = pd.DataFrame({'k': [9, 5, 1], 'b': [90, 50, 10]})

    frame = join_dataframes(left, 'k', right, 'k', 'outer')

    assert frame['k'].tolist() == [5, 1, 5, 9]
    assert frame_rows(frame, ['a', 'b']) == [
        ('p', 50), ('q', 10), ('r', 50), (None, 90)]


def test_join_dataframes_right_runs_in_right_row_order():
    """Verify a right join orders by the right frame, not the left.

    Mutation: sorting on the left position under every join type, which
        sends the unmatched right row to the end because its left
        position is null.
    Oracle: hand-computed [9, 5, 1] - the right frame's own order - where
        ordering by the left side would give [5, 1, 9].
    """
    left = pd.DataFrame({'k': [5, 1], 'a': ['p', 'q']})
    right = pd.DataFrame({'k': [9, 5, 1], 'b': [90, 50, 10]})

    frame = join_dataframes(left, 'k', right, 'k', 'right')

    assert frame['k'].tolist() == [9, 5, 1]
    assert frame_rows(frame, ['a']) == [(None,), ('p',), ('q',)]


def test_join_dataframes_fans_out_a_repeated_key(df_keys_12_bc, df_dup_key1_de):
    """Verify a repeated key produces one row per pairing.

    Mutation: de-duplicating on the join key, which collapses the two
        key-1 rows on the right and loses a pairing.
    Oracle: hand-computed 2 rows carrying e 3 and 4 against the single
        left key-1 row, and no key-2 row at all under inner.
    """
    frame = join_dataframes(df_keys_12_bc, 'key', df_dup_key1_de, 'key')

    assert frame['key'].tolist() == [1, 1]
    assert frame['e'].tolist() == [3, 4]
    assert frame['b'].tolist() == [1, 1]


def test_join_dataframes_coalesces_a_shared_column_left_first():
    """Verify a column both sides carry stays one column, left value first.

    Mutation: keeping pandas' v_x/v_y pair instead of folding them; or
        taking the left value outright, so the left null blanks the cell
        the right side could have filled.
    Oracle: hand-computed [10, 99, 30] - left wins rows 1 and 3, the
        right fills row 2 where the left is null.
    """
    left = pd.DataFrame({'key': [1, 2, 3], 'v': [10, None, 30]})
    right = pd.DataFrame({'key': [1, 2, 3], 'v': [77, 99, None]})

    frame = join_dataframes(left, 'key', right, 'key')

    assert list(frame.columns) == ['key', 'v']
    assert frame['v'].tolist() == [10, 99, 30]


def test_join_dataframes_prefer_right_flips_the_coalesce_but_not_the_fallback():
    """Verify prefer='right' lets the right side win, still falling back.

    Mutation: ignoring prefer so the left always wins; or dropping the
        fallback inside the right branch, leaving a null wherever the
        right side is empty rather than the left value.
    Oracle: hand-computed [77, 99, 30] against the left-preferring
        [10, 99, 30] over the same two frames.
    """
    left = pd.DataFrame({'key': [1, 2, 3], 'v': [10, None, 30]})
    right = pd.DataFrame({'key': [1, 2, 3], 'v': [77, 99, None]})

    frame = join_dataframes(left, 'key', right, 'key', prefer='right')

    assert frame['v'].tolist() == [77, 99, 30]
    assert join_dataframes(left, 'key', right, 'key')['v'].tolist() == [10, 99, 30]


def test_join_dataframes_coalesce_keeps_a_falsy_value():
    """Verify 0, empty string, and False win a coalesce, never fall through.

    Mutation: testing the winning side for truth rather than for a null,
        which swaps in the right side's 9, 'x', and True wherever the
        left holds a falsy value.
    Oracle: hand-computed 0, '' and False on the left against 9, 'x' and
        True on the right.
    """
    left = pd.DataFrame({'key': [1], 'n': [0], 's': [''], 'f': [False]})
    right = pd.DataFrame({'key': [1], 'n': [9], 's': ['x'], 'f': [True]})

    frame = join_dataframes(left, 'key', right, 'key')

    assert frame_rows(frame, ['n', 's', 'f']) == [(0, '', False)]


def test_join_dataframes_suffixes_every_column_including_the_keys(df_keys_12_bc,
                                                                  df_keys_12_de):
    """Verify lsuffix and rsuffix rename every column on each side.

    Mutation: suffixing only the columns that collide, which leaves one
        bare key and silently merges the two key columns into it.
    Oracle: hand-listed names key_a, b_a, c_a, key_b, d_b, e_b - both
        keys present and every column suffixed.
    """
    frame = join_dataframes(df_keys_12_bc, 'key', df_keys_12_de, 'key',
                            lsuffix='_a', rsuffix='_b')

    assert list(frame.columns) == ['key_a', 'b_a', 'c_a', 'key_b', 'd_b', 'e_b']
    assert frame['key_a'].tolist() == frame['key_b'].tolist() == [1, 2]


def test_join_dataframes_key_left_out_of_lcols_still_matches(df_keys_12_bc,
                                                             df_dup_key1_de):
    """Verify a key excluded by lcols drives the match yet leaves the result.

    Mutation: cutting the columns before the merge, so the key is gone by
        the time pandas needs it and the call raises or matches nothing;
        or keeping the key in the output regardless of lcols.
    Oracle: hand-listed columns b, key, d, e - the left key dropped while
        the right key survives - over the 2 rows the key-1 match yields.
    """
    frame = join_dataframes(df_keys_12_bc, 'key', df_dup_key1_de, 'key',
                            lcols=['b'])

    assert list(frame.columns) == ['b', 'key', 'd', 'e']
    assert frame['b'].tolist() == [1, 1]
    assert frame['e'].tolist() == [3, 4]


def test_join_dataframes_rcols_drops_unlisted_right_columns(df_keys_12_bc,
                                                            df_keys_12_de):
    """Verify rcols restricts which right columns are carried over.

    Mutation: ignoring rcols, which pulls d into the result and widens
        every downstream frame.
    Oracle: hand-listed columns key, b, c, e with d absent, and e
        carrying the right-side values.
    """
    frame = join_dataframes(df_keys_12_bc, 'key', df_keys_12_de, 'key', 'left',
                            rcols=['e'])

    assert list(frame.columns) == ['key', 'b', 'c', 'e']
    assert frame['e'].tolist() == [2, 2]


def test_join_dataframes_reads_a_bare_string_lcols_as_one_column():
    """Verify lcols='id' keeps the id column, not every one-letter name.

    Mutation: testing membership against the string itself, so 'id'
        becomes the character set {i, d} and keeps a column named 'd'
        while dropping 'id'.
    Oracle: hand-listed columns id, k, b - 'd' present in the left frame
        purely to catch the character-wise reading.
    """
    left = pd.DataFrame({'k': [1], 'id': [9], 'd': [7]})
    right = pd.DataFrame({'k': [1], 'b': [3]})

    frame = join_dataframes(left, 'k', right, 'k', lcols='id')

    assert list(frame.columns) == ['id', 'k', 'b']


def test_join_dataframes_orders_columns_left_then_right_only():
    """Verify column order is the left frame's, then the right's newcomers.

    Mutation: taking pandas' merge order, which appends the shared column
        v at the end rather than leaving it in its left position.
    Oracle: hand-listed ['key', 'v', 'a', 'z', 'b'] - the left's three in
        their own order, then only the right columns the left lacks.
    """
    left = pd.DataFrame({'key': [1], 'v': [1], 'a': [1]})
    right = pd.DataFrame({'key': [1], 'z': [2], 'v': [2], 'b': [2]})

    frame = join_dataframes(left, 'key', right, 'key')

    assert list(frame.columns) == ['key', 'v', 'a', 'z', 'b']


def test_join_dataframes_matches_a_null_key_against_a_null_key():
    """Verify null keys pair with each other rather than dropping out.

    Mutation: dropping the null-keyed rows before the merge, as a SQL
        join would, which leaves A_none and B_none unmatched and turns
        one outer row into two.
    Oracle: hand-computed 4 outer rows where the null-key row carries
        both A_none and B_none, against keys 1, 2 and 3.
    """
    left = pd.DataFrame({'key': [1, None, 2], 'va': ['A1', 'A_none', 'A2']})
    right = pd.DataFrame({'key': [1, None, 3], 'vb': ['B1', 'B_none', 'B3']})

    frame = join_dataframes(left, 'key', right, 'key', 'outer')

    assert frame_rows(frame, ['va', 'vb']) == [
        ('A1', 'B1'), ('A_none', 'B_none'), ('A2', None), (None, 'B3')]


def test_join_dataframes_matches_an_object_key_against_a_numeric_one():
    """Verify an all-null object key column joins a float key without raising.

    Mutation: handing both key columns straight to pandas, which refuses
        to merge object against float64 and raises ValueError.
    Oracle: hand-computed 2 left rows kept with no right match under a
        left join, and 2 right rows kept under a right join.
    """
    left = pd.DataFrame({'k': [None, None], 'a': [1, 2]})
    right = pd.DataFrame({'k': [1.0, 2.0], 'b': [10, 20]})
    assert left['k'].dtype == object

    frame = join_dataframes(left, 'k', right, 'k', 'left')

    assert len(frame) == 2
    assert frame['b'].isna().all()
    assert len(join_dataframes(left, 'k', right, 'k', 'right')) == 2


def test_join_dataframes_ignores_the_input_index():
    """Verify a non-default index on either side does not misplace values.

    Mutation: skipping reset_index on the working frames, so pandas
        aligns the key column by label and pairs each row with another
        row's key.
    Oracle: hand-computed pairs p-10, q-20 - the same answer the default
        index gives - against reversed and repeated input indexes.
    """
    left = pd.DataFrame({'k': [1, 2], 'a': ['p', 'q']}, index=[9, 4])
    right = pd.DataFrame({'k': [2, 1], 'b': [20, 10]}, index=[7, 7])

    frame = join_dataframes(left, 'k', right, 'k')

    assert list(frame.index) == [0, 1]
    assert frame_rows(frame, ['a', 'b']) == [('p', 10), ('q', 20)]


def test_join_dataframes_leaves_both_inputs_untouched():
    """Verify neither input frame is reordered, reindexed, or widened.

    Mutation: building the working frames as views and writing the key
        and position helpers into them, which plants __lpos and __seq
        columns in the caller's own data.
    Oracle: differential against copies taken before the call, over the
        outer-with-first path that adds the most helper columns.
    """
    left = pd.DataFrame({'k': [1, 1], 'a': ['p', 'q']}, index=[9, 4])
    right = pd.DataFrame({'k': [1, 2], 'b': [10, 20]})
    before = left.copy(), right.copy()

    join_dataframes(left, 'k', right, 'k', 'outer', first=True)

    assert left.equals(before[0])
    assert right.equals(before[1])


@pytest.mark.parametrize(('how', 'expected_e'), [
    ('inner', [3]),
    ('left', [3, None]),
    ('right', [3, 4]),
    ('outer', [3, None, 4]),
])
def test_join_dataframes_first_pairs_by_position_within_the_key(
        df_keys_12_bc, df_dup_key1_de, how, expected_e):
    """Verify first pairs the nth row of a key group with the nth row.

    Mutation: de-duplicating either side on the key rather than merging
        on a within-key ordinal, which restores the cartesian product for
        outer and keeps the wrong row for right; or taking the last row
        of the group, which surfaces e 4 where the contract says 3.
    Oracle: hand-computed e per join type over a 1-by-2 key-1 group - the
        min, nleft, nright and max rows respectively.
    """
    frame = join_dataframes(df_keys_12_bc, 'key', df_dup_key1_de, 'key', how,
                            first=True)

    assert [None if pd.isna(v) else v for v in frame['e'].tolist()] == expected_e


def test_join_dataframes_first_is_not_the_cartesian_product():
    """Verify first collapses a fan-out both sides share.

    Mutation: dropping the ordinal from the merge keys, which makes first
        a no-op and returns every pairing.
    Oracle: differential - the same outer join without first, whose
        2-by-2 product is 4 rows against first's 2 - plus the
        hand-computed pairs 10-3 and 20-4.
    """
    left = pd.DataFrame({'key': [1, 1], 'b': [10, 20]})
    right = pd.DataFrame({'key': [1, 1], 'd': [3, 4]})

    assert len(join_dataframes(left, 'key', right, 'key', 'outer')) == 4

    frame = join_dataframes(left, 'key', right, 'key', 'outer', first=True)
    assert frame_rows(frame, ['b', 'd']) == [(10, 3), (20, 4)]


def test_join_dataframes_first_warns_with_the_dropped_counts(caplog):
    """Verify first reports how many rows each side lost.

    Mutation: counting result rows rather than distinct source positions,
        which reports 0 dropped because the merge has already collapsed
        them; or losing the warning, leaving a silent data loss.
    Oracle: hand-computed 1 left row and 0 right rows dropped from a
        2-by-1 inner pairing on one key.
    """
    left = pd.DataFrame({'key': [1, 1], 'b': [10, 20]})
    right = pd.DataFrame({'key': [1], 'd': [5]})

    with caplog.at_level(logging.WARNING, logger='rollups.frame'):
        frame = join_dataframes(left, 'key', right, 'key', 'inner', first=True)

    assert len(frame) == 1
    assert 'dropped 1 left and 0 right rows' in caplog.text


def test_join_dataframes_cross_pairs_every_row_with_every_row():
    """Verify how='cross' returns the product, left row major.

    Mutation: mapping cross to inner, which raises for want of a key; or
        building the product right major, which reorders every row.
    Oracle: hand-enumerated 2-by-3 product in the order 1x10, 1x20, 1x30,
        2x10, 2x20, 2x30.
    """
    left = pd.DataFrame({'v': [1, 2]})
    right = pd.DataFrame({'w': [10, 20, 30]})

    frame = join_dataframes(left, None, right, None, 'cross')

    assert frame_rows(frame, ['v', 'w']) == [
        (1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)]


@pytest.mark.parametrize(('kwargs', 'message'), [
    ({'how': 'sideways'}, 'not supported sideways'),
    ({'prefer': 'middle'}, "prefer must be 'left' or 'right'"),
    ({'lkey': None}, "pass how='cross'"),
    ({'lkey': ['key', 'b']}, 'differ in length'),
    ({'rkey': 'nope'}, "right frame has no column 'nope'"),
    ({'lcols': ['key', 'gone']}, "left frame has no column 'gone'"),
    ({'how': 'cross'}, 'takes no key columns'),
])
def test_join_dataframes_rejects_a_malformed_call(df_keys_12_bc, df_keys_12_de,
                                                  kwargs, message):
    """Verify each malformed call raises rather than guessing.

    Mutation: falling back to inner on an unknown how, to 'left' on an
        unknown prefer, or letting a missing column through to a bare
        pandas KeyError - each hides a caller's typo behind plausible
        output or an undiagnosable error. Ignoring a key under cross
        hands a caller who meant an inner join the whole product.
    Oracle: hand-specified ValueError text per malformed argument.
    """
    call = {'lkey': 'key', 'rkey': 'key', 'how': 'inner'} | kwargs
    with pytest.raises(ValueError, match=message):
        join_dataframes(df_keys_12_bc, right=df_keys_12_de, **call)


def test_join_dataframes_rejects_a_frame_that_repeats_a_column_name():
    """Verify a duplicate column label raises rather than fanning out.

    Mutation: letting the duplicate through, after which selecting it
        returns both copies, the merge widens by one column per copy, and
        the result silently carries values under the wrong names.
    Oracle: hand-specified ValueError naming 'v', the label the left
        frame carries twice.
    """
    left = pd.DataFrame([[1, 2, 3]], columns=['key', 'v', 'v'])
    right = pd.DataFrame({'key': [1], 'd': [5]})

    with pytest.raises(ValueError, match="left frame carries 'v' more than once"):
        join_dataframes(left, 'key', right, 'key')


def test_join_dataframes_survives_a_column_named_like_a_merge_suffix():
    """Verify a column already named v__l does not collide with the merge.

    Mutation: fixing the internal merge suffixes at __l and __r without
        checking that the SUFFIXED name is free, which makes pandas raise
        MergeError the moment a caller carries a column of that name
        beside the column it collides with.
    Oracle: hand-listed columns key, v, v__l with v coalesced to the left
        value 10 and v__l left at its own 99.
    """
    left = pd.DataFrame({'key': [1], 'v': [10], 'v__l': [99]})
    right = pd.DataFrame({'key': [1], 'v': [20]})

    frame = join_dataframes(left, 'key', right, 'key')

    assert frame.to_dict('list') == {'key': [1], 'v': [10], 'v__l': [99]}


def test_join_dataframes_rejects_first_under_cross():
    """Verify first raises under cross rather than being ignored.

    Mutation: letting the flag through, where it silently does nothing
        because a cross join has no key groups to pair within - a caller
        who asked for one pairing gets the whole product instead.
    Oracle: hand-specified ValueError against the 4-row product the same
        call returns without first.
    """
    left = pd.DataFrame({'a': [1, 2]})
    right = pd.DataFrame({'b': [3, 4]})

    assert len(join_dataframes(left, None, right, None, 'cross')) == 4
    with pytest.raises(ValueError, match='first pairs rows within a key group'):
        join_dataframes(left, None, right, None, 'cross', first=True)


def test_join_dataframes_reads_the_overlap_after_the_suffix_lands():
    """Verify which columns coalesce is decided on the suffixed names.

    Mutation: computing the overlap from the columns as the caller named
        them rather than as the suffix leaves them, which misses that
        left's suffixed 'rate_a' is the very name the right side already
        carries - the two then survive as a mangled pair instead of
        coalescing. A left column that merely LOOKS suffixed catches the
        opposite defect, matching names by their shape rather than by
        membership of both sides.
    Oracle: hand-listed columns key_a, rate_a, key with rate_a holding
        left's 0.5 and 0.25, against a second join where rate_a belongs
        to the left alone and passes through beside an untouched d.
    """
    left = pd.DataFrame({'key': [1, 2], 'rate': [0.5, 0.25]})
    right = pd.DataFrame({'key': [1, 2], 'rate_a': [None, 9.0]})

    collided = join_dataframes(left, 'key', right, 'key', lsuffix='_a')

    assert list(collided.columns) == ['key_a', 'rate_a', 'key']
    assert collided['rate_a'].tolist() == [0.5, 0.25]

    lone = join_dataframes(pd.DataFrame({'key': [1, 2], 'rate_a': [0.5, 0.25]}),
                           'key', pd.DataFrame({'key': [1, 2], 'd': [5, 6]}), 'key')

    assert list(lone.columns) == ['key', 'rate_a', 'd']
    assert lone['rate_a'].tolist() == [0.5, 0.25]


def test_join_dataframes_coalesces_an_extension_dtype_column():
    """Verify a categorical or nullable column coalesces instead of raising.

    Mutation: coalescing with Series.where alone, which refuses to widen
        a categorical and refuses a value from another dtype family -
        the join then raises TypeError or ValueError on input it is
        supposed to handle.
    Oracle: hand-computed c values x, y, z across an outer join of two
        categorical columns whose categories differ, plus 1 and 'z' from
        a nullable Int64 column coalesced against a string one.
    """
    left = pd.DataFrame({'k': [1, 2], 'c': pd.Series(['x', 'y'], dtype='category')})
    right = pd.DataFrame({'k': [2, 3], 'c': pd.Series(['y', 'z'], dtype='category')})

    assert join_dataframes(left, 'k', right, 'k', 'outer')['c'].tolist() == \
        ['x', 'y', 'z']

    mixed = join_dataframes(
        pd.DataFrame({'k': [1, 2], 'v': pd.array([1, None], dtype='Int64')}), 'k',
        pd.DataFrame({'k': [1, 2], 'v': pd.Series(['y', 'z'], dtype='str')}), 'k')

    assert mixed['v'].tolist() == [1, 'z']


def test_join_dataframes_leaves_a_non_string_column_label_alone():
    """Verify an integer column label survives a join that adds no suffix.

    Mutation: building the output names with an f-string unconditionally,
        which stringifies every label even when no suffix was asked for -
        frame[0] then raises KeyError because the column is named '0'.
    Oracle: hand-listed integer labels [0, 1, 2], read back off the
        result by the same integers the caller passed in.
    """
    left = pd.DataFrame({0: [1, 2], 1: [10, 20]})
    right = pd.DataFrame({0: [1, 2], 2: [30, 40]})

    frame = join_dataframes(left, [0], right, [0])

    assert list(frame.columns) == [0, 1, 2]
    assert frame[1].tolist() == [10, 20]


def test_join_dataframes_rejects_a_repeated_key_the_column_filter_hid():
    """Verify a duplicated key column raises even when lcols drops it.

    Mutation: checking only the columns lcols and rcols kept, which lets
        a duplicated key through to the merge and surfaces an internal
        helper name in the error a caller has to read.
    Oracle: hand-specified message naming 'a', against the internal
        '__lmatch0' the unguarded path leaks.
    """
    left = pd.DataFrame(np.array([[1, 1, 5], [2, 2, 6]]), columns=['a', 'a', 'b'])
    right = pd.DataFrame({'a': [1, 2], 'r': [7, 8]})

    with pytest.raises(ValueError, match="left frame carries 'a' more than once"):
        join_dataframes(left, 'a', right, 'a', lcols=['b'])


def test_join_dataframes_rejects_two_labels_a_suffix_lands_on_one_name():
    """Verify a suffix that stringifies two labels onto one name raises.

    Mutation: checking the labels only as they arrived, so the integer 0
        and the string '0' pass - the suffix then makes both '0_a' and
        the result carries the same values under duplicated columns.
    Oracle: hand-specified message naming '0_a', against the six-column
        frame the unguarded path returns for a three-by-two join.
    """
    left = pd.DataFrame({'k': [1, 2]})
    left[0] = [10, 20]
    left['0'] = ['a', 'b']
    right = pd.DataFrame({'k': [1, 2], 'z': [9, 8]})

    assert list(join_dataframes(left, 'k', right, 'k').columns) == ['k', 0, '0', 'z']

    with pytest.raises(ValueError, match="left frame carries '0_a' more than once"):
        join_dataframes(left, 'k', right, 'k', lsuffix='_a')


def test_join_dataframes_joins_on_two_key_columns():
    """Verify a two-column key matches on the pair, not on either part.

    Mutation: keying on the first column alone, which pairs ('a', 2) with
        ('a', 1) and returns a row the pair-wise match must not.
    Oracle: hand-computed single match on ('a', 1), with ('a', 2) and
        ('b', 1) both unmatched.
    """
    left = pd.DataFrame({'k1': ['a', 'a'], 'k2': [1, 2], 'v': [10, 20]})
    right = pd.DataFrame({'k1': ['a', 'b'], 'k2': [1, 1], 'w': [30, 40]})

    frame = join_dataframes(left, ['k1', 'k2'], right, ['k1', 'k2'])

    assert frame_rows(frame, ['k1', 'k2', 'v', 'w']) == [('a', 1, 10, 30)]


def test_join_dataframes_joins_on_differently_named_keys(df_keys_12_bc,
                                                         df_keys_12_de):
    """Verify differently named keys match and both columns survive.

    Mutation: merging on the left name against itself, which matches key
        to key and gives 2 rows rather than the 3 the key-against-d match
        yields.
    Oracle: hand-computed keys 1, 2, 2 - the left key 1 finds no d of 1
        and comes back null, the left key 2 finds both d values of 2.
    """
    frame = join_dataframes(df_keys_12_bc, 'key', df_keys_12_de, 'd', 'left')

    assert frame['key'].tolist() == [1, 2, 2]
    assert frame['d'].isna().tolist() == [True, False, False]
    assert frame.loc[frame['key'] == 2, 'd'].tolist() == [2.0, 2.0]


def test_join_dataframes_leaves_an_empty_side_with_the_full_column_set():
    """Verify a zero-row side still shapes the result's columns.

    Mutation: returning early on an empty side, which loses that side's
        columns and breaks a caller reading them.
    Oracle: hand-listed columns key, b, key2, d over zero rows for inner
        and one all-null right half for left.
    """
    left = pd.DataFrame({'key': [1], 'b': [2]})
    empty = pd.DataFrame({'key2': pd.Series(dtype='int64'),
                          'd': pd.Series(dtype='int64')})

    inner = join_dataframes(left, 'key', empty, 'key2')
    assert list(inner.columns) == ['key', 'b', 'key2', 'd']
    assert len(inner) == 0

    kept = join_dataframes(left, 'key', empty, 'key2', 'left')
    assert len(kept) == 1
    assert kept['d'].isna().all()


@pytest.mark.parametrize('how', ['inner', 'outer', 'left', 'right'])
@pytest.mark.parametrize('prefer', ['left', 'right'])
def test_join_dataframes_agrees_with_dataset_join(how, prefer):
    """Verify the frame join returns DataSet.join's rows and columns.

    Mutation: any divergence in which rows survive, which columns appear,
        or how a shared column resolves - the frame path silently
        answering something the row path does not.
    Oracle: differential against DataSet.join over rows carrying repeated
        keys, unmatched keys on both sides, a null key, and a shared
        column c that each side populates in turn.
    """
    rows_a = [{'key': 1, 'b': 1, 'c': None}, {'key': 1, 'b': 2, 'c': 5},
              {'key': 2, 'b': 3, 'c': 6}, {'key': None, 'b': 4, 'c': None}]
    rows_b = [{'key': 1, 'c': 7, 'e': 8}, {'key': 3, 'c': None, 'e': 9},
              {'key': None, 'c': 10, 'e': 11}]

    joined = DataSet.join(DataSet([dict(r) for r in rows_a]), 'key',
                          DataSet([dict(r) for r in rows_b]), 'key', how,
                          bfirst=(prefer == 'right'))
    frame = join_dataframes(pd.DataFrame(rows_a), 'key',
                            pd.DataFrame(rows_b), 'key', how, prefer=prefer)

    assert list(frame.columns) == joined.cols
    assert tokens(frame_rows(frame, joined.cols)) == tokens(dataset_rows(joined))


if __name__ == '__main__':
    pytest.main([__file__])
