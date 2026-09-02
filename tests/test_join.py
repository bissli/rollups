"""Tests for DataSet.join and match_rows.

Sample datasets shared by the join tests:

    a                b                c                d
    +-----+---+---+  +-----+---+---+  +-----+---+---+  +-----+---+---+
    | key | b | c |  | key | d | e |  | key | d | e |  | key | x | y |
    +-----+---+---+  +-----+---+---+  +-----+---+---+  +-----+---+---+
    |  1  | 1 | 1 |  |  1  | 2 | 2 |  |  1  | 3 | 3 |  |  2  | 3 | 3 |
    |  2  | 1 | 1 |  |  2  | 2 | 2 |  |  1  | 4 | 4 |  |  2  | 4 | 4 |
    +-----+---+---+  +-----+---+---+  +-----+---+---+  +-----+---+---+

Notes
-----
- inner keeps keys held by both sides, outer keeps keys held by
  either, left keeps the left keys, right keeps the right keys.
- A repeated key fans out into the full cartesian product unless
  first=True, which pairs rows sequentially with pop() - taking from
  the END of each list - and drops the surplus on the longer side.
- An overlapping column takes the left value unless that value is
  None, in which case the right fills in. bfirst reverses the order.
"""
import logging

import pytest
from opendate import UTC, Date, DateTime, Time
from rollups import DataSet, match_rows

# --- Helpers ---


def dataset_rows(ds: DataSet) -> list[tuple]:
    """Rows of `ds` as tuples in column order.
    """
    return [tuple(row[col] for col in ds.cols) for row in ds]


# --- Fixtures ---


@pytest.fixture
def ds_basic_left():
    """Basic left dataset with name/value."""
    ds = DataSet([
        {'name': 'A', 'value': 100},
        {'name': 'B', 'value': 200},
        {'name': 'C', 'value': 300},
        {'name': 'D', 'value': 400},
        {'name': 'E', 'value': 500}])
    ds.columns = (('name', str), ('value', int))
    return ds


@pytest.fixture
def ds_basic_right():
    """Basic right dataset with name/extra."""
    ds = DataSet([
        {'name': 'A', 'extra': 'X'},
        {'name': 'C', 'extra': 'Y'},
        {'name': 'E', 'extra': 'Z'},
        {'name': 'F', 'extra': 'W'}])
    ds.columns = (('name', str), ('extra', str))
    return ds


@pytest.fixture
def ds_keys_12_bc():
    """Two rows, unique keys 1 and 2."""
    return DataSet([{'key': 1, 'b': 1, 'c': 1}, {'key': 2, 'b': 1, 'c': 1}])


@pytest.fixture
def ds_keys_12_de():
    """Two rows, unique keys 1 and 2, disjoint columns."""
    return DataSet([{'key': 1, 'd': 2, 'e': 2}, {'key': 2, 'd': 2, 'e': 2}])


@pytest.fixture
def ds_dup_key1_de():
    """Two rows sharing key 1, so a join against it fans out."""
    return DataSet([{'key': 1, 'd': 3, 'e': 3}, {'key': 1, 'd': 4, 'e': 4}])


@pytest.fixture
def ds_dup_key1_b():
    """Two rows sharing key 1, so the LEFT side of a join fans out."""
    return DataSet([{'key': 1, 'b': 10}, {'key': 1, 'b': 20}])


@pytest.fixture
def ds_dup_key2_xy():
    """Two rows sharing key 2."""
    return DataSet([{'key': 2, 'x': 3, 'y': 3}, {'key': 2, 'x': 4, 'y': 4}])


@pytest.fixture
def left_ds():
    """Left side, holding key 1 which the right side does not."""
    return DataSet([
        {'id': 1, 'x': 'p'},
        {'id': 2, 'x': 'q'},
        {'id': 3, 'x': 'r'},
    ])


@pytest.fixture
def right_ds():
    """Right side, holding key 4 which the left side does not."""
    return DataSet([
        {'id': 2, 'y': 20},
        {'id': 3, 'y': 30},
        {'id': 4, 'y': 40},
    ])


# --- Basic Join Type Tests (Parameterized) ---

@pytest.mark.parametrize(
    ('join_type', 'expected_len', 'expected_names', 'expected_extras'),
    [
        ('inner', 3, ['A', 'C', 'E'], ['X', 'Y', 'Z']),
        ('outer', 6, ['A', 'B', 'C', 'D', 'E', 'F'],
         ['X', None, 'Y', None, 'Z', 'W']),
        ('left', 5, ['A', 'B', 'C', 'D', 'E'],
         ['X', None, 'Y', None, 'Z']),
        ('right', 4, ['A', 'C', 'E', 'F'], ['X', 'Y', 'Z', 'W']),
    ])
def test_join_types(ds_basic_left, ds_basic_right, join_type, expected_len,
                    expected_names, expected_extras):
    """Verify each join type keeps the right keys and fills the gaps.

    Mutation: outer narrowed to the key intersection, left reading
        bdict, or the placeholder row for an unmatched key dropped.
    Oracle: hand-computed names and extras for left A-E against
        right A, C, E, F.
    """
    if join_type == 'inner':
        jd = DataSet.join(ds_basic_left, 'name', ds_basic_right, 'name')
    else:
        jd = DataSet.join(ds_basic_left, 'name', ds_basic_right, 'name', join_type)
    jd.sort_data('name')
    assert len(jd) == expected_len
    assert [r['name'] for r in jd] == expected_names
    assert [r['extra'] for r in jd] == expected_extras


# --- Column Modifier Tests ---

def test_join_with_column_modifiers():
    """Verify amod and bmod keep both sides' columns side by side.

    Mutation: amod used in place of bmod when building the right
        row, so the right side overwrites the left's columns.
    Oracle: hand-computed suffixed pairs, 100/10 and 'A1'/'B1'.
    """
    ds_a = DataSet([
        {'id': 1, 'value': 100, 'name': 'A1'},
        {'id': 2, 'value': 200, 'name': 'A2'},
    ])
    ds_b = DataSet([
        {'id': 1, 'value': 10, 'name': 'B1'},
        {'id': 2, 'value': 20, 'name': 'B2'},
    ])
    result = DataSet.join(ds_a, 'id', ds_b, 'id', amod='_a', bmod='_b')
    result.sort_data('id_a')
    assert [(r.id_a, r.id_b, r.value_a, r.value_b, r.name_a, r.name_b)
            for r in result] == [
        (1, 1, 100, 10, 'A1', 'B1'),
        (2, 2, 200, 20, 'A2', 'B2')]


# --- Multiple Key Column Tests ---

def test_join_multiple_key_columns():
    """Verify a composite key matches only on every key column.

    Mutation: the key tuple truncated to akey[:1], which would also
        pair East/B with West/B on the region alone.
    Oracle: hand-computed - only East/A and West/A sit on both sides.
    """
    ds_a = DataSet([
        {'region': 'East', 'product': 'A', 'sales': 100},
        {'region': 'East', 'product': 'B', 'sales': 150},
        {'region': 'West', 'product': 'A', 'sales': 200},
    ])
    ds_b = DataSet([
        {'region': 'East', 'product': 'A', 'cost': 80},
        {'region': 'West', 'product': 'A', 'cost': 160},
        {'region': 'West', 'product': 'B', 'cost': 120},
    ])
    result = DataSet.join(ds_a, ['region', 'product'], ds_b, ['region', 'product'])
    result.sort_data('region', 'product')
    assert [(r.region, r.product, r.sales, r.cost) for r in result] == [
        ('East', 'A', 100, 80),
        ('West', 'A', 200, 160)]


# --- None Key Tests ---

def test_join_with_none_in_keys():
    """Verify an outer join treats a None key value as joinable.

    Mutation: outer narrowed to the key intersection, which would
        drop the unmatched keys 2 and 3.
    Oracle: hand-computed 4 rows - None, 1, 2, 3 - with the None row
        carrying both sides.
    """
    ds_a = DataSet([
        {'key': 1, 'value_a': 'A1'},
        {'key': None, 'value_a': 'A_none'},
        {'key': 2, 'value_a': 'A2'},
    ])
    ds_b = DataSet([
        {'key': 1, 'value_b': 'B1'},
        {'key': None, 'value_b': 'B_none'},
        {'key': 3, 'value_b': 'B3'},
    ])
    result = DataSet.join(ds_a, 'key', ds_b, 'key', 'outer')
    result.sort_data('key')
    assert [(r.key, r.value_a, r.value_b) for r in result] == [
        (None, 'A_none', 'B_none'),
        (1, 'A1', 'B1'),
        (2, 'A2', None),
        (3, None, 'B3')]


def test_join_none_key_cartesian_product():
    """Verify a None key pairs every left row with every right row.

    Mutation: the `None in akey` guard removed, so a None key falls
        through to per-row keying and matches nothing.
    Oracle: hand-enumerated 2 x 3 product.
    """
    ds_a = DataSet([{'value_a': 1}, {'value_a': 2}])
    ds_b = DataSet([{'value_b': 10}, {'value_b': 20}, {'value_b': 30}])
    result = DataSet.join(ds_a, None, ds_b, None)
    assert sorted((r.value_a, r.value_b) for r in result) == [
        (1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)]
    assert len(ds_a) == 2
    assert len(ds_b) == 3


# --- Duplicate Key and Cartesian Product Tests ---

def test_join_duplicate_keys_cartesian():
    """Verify a repeated key fans out into the full cartesian product.

    Mutation: inner widened to the key union, which would add a row
        for the unmatched key 2.
    Oracle: hand-enumerated 2 x 3 product of A1, A2 against B1-B3.
    """
    ds_a = DataSet([
        {'key': 1, 'value_a': 'A1'},
        {'key': 1, 'value_a': 'A2'},
        {'key': 2, 'value_a': 'A3'},
    ])
    ds_b = DataSet([
        {'key': 1, 'value_b': 'B1'},
        {'key': 1, 'value_b': 'B2'},
        {'key': 1, 'value_b': 'B3'},
    ])
    result = DataSet.join(ds_a, 'key', ds_b, 'key')
    assert sorted((r.key, r.value_a, r.value_b) for r in result) == [
        (1, 'A1', 'B1'), (1, 'A1', 'B2'), (1, 'A1', 'B3'),
        (1, 'A2', 'B1'), (1, 'A2', 'B2'), (1, 'A2', 'B3')]


def test_join_outer_with_multiple_duplicate_keys(ds_dup_key1_de, ds_dup_key2_xy):
    """Verify an outer join keeps every duplicate row from both sides.

    Mutation: outer narrowed to the key intersection, or the
        placeholder row for an unmatched key dropped.
    Oracle: hand-computed - key 1 rows carry d/e and no x/y, key 2
        rows carry x/y and no d/e.
    """
    result = DataSet.join(ds_dup_key1_de, 'key', ds_dup_key2_xy, 'key', 'outer')
    assert [(r.key, r.d, r.e, r.x, r.y) for r in result] == [
        (1, 3, 3, None, None),
        (1, 4, 4, None, None),
        (2, None, None, 3, 3),
        (2, None, None, 4, 4)]


# --- bfirst Parameter Tests ---

def test_join_with_bfirst_parameter():
    """Verify bfirst takes the right value unless that value is None.

    Mutation: the bfirst branch ignored, so the left always wins.
    Oracle: hand-computed - row 1 takes 150/'completed' from the
        right, row 2 keeps 200 because the right value is None.
    """
    ds_a = DataSet([
        {'id': 1, 'value': 100, 'status': 'pending'},
        {'id': 2, 'value': 200, 'status': 'pending'},
    ])
    ds_b = DataSet([
        {'id': 1, 'value': 150, 'status': 'completed'},
        {'id': 2, 'value': None, 'status': 'cancelled'},
    ])
    result = DataSet.join(ds_a, 'id', ds_b, 'id', bfirst=True)
    result.sort_data('id')
    assert [(r.id, r.value, r.status) for r in result] == [
        (1, 150, 'completed'),
        (2, 200, 'cancelled')]


def test_join_overlapping_columns_default_priority():
    """Verify the left value wins unless the left value is None.

    Mutation: the default flipped to right-first, or the None
        fallback dropped so a None left value survives.
    Oracle: hand-computed - row 1 keeps 'A1'/100, row 2 falls back
        to 'B2'/20.
    """
    ds_a = DataSet([
        {'id': 1, 'value': 'A1', 'shared': 100},
        {'id': 2, 'value': None, 'shared': None},
    ])
    ds_b = DataSet([
        {'id': 1, 'value': 'B1', 'shared': 10},
        {'id': 2, 'value': 'B2', 'shared': 20},
    ])
    result = DataSet.join(ds_a, 'id', ds_b, 'id')
    result.sort_data('id')
    assert [(r.id, r.value, r.shared) for r in result] == [
        (1, 'A1', 100),
        (2, 'B2', 20)]


# --- Column Filtering Tests ---

def test_join_column_filtering_both_sides():
    """Verify acol and bcol drop the columns they leave out.

    Mutation: the `if k in acol` filter dropped, letting age and
        city through into the joined schema.
    Oracle: hand-listed column set {id, name, dept}.
    """
    ds_a = DataSet([
        {'id': 1, 'name': 'A', 'age': 30, 'city': 'NYC'},
        {'id': 2, 'name': 'B', 'age': 25, 'city': 'LA'},
    ])
    ds_b = DataSet([
        {'id': 1, 'dept': 'DeptA', 'level': 50000, 'extra': 5000},
        {'id': 2, 'dept': 'DeptB', 'level': 60000, 'extra': 6000},
    ])
    result = DataSet.join(
        ds_a, 'id', ds_b, 'id',
        acol=['id', 'name'],
        bcol=['id', 'dept'])
    result.sort_data('id')
    assert set(result.cols) == {'id', 'name', 'dept'}
    assert [(r.id, r.name, r.dept) for r in result] == [
        (1, 'A', 'DeptA'),
        (2, 'B', 'DeptB')]
    assert 'age' not in result[0]
    assert 'level' not in result[0]


# --- Empty Dataset Tests ---

def test_join_empty_datasets():
    """Verify an empty side yields no inner rows but keeps left rows.

    Mutation: inner widened to the key union, or left reading bdict,
        either of which changes one of the two counts.
    Oracle: hand-computed 0 inner rows, and 1 left row whose right
        columns are None.
    """
    ds_empty = DataSet([], columns=[('id', int), ('value', str)])
    ds_with_data = DataSet([{'id': 1, 'count': 100}])
    result = DataSet.join(ds_empty, 'id', ds_with_data, 'id')
    assert len(result) == 0
    result = DataSet.join(ds_with_data, 'id', ds_empty, 'id', 'left')
    assert len(result) == 1
    assert result[0].count == 100
    assert result[0].value is None


# --- Type Preservation Tests ---

def test_join_type_preservation():
    """Verify the joined schema keeps each side's declared types.

    Mutation: the bcols merge loop dropped, so a column held only by
        the right side never reaches the joined schema.
    Oracle: the types declared on the two source datasets.
    """
    ds_a = DataSet([
        {'id': 1, 'date': Date(2024, 1, 1), 'count': 100},
        {'id': 2, 'date': Date(2024, 1, 2), 'count': 200},
    ])
    ds_a.columns = [('id', int), ('date', Date), ('count', int)]
    ds_b = DataSet([
        {'id': 1, 'timestamp': DateTime(2024, 1, 1, 10, 0, 0), 'rate': 50.5},
        {'id': 2, 'timestamp': DateTime(2024, 1, 2, 11, 0, 0), 'rate': 55.5},
    ])
    ds_b.columns = [('id', int), ('timestamp', DateTime), ('rate', float)]
    result = DataSet.join(ds_a, 'id', ds_b, 'id')
    assert result.colmap == {'id': int, 'date': Date, 'count': int,
                             'timestamp': DateTime, 'rate': float}


def test_join_column_ordering():
    """Verify joined columns run left side first, then right.

    Mutation: jcols_dict seeded from bcols so the right side leads.
    Oracle: hand-listed order ['id', 'a1', 'a2', 'b1', 'b2'].
    """
    ds_a = DataSet([{'id': 1, 'a1': 1, 'a2': 2}])
    ds_a.columns = [('id', int), ('a1', int), ('a2', int)]
    ds_b = DataSet([{'id': 1, 'b1': 10, 'b2': 20}])
    ds_b.columns = [('id', int), ('b1', int), ('b2', int)]
    result = DataSet.join(ds_a, 'id', ds_b, 'id')
    assert result.cols == ['id', 'a1', 'a2', 'b1', 'b2']


# --- first=True Parameter Tests ---

def test_join_with_first_limits_to_one_match():
    """Verify first=True emits one inner row, pairing from the end.

    Mutation: pop() changed to pop(0), or the inner break removed so
        the surplus left rows are emitted too.
    Oracle: hand-computed - the last row of each side, seq_a 3 with
        seq_b 2.
    """
    ds_a = DataSet([
        {'key': 1, 'seq_a': 1, 'value_a': 10},
        {'key': 1, 'seq_a': 2, 'value_a': 20},
        {'key': 1, 'seq_a': 3, 'value_a': 30},
    ])
    ds_b = DataSet([
        {'key': 1, 'seq_b': 1, 'value_b': 100},
        {'key': 1, 'seq_b': 2, 'value_b': 200},
    ])
    result = DataSet.join(ds_a, 'key', ds_b, 'key', first=True)
    assert [(r.key, r.seq_a, r.seq_b, r.value_a, r.value_b) for r in result] == [
        (1, 3, 2, 30, 200)]


def test_join_left_with_first_preserves_left_rows():
    """Verify a left join with first=True keeps every left key once.

    Mutation: pop() changed to pop(0), giving 'B1' instead of 'B2',
        or the left break testing brows instead of arows.
    Oracle: hand-computed - B2 pairs with key 1, keys 2 and 3 carry
        None.
    """
    ds_a = DataSet([
        {'key': 1, 'value_a': 'A1'},
        {'key': 2, 'value_a': 'A2'},
        {'key': 3, 'value_a': 'A3'},
    ])
    ds_b = DataSet([
        {'key': 1, 'value_b': 'B1'},
        {'key': 1, 'value_b': 'B2'},
    ])
    result = DataSet.join(ds_a, 'key', ds_b, 'key', 'left', first=True)
    result.sort_data('key')
    assert [(r.key, r.value_a, r.value_b) for r in result] == [
        (1, 'A1', 'B2'),
        (2, 'A2', None),
        (3, 'A3', None)]


def test_join_outer_with_first_includes_all_keys():
    """Verify an outer join with first=True emits every surplus row.

    Mutation: the `while arows or brows` loop narrowed to `and`, or
        pop() changed to pop(0).
    Oracle: hand-computed 4 rows, including key 1's leftover A1 row
        with no right value.
    """
    ds_a = DataSet([
        {'key': 1, 'value_a': 'A1'},
        {'key': 1, 'value_a': 'A2'},
        {'key': 2, 'value_a': 'A3'},
    ])
    ds_b = DataSet([
        {'key': 1, 'value_b': 'B1'},
        {'key': 3, 'value_b': 'B2'},
    ])
    result = DataSet.join(ds_a, 'key', ds_b, 'key', 'outer', first=True)
    result.sort_data('key')
    assert [(r.key, r.value_a, r.value_b) for r in result] == [
        (1, 'A2', 'B1'),
        (1, 'A1', None),
        (2, 'A3', None),
        (3, None, 'B2')]


# --- Self Join Tests ---

def test_join_self_join():
    """Verify a self join links each child row to its parent row.

    Mutation: amod used in place of bmod when building the right
        row, collapsing name_parent onto name.
    Oracle: hand-computed - both children map to 'Root', and the
        root row itself has no parent.
    """
    ds = DataSet([
        {'id': 1, 'parent_id': None, 'name': 'Root'},
        {'id': 2, 'parent_id': 1, 'name': 'Child1'},
        {'id': 3, 'parent_id': 1, 'name': 'Child2'},
    ])
    result = DataSet.join(ds, 'parent_id', ds, 'id', 'left',
                          amod='', bmod='_parent',
                          acol=['id', 'parent_id', 'name'],
                          bcol=['id', 'name'])
    result.sort_data('id')
    assert [(r.id, r.name, r.id_parent, r.name_parent) for r in result] == [
        (1, 'Root', None, None),
        (2, 'Child1', 1, 'Root'),
        (3, 'Child2', 1, 'Root')]


# --- Join Behavior Tests ---

def test_join_left_with_first_two_duplicates_in_right(ds_keys_12_bc, ds_dup_key1_de):
    """Verify a left join with first=True drops the surplus right rows.

    Mutation: pop() changed to pop(0), which keeps d=3 rather than
        d=4, or the left break testing brows instead of arows.
    Oracle: hand-computed - key 1 keeps only c's last row, key 2 has
        no right match.
    """
    result = DataSet.join(
        ds_keys_12_bc, 'key', ds_dup_key1_de, 'key',
        'left', first=True)
    assert [(r.key, r.b, r.c, r.d, r.e) for r in result] == [
        (1, 1, 1, 4, 4),
        (2, 1, 1, None, None)]


def test_join_outer_with_first_extra_row_for_unmatched(ds_keys_12_bc, ds_dup_key1_de):
    """Verify an outer join with first=True emits the surplus right row.

    Mutation: the `while arows or brows` loop narrowed to `and`,
        which would drop c's leftover d=3 row.
    Oracle: hand-computed 3 rows - the surplus row carries only the
        right side's values.
    """
    result = DataSet.join(
        ds_keys_12_bc, 'key', ds_dup_key1_de, 'key',
        'outer', first=True)
    assert [(r.key, r.b, r.c, r.d, r.e) for r in result] == [
        (1, 1, 1, 4, 4),
        (1, None, None, 3, 3),
        (2, 1, 1, None, None)]


def test_join_outer_with_first_no_dropped_rows(ds_keys_12_bc, ds_keys_12_de):
    """Verify first=True drops nothing when neither side repeats a key.

    Mutation: the merge walking only the left row's keys, which
        leaves d and e out of every joined row.
    Oracle: hand-computed 2 rows, each carrying all five columns.
    """
    result = DataSet.join(
        ds_keys_12_bc, 'key', ds_keys_12_de, 'key',
        'outer', first=True)
    assert [(r.key, r.b, r.c, r.d, r.e) for r in result] == [
        (1, 1, 1, 2, 2),
        (2, 1, 1, 2, 2)]


def test_join_left_with_none_values_in_right(ds_keys_12_bc):
    """Verify left join(a, e) matches the mirror right join(e, a).

    Mutation: the None fallback dropped, so the mirror join keeps
        e's None values and the two containers diverge.
    Oracle: the mirrored join, plus hand-computed b=1 and c=1 on
        key 1.
    """
    e = DataSet([
        {'key': 1, 'b': None, 'c': 1},
        {'key': 2, 'b': 1, 'c': None}
    ])
    left_result = DataSet.join(ds_keys_12_bc, 'key', e, 'key', 'left')
    right_result = DataSet.join(e, 'key', ds_keys_12_bc, 'key', 'right')
    assert left_result.container == right_result.container
    left_result.sort_data('key')
    assert left_result[0].b == 1
    assert left_result[0].c == 1


def test_join_left_with_non_matching_key_columns(ds_keys_12_bc, ds_keys_12_de):
    """Verify a left join on unrelated key columns still fans out.

    Mutation: left reading bdict, or the placeholder row for an
        unmatched key dropped.
    Oracle: hand-computed - a's key 1 finds no b row with d=1, and
        a's key 2 matches both b rows with d=2.
    """
    result = DataSet.join(ds_keys_12_bc, 'key', ds_keys_12_de, 'd', 'left')
    assert [(r.key, r.d, r.e) for r in result] == [
        (1, None, None),
        (2, 2, 2),
        (2, 2, 2)]


# --- Type Deduplication Tests (Parameterized) ---

@pytest.mark.parametrize(
    ('a_type', 'b_type', 'expected_type', 'expected_value', 'bfirst'),
    [
        (type(None), float, float, 50.5, False),
        (float, type(None), float, 100.0, False),
        (type(None), type(None), type(None), None, False),
        (int, float, int, 100, False),
        (int, float, float, 50.5, True),
        (str, int, str, 'text', False),
        (float, type(None), float, 100.0, True),
        ])
def test_join_type_deduplication_parameterized(a_type, b_type, expected_type,
                                               expected_value, bfirst):
    """Verify an overlapping column resolves to one type and one value.

    Mutation: the NoneType check reading `typ` rather than
        `existing_typ`, or the bfirst branch losing its
        `typ is not NoneType` guard.
    Oracle: hand-computed type and value for each pairing.
    """
    a_value = (
        None if a_type is type(None)
        else (100 if a_type in {int, float} else 'text'))
    b_value = (
        None if b_type is type(None)
        else (50.5 if b_type == float
              else (50 if b_type == int else 'other')))

    ds_a = DataSet([{'id': 1, 'value': a_value}])
    ds_a.columns = [('id', int), ('value', a_type)]
    ds_b = DataSet([{'id': 1, 'value': b_value}])
    ds_b.columns = [('id', int), ('value', b_type)]

    result = DataSet.join(ds_a, 'id', ds_b, 'id', bfirst=bfirst)

    column_names = [name for name, _ in result.columns]
    assert column_names.count('value') == 1
    assert result.colmap['value'] is expected_type
    assert result[0].value == expected_value


def test_join_type_deduplication_nonetype_vs_concrete():
    """Verify a NoneType column is promoted to the other side's type.

    Mutation: the NoneType check reading `typ` rather than
        `existing_typ`, leaving metric typed NoneType.
    Oracle: the float declared on the right, plus its value 50.5.
    """
    ds_a = DataSet([
        {'id': 1, 'metric': None, 'value_a': 100},
        {'id': 2, 'metric': None, 'value_a': 200},
    ])
    ds_a.columns = [('id', int), ('metric', type(None)), ('value_a', int)]

    ds_b = DataSet([
        {'id': 1, 'metric': 50.5, 'value_b': 10},
        {'id': 2, 'metric': 55.25, 'value_b': 20},
    ])
    ds_b.columns = [('id', int), ('metric', float), ('value_b', int)]

    result = DataSet.join(ds_a, 'id', ds_b, 'id')
    result.sort_data('id')

    assert len(result.columns) == 4
    column_names = [name for name, _ in result.columns]
    assert column_names.count('metric') == 1
    assert result.colmap['metric'] == float
    assert [r.metric for r in result] == [50.5, 55.25]


def test_join_type_deduplication_multiple_columns():
    """Verify each overlapping column resolves its type on its own.

    Mutation: the bfirst guard inverted, or jcols_dict seeded from
        bcols, either of which retypes col2 as float.
    Oracle: hand-declared types - col1 promotes NoneType to float,
        col2 keeps the left's int - and the values that follow.
    """
    ds_a = DataSet([
        {'id': 1, 'col1': None, 'col2': 100, 'col3': None, 'col4': 'text'},
    ])
    ds_a.columns = [('id', int), ('col1', type(None)), ('col2', int),
                    ('col3', type(None)), ('col4', str)]

    ds_b = DataSet([
        {'id': 1, 'col1': 50.5, 'col2': 99.5, 'col3': None, 'col4': 'other'},
    ])
    ds_b.columns = [('id', int), ('col1', float), ('col2', float),
                    ('col3', type(None)), ('col4', str)]

    result = DataSet.join(ds_a, 'id', ds_b, 'id')

    column_names = [name for name, _ in result.columns]
    assert column_names.count('col1') == 1
    assert column_names.count('col2') == 1
    assert result.colmap['col1'] == float
    assert result.colmap['col2'] == int
    assert (result[0].col1, result[0].col2) == (50.5, 100)
    assert (result[0].col3, result[0].col4) == (None, 'text')


def test_join_type_deduplication_with_modifiers():
    """Verify a modifier suffix keeps the two sides' columns apart.

    Mutation: the modifier left off the joined column names, or the
        right row built with amod, either of which merges rate_a and
        rate_b into one promoted column.
    Oracle: hand-declared types - rate_a stays NoneType, rate_b is
        float - and both values survive side by side.
    """
    ds_a = DataSet([{'id': 1, 'rate': None, 'value': 100}])
    ds_a.columns = [('id', int), ('rate', type(None)), ('value', int)]
    ds_b = DataSet([{'id': 1, 'rate': 50.5, 'value': 200}])
    ds_b.columns = [('id', int), ('rate', float), ('value', int)]

    result = DataSet.join(ds_a, 'id', ds_b, 'id', amod='_a', bmod='_b')

    assert result.colmap['rate_a'] is type(None)
    assert result.colmap['rate_b'] is float
    assert result[0].rate_a is None
    assert result[0].rate_b == 50.5
    assert (result[0].value_a, result[0].value_b) == (100, 200)


@pytest.mark.parametrize('join_type', ['outer', 'left', 'right'])
def test_join_type_deduplication_all_join_types(join_type):
    """Verify type promotion holds for every join type.

    Mutation: the NoneType check reading `typ` rather than
        `existing_typ`, leaving metric typed NoneType.
    Oracle: the float declared on the right side.
    """
    ds_a = DataSet([{'id': 1, 'metric': None}, {'id': 2, 'metric': None}])
    ds_a.columns = [('id', int), ('metric', type(None))]
    ds_b = DataSet([{'id': 2, 'metric': 100.5}, {'id': 3, 'metric': 200.5}])
    ds_b.columns = [('id', int), ('metric', float)]

    result = DataSet.join(ds_a, 'id', ds_b, 'id', join_type)

    column_names = [name for name, _ in result.columns]
    assert column_names.count('metric') == 1
    assert result.colmap['metric'] == float


def test_join_type_deduplication_complex_types():
    """Verify promotion also lifts NoneType to Date and to Time.

    Mutation: the NoneType check reading `typ` rather than
        `existing_typ`, leaving both columns typed NoneType.
    Oracle: the Date and Time types declared on the right, plus
        their values.
    """
    ds_a = DataSet([{'id': 1, 'date_col': None, 'time_col': None}])
    ds_a.columns = [('id', int), ('date_col', type(None)), ('time_col', type(None))]
    ds_b = DataSet([{
        'id': 1,
        'date_col': Date(2024, 1, 1),
        'time_col': Time(10, 30, 0, tzinfo=UTC),
    }])
    ds_b.columns = [('id', int), ('date_col', Date), ('time_col', Time)]

    result = DataSet.join(ds_a, 'id', ds_b, 'id')

    assert result.colmap['date_col'] == Date
    assert result.colmap['time_col'] == Time
    assert result[0].date_col == Date(2024, 1, 1)
    assert result[0].time_col == Time(10, 30, 0, tzinfo=UTC)


# --- Edge Case Tests ---

def test_join_single_row_datasets():
    """Verify a one-row inner join carries both sides' values.

    Mutation: the merge walking only the left row's keys, which
        drops value_b.
    Oracle: hand-computed 100 and 200.
    """
    ds_a = DataSet([{'id': 1, 'value_a': 100}])
    ds_b = DataSet([{'id': 1, 'value_b': 200}])
    result = DataSet.join(ds_a, 'id', ds_b, 'id')
    assert len(result) == 1
    assert result[0].value_a == 100
    assert result[0].value_b == 200


def test_join_empty_string_keys():
    """Verify a falsy key or value is not read as a missing one.

    Mutation: the merge testing `_aval` for truth instead of
        `is not None`, which would take the right's 99 over the
        left's 0.
    Oracle: hand-computed - the '' key matches and the left 0 wins.
    """
    ds_a = DataSet([{'key': '', 'value': 0, 'note': 'left'}])
    ds_b = DataSet([{'key': '', 'value': 99, 'note': 'right'}])
    result = DataSet.join(ds_a, 'key', ds_b, 'key')
    assert len(result) == 1
    assert result[0].key == ''
    assert result[0].value == 0
    assert result[0].note == 'left'


def test_join_many_keys():
    """Verify a large inner join keeps exactly the overlapping ids.

    Mutation: inner widened to the key union, or rows paired by
        position rather than by key.
    Oracle: hand-computed - ids 50..99, each row holding id * 10 on
        the left and id * 100 on the right.
    """
    ds_a = DataSet([{'id': i, 'value_a': i * 10} for i in range(100)])
    ds_b = DataSet([{'id': i, 'value_b': i * 100} for i in range(50, 150)])
    result = DataSet.join(ds_a, 'id', ds_b, 'id')
    assert sorted(r.id for r in result) == list(range(50, 100))
    assert all(r.value_a == r.id * 10 for r in result)
    assert all(r.value_b == r.id * 100 for r in result)


def test_join_preserves_column_with_summary():
    """Verify a declared summary neither joins nor carries over.

    Mutation: the joined dataset inheriting adataset._summary_args,
        so its summary totals only the left side's columns.
    Oracle: hand-computed totals - 300 down value, 30 down extra.
    """
    ds_a = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200}
    ])
    ds_a.columns = [('id', int), ('value', int)]
    ds_a.add_summary_row()

    ds_b = DataSet([
        {'id': 1, 'extra': 10},
        {'id': 2, 'extra': 20}
    ])

    result = DataSet.join(ds_a, 'id', ds_b, 'id')
    assert len(result) == 2
    assert result.summary.value == 300
    assert result.summary.extra == 30
    assert ds_a.summary.value == 300


def test_join_different_key_column_names():
    """Verify a join pairs rows across differently named key columns.

    Mutation: bdict keyed by akey, which cannot read the right rows.
    Oracle: hand-computed pairs - 1 with 10, 2 with 20 - with both
        key columns kept.
    """
    ds_a = DataSet([
        {'id_a': 1, 'value_a': 100},
        {'id_a': 2, 'value_a': 200}
    ])
    ds_b = DataSet([
        {'id_b': 1, 'value_b': 10},
        {'id_b': 2, 'value_b': 20}
    ])
    result = DataSet.join(ds_a, 'id_a', ds_b, 'id_b')
    result.sort_data('id_a')
    assert [(r.id_a, r.id_b, r.value_a, r.value_b) for r in result] == [
        (1, 1, 100, 10),
        (2, 2, 200, 20)]


def test_join_float_keys():
    """Verify float key equality alone decides the match.

    Mutation: inner widened to the key union, which would add the
        unmatched 2.5 and 3.5 rows.
    Oracle: hand-computed - only 1.5 sits on both sides.
    """
    ds_a = DataSet([
        {'key': 1.5, 'value_a': 100},
        {'key': 2.5, 'value_a': 200}
    ])
    ds_a.columns = [('key', float), ('value_a', int)]
    ds_b = DataSet([
        {'key': 1.5, 'value_b': 10},
        {'key': 3.5, 'value_b': 30}
    ])
    ds_b.columns = [('key', float), ('value_b', int)]
    result = DataSet.join(ds_a, 'key', ds_b, 'key')
    assert [(r.key, r.value_a, r.value_b) for r in result] == [(1.5, 100, 10)]


def test_join_date_keys():
    """Verify Date key equality alone decides the match.

    Mutation: inner widened to the key union, which would add the
        unmatched 2024-06-30 and 2024-12-31 rows.
    Oracle: hand-computed - only 2024-01-15 sits on both sides.
    """
    ds_a = DataSet([
        {'date': Date(2024, 1, 15), 'value_a': 100},
        {'date': Date(2024, 6, 30), 'value_a': 200}
    ])
    ds_a.columns = [('date', Date), ('value_a', int)]
    ds_b = DataSet([
        {'date': Date(2024, 1, 15), 'value_b': 10},
        {'date': Date(2024, 12, 31), 'value_b': 30}
    ])
    ds_b.columns = [('date', Date), ('value_b', int)]
    result = DataSet.join(ds_a, 'date', ds_b, 'date')
    assert [(r.date, r.value_a, r.value_b) for r in result] == [
        (Date(2024, 1, 15), 100, 10)]


def test_join_mismatched_key_lengths_no_match():
    """Verify keys of different arity never match.

    Mutation: the key tuple truncated to akey[:1], which would let
        the two-column left key match the one-column right key.
    Oracle: hand-computed - a left join keeps both left rows and
        leaves baz None.
    """
    A = DataSet([
        {'foo': 'x', 'bar': 1, 'date': Date(2024, 1, 1)},
        {'foo': 'y', 'bar': 2, 'date': Date(2024, 1, 2)},
    ], cols=['foo', 'bar', 'date'], typs=[str, int, Date])

    B = DataSet([
        {'foo': 'x', 'baz': 100},
        {'foo': 'y', 'baz': 200},
    ], cols=['foo', 'baz'], typs=[str, int])

    result = DataSet.join(A, ['foo', 'date'], B, ['foo'], jointype='left')

    assert len(result) == 2
    for row in result:
        assert row.get('baz') is None, 'baz should be None due to key length mismatch'


def test_join_matched_single_key_works():
    """Verify a left join on one shared key pairs every row.

    Mutation: the merge walking only the left row's keys, which
        drops baz.
    Oracle: hand-computed - x with 100, y with 200.
    """
    A = DataSet([
        {'foo': 'x', 'bar': 1, 'date': Date(2024, 1, 1)},
        {'foo': 'y', 'bar': 2, 'date': Date(2024, 1, 2)},
    ], cols=['foo', 'bar', 'date'], typs=[str, int, Date])

    B = DataSet([
        {'foo': 'x', 'baz': 100},
        {'foo': 'y', 'baz': 200},
    ], cols=['foo', 'baz'], typs=[str, int])

    result = DataSet.join(A, ['foo'], B, ['foo'], jointype='left')
    result.sort_data('foo')

    assert len(result) == 2
    assert result[0]['baz'] == 100
    assert result[1]['baz'] == 200


# --- match_rows tests ---

def test_match_rows_three_way_split():
    """Verify rows split into a-only, b-only, and matched pairs.

    Mutation: inverting `if not b`, which routes matched rows to onlya
        and leaves inboth empty.
    Oracle: hand-computed split of keys {1,2} against {2,3}.
    """
    arows = [{'k': 1, 'v': 'a1'}, {'k': 2, 'v': 'a2'}]
    brows = [{'k': 2, 'v': 'b2'}, {'k': 3, 'v': 'b3'}]
    key = lambda r: r['k']

    onlya, onlyb, inboth = match_rows(arows, key, brows, key)

    assert [r['v'] for r in onlya] == ['a1']
    assert [r['v'] for r in onlyb] == ['b3']
    assert [(a['v'], b['v']) for a, b in inboth] == [('a2', 'b2')]


def test_match_rows_last_row_wins_on_repeated_key():
    """Verify a repeated key keeps the last row on that side.

    Mutation: building amap first-wins, e.g. via setdefault.
    Oracle: hand-computed - of a1/a2 sharing key 1, a2 is the survivor.
    """
    arows = [{'k': 1, 'v': 'a1'}, {'k': 1, 'v': 'a2'}]
    brows = [{'k': 1, 'v': 'b1'}]
    key = lambda r: r['k']

    _, _, inboth = match_rows(arows, key, brows, key)

    assert [(a['v'], b['v']) for a, b in inboth] == [('a2', 'b1')]


def test_match_rows_walks_keys_in_sorted_order():
    """Verify matched pairs come back ordered by key, not by input order.

    Mutation: iterating amap.keys() unsorted, which preserves the 3,1,2
        insertion order.
    Oracle: hand-computed [1, 2, 3] against the shuffled input.
    """
    arows = [{'k': 3}, {'k': 1}, {'k': 2}]
    brows = [{'k': 2}, {'k': 3}, {'k': 1}]
    key = lambda r: r['k']

    _, _, inboth = match_rows(arows, key, brows, key)

    assert [a['k'] for a, _ in inboth] == [1, 2, 3]


# --- join warning and control-flow tests ---

def test_join_rejects_unknown_jointype(ds_keys_12_bc, ds_keys_12_de):
    """Verify an unsupported join type raises and names itself.

    Mutation: raising a bare ValueError with no message, which leaves a
        caller's typo undiagnosable.
    Oracle: hand-specified message text for a name outside the four.
    """
    with pytest.raises(ValueError, match='join type is not supported sideways'):
        DataSet.join(ds_keys_12_bc, 'key', ds_keys_12_de, 'key', 'sideways')


def test_join_first_inner_warns_when_one_side_has_the_surplus(ds_dup_key1_b,
                                                              caplog):
    """Verify an inner join with first warns about the rows it discards.

    Mutation: requiring BOTH sides to hold surplus rows before warning,
        which silences the common case of one side fanning out; or
        losing the key from the message, leaving the caller nothing to
        chase.
    Oracle: hand-computed 1 row out of a 2-by-1 pairing, and the key
        (1,) named in the warning.
    """
    b = DataSet([{'key': 1, 'd': 5}])

    with caplog.at_level(logging.WARNING, logger='rollups.join'):
        j = DataSet.join(ds_dup_key1_b, 'key', b, 'key', 'inner', first=True)

    assert len(j) == 1
    assert 'Dropped rows for key (1,)' in caplog.text


def test_join_first_left_warns_about_the_dropped_right_rows(caplog):
    """Verify a left join with first names the key whose right rows go.

    Mutation: losing the key from the message, which reduces the
        warning to noise a caller cannot act on.
    Oracle: hand-computed 1 row out of a 1-by-2 pairing, and the key
        (1,) named in the warning.
    """
    a = DataSet([{'key': 1, 'b': 10}])
    b = DataSet([{'key': 1, 'd': 5}, {'key': 1, 'd': 6}])

    with caplog.at_level(logging.WARNING, logger='rollups.join'):
        j = DataSet.join(a, 'key', b, 'key', 'left', first=True)

    assert len(j) == 1
    assert 'Dropped brows for key (1,)' in caplog.text


def test_join_first_right_stops_once_the_right_rows_run_out(ds_dup_key1_b,
                                                            caplog):
    """Verify a right join with first emits one row and warns once.

    Mutation: never matching the 'right' branch, inverting its
        exhausted-right test, or returning out of the loop instead of
        breaking - the first two let the surplus left row through as a
        second, half-empty row, the third returns None.
    Oracle: hand-computed single row (1, 20, 5) - pop() takes the last
        left row - and the key (1,) named in the warning.
    """
    b = DataSet([{'key': 1, 'd': 5}])

    with caplog.at_level(logging.WARNING, logger='rollups.join'):
        j = DataSet.join(ds_dup_key1_b, 'key', b, 'key', 'right', first=True)

    assert j is not None
    assert dataset_rows(j) == [(1, 20, 5)]
    assert 'Dropped arows for key (1,)' in caplog.text


if __name__ == '__main__':
    pytest.main([__file__])


class _Sub(DataSet):
    """Subclass carrying a marker a plain DataSet cannot fake."""

    marker = 'sub'


def test_join_result_takes_the_class_the_method_was_reached_through():
    """Verify join returns cls, so a subclass survives and a base narrows.

    Mutation: the result built as a bare `DataSet(columns=jcols)`, so
    `_Sub.join` silently degrades to the base class.
    Oracle: exact types both ways -- reaching through the subclass gives
    the subclass, reaching through the base gives the base, for the same
    two inputs.
    """
    a = _Sub([{'id': 1, 'x': 10}])
    a.columns = [('id', int), ('x', int)]
    b = _Sub([{'id': 1, 'y': 20}])
    b.columns = [('id', int), ('y', int)]

    assert type(_Sub.join(a, 'id', b, 'id')) is _Sub
    assert type(DataSet.join(a, 'id', b, 'id')) is DataSet


def test_join_datasets_is_reachable_standalone_and_matches_the_method():
    """Verify the free function is exported and agrees with the method.

    Mutation: join_datasets left out of __all__, leaving a one-caller
    module helper; or the delegate passing arguments in an order the
    free function does not expect, so the two paths disagree.
    Oracle: a hand-written joined row, asserted against both paths.
    """
    import rollups as pkg

    a = DataSet([{'id': 1, 'x': 10}])
    a.columns = [('id', int), ('x', int)]
    b = DataSet([{'id': 1, 'y': 20}])
    b.columns = [('id', int), ('y', int)]

    assert 'join_datasets' in pkg.__all__

    direct = pkg.join_datasets(a, 'id', b, 'id')
    method = DataSet.join(a, 'id', b, 'id')

    assert [dict(r) for r in direct] == [{'id': 1, 'x': 10, 'y': 20}]
    assert [dict(r) for r in direct] == [dict(r) for r in method]


def test_join_keeps_a_column_named_none():
    """Verify a column whose name is None survives the merge.

    Mutation: marking a column the other side lacks with None rather
        than a distinct sentinel, so a column literally named None
        reads as absent and its value is dropped.
    Oracle: hand-computed - the left row holds 1 under the None column,
        which the merge renders under the key 'None'.
    """
    a = DataSet([{None: 1, 'k': 2}])
    b = DataSet([{'y': 9, 'k': 2}])

    joined = DataSet.join(a, ('k',), b, ('k',), jointype='inner')

    assert dict(joined[0]) == {'None': 1, 'k': 2, 'y': 9}


def test_join_reads_a_missing_column_as_none_not_a_dict_method():
    """Verify a column a row lacks joins as None, whatever it is named.

    Mutation: reading the row through its own get() while the mapping it
        derives from lets a method name answer as a key, so a column
        named after one lands a bound method in the joined row.
    Oracle: hand-computed - neither side carries 'items', so it joins
        as None rather than as the method of that name.
    """
    a = DataSet([{'k': 1, 'x': 'left'}])
    b = DataSet([{'k': 1, 'y': 'right'}])

    joined = DataSet.join(a, ('k',), b, ('k',), acol=['k', 'x', 'items'])

    assert joined[0]['items'] is None
