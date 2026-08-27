"""Tests for DataSet bucket (grouping and aggregation) operations.

Aggregation Format Options
--------------------------
The bucket method accepts several shapes for one aggregation:

1. Column name (string): ['sales', 'quantity']
   -> sum, with None values dropped first
2. One-item tuple: [('sales',)]
   -> sum, with None values dropped first
3. Two-item tuple (column, operation): [('sales', max)]
4. Three-item tuple, read by the third item's type:
   - a string is an alias: [('sales', min, 'min_sales')]
   - a callable is a filter, handed the group's rows and returning
     the values to aggregate
5. Four-item tuple (column, operation, filter, alias)

Default Aggregation Behavior
----------------------------
- Default operation: sum
- None values: dropped by the default filter
- A filter that can match nothing must return a one-item fallback
  list, or the operation has nothing to work on
- Result type: the source column type unless the operation changed it
"""
import operator
import pickle

import pytest
from opendate import UTC, Date, DateTime, Time
from rollups import DataSet

from libb import attrdict
from libb.stats import safe_max, safe_min

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Basic dataset for bucket tests."""
    ds = DataSet([
        {'id': 1, 'category': 'X', 'region': 'North', 'value': 100},
        {'id': 2, 'category': 'Y', 'region': 'South', 'value': 200},
        {'id': 3, 'category': 'X', 'region': 'East', 'value': 150},
        {'id': 4, 'category': 'X', 'region': 'North', 'value': 250},
        {'id': 5, 'category': 'Y', 'region': 'South', 'value': 300}])
    ds.columns = (('id', int), ('category', str), ('region', str), ('value', int))
    return ds


@pytest.fixture
def category_value_dataset():
    """Dataset with category and value columns for aggregation tests."""
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 200},
        {'category': 'A', 'value': 300},
        {'category': 'B', 'value': 400},
        {'category': 'B', 'value': 500},
    ])
    ds.columns = [('category', str), ('value', int)]
    return ds


@pytest.fixture
def mixed_type_dataset():
    """Dataset with various column types."""
    ds = DataSet([
        {'id': 'A', 'int_col': 10, 'float_col': 10.5, 'date_col': Date(2024, 1, 1)},
        {'id': 'A', 'int_col': 20, 'float_col': 20.5, 'date_col': Date(2024, 1, 15)},
        {'id': 'B', 'int_col': 30, 'float_col': 30.5, 'date_col': Date(2024, 2, 1)},
    ])
    ds.columns = [
        ('id', str),
        ('int_col', int),
        ('float_col', float),
        ('date_col', Date),
        ]
    return ds


@pytest.fixture
def ds():
    """Three rows over two groups, key 1 holding two of them."""
    return DataSet([
        attrdict(key=1, b=2, c=4),
        attrdict(key=1, b=3, c=5),
        attrdict(key=2, b=4, c=7),
        ])


@pytest.fixture
def bucket_ds():
    """Groups a and b hold values; group c holds only a null weight."""
    return DataSet([
        {'k': 'a', 'v': 1, 'w': 10.0},
        {'k': 'a', 'v': 2, 'w': 20.0},
        {'k': 'b', 'v': 3, 'w': 30.0},
        {'k': 'b', 'v': 4, 'w': 40.0},
        {'k': 'c', 'v': 5, 'w': None},
    ])


# --- Basic Bucket Tests ---

def test_bucket_simple_grouping_no_aggregation():
    """Verify an empty aggregation list returns the distinct keys.

    Mutation: dropping `data.sort(key=sort_key)`, so interleaved keys
        group only where they are adjacent; or `data = self.container`
        in place of the slice copy, re-sorting the caller's own rows.
    Oracle: hand-counted 2 distinct categories over interleaved rows,
        and the caller's row order recorded before the call.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100, 'price': 50.0},
        {'category': 'B', 'value': 300, 'price': 60.0},
        {'category': 'A', 'value': 200, 'price': 55.0},
    ])
    order_before = [r.value for r in ds]

    result = ds.bucket(['category'], [])

    assert len(result) == 2
    assert sorted([r.category for r in result]) == ['A', 'B']
    assert result.cols == ['category']
    assert [r.value for r in ds] == order_before


def test_bucket_default_sum_aggregation(basic_dataset):
    """Verify a bare column name sums it within each key group.

    Mutation: the default operation changed from sum to max, or the
        pre-group sort dropped so the two North rows and the two South
        rows never meet.
    Oracle: hand-computed per-region totals over interleaved rows.
    """
    bd = basic_dataset.bucket(['region'], ['id', 'value'])
    bd.sort_data('region')
    assert len(bd) == 3
    assert bd[0]['id'] == 3
    assert bd[0]['value'] == 150
    assert bd[1]['id'] == 5
    assert bd[1]['value'] == 350
    assert bd[2]['id'] == 7
    assert bd[2]['value'] == 500


def test_bucket_custom_aggregation_functions(basic_dataset):
    """Verify a custom operation per column across two key columns.

    Mutation: keyfn grouping on the first key column only, or zipping
        the key values onto the key columns reversed.
    Oracle: hand-computed min id and mean value per (category, region).
    """
    def avg(v):
        v = list(v)
        return sum(v) / len(v)
    bd = basic_dataset.bucket(
        ['category', 'region'],
        [('id', safe_min), ('value', avg)])
    bd.sort(operator.itemgetter('category', 'region'))
    assert len(bd) == 3
    assert (bd[0]['category'], bd[0]['region']) == ('X', 'East')
    assert (bd[1]['category'], bd[1]['region']) == ('X', 'North')
    assert (bd[2]['category'], bd[2]['region']) == ('Y', 'South')
    assert bd[0]['id'] == 3
    assert bd[0]['value'] == 150.0
    assert bd[1]['id'] == 1
    assert bd[1]['value'] == 175.0
    assert bd[2]['id'] == 2
    assert bd[2]['value'] == 250.0


def test_bucket_with_alias():
    """Verify a 3-tuple's string third item renames the result column.

    Mutation: parse_aggregation reading a 3-tuple's third item as a
        filter rather than an alias, so both totals land on the source
        column names.
    Oracle: hand-summed 300 and 30 under the aliased names.
    """
    ds = DataSet([
        {'category': 'A', 'value1': 100, 'value2': 10},
        {'category': 'A', 'value1': 200, 'value2': 20},
        {'category': 'B', 'value1': 300, 'value2': 30},
    ])
    result = ds.bucket(
        ['category'],
        [('value1', sum, 'total1'), ('value2', sum, 'total2')])
    assert len(result) == 2
    assert result.cols == ['category', 'total1', 'total2']
    result.sort_data('category')
    assert result[0].total1 == 300
    assert result[0].total2 == 30


def test_bucket_with_filter_function():
    """Verify a 3-tuple's callable third item filters the rows first.

    Mutation: parse_aggregation ignoring the filter and falling back to
        the default non_none filter over every row in the group.
    Oracle: hand-summed active-only values, 10+30 and 40.
    """
    ds = DataSet([
        {'id': 1, 'value': 10, 'status': 'active'},
        {'id': 1, 'value': 20, 'status': 'inactive'},
        {'id': 1, 'value': 30, 'status': 'active'},
        {'id': 2, 'value': 40, 'status': 'active'},
        {'id': 2, 'value': 50, 'status': 'inactive'},
    ])
    active_filter = lambda rows: [r.value for r in rows if r.status == 'active']
    result = ds.bucket(['id'], [('value', sum, active_filter)])
    result.sort_data('id')
    assert len(result) == 2
    assert result[0].value == 40
    assert result[1].value == 40


def test_bucket_with_filter_and_alias():
    """Verify a 4-tuple applies both the filter and the alias.

    Mutation: parse_aggregation returning the column name as a 4-tuple
        alias, so the aliased column never appears.
    Oracle: hand-summed valid-only values, 100+300 and 400.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100, 'is_valid': True},
        {'category': 'A', 'value': 200, 'is_valid': False},
        {'category': 'A', 'value': 300, 'is_valid': True},
        {'category': 'B', 'value': 400, 'is_valid': True},
    ])
    valid_filter = lambda rows: [r.value for r in rows if r.is_valid]
    result = ds.bucket(['category'], [('value', sum, valid_filter, 'valid_total')])
    result.sort_data('category')
    assert result[0].valid_total == 400
    assert result[1].valid_total == 400


def test_bucket_empty_key_aggregates_all():
    """Verify an empty key list totals every row into one group.

    Mutation: the default operation changed from sum to max, which
        would return 300 and 70.0.
    Oracle: hand-summed 600 and 180.0 over all three rows.
    """
    ds = DataSet([
        {'value1': 100, 'value2': 50.0},
        {'value1': 200, 'value2': 60.0},
        {'value1': 300, 'value2': 70.0},
    ])
    result = ds.bucket([], ['value1', 'value2'])
    assert len(result) == 1
    assert result[0].value1 == 600
    assert result[0].value2 == pytest.approx(180.0, abs=0.01)


# --- Aggregation Function Tests (Parameterized) ---

@pytest.mark.parametrize(('agg_func', 'expected_a', 'expected_b'), [
    (list, ['X', 'Y'], ['Z']),
    (set, {'X', 'Y'}, {'Z'}),
])
def test_bucket_collection_aggregations(agg_func, expected_a, expected_b):
    """Verify list() and set() collect a group's column values.

    Mutation: parse_aggregation dropping the default filter, so the
        operation receives row dicts instead of column values.
    Oracle: hand-listed values per category.
    """
    ds = DataSet([
        {'category': 'A', 'value': 'X'},
        {'category': 'A', 'value': 'Y'},
        {'category': 'B', 'value': 'Z'},
    ])
    result = ds.bucket(['category'], [('value', agg_func)])
    result.sort_data('category')
    assert len(result) == 2
    assert result[0].value == expected_a
    assert result[1].value == expected_b


def test_bucket_multiple_aggregations_same_column(category_value_dataset):
    """Verify one column aggregated three ways under three aliases.

    Mutation: parse_aggregation reading a 3-tuple's string third item
        as a filter, so all three results collapse onto 'value'.
    Oracle: hand-computed sum, max and min per category.
    """
    result = category_value_dataset.bucket(['category'], [
        ('value', sum, 'total'),
        ('value', safe_max, 'maximum'),
        ('value', safe_min, 'minimum'),
    ])
    result.sort_data('category')
    assert result[0].total == 600
    assert result[0].maximum == 300
    assert result[0].minimum == 100
    assert result[1].total == 900
    assert result[1].maximum == 500
    assert result[1].minimum == 400


def test_bucket_handles_none_values():
    """Verify the default filter drops None before summing.

    Mutation: non_none keeping the None values, so sum raises TypeError
        and the safe wrapper turns the group's total into None.
    Oracle: hand-summed 100+200 and 300.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': None},
        {'category': 'A', 'value': 200},
        {'category': 'B', 'value': None},
        {'category': 'B', 'value': 300},
    ])
    result = ds.bucket(['category'], ['value'])
    result.sort_data('category')
    assert len(result) == 2
    assert result[0].value == 300
    assert result[1].value == 300


def test_bucket_multiple_key_columns():
    """Verify grouping on two key columns keys on the pair.

    Mutation: keyfn using only the first key column, merging East/A
        with East/B into one 600 group.
    Oracle: hand-summed 300, 300 and 400.
    """
    ds = DataSet([
        {'region': 'East', 'category': 'A', 'value': 100},
        {'region': 'East', 'category': 'A', 'value': 200},
        {'region': 'East', 'category': 'B', 'value': 300},
        {'region': 'West', 'category': 'A', 'value': 400},
    ])
    result = ds.bucket(['region', 'category'], ['value'])
    result.sort_data('region', 'category')
    assert len(result) == 3
    assert result[0].value == 300
    assert result[1].value == 300
    assert result[2].value == 400


def test_bucket_with_set_aggregation():
    """Verify set() collapses a value repeated within a group.

    Mutation: parse_aggregation dropping the default filter, so set()
        receives unhashable row dicts and the safe wrapper hides it.
    Oracle: hand-listed {'X', 'Y'} from three A rows, two of them 'X'.
    """
    ds = DataSet([
        {'category': 'A', 'tag': 'X'},
        {'category': 'A', 'tag': 'Y'},
        {'category': 'A', 'tag': 'X'},
        {'category': 'B', 'tag': 'Z'},
    ])
    result = ds.bucket(['category'], [('tag', set)])
    result.sort_data('category')
    assert len(result) == 2
    assert result[0].tag == {'X', 'Y'}
    assert result[1].tag == {'Z'}


# --- Custom Aggregation Tests ---

def test_bucket_custom_function_with_closure():
    """Verify a closure aggregation reading two columns off the rows.

    Mutation: parse_aggregation returning the column name as a 4-tuple
        alias, so 'rate' overwrites the summed 'profit'.
    Oracle: hand-computed 3000/30000 -> 10% and -500/5000 -> -10%.
    """
    ds = DataSet([
        {'group': 'A', 'profit': 1000, 'amount': 10000},
        {'group': 'A', 'profit': 2000, 'amount': 20000},
        {'group': 'B', 'profit': -500, 'amount': 5000},
    ])
    rate_calc = (
        lambda rows: sum(r.profit for r in rows) / sum(r.amount for r in rows) * 100)
    result = ds.bucket(
        ['group'],
        [('profit', sum), ('profit', rate_calc, lambda x: x, 'rate')])
    result.sort_data('group')
    assert result[0].profit == 3000
    assert result[0].rate == pytest.approx(10.0, abs=0.01)
    assert result[1].profit == -500
    assert result[1].rate == pytest.approx(-10.0, abs=0.01)


def test_bucket_with_max_filter():
    """Verify max runs over the filtered values, not the whole group.

    Mutation: parse_aggregation ignoring the filter, which would take
        the unfiltered max 35 for id 1; or reading the callable third
        item as an alias, which drops the 'value' column.
    Oracle: hand-picked priority-2 values, max 20 and 30.
    """
    ds = DataSet([
        {'id': 1, 'value': 10, 'priority': 1},
        {'id': 1, 'value': 20, 'priority': 2},
        {'id': 1, 'value': 35, 'priority': 1},
        {'id': 2, 'value': 30, 'priority': 2},
    ])
    high_priority_filter = lambda rows: [r.value for r in rows if r.priority == 2]
    result = ds.bucket(['id'], [('value', max, high_priority_filter)])
    result.sort_data('id')
    assert result[0].value == 20
    assert result[1].value == 30


def test_bucket_returns_empty_for_filtered_groups():
    """Verify a filter's [None] fallback yields None for that group.

    Mutation: parse_aggregation ignoring the filter, so group 2 returns
        its own value 30 instead of None.
    Oracle: hand-computed max 20 for group 1 and None for group 2.
    """
    ds = DataSet([
        {'id': 1, 'value': 10, 'type': 'A'},
        {'id': 1, 'value': 20, 'type': 'A'},
        {'id': 2, 'value': 30, 'type': 'B'},
    ])
    type_a_filter = lambda rows: [r.value for r in rows if r.type == 'A'] or [None]
    result = ds.bucket(['id'], [('value', max, type_a_filter)])
    result.sort_data('id')
    assert len(result) == 2
    assert result[0].value == 20
    assert result[1].value is None


def test_bucket_multiple_keys_with_none_values():
    """Verify a None in a key column forms its own group.

    Mutation: keyfn using only the first key column, merging (1, 'A')
        with (1, None) into one 350 group.
    Oracle: hand-summed 50 for (1, None), 300 for (1, 'A') and 150 for
        (2, 'B').
    """
    ds = DataSet([
        {'id1': 1, 'id2': 'A', 'value': 100},
        {'id1': 1, 'id2': 'A', 'value': 200},
        {'id1': 1, 'id2': None, 'value': 50},
        {'id1': 2, 'id2': 'B', 'value': 150},
    ])
    result = ds.bucket(['id1', 'id2'], ['value'])
    result.sort_data('id1', 'id2')
    assert len(result) == 3
    assert result[0].value == 50
    assert result[1].value == 300
    assert result[2].value == 150


def test_bucket_preserves_column_order():
    """Verify key columns lead the result and aggregations follow.

    Mutation: the aggregation loop inserting at index 0, which puts the
        aggregated columns ahead of the keys.
    Oracle: the hand-written order ['category', 'value', 'count'].
    """
    ds = DataSet([
        {'category': 'A', 'value': 100, 'count': 10},
        {'category': 'A', 'value': 200, 'count': 20},
    ])
    result = ds.bucket(['category'], ['value', 'count'])
    assert result.cols == ['category', 'value', 'count']


def test_bucket_with_lambda_aggregation():
    """Verify a pass-through filter hands whole rows to the operation.

    Mutation: parse_aggregation ignoring the filter, so the lambda gets
        the column's values and cannot read `.data` off a row.
    Oracle: hand-flattened [1, 2, 3, 4, 5] and [6].
    """
    ds = DataSet([
        {'id': 1, 'data': [1, 2, 3]},
        {'id': 1, 'data': [4, 5]},
        {'id': 2, 'data': [6]},
    ])
    flatten = (
        lambda rows: [item for r in rows for item in r.data])
    result = ds.bucket(
        ['id'],
        [('data', lambda x: list(flatten(x)), lambda rows: rows)])
    result.sort_data('id')
    assert result[0].data == [1, 2, 3, 4, 5]
    assert result[1].data == [6]


def test_bucket_weighted_average():
    """Verify a weighted mean computed from two columns of the group.

    Mutation: parse_aggregation ignoring the filter, so the operation
        sees values rather than rows and cannot read the weights.
    Oracle: hand-computed (10*2 + 20*3)/5 = 16.0 and 30.0.
    """
    ds = DataSet([
        {'category': 'A', 'value': 10, 'weight': 2},
        {'category': 'A', 'value': 20, 'weight': 3},
        {'category': 'B', 'value': 30, 'weight': 1},
    ])
    weighted_avg = (
        lambda rows: sum(r.value * r.weight for r in rows) /
        sum(r.weight for r in rows))
    result = ds.bucket(['category'], [('value', weighted_avg, lambda x: x, 'wavg')])
    result.sort_data('category')
    assert result[0].wavg == pytest.approx(16.0, abs=0.01)
    assert result[1].wavg == pytest.approx(30.0, abs=0.01)


def test_bucket_with_count_aggregation(category_value_dataset):
    """Verify len() over the passed-through rows counts group size.

    Mutation: parse_aggregation returning the column name as a 4-tuple
        alias, so 'count' overwrites the summed 'value'.
    Oracle: hand-counted 3 rows for A and 2 for B, beside the sums.
    """
    result = category_value_dataset.bucket(
        ['category'],
        [('value', sum), ('value', len, lambda x: x, 'count')]
    )
    result.sort_data('category')
    assert result[0].value == 600
    assert result[0].count == 3
    assert result[1].value == 900
    assert result[1].count == 2


def test_bucket_string_concatenation():
    """Verify a join over the group's rows produces one string.

    Mutation: parse_aggregation ignoring the filter, so the join gets
        values and cannot read `.name` off a row.
    Oracle: hand-joined 'X, Y' and 'Z'.
    """
    ds = DataSet([
        {'category': 'A', 'name': 'X'},
        {'category': 'A', 'name': 'Y'},
        {'category': 'B', 'name': 'Z'},
    ])
    join_names = lambda rows: ', '.join(r.name for r in rows)
    result = ds.bucket(['category'], [('name', join_names, lambda x: x, 'names')])
    result.sort_data('category')
    assert result[0].names == 'X, Y'
    assert result[1].names == 'Z'


def test_bucket_select_from_max_row(basic_dataset):
    """Verify picking fields off the row with the largest metric.

    Mutation: parse_aggregation ignoring the filter, so the operation
        receives values and cannot reach the row's other columns.
    Oracle: hand-picked id 4 / value 250 for KEY-X and id 5 / value 300
        for KEY-Y.
    """
    basic_dataset.add_column('group_key', str, value=lambda x: f'KEY-{x.category}')
    basic_dataset.add_column('metric', int, value=lambda x: x.value * 10)
    max_metric = lambda col: lambda rows: max(rows, key=lambda r: r.metric)[col]
    passit = lambda x: x
    result = basic_dataset.bucket(['group_key'], [
        ('id', max_metric('id'), passit),
        ('value', max_metric('value'), passit),
        ('metric', sum),
    ])
    result.sort_data('group_key')
    by_key = {r.group_key: r for r in result}
    assert by_key['KEY-X'].id == 4
    assert by_key['KEY-X'].value == 250
    assert by_key['KEY-X'].metric == 5000
    assert by_key['KEY-Y'].id == 5
    assert by_key['KEY-Y'].value == 300
    assert by_key['KEY-Y'].metric == 5000


# --- Format Variant Tests ---

def test_bucket_single_item_tuple_format():
    """Verify a one-item tuple aggregation defaults to sum.

    Mutation: parse_aggregation defaulting a one-item tuple to max, or
        `keycols = list(keycols)` splitting the string 'category' into
        single characters.
    Oracle: hand-summed 300 for A and 300 for B.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 200},
        {'category': 'B', 'value': 300},
    ])
    result = ds.bucket('category', [('value',)])
    result.sort_data('category')
    assert len(result) == 2
    assert result[0].value == 300
    assert result[1].value == 300


def test_bucket_filter_returning_none_for_no_matches():
    """Verify a filter's [None] fallback when nothing passes it.

    Mutation: parse_aggregation reading the 3-tuple's callable third
        item as an alias, so the fallback never runs.
    Oracle: hand-picked max 5 for id 1, and None for id 2 whose only
        row holds the excluded c == 7.
    """
    ds = DataSet([
        attrdict(id=1, b=2, c=4),
        attrdict(id=1, b=3, c=5),
        attrdict(id=2, b=4, c=7),
    ])
    fn = lambda rows: [r.c for r in rows if r.c != 7] or [None]
    bkt = ds.bucket('id', [
        ('b', max, fn),
        ('c', max, fn)
    ])
    bkt.sort_data('id')
    assert len(bkt) == 2
    assert bkt[0].b == 5
    assert bkt[0].c == 5
    assert bkt[1].b is None
    assert bkt[1].c is None


def test_bucket_multiple_keys_with_none_and_custom_filter():
    """Verify a per-column filter skipping falsy values on two keys.

    Mutation: parse_aggregation reading the 3-tuple's callable third
        item as an alias, so None reaches sum and the group returns
        None.
    Oracle: hand-summed a = 2+4 = 6 and b = 3 + -1 = 2.
    """
    ds = DataSet([
        attrdict(id1=1, id2=2, a=2, b=None),
        attrdict(id1=1, id2=2, a=None, b=3),
        attrdict(id1=1, id2=2, a=4, b=-1),
    ])
    fn = lambda col: lambda rows: [r[col] for r in rows if r[col]] or [0.]
    bkt = ds.bucket(['id1', 'id2'], [
        ('a', sum, fn('a')),
        ('b', sum, fn('b'))
    ])
    assert len(bkt) == 1
    assert bkt[0].a == 6
    assert bkt[0].b == 2


def test_bucket_no_keys_with_tuple_formats():
    """Verify a tuple and a list of column names aggregate alike.

    Mutation: the default operation changed from sum to max, which
        would return 4 rather than 7 for column a.
    Oracle: hand-summed a = 2+1+4 = 7 and b = 1+3-1 = 3.
    """
    ds = DataSet([
        attrdict(a=2, b=1),
        attrdict(a=1, b=3),
        attrdict(a=4, b=-1),
    ])
    bkt1 = ds.bucket([], ('a', 'b'))
    bkt2 = ds.bucket([], [('a'), ('b')])
    assert len(bkt1) == 1
    assert len(bkt2) == 1
    assert bkt1[0].a == bkt2[0].a
    assert bkt1[0].b == bkt2[0].b
    assert bkt1[0].a == 7
    assert bkt1[0].b == 3


def test_bucket_mixed_aggregation_formats():
    """Verify one call mixing all four aggregation shapes.

    Mutation: parse_aggregation returning the column name as a 4-tuple
        alias, so 'positive_sum' never appears.
    Oracle: hand-computed 300/2000/10/[1, 2] for A and 300/3000/30/[3]
        for B.
    """
    ds = DataSet([
        {'category': 'A', 'value1': 100, 'value2': 1000, 'value3': 10, 'value4': 1},
        {'category': 'A', 'value1': 200, 'value2': 2000, 'value3': -20, 'value4': 2},
        {'category': 'B', 'value1': 300, 'value2': 3000, 'value3': 30, 'value4': 3},
    ])
    positive_filter = lambda rows: [r.value3 for r in rows if r.value3 > 0]
    result = ds.bucket(['category'], [
        'value1',
        ('value2', max),
        ('value3', sum, positive_filter, 'positive_sum'),
        ('value4', list),
    ])
    result.sort_data('category')
    assert result[0].value1 == 300
    assert result[0].value2 == 2000
    assert result[0].positive_sum == 10
    assert result[0].value4 == [1, 2]
    assert result[1].value1 == 300
    assert result[1].value2 == 3000
    assert result[1].positive_sum == 30
    assert result[1].value4 == [3]


def test_bucket_inplace_replaces_source():
    """Verify inplace=True rewrites the caller and returns it.

    Mutation: the inplace branch dropped, so bucket hands back a new
        DataSet and leaves the caller's rows alone.
    Oracle: hand-summed 300 for A and 300 for B, read back off the
        original object.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'B', 'value': 300},
        {'category': 'A', 'value': 200},
    ])
    ds.columns = [('category', str), ('value', int)]

    result = ds.bucket(['category'], ['value'], inplace=True)

    assert result is ds
    assert len(ds) == 2
    ds.sort_data('category')
    assert [(r.category, r.value) for r in ds] == [('A', 300), ('B', 300)]


# --- Empty Dataset Tests ---

def test_bucket_empty_dataset():
    """Verify bucketing an empty DataSet keeps the declared columns.

    Mutation: the aggregation loop inserting at index 0, which orders
        the result ['value', 'id'].
    Oracle: the hand-written column order ['id', 'value'].
    """
    ds = DataSet([], columns=[('id', str), ('value', int)])
    result = ds.bucket(['id'], ['value'])
    assert len(result) == 0
    assert result.cols == ['id', 'value']


# --- Serialization Tests ---

def test_bucket_result_survives_pickling():
    """Verify a pickled bucket result restores rows, names and types.

    Mutation: get_type reporting the source column type rather than the
        aggregate's, which types the set column str and stringifies it
        on the next read.
    Oracle: the pre-pickle DataSet plus hand-written column types.
    """
    ds = DataSet([
        {'category': 'A', 'amount': 100.0, 'tag': 'X'},
        {'category': 'A', 'amount': 200.0, 'tag': 'Y'},
        {'category': 'B', 'amount': 300.0, 'tag': 'Z'}])
    ds.columns = (('category', str), ('amount', float), ('tag', str))
    bkt = ds.bucket(['category'], [('amount', sum, 'total'), ('tag', set)])
    bkt.sort_data('category')

    restored = pickle.loads(pickle.dumps(bkt))

    assert restored.columns == [('category', str), ('total', float), ('tag', set)]
    assert restored.container == bkt.container
    assert restored[0].total == 300.0
    assert restored[0].tag == {'X', 'Y'}


# --- Complex Calculation Tests ---

def test_bucket_complex_ratio_calculation():
    """Verify a ratio aggregation feeding a summary row.

    Mutation: parse_aggregation ignoring the filter, so the ratio
        operation receives values instead of rows.
    Oracle: hand-summed value 2200, and the per-category ratios
        10 + 20 + 10 = 40 that the summary row adds up.
    """
    ds = DataSet([
        {'category': 'A', 'total': 1000, 'value': 100},
        {'category': 'A', 'total': 2000, 'value': 200},
        {'category': 'B', 'total': 3000, 'value': 600},
        {'category': 'B', 'total': 4000, 'value': 800},
        {'category': 'C', 'total': 5000, 'value': 500},
    ])
    ds.columns = [('category', str), ('total', float), ('value', float)]
    passit = lambda x: x
    ratio_fn = (
        lambda rows: sum(r.value for r in rows) / sum(r.total for r in rows) *
        100.)
    ds.add_column('ratio', float)
    ds = ds.bucket(['category'], [('value', sum), ('ratio', passit, ratio_fn)])
    ds.add_summary_row(label_idx=ds.cols.index('category'), columns=('ratio', 'value'))
    assert ds.summary.value == pytest.approx(2200, abs=0.1)
    assert ds.summary.ratio == pytest.approx(40.0, abs=0.1)


# --- Type Preservation Tests (Parameterized) ---

@pytest.mark.parametrize(('type_cls', 'values', 'expected_max'), [
    (Date, [Date(2024, 1, 1), Date(2024, 1, 3), Date(2024, 1, 2)], Date(2024, 1, 3)),
    (Time, [Time(9, 30, 0, tzinfo=UTC), Time(14, 15, 0, tzinfo=UTC),
            Time(10, 0, 0, tzinfo=UTC)], Time(14, 15, 0, tzinfo=UTC)),
    (bool, [True, False, False], True),
])
def test_bucket_type_preservation(type_cls, values, expected_max):
    """Verify max over a typed column keeps the type and the maximum.

    Mutation: parse_aggregation dropping the default filter, so max
        compares row dicts; or get_type reporting the source type where
        the operation changed it.
    Oracle: the hand-picked maximum of group A, per type.
    """
    ds = DataSet([
        {'category': 'A', 'typed_col': values[0], 'value': 100},
        {'category': 'A', 'typed_col': values[1], 'value': 200},
        {'category': 'B', 'typed_col': values[2], 'value': 150},
    ])
    ds.columns = [('category', str), ('typed_col', type_cls), ('value', int)]
    result = ds.bucket(['category'], [('typed_col', max), ('value', sum)])
    assert len(result) == 2
    assert result.colmap['typed_col'] == type_cls
    for row in result:
        assert isinstance(row['typed_col'], type_cls)
    result.sort_data('category')
    assert result[0].typed_col == expected_max
    assert result[0].value == 300


def test_bucket_max_over_date_column_skips_none():
    """Verify max over a Date column ignores a None in the group.

    Mutation: non_none keeping the None values, so max raises TypeError
        and the safe wrapper turns the group's date into None.
    Oracle: hand-picked latest date per category, 2024-01-03 and
        2024-01-02.
    """
    ds = DataSet([
        {'category': 'A', 'date': Date(2024, 1, 1), 'value': 100},
        {'category': 'A', 'date': None, 'value': 200},
        {'category': 'A', 'date': Date(2024, 1, 3), 'value': 150},
        {'category': 'B', 'date': Date(2024, 1, 2), 'value': 50},
    ])
    ds.columns = [('category', str), ('date', Date), ('value', int)]

    result = ds.bucket(['category'], [('date', max), ('value', sum)])

    result.sort_data('category')
    assert result.colmap['date'] == Date
    assert isinstance(result[0].date, Date)
    assert result[0].date == Date(2024, 1, 3)
    assert result[1].date == Date(2024, 1, 2)


def test_bucket_preserves_int_not_float():
    """Verify summing an int column reports int, not float.

    Mutation: the default operation changed from sum to max, or
        get_type promoting every int result to float.
    Oracle: hand-summed 300 against the declared int type.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 200},
    ])
    ds.columns = [('category', str), ('value', int)]
    result = ds.bucket(['category'], ['value'])
    assert result.colmap['value'] == int
    assert isinstance(result[0].value, int)
    assert result[0].value == 300


def test_bucket_datetime_type_preservation():
    """Verify a midnight DateTime max keeps the DateTime type.

    Mutation: get_type collapsing a temporal result to Date, as
        smart_type already does for a midnight datetime.datetime.
    Oracle: hand-picked latest timestamps 2024-01-03 00:00 and
        2024-01-02 00:00, both landing on midnight.
    """
    ds = DataSet([
        {'category': 'A', 'timestamp': DateTime(2024, 1, 1, 0, 0, 0), 'value': 100},
        {'category': 'A', 'timestamp': DateTime(2024, 1, 2, 14, 30, 0), 'value': 200},
        {'category': 'A', 'timestamp': DateTime(2024, 1, 3, 0, 0, 0), 'value': 150},
        {'category': 'B', 'timestamp': DateTime(2024, 1, 1, 10, 15, 0), 'value': 50},
        {'category': 'B', 'timestamp': DateTime(2024, 1, 2, 0, 0, 0), 'value': 75},
    ])
    ds.columns = [('category', str), ('timestamp', DateTime), ('value', int)]

    result = ds.bucket(['category'], [('timestamp', max), ('value', sum)])

    assert len(result) == 2
    assert result.colmap['timestamp'] == DateTime
    for row in result:
        assert isinstance(row['timestamp'], DateTime)

    result_a = [r for r in result if r['category'] == 'A'][0]
    assert result_a['timestamp'] == DateTime(2024, 1, 3, 0, 0, 0, tzinfo=UTC)
    assert result_a['value'] == 450

    result_b = [r for r in result if r['category'] == 'B'][0]
    assert result_b['timestamp'] == DateTime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
    assert result_b['value'] == 125


def test_bucket_float_type_preservation_with_int_input():
    """Verify int inputs in a float column come back float.

    Mutation: get_type promoting every int result to float, which would
        report the int quantity column as float too.
    Oracle: hand-summed 300.0 price and 15 quantity, against the
        declared float and int types.
    """
    ds = DataSet([
        {'category': 'A', 'price': 100, 'quantity': 10},
        {'category': 'A', 'price': 200, 'quantity': 5},
        {'category': 'B', 'price': 150, 'quantity': 20},
    ])
    ds.columns = [('category', str), ('price', float), ('quantity', int)]

    result = ds.bucket(['category'], [('price', sum), ('quantity', sum)])

    assert result.colmap['price'] == float
    assert result.colmap['quantity'] == int

    result.sort_data('category')
    assert isinstance(result[0].price, float)
    assert isinstance(result[0].quantity, int)
    assert result[0].price == 300.0
    assert result[0].quantity == 15


def test_bucket_float_from_division():
    """Verify a division aggregation reports float beside an int sum.

    Mutation: get_type reporting the source column type, so 'average'
        would come back int like the column it reads.
    Oracle: hand-computed 300/7 = 42.857 beside the int sum 300.
    """
    ds = DataSet([
        {'category': 'A', 'total': 100, 'count': 3},
        {'category': 'A', 'total': 200, 'count': 4},
        {'category': 'B', 'total': 150, 'count': 2},
    ])
    ds.columns = [('category', str), ('total', int), ('count', int)]

    avg_fn = lambda rows: sum(r.total for r in rows) / sum(r.count for r in rows)
    result = ds.bucket(['category'], [
        ('total', sum),
        ('total', avg_fn, lambda x: x, 'average')
    ])

    assert result.colmap['total'] == int
    assert result.colmap['average'] == float

    result.sort_data('category')
    assert isinstance(result[0].total, int)
    assert isinstance(result[0].average, float)
    assert result[0].total == 300
    assert result[0].average == pytest.approx(42.857, abs=0.001)


def test_bucket_string_type_preservation():
    """Verify a row-picking operation keeps the str type and the order.

    Mutation: parse_aggregation ignoring the filter, so the operation
        receives values and cannot read `.name` off a row.
    Oracle: hand-picked first name 'Alice' from group A.
    """
    ds = DataSet([
        {'category': 'A', 'name': 'Alice', 'code': 'X1'},
        {'category': 'A', 'name': 'Bob', 'code': 'X2'},
        {'category': 'B', 'name': 'Charlie', 'code': 'Y1'},
    ])
    ds.columns = [('category', str), ('name', str), ('code', str)]

    first_name = lambda rows: list(rows)[0].name
    result = ds.bucket(['category'], [
        ('name', first_name, lambda x: x, 'first_name'),
        ('code', list, lambda x: x, 'codes')
    ])

    assert result.colmap['first_name'] == str

    result.sort_data('category')
    assert isinstance(result[0].first_name, str)
    assert result[0].first_name == 'Alice'


def test_bucket_none_type_handling():
    """Verify an all-None column keeps its declared type.

    Mutation: infer_aggregation_type ignoring the declared type, or
        get_type returning str where the group's result is None.
    Oracle: the declared int type against a None result per group.
    """
    ds = DataSet([
        {'category': 'A', 'value': None},
        {'category': 'A', 'value': None},
        {'category': 'B', 'value': None},
    ])
    ds.columns = [('category', str), ('value', int)]

    result = ds.bucket(['category'], ['value'])

    assert result.colmap['value'] == int
    result.sort_data('category')
    assert result[0].value is None
    assert result[1].value is None


def test_bucket_type_inference_with_min_max():
    """Verify aliased min and max over one float column.

    Mutation: parse_aggregation reading a 3-tuple's string third item
        as a filter, so both results collapse onto 'value'.
    Oracle: hand-picked 200.7 max and 100.5 min for category A.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100.5},
        {'category': 'A', 'value': 200.7},
        {'category': 'B', 'value': 150.3},
    ])
    ds.columns = [('category', str), ('value', float)]

    result = ds.bucket(['category'], [
        ('value', max, 'max_value'),
        ('value', min, 'min_value'),
    ])

    assert result.colmap['max_value'] == float
    assert result.colmap['min_value'] == float

    result.sort_data('category')
    assert isinstance(result[0].max_value, float)
    assert isinstance(result[0].min_value, float)
    assert result[0].max_value == 200.7
    assert result[0].min_value == 100.5


def test_bucket_mixed_numeric_types():
    """Verify an int and a float column keep their own types.

    Mutation: the default operation changed from sum to max, which
        would give 200 and 20.7 for category A.
    Oracle: hand-summed 300 and 31.2.
    """
    ds = DataSet([
        {'category': 'A', 'int_val': 100, 'float_val': 10.5},
        {'category': 'A', 'int_val': 200, 'float_val': 20.7},
        {'category': 'B', 'int_val': 150, 'float_val': 15.3},
    ])
    ds.columns = [('category', str), ('int_val', int), ('float_val', float)]

    result = ds.bucket(['category'], ['int_val', 'float_val'])

    assert result.colmap['int_val'] == int
    assert result.colmap['float_val'] == float

    result.sort_data('category')
    assert isinstance(result[0].int_val, int)
    assert isinstance(result[0].float_val, float)
    assert result[0].int_val == 300
    assert result[0].float_val == pytest.approx(31.2, abs=0.01)


def test_bucket_preserves_float_over_int_in_results():
    """Verify summing whole-number floats stays float.

    Mutation: the default operation changed from sum to max, giving
        200.0 instead of 300.0.
    Oracle: hand-summed 300.0 against the declared float type.
    """
    ds = DataSet([
        {'category': 'A', 'amount': 100.0},
        {'category': 'A', 'amount': 200.0},
        {'category': 'B', 'amount': 300.0},
    ])
    ds.columns = [('category', str), ('amount', float)]

    result = ds.bucket(['category'], ['amount'])

    assert result.colmap['amount'] == float
    result.sort_data('category')
    assert isinstance(result[0].amount, float)
    assert result[0].amount == 300.0


def test_bucket_date_with_different_aggregations():
    """Verify aliased max and min over one Date column.

    Mutation: parse_aggregation reading a 3-tuple's string third item
        as a filter, so both results collapse onto 'date'.
    Oracle: hand-picked 2024-01-15 max and 2024-01-01 min for A.
    """
    ds = DataSet([
        {'category': 'A', 'date': Date(2024, 1, 1), 'value': 100},
        {'category': 'A', 'date': Date(2024, 1, 15), 'value': 200},
        {'category': 'A', 'date': Date(2024, 1, 10), 'value': 150},
        {'category': 'B', 'date': Date(2024, 2, 1), 'value': 300},
    ])
    ds.columns = [('category', str), ('date', Date), ('value', int)]

    result = ds.bucket(['category'], [
        ('date', max, 'max_date'),
        ('date', min, 'min_date'),
        ('value', sum)
    ])

    assert result.colmap['max_date'] == Date
    assert result.colmap['min_date'] == Date

    result.sort_data('category')
    assert isinstance(result[0].max_date, Date)
    assert isinstance(result[0].min_date, Date)
    assert result[0].max_date == Date(2024, 1, 15)
    assert result[0].min_date == Date(2024, 1, 1)


def test_bucket_datetime_with_non_midnight_values():
    """Verify max over DateTime picks the later time within a day.

    Mutation: parse_aggregation dropping the default filter, so max
        compares row dicts instead of timestamps.
    Oracle: hand-picked 2024-01-01 14:45 over 09:30 the same day.
    """
    ds = DataSet([
        {'category': 'A', 'timestamp': DateTime(2024, 1, 1, 9, 30, 0), 'value': 100},
        {'category': 'A', 'timestamp': DateTime(2024, 1, 1, 14, 45, 0), 'value': 200},
        {'category': 'B', 'timestamp': DateTime(2024, 1, 2, 10, 15, 0), 'value': 150},
    ])
    ds.columns = [('category', str), ('timestamp', DateTime), ('value', int)]

    result = ds.bucket(['category'], [('timestamp', max), ('value', sum)])

    assert result.colmap['timestamp'] == DateTime

    result.sort_data('category')
    assert isinstance(result[0].timestamp, DateTime)
    assert result[0].timestamp == DateTime(2024, 1, 1, 14, 45, 0, tzinfo=UTC)
    assert result[0].value == 300


def test_bucket_time_with_different_operations():
    """Verify aliased min and max over one Time column.

    Mutation: parse_aggregation reading a 3-tuple's string third item
        as a filter, so both results collapse onto 'start_time'.
    Oracle: hand-picked 09:00 earliest and 10:30 latest for A.
    """
    ds = DataSet([
        {'category': 'A', 'start_time': Time(9, 0, 0, tzinfo=UTC), 'duration': 60},
        {'category': 'A', 'start_time': Time(10, 30, 0, tzinfo=UTC), 'duration': 45},
        {'category': 'B', 'start_time': Time(14, 15, 0, tzinfo=UTC), 'duration': 90},
    ])
    ds.columns = [('category', str), ('start_time', Time), ('duration', int)]

    result = ds.bucket(['category'], [
        ('start_time', min, 'earliest'),
        ('start_time', max, 'latest'),
        ('duration', sum)
    ])

    assert result.colmap['earliest'] == Time
    assert result.colmap['latest'] == Time

    result.sort_data('category')
    assert isinstance(result[0].earliest, Time)
    assert isinstance(result[0].latest, Time)
    assert result[0].earliest == Time(9, 0, 0, tzinfo=UTC)
    assert result[0].latest == Time(10, 30, 0, tzinfo=UTC)
    assert result[0].duration == 105


def test_bucket_type_with_custom_class():
    """Verify an operation returning str is typed str, not the source.

    Mutation: get_type reporting the source column type, so 'total_str'
        would be typed int and parsed back into one on the next read.
    Oracle: hand-computed 300 and its string form '300'.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 200},
        {'category': 'B', 'value': 300},
    ])
    ds.columns = [('category', str), ('value', int)]

    to_string = lambda rows: str(sum(r.value for r in rows))
    result = ds.bucket(['category'], [
        ('value', sum, 'total'),
        ('value', to_string, lambda x: x, 'total_str')
    ])

    assert result.colmap['total'] == int
    assert result.colmap['total_str'] == str

    result.sort_data('category')
    assert isinstance(result[0].total, int)
    assert isinstance(result[0].total_str, str)
    assert result[0].total == 300
    assert result[0].total_str == '300'


def test_bucket_empty_groups_type_inference():
    """Verify a [None] fallback leaves the declared type intact.

    Mutation: the safe wrapper returning 0 rather than None when the
        operation fails on [None].
    Oracle: hand-computed 100, None and 300 per category, with int
        typing throughout.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100, 'type': 'X'},
        {'category': 'B', 'value': 200, 'type': 'Y'},
        {'category': 'C', 'value': 300, 'type': 'X'},
    ])
    ds.columns = [('category', str), ('value', int), ('type', str)]

    filter_x = lambda rows: [r.value for r in rows if r.type == 'X'] or [None]
    result = ds.bucket(['category'], [('value', sum, filter_x)])

    assert result.colmap['value'] == int
    result.sort_data('category')
    assert result[0].value == 100
    assert result[1].value is None
    assert result[2].value == 300


def test_bucket_multiple_types_same_aggregation(mixed_type_dataset):
    """Verify max over int, float and Date columns in one call.

    Mutation: parse_aggregation dropping the default filter, so max
        compares row dicts instead of column values.
    Oracle: hand-picked 20, 20.5 and 2024-01-15 for group A.
    """
    result = mixed_type_dataset.bucket(['id'], [
        ('int_col', max),
        ('float_col', max),
        ('date_col', max)
    ])

    assert result.colmap['int_col'] == int
    assert result.colmap['float_col'] == float
    assert result.colmap['date_col'] == Date

    result.sort_data('id')
    assert isinstance(result[0].int_col, int)
    assert isinstance(result[0].float_col, float)
    assert isinstance(result[0].date_col, Date)
    assert result[0].int_col == 20
    assert result[0].float_col == 20.5
    assert result[0].date_col == Date(2024, 1, 15)


# --- Set Aggregation Preservation Tests ---

def test_bucket_set_aggregation_remains_set():
    """Verify set aggregation yields real sets, not their repr.

    Mutation: get_type reporting the source str type for a set result,
        which stringifies the set on the next read.
    Oracle: a set intersection against a hand-written comparand.
    """
    ds = DataSet([
        {'id': 1, 'category': 'A', 'tag': 'X'},
        {'id': 1, 'category': 'A', 'tag': 'Y'},
        {'id': 2, 'category': 'B', 'tag': 'Z'},
    ])
    ds.columns = [('id', int), ('category', str), ('tag', str)]

    result = ds.bucket(['id'], [('tag', set), ('category', set)])

    result.sort_data('id')
    assert len(result) == 2

    assert isinstance(result[0].tag, set), f'Expected set, got {type(result[0].tag)}'
    assert isinstance(result[0].category, set), (
        f'Expected set, got {type(result[0].category)}')

    other_tags = {'X', 'A'}
    assert result[0].tag.intersection(other_tags) == {'X'}


def test_bucket_set_aggregation_with_subsequent_bucket():
    """Verify sets survive being bucketed a second time.

    Mutation: the safe wrapper dropping its set-union fallback, so
        set() over a column of sets returns None.
    Oracle: hand-listed {'C1', 'C2'} intersected with a wider set.
    """
    ds = DataSet([
        {'group': 'A', 'subgroup': 'X', 'code': 'C1', 'value': 100},
        {'group': 'A', 'subgroup': 'X', 'code': 'C2', 'value': 200},
        {'group': 'A', 'subgroup': 'Y', 'code': 'C3', 'value': 150},
        {'group': 'B', 'subgroup': 'Z', 'code': 'C4', 'value': 300},
    ])
    ds.columns = [('group', str), ('subgroup', str), ('code', str), ('value', int)]

    first_bucket = ds.bucket(['group', 'subgroup'], [('code', set), ('value', sum)])

    second_bucket = first_bucket.bucket(
        ['group'],
        [('code', set), ('subgroup', set), ('value', max)])

    second_bucket.sort_data('group')
    assert len(second_bucket) == 2

    assert isinstance(second_bucket[0].code, set), (
        f'Expected set, got {type(second_bucket[0].code)}')
    assert isinstance(second_bucket[0].subgroup, set), (
        f'Expected set, got {type(second_bucket[0].subgroup)}')

    valid_codes = {'C1', 'C2', 'C5'}
    assert second_bucket[0].code.intersection(valid_codes) == {'C1', 'C2'}
    assert second_bucket[0].subgroup == {'X', 'Y'}


# --- Type Inference Tests (Parameterized) ---

@pytest.mark.parametrize(('value_col', 'first_val', 'expected_type'), [
    ('value', 100, int),
    ('value', 123.45, float),
    ('value', True, bool),
    ('date_col', Date(2024, 1, 15), Date),
    ('timestamp', DateTime(2024, 1, 15, 10, 30, 0, tzinfo=UTC), DateTime),
    ('time_col', Time(14, 30, 0, tzinfo=UTC), Time),
])
def test_bucket_declared_type_survives_none_group(value_col, first_val, expected_type):
    """Verify a group aggregating to None keeps the guessed type.

    Mutation: get_type returning str rather than the source type where
        the first group's result is None.
    Oracle: the type guessed from the second row, per parameter.
    """
    ds = DataSet([
        {'id': 1, 'category': 'A', value_col: None},
        {'id': 2, 'category': 'B', value_col: first_val},
    ])

    result = ds.bucket(['category'], [(value_col, max)])

    assert result.colmap[value_col] == expected_type
    result.sort_data('category')
    assert result[0][value_col] is None
    assert result[1][value_col] == first_val


def test_bucket_infer_aggregation_type_with_known_original():
    """Verify a declared type wins over an all-None result.

    Mutation: infer_aggregation_type skipping the declared type and
        inferring from the bucket results, which are all None.
    Oracle: the declared float type.
    """
    ds = DataSet([
        {'category': 'A', 'value': None},
        {'category': 'A', 'value': None},
    ])
    ds.columns = [('category', str), ('value', float)]

    result = ds.bucket(['category'], ['value'])

    assert result.colmap['value'] == float


def test_bucket_infer_aggregation_type_defaults_to_string():
    """Verify an unknown, all-None column falls back to str.

    Mutation: infer_aggregation_type's fallback changed from str to
        NoneType.
    Oracle: the hand-written str fallback.
    """
    ds = DataSet([
        {'category': 'A', 'unknown': None},
        {'category': 'A', 'unknown': None},
        {'category': 'B', 'unknown': None},
    ])

    result = ds.bucket(['category'], ['unknown'])

    assert result.colmap['unknown'] == str


def test_bucket_infer_aggregation_type_with_list_aggregation():
    """Verify a list aggregation types the column list.

    Mutation: get_type reporting the source str type, which stringifies
        the list on the next read.
    Oracle: hand-listed ['X', 'Y'] for category A.
    """
    ds = DataSet([
        {'category': 'A', 'tag': 'X'},
        {'category': 'A', 'tag': 'Y'},
        {'category': 'B', 'tag': 'Z'},
    ])

    result = ds.bucket(['category'], [('tag', list)])

    assert result.colmap['tag'] == list
    result.sort_data('category')
    assert result[0].tag == ['X', 'Y']


def test_bucket_infer_aggregation_type_with_set_aggregation():
    """Verify a set aggregation types the column set.

    Mutation: get_type reporting the source str type, which stringifies
        the set on the next read.
    Oracle: the hand-written set type.
    """
    ds = DataSet([
        {'category': 'A', 'item': 'X'},
        {'category': 'A', 'item': 'X'},
        {'category': 'A', 'item': 'Y'},
    ])

    result = ds.bucket(['category'], [('item', set)])

    assert result.colmap['item'] == set


def test_bucket_infer_aggregation_type_multiple_none_groups():
    """Verify the bucket results supply a type the source column lacks.

    Mutation: infer_aggregation_type returning on the first bucket
        rather than on the first bucket holding a value - bucket A
        holds None, so it would fall through to the str default.
    Oracle: the float type of the lone 99.9 measurement, two buckets
        past the first.
    """
    rows = [
        {'id': 0, 'category': 'A', 'measure': None},
        {'id': 1, 'category': 'B', 'measure': None},
        {'id': 2, 'category': 'C', 'measure': 99.9},
        ]
    # Declared rather than inferred: the scan would read the 99.9 and
    # type the column float, leaving nothing for the bucket pass to
    # supply.
    ds = DataSet(rows, cols=['id', 'category', 'measure'],
                 typs=[int, str, type(None)])
    assert ds.colmap['measure'] is type(None)

    result = ds.bucket(['category'], ['measure'])

    assert result.colmap['measure'] == float
    result.sort_data('category')
    assert result[0].measure is None
    assert result[1].measure is None
    assert result[2].measure == 99.9


def test_bucket_float_column_keeps_float_on_int_result():
    """Verify a float column stays float when the result is an int.

    Mutation: get_type dropping its int-to-float promotion, or the
        aggregation loop reading the declared type by alias rather than
        by source column.
    Oracle: hand-computed round(100.5 + 200.5) -> 301, reported 301.0.
    """
    ds = DataSet([
        {'category': 'A', 'amount': 100.5},
        {'category': 'A', 'amount': 200.5},
    ])
    ds.columns = [('category', str), ('amount', float)]

    rounded_total = lambda rows: round(sum(r.amount for r in rows))
    result = ds.bucket(['category'], [
        ('amount', rounded_total, lambda x: x, 'rounded')])

    assert result.colmap['rounded'] == float
    assert isinstance(result[0].rounded, float)
    assert result[0].rounded == 301.0


def test_bucket_infer_aggregation_type_with_custom_operation_returning_string():
    """Verify a str-returning operation types its alias str.

    Mutation: parse_aggregation returning the column name as a 4-tuple
        alias, so 'csv_values' never appears.
    Oracle: hand-joined '100,200'.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 200},
    ])

    to_csv = lambda rows: ','.join(str(r.value) for r in rows)
    result = ds.bucket(['category'], [('value', to_csv, lambda x: x, 'csv_values')])

    assert result.colmap['csv_values'] == str
    assert result[0].csv_values == '100,200'


def test_bucket_infer_aggregation_type_empty_buckets():
    """Verify an empty DataSet keeps the declared aggregation type.

    Mutation: infer_aggregation_type ignoring the declared type, which
        leaves str once the empty bucket list yields nothing.
    Oracle: the declared int type.
    """
    ds = DataSet([], columns=[('id', str), ('value', int)])

    result = ds.bucket(['id'], ['value'])

    assert result.colmap['value'] == int
    assert len(result) == 0


# --- Edge Case Tests ---

def test_bucket_with_unhashable_group_key():
    """Verify an unorderable key column groups without raising.

    Mutation: dropping the Hashable guard in sort_key, so the sort
        compares two dicts and raises TypeError.
    Oracle: hand-counted groups -- adjacent equal keys merge, and a
        repeat further down the rows forms a group of its own.
    """
    ds = DataSet([
        {'key': {'a': 1}, 'value': 100},
        {'key': {'a': 1}, 'value': 200},
        {'key': {'b': 2}, 'value': 300},
        {'key': {'a': 1}, 'value': 400},
    ])

    result = ds.bucket(['key'], ['value'])

    assert [(r.key, r.value) for r in result] == [
        ({'a': 1}, 300), ({'b': 2}, 300), ({'a': 1}, 400)]


def test_bucket_with_lambda_raising_exception():
    """Verify an operation's own error is not swallowed.

    Mutation: the safe wrapper catching Exception rather than
        ValueError and TypeError, turning the error into a None cell.
    Oracle: pytest.raises on a hand-planted division by zero.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 0},
        {'category': 'B', 'value': 200},
    ])

    def bad_agg(rows):
        return sum(100 / r.value for r in rows)

    with pytest.raises(ZeroDivisionError):
        ds.bucket(['category'], [('value', bad_agg, lambda x: x, 'result')])


def test_bucket_very_large_number_of_groups():
    """Verify a thousand single-row groups each keep their own key.

    Mutation: key columns typed str rather than from the source colmap,
        which turns r.id into a string.
    Oracle: the hand-written r.value == r.id * 10 relation per row.
    """
    ds = DataSet([{'id': i, 'value': i * 10} for i in range(1000)])
    ds.columns = [('id', int), ('value', int)]

    result = ds.bucket(['id'], ['value'])

    assert len(result) == 1000
    assert all(r.value == r.id * 10 for r in result)


def test_bucket_with_nested_dataset_values():
    """Verify list aggregation collects nested DataSets in row order.

    Mutation: get_type reporting the source column type, so the column
        is typed DataSet and rebuilt from the list on the next read.
    Oracle: the two hand-placed inner DataSets, in row order.
    """
    first_ds = DataSet([{'x': 1}])
    second_ds = DataSet([{'x': 2}])
    ds = DataSet([
        {'category': 'A', 'nested': first_ds},
        {'category': 'A', 'nested': second_ds},
    ])

    result = ds.bucket(['category'], [('nested', list)])

    assert len(result) == 1
    assert result.colmap['nested'] == list
    assert result[0].nested == [first_ds, second_ds]


def test_bucket_key_is_tuple_column():
    """Verify a tuple-valued key column groups on the whole tuple.

    Mutation: key columns typed str rather than from the source colmap,
        which stringifies the tuple keys.
    Oracle: hand-summed 300 per tuple key.
    """
    ds = DataSet([
        {'key': (1, 2), 'value': 100},
        {'key': (1, 2), 'value': 200},
        {'key': (3, 4), 'value': 300},
    ])

    result = ds.bucket(['key'], ['value'])

    assert len(result) == 2
    result_by_key = {r.key: r.value for r in result}
    assert result_by_key[(1, 2)] == 300
    assert result_by_key[(3, 4)] == 300


def test_bucket_single_row_dataset():
    """Verify a one-row group aggregates to that row's own value.

    Mutation: an off-by-one on the group's rows (`list(grouped)[1:]`),
        which empties a single-row group and yields None.
    Oracle: the row's own value, 100, and the declared column types.
    """
    ds = DataSet([{'category': 'A', 'value': 100}])
    ds.columns = [('category', str), ('value', int)]

    result = ds.bucket(['category'], ['value'])

    assert len(result) == 1
    assert result[0].category == 'A'
    assert result[0].value == 100
    assert result.colmap == {'category': str, 'value': int}


def test_bucket_all_rows_same_group():
    """Verify one key value collapses every row into one result row.

    Mutation: the default operation changed from sum to max, giving 300
        instead of 600.
    Oracle: hand-summed 100+200+300.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 200},
        {'category': 'A', 'value': 300},
    ])
    ds.columns = [('category', str), ('value', int)]

    result = ds.bucket(['category'], ['value'])

    assert len(result) == 1
    assert result[0].value == 600


def test_bucket_empty_string_category():
    """Verify '' and None are distinct keys and neither breaks the sort.

    Mutation: dropping the `row[col] is None` leg of sort_key, so the
        sort compares None with a string and raises TypeError.
    Oracle: hand-summed 300 for '', 300 for 'A' and 400 for None.
    """
    ds = DataSet([
        {'category': '', 'value': 100},
        {'category': 'A', 'value': 300},
        {'category': None, 'value': 400},
        {'category': '', 'value': 200},
    ])

    result = ds.bucket(['category'], ['value'])

    assert len(result) == 3
    result_by_cat = {r.category: r.value for r in result}
    assert result_by_cat[''] == 300
    assert result_by_cat['A'] == 300
    assert result_by_cat[None] == 400


def test_bucket_aggregation_returning_none():
    """Verify an operation returning None yields a None cell.

    Mutation: parse_aggregation returning the column name as a 4-tuple
        alias, so 'result' never appears.
    Oracle: the hand-written None the operation returns.
    """
    ds = DataSet([
        {'category': 'A', 'value': 100},
        {'category': 'A', 'value': 200},
    ])

    always_none = lambda rows: None
    result = ds.bucket(['category'], [('value', always_none, lambda x: x, 'result')])

    assert len(result) == 1
    assert result[0].result is None


def test_bucket_with_boolean_key():
    """Verify True and False form separate key groups.

    Mutation: key columns typed str rather than from the source colmap,
        so the keys become 'True' and 'False'.
    Oracle: hand-summed 300 under each flag.
    """
    ds = DataSet([
        {'is_active': True, 'value': 100},
        {'is_active': True, 'value': 200},
        {'is_active': False, 'value': 300},
    ])
    ds.columns = [('is_active', bool), ('value', int)]

    result = ds.bucket(['is_active'], ['value'])

    assert len(result) == 2
    result_by_flag = {r.is_active: r.value for r in result}
    assert result_by_flag[True] == 300
    assert result_by_flag[False] == 300


def test_bucket_preserves_order_within_groups():
    """Verify the pre-group sort keeps each group's rows in row order.

    Mutation: reversing a group's rows (`list(grouped)[::-1]`) before
        the filter runs.
    Oracle: hand-written sequences [1, 2, 3] for A and [10, 11] for B,
        read off interleaved input rows.
    """
    ds = DataSet([
        {'category': 'A', 'seq': 1},
        {'category': 'B', 'seq': 10},
        {'category': 'A', 'seq': 2},
        {'category': 'B', 'seq': 11},
        {'category': 'A', 'seq': 3},
    ])

    get_seqs = lambda rows: [r.seq for r in rows]
    result = ds.bucket(['category'], [('seq', get_seqs, lambda x: x, 'sequences')])
    result.sort_data('category')

    assert result[0].sequences == [1, 2, 3]
    assert result[1].sequences == [10, 11]


def test_bucket_with_float_key():
    """Verify float key values group by exact equality.

    Mutation: key columns typed str rather than from the source colmap,
        so 1.1 becomes '1.1' and the lookup misses.
    Oracle: hand-summed 300 for 1.1 and 300 for 2.2.
    """
    ds = DataSet([
        {'key': 1.1, 'value': 100},
        {'key': 2.2, 'value': 300},
        {'key': 1.1, 'value': 200},
    ])
    ds.columns = [('key', float), ('value', int)]

    result = ds.bucket(['key'], ['value'])

    assert len(result) == 2
    result_by_key = {r.key: r.value for r in result}
    assert result_by_key[1.1] == 300
    assert result_by_key[2.2] == 300


def test_bucket_date_key():
    """Verify a Date key column groups by date.

    Mutation: key columns typed str rather than from the source colmap,
        so the Date keys become strings.
    Oracle: hand-summed 300 per date.
    """
    ds = DataSet([
        {'date': Date(2024, 1, 1), 'value': 100},
        {'date': Date(2024, 1, 1), 'value': 200},
        {'date': Date(2024, 1, 2), 'value': 300},
    ])
    ds.columns = [('date', Date), ('value', int)]

    result = ds.bucket(['date'], ['value'])

    assert len(result) == 2
    result_by_date = {r.date: r.value for r in result}
    assert result_by_date[Date(2024, 1, 1)] == 300
    assert result_by_date[Date(2024, 1, 2)] == 300


if __name__ == '__main__':
    pytest.main([__file__])


class _Sub(DataSet):
    """Subclass carrying a marker a plain DataSet cannot fake."""

    marker = 'sub'


def test_bucket_returns_the_receivers_class():
    """Verify bucket builds its result through self.__class__.

    Mutation: the result built as a bare `DataSet(buckets)`, so a
    subclass loses its identity on every group-by.
    Oracle: the exact type of the result, and the aggregated value
    beside it so the test still pins behavior.
    """
    ds = _Sub([{'g': 'a', 'v': 1}, {'g': 'a', 'v': 2}])
    ds.columns = [('g', str), ('v', int)]

    grouped = ds.bucket(['g'], ['v'])
    assert type(grouped) is _Sub
    assert grouped[0].v == 3


def test_bucket_dataset_is_reachable_standalone_and_matches_the_method():
    """Verify the free function is exported and agrees with the method.

    The extraction only earns its keep if a caller can reach the
    algorithm without going through the class as a namespace.

    Mutation: the method doing work the free function does not (an
    extra sort, a dropped column), so the two paths diverge; or
    bucket_dataset left out of __all__, so there is no second entry
    point and the module holds a one-caller helper.
    Oracle: hand-computed group totals, asserted against both paths.
    """
    import rollups as pkg

    ds = DataSet([{'g': 'a', 'v': 1}, {'g': 'a', 'v': 2}, {'g': 'b', 'v': 4}])
    ds.columns = [('g', str), ('v', int)]

    assert 'bucket_dataset' in pkg.__all__

    direct = pkg.bucket_dataset(ds, ['g'], ['v'])
    method = ds.bucket(['g'], ['v'])

    assert {r.g: r.v for r in direct} == {'a': 3, 'b': 4}
    assert [dict(r) for r in direct] == [dict(r) for r in method]


def test_bucket_counts_no_value_for_a_column_no_row_carries():
    """Verify an aggregated column absent from every row holds nothing.

    Mutation: reading each row through its own get(), which answers a
        missing key with the dict attribute of that name, so every row
        contributes a bound method and the group looks populated.
    Oracle: hand-computed - two rows carry no column called 'values',
        so len() over the group sees the one-item fallback, not two.
    """
    ds = DataSet([{'g': 'x', 'v': 1}, {'g': 'x', 'v': 2}])

    out = ds.bucket('g', [('values', len)])

    assert [dict(row) for row in out] == [{'g': 'x', 'values': 1}]
