from collections import defaultdict

import pytest
from opendate import UTC, Date, DateTime
from rollups import DataSet

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Basic dataset for partition tests."""
    ds = DataSet([
        {'id': 1, 'name': 'A', 'category': 'X', 'value': 100},
        {'id': 2, 'name': 'B', 'category': 'Y', 'value': 200},
        {'id': 3, 'name': 'C', 'category': 'X', 'value': 300},
        {'id': 4, 'name': 'D', 'category': 'X', 'value': 400},
        {'id': 5, 'name': 'E', 'category': 'Y', 'value': 500}])
    ds.columns = (('id', int), ('name', str), ('category', str), ('value', int))
    return ds


@pytest.fixture
def region_dataset():
    """Dataset with category and region for multi-key partition tests."""
    ds = DataSet([
        {'id': 1, 'category': 'X', 'region': 'North', 'value': 100},
        {'id': 2, 'category': 'Y', 'region': 'South', 'value': 200},
        {'id': 3, 'category': 'X', 'region': 'East', 'value': 300},
        {'id': 4, 'category': 'X', 'region': 'North', 'value': 400},
        {'id': 5, 'category': 'Y', 'region': 'South', 'value': 500}])
    ds.columns = (('id', int), ('category', str), ('region', str), ('value', int))
    return ds


# --- Basic Partition Tests ---

def test_partition_by_single_attribute(basic_dataset):
    """Verify partitioning by single attribute creates correct groups.

    Mutation: dropping empty=True from the defaultdict factory, so each
        group starts with every source row.
    Oracle: hand-computed id lists per category.
    """
    by_category = basic_dataset.partition(lambda x: x.category)

    assert set(by_category) == {'X', 'Y'}
    assert [row.id for row in by_category['X']] == [1, 3, 4]
    assert [row.id for row in by_category['Y']] == [2, 5]
    assert all(row.category == 'X' for row in by_category['X'])
    assert all(row.category == 'Y' for row in by_category['Y'])


def test_partition_by_multiple_attributes(region_dataset):
    """Verify partitioning by tuple of attributes creates correct groups.

    Mutation: keying on partition_func(row)[0], collapsing the tuple to
        its first element.
    Oracle: hand-computed id lists for the three (category, region) pairs.
    """
    by_category_region = region_dataset.partition(lambda x: (x.category, x.region))

    assert set(by_category_region) == {('X', 'North'), ('X', 'East'), ('Y', 'South')}
    assert [row.id for row in by_category_region[('X', 'North')]] == [1, 4]
    assert [row.id for row in by_category_region[('X', 'East')]] == [3]
    assert [row.id for row in by_category_region[('Y', 'South')]] == [2, 5]


def test_partition_by_reference():
    """Verify partition results hold the original rows, not copies.

    Mutation: appending lazydict(row) rather than row, giving each group
        its own row objects.
    Oracle: writes made through the group, read back on the source.
    """
    ds = DataSet([
        {'id': 1, 'category': 'A', 'value': 100},
        {'id': 2, 'category': 'B', 'value': 200},
        {'id': 3, 'category': 'A', 'value': 300},
        {'id': 4, 'category': 'A', 'value': 400},
        {'id': 5, 'category': 'B', 'value': 500}])
    ds.columns = (('id', int), ('category', str), ('value', int))
    by_category = ds.partition(lambda x: x.category)

    for row in by_category['A']:
        row.value = None

    assert ds[0].value is None
    assert ds[2].value is None
    assert ds[3].value is None
    assert ds[1].value == 200
    assert ds[4].value == 500


def test_partition_unpacking(region_dataset):
    """Verify items() covers every source row exactly once, by identity.

    Mutation: rebinding partitions[key] to a fresh dataset per row, so
        only the last row of each key survives.
    Oracle: source row object ids, counted across every group.
    """
    partitions = region_dataset.partition(lambda x: (x.category, x.region))

    placed = sorted(id(row) for rows in partitions.values() for row in rows)
    assert placed == sorted(id(row) for row in region_dataset)
    assert set(partitions) == {('X', 'North'), ('X', 'East'), ('Y', 'South')}
    assert all(isinstance(rows, DataSet) for rows in partitions.values())


def test_partition_with_lambda():
    """Verify partitioning by a derived key keeps rows in source order.

    Mutation: iterating reversed(self.container), or skipping the last
        row.
    Oracle: hand-computed id lists, with value 150 on the high side of
        the threshold.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
        {'id': 3, 'value': 50},
        {'id': 4, 'value': 300},
        {'id': 5, 'value': 150}])
    ds.columns = (('id', int), ('value', int))
    by_range = ds.partition(lambda x: 'low' if x.value < 150 else 'high')

    assert set(by_range) == {'low', 'high'}
    assert [row.id for row in by_range['low']] == [1, 3]
    assert [row.id for row in by_range['high']] == [2, 4, 5]


# --- Empty and Single Group Tests ---

def test_partition_empty_dataset():
    """Verify an empty dataset yields no groups and no key-function calls.

    Mutation: iterating self.container + [self.summary], seeding a
        phantom group from the summary row.
    Oracle: call log of the key function, empty by construction.
    """
    calls = []

    def record_key(row):
        calls.append(row)
        return row.get('key')

    ds = DataSet([])
    partitions = ds.partition(record_key)

    assert calls == []
    assert len(partitions) == 0
    assert isinstance(partitions, defaultdict)


def test_partition_single_group():
    """Verify one shared key gives one group that leaves the source alone.

    Mutation: using self in place of self.copy(empty=True) as the
        factory, so rows are appended back onto the source.
    Oracle: source length and id order, hand-computed.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
        {'id': 3, 'value': 300},
        {'id': 4, 'value': 400},
        {'id': 5, 'value': 500}])
    ds.columns = (('id', int), ('value', int))
    by_constant = ds.partition(lambda x: 'all')

    assert set(by_constant) == {'all'}
    assert by_constant['all'] is not ds
    assert [row.id for row in by_constant['all']] == [1, 2, 3, 4, 5]
    assert len(ds) == 5


# --- Column Preservation Tests ---

def test_partition_preserves_columns():
    """Verify partition results preserve column definitions.

    Mutation: factory returning DataSet() rather than self.copy, so a
        group infers its columns from the rows it happens to receive.
    Oracle: the source column list, compared name by name and type by
        type.
    """
    ds = DataSet([
        {'id': 1, 'category': 'A', 'value': 100},
        {'id': 2, 'category': 'B', 'value': 200},
        {'id': 3, 'category': 'A', 'value': 300},
        {'id': 4, 'category': 'A', 'value': 400},
        {'id': 5, 'category': 'B', 'value': 500}])
    ds.columns = (('id', int), ('category', str), ('value', int))
    by_category = ds.partition(lambda x: x.category)

    for group in by_category.values():
        assert group.cols == ['id', 'category', 'value']
        assert group.typs == [int, str, int]
        assert group.columns == ds.columns


def test_partition_preserves_order():
    """Verify partition maintains row order within groups.

    Mutation: iterating sorted(self, ...) or reversed(self.container).
    Oracle: hand-computed sequence numbers, [1, 3, 5] and [2, 4].
    """
    ds = DataSet([
        {'key': 'a', 'seq': 1},
        {'key': 'b', 'seq': 2},
        {'key': 'a', 'seq': 3},
        {'key': 'b', 'seq': 4},
        {'key': 'a', 'seq': 5}
    ])
    by_key = ds.partition(lambda x: x.key)

    assert [row.seq for row in by_key['a']] == [1, 3, 5]
    assert [row.seq for row in by_key['b']] == [2, 4]


# --- Key Type Tests ---

def test_partition_with_none_values():
    """Verify a None key gets its own group rather than being dropped.

    Mutation: guarding the append with `if key:`, so falsy keys never
        land anywhere.
    Oracle: hand-computed values per key, totaling all four rows.
    """
    ds = DataSet([
        {'key': 'a', 'value': 1},
        {'key': None, 'value': 2},
        {'key': 'b', 'value': 3},
        {'key': None, 'value': 4}
    ])
    by_key = ds.partition(lambda x: x.key)

    assert set(by_key) == {'a', 'b', None}
    assert [row.value for row in by_key[None]] == [2, 4]
    assert [row.value for row in by_key['a']] == [1]
    assert [row.value for row in by_key['b']] == [3]


def test_partition_numeric_keys():
    """Verify numeric keys stay numeric and are not coerced to text.

    Mutation: keying on str(partition_func(row)).
    Oracle: the hand-listed set of int values, plus a type check on each
        key.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
        {'id': 3, 'value': 300},
        {'id': 4, 'value': 400},
        {'id': 5, 'value': 500}])
    ds.columns = (('id', int), ('value', int))
    by_value = ds.partition(lambda x: x.value)

    assert set(by_value) == {100, 200, 300, 400, 500}
    assert all(isinstance(key, int) for key in by_value)
    assert all(len(rows) == 1 for rows in by_value.values())


def test_partition_with_boolean_key():
    """Verify a False key gets its own group at the threshold boundary.

    Mutation: guarding the append with `if key:`, dropping every False
        row.
    Oracle: hand-computed id lists, with value 200 sitting on the False
        side of `> 200`.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 250},
        {'id': 3, 'value': 150},
        {'id': 4, 'value': 300},
        {'id': 5, 'value': 200}])
    ds.columns = (('id', int), ('value', int))
    by_threshold = ds.partition(lambda x: x.value > 200)

    assert set(by_threshold) == {True, False}
    assert [row.id for row in by_threshold[True]] == [2, 4]
    assert [row.id for row in by_threshold[False]] == [1, 3, 5]


# --- DefaultDict Behavior Tests ---

def test_partition_returns_defaultdict():
    """Verify reading an absent key mints an empty dataset, not a KeyError.

    Mutation: returning a plain dict built from the same loop.
    Oracle: a key no row produced, read back after partitioning.
    """
    ds = DataSet([{'a': 1, 'b': 2}])
    partitions = ds.partition(lambda x: x.a)

    nonexistent = partitions[999]
    assert isinstance(nonexistent, DataSet)
    assert len(nonexistent) == 0
    assert nonexistent.cols == ds.cols


def test_partition_defaultdict_preserves_types():
    """Verify a minted group carries declared types and stands on its own.

    Mutation: hoisting self.copy(empty=True) out of the lambda, so every
        key shares one dataset.
    Oracle: appending to the minted group and re-reading the real group.
    """
    ds = DataSet([{'a': 1}], columns=[('a', int), ('b', float)])
    partitions = ds.partition(lambda x: x.a)

    new_group = partitions['nonexistent']

    assert new_group.cols == ['a', 'b']
    assert new_group.typs == [int, float]

    new_group.append({'a': 2, 'b': 3.0})
    assert len(partitions[1]) == 1
    assert len(new_group) == 1


# --- Edge Case Tests ---

def test_partition_with_date_keys():
    """Verify Date keys come from converted values, so a text date joins
    its Date twin.

    Mutation: iterating self.container in place of self, skipping the
        lazy type conversion __iter__ triggers.
    Oracle: hand-computed values per date, with '2024-01-15' landing
        beside Date(2024, 1, 15).
    """
    ds = DataSet([
        {'date': Date(2024, 1, 15), 'value': 100},
        {'date': '2024-01-15', 'value': 200},
        {'date': Date(2024, 6, 30), 'value': 300},
        {'date': Date(2024, 12, 31), 'value': 400}])
    ds.columns = (('date', Date), ('value', int))

    by_date = ds.partition(lambda x: x.date)

    assert set(by_date) == {Date(2024, 1, 15), Date(2024, 6, 30), Date(2024, 12, 31)}
    assert [row.value for row in by_date[Date(2024, 1, 15)]] == [100, 200]
    assert [row.value for row in by_date[Date(2024, 6, 30)]] == [300]
    assert [row.value for row in by_date[Date(2024, 12, 31)]] == [400]


def test_partition_with_datetime_keys():
    """Verify DateTime keys keep their time of day rather than collapsing
    to the date.

    Mutation: keying on str(partition_func(row)) or on the date part
        alone.
    Oracle: two timestamps sharing 2024-01-15 kept apart, hand-computed.
    """
    ds = DataSet([
        {'dt': DateTime(2024, 1, 15, 10, 30, tzinfo=UTC), 'value': 100},
        {'dt': DateTime(2024, 1, 15, 10, 30, tzinfo=UTC), 'value': 200},
        {'dt': DateTime(2024, 1, 15, 16, 0, tzinfo=UTC), 'value': 300},
        {'dt': DateTime(2024, 6, 30, 14, 45, tzinfo=UTC), 'value': 400}])
    ds.columns = (('dt', DateTime), ('value', int))

    by_dt = ds.partition(lambda x: x.dt)

    assert len(by_dt) == 3
    first = by_dt[DateTime(2024, 1, 15, 10, 30, tzinfo=UTC)]
    second = by_dt[DateTime(2024, 1, 15, 16, 0, tzinfo=UTC)]
    third = by_dt[DateTime(2024, 6, 30, 14, 45, tzinfo=UTC)]
    assert [row.value for row in first] == [100, 200]
    assert [row.value for row in second] == [300]
    assert [row.value for row in third] == [400]


def test_partition_function_raises_exception():
    """Verify a raising key function propagates rather than being swallowed.

    Mutation: wrapping partition_func(row) in try/except and filing the
        failures under a None key.
    Oracle: ZeroDivisionError on the row whose value is 0.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 0},
        {'id': 3, 'value': 300}])

    def bad_key_func(x):
        return 1000 / x.value

    with pytest.raises(ZeroDivisionError):
        ds.partition(bad_key_func)


def test_partition_many_groups():
    """Verify each of a hundred unique keys gets its own single-row group.

    Mutation: hoisting self.copy(empty=True) out of the lambda, so every
        key shares one dataset holding all hundred rows.
    Oracle: value == id * 10, recomputed per group.
    """
    ds = DataSet([{'id': i, 'value': i * 10} for i in range(100)])
    ds.columns = (('id', int), ('value', int))

    by_id = ds.partition(lambda x: x.id)

    assert set(by_id) == set(range(100))
    assert all(len(rows) == 1 for rows in by_id.values())
    assert all(by_id[i][0].value == i * 10 for i in range(100))


def test_partition_mixed_type_keys():
    """Verify keys come from converted values, not the raw input.

    Mutation: iterating self.container in place of self, skipping the
        lazy type conversion __iter__ triggers.
    Oracle: the inferred float column makes 1, '1' and 1.0 one key.
    """
    ds = DataSet([
        {'key': 1, 'value': 100},
        {'key': '1', 'value': 200},
        {'key': 1.0, 'value': 300}])

    by_key = ds.partition(lambda x: x.key)

    assert len(by_key) == 1
    assert isinstance(next(iter(by_key)), float)
    assert [row.value for row in by_key[1.0]] == [100, 200, 300]


def test_partition_single_row_dataset():
    """Verify a one-row dataset gives one group holding that same row.

    Mutation: dropping empty=True from the defaultdict factory, so the
        group starts with the source row already in it.
    Oracle: group length of one, and row identity against the source.
    """
    ds = DataSet([{'id': 1, 'category': 'A', 'value': 100}])
    ds.columns = (('id', int), ('category', str), ('value', int))

    by_category = ds.partition(lambda x: x.category)

    assert set(by_category) == {'A'}
    assert len(by_category['A']) == 1
    assert by_category['A'][0] is ds[0]


def test_partition_key_func_returns_tuple_with_none():
    """Verify a None inside a tuple key keeps that key distinct.

    Mutation: normalizing the key, e.g. '-'.join(map(str, key)) or
        dropping None members.
    Oracle: ('X', None) and ('X', 1) held apart, one row each.
    """
    ds = DataSet([
        {'a': 'X', 'b': 1},
        {'a': 'X', 'b': None},
        {'a': 'Y', 'b': 1}])

    by_key = ds.partition(lambda x: (x.a, x.b))

    assert set(by_key) == {('X', 1), ('X', None), ('Y', 1)}
    assert all(len(rows) == 1 for rows in by_key.values())


def test_partition_with_empty_string_key():
    """Verify an empty-string key gets its own group rather than dropped.

    Mutation: guarding the append with `if key:`, so falsy keys never
        land anywhere.
    Oracle: hand-computed values, [100, 300] under the empty key.
    """
    ds = DataSet([
        {'key': '', 'value': 100},
        {'key': 'a', 'value': 200},
        {'key': '', 'value': 300}])

    by_key = ds.partition(lambda x: x.key)

    assert set(by_key) == {'', 'a'}
    assert [row.value for row in by_key['']] == [100, 300]
    assert [row.value for row in by_key['a']] == [200]


def test_partition_preserves_summary():
    """Verify each group inherits the summary declaration and totals only
    its own rows.

    Mutation: factory returning DataSet(columns=self.columns), losing
        _summary_args so the group falls back to default totals.
    Oracle: hand-computed per-group totals, 400 and 200 against the
        source total of 600.
    """
    ds = DataSet([
        {'id': 1, 'category': 'A', 'value': 100},
        {'id': 2, 'category': 'B', 'value': 200},
        {'id': 3, 'category': 'A', 'value': 300}])
    ds.columns = (('id', int), ('category', str), ('value', int))
    ds.add_summary_row(label_idx=1)

    by_category = ds.partition(lambda x: x.category)

    assert not any(DataSet.is_summary_row(row)
                   for group in by_category.values() for row in group)
    assert by_category['A'].summary.category == 'Total'
    assert by_category['A'].summary.value == 400
    assert by_category['B'].summary.value == 200
    assert ds.summary.value == 600


def test_partition_key_func_accesses_multiple_attributes():
    """Verify a key computed from several columns groups by that value.

    Mutation: keying on str(partition_func(row)).
    Oracle: hand-computed a + b sums, 3 twice and 4 once.
    """
    ds = DataSet([
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 1, 'b': 3, 'c': 2},
        {'a': 2, 'b': 1, 'c': 3}])

    by_sum = ds.partition(lambda x: x.a + x.b)

    assert set(by_sum) == {3, 4}
    assert [row.c for row in by_sum[3]] == [3, 3]
    assert [row.c for row in by_sum[4]] == [2]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
