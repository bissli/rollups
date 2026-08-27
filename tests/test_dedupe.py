import datetime

import pytest
from opendate import Date, DateTime
from rollups import DataSet

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Basic dataset with duplicates for dedupe tests."""
    return DataSet([
        {'foo': 1, 'bar': 2},
        {'foo': 1, 'bar': 3},
        {'foo': 2, 'bar': 4}
    ])


@pytest.fixture
def priority_dataset():
    """Dataset with priority column for filter tests."""
    return DataSet([
        {'foo': 1, 'bar': 2, 'priority': 1},
        {'foo': 1, 'bar': 3, 'priority': 2},
        {'foo': 2, 'bar': 4, 'priority': 1}
    ])


# --- Basic Dedupe Tests ---

def test_dedupe_basic_single_key(basic_dataset):
    """Verify dedupe keys on the named column alone, keeping the first row.

    Mutation: unwinding every column instead of the key columns, which
        would call all three rows distinct.
    Oracle: hand-computed rows, one per distinct foo, first occurrence.
    """
    result = basic_dataset.dedupe('foo')
    assert len(result) == 2
    assert result[0] == {'foo': 1, 'bar': 2}
    assert result[1] == {'foo': 2, 'bar': 4}
    assert result.cols == ['foo', 'bar']


@pytest.mark.parametrize('keys', [
    ('foo', 'bar'),   # tuple
    ['foo', 'bar'],   # list
])
def test_dedupe_multiple_keys_formats(keys):
    """Verify a tuple of key columns behaves the same as a list.

    Mutation: isinstance(keys, list) in place of isinstance(keys, list |
        tuple), which wraps a tuple key into a one-element list.
    Oracle: hand-computed two-column dedupe, identical for both forms.
    """
    ds = DataSet([
        {'foo': 1, 'bar': 2},
        {'foo': 1, 'bar': 2},
        {'foo': 1, 'bar': 3}
    ])
    result = ds.dedupe(keys)
    assert [dict(row) for row in result] == [
        {'foo': 1, 'bar': 2},
        {'foo': 1, 'bar': 3},
        ]


def test_dedupe_basic_multiple_keys_list():
    """Verify every listed key column takes part in the duplicate test.

    Mutation: keying on keys[0] alone, which would merge the two
        key1='a' rows that differ only on key2.
    Oracle: hand-computed 3 groups, first value of each.
    """
    ds = DataSet([
        {'key1': 'a', 'key2': 1, 'value': 100},
        {'key1': 'a', 'key2': 1, 'value': 200},
        {'key1': 'a', 'key2': 2, 'value': 300},
        {'key1': 'b', 'key2': 1, 'value': 400}
    ])
    result = ds.dedupe(['key1', 'key2'])
    assert [row['value'] for row in result] == [100, 300, 400]


def test_dedupe_no_duplicates():
    """Verify every row survives, in input order, when no key repeats.

    Mutation: inverting the seen-guard to `if row in d`, which records
        only repeated keys and so empties this result.
    Oracle: hand-computed input order 3, 1, 2 - unsorted on purpose.
    """
    ds = DataSet([
        {'id': 3, 'value': 'a'},
        {'id': 1, 'value': 'b'},
        {'id': 2, 'value': 'c'}
    ])
    result = ds.dedupe('id')
    assert [row['id'] for row in result] == [3, 1, 2]


def test_dedupe_all_duplicates():
    """Verify identical rows collapse to one and the row total follows.

    Mutation: building the result with self.copy(), which carries the
        source's stale total of 3.
    Oracle: hand-computed single row; total equals the kept row count.
    """
    ds = DataSet([
        {'key': 1, 'value': 'same'},
        {'key': 1, 'value': 'same'},
        {'key': 1, 'value': 'same'}
    ])
    result = ds.dedupe('key')
    assert len(result) == 1
    assert result[0] == {'key': 1, 'value': 'same'}
    assert result.total == 1


# --- Filter Function Tests ---

def test_dedupe_with_filter_function(priority_dataset):
    """Verify the filter picks which row of a duplicate group is kept.

    Mutation: negating the filter test to `if not filter_fn(row)`, which
        would keep priority 1 for foo=1.
    Oracle: hand-picked rows - priority 2 for foo=1, the lone foo=2 row.
    """
    ds_filtered = priority_dataset.dedupe('foo', lambda row: row['priority'] == 2)
    assert len(ds_filtered) == 2
    assert ds_filtered[0] == {'foo': 1, 'bar': 3, 'priority': 2}
    assert ds_filtered[1] == {'foo': 2, 'bar': 4, 'priority': 1}


def test_dedupe_filter_selects_first_match():
    """Verify the first matching row of a group wins, not the last.

    Mutation: dropping the break after a match, which appends both
        matching rows of the key=1 group.
    Oracle: hand-computed rows - value 20 beats value 30 on key=1.
    """
    ds = DataSet([
        {'key': 1, 'value': 10, 'flag': False},
        {'key': 1, 'value': 20, 'flag': True},
        {'key': 1, 'value': 30, 'flag': True},
        {'key': 2, 'value': 40, 'flag': True}
    ])
    result = ds.dedupe('key', lambda row: row['flag'])
    assert [row['value'] for row in result] == [20, 40]


def test_dedupe_filter_fallback_to_first():
    """Verify a group with no matching row falls back to its first row.

    Mutation: group_rows[-1] in place of group_rows[0] in the fallback,
        which would keep value 20 for key=1.
    Oracle: hand-computed rows - the first of each unmatched group.
    """
    ds = DataSet([
        {'key': 1, 'value': 10, 'type': 'A'},
        {'key': 1, 'value': 20, 'type': 'A'},
        {'key': 2, 'value': 30, 'type': 'B'}
    ])
    result = ds.dedupe('key', lambda row: row['type'] == 'C')
    assert [row['value'] for row in result] == [10, 30]


def test_dedupe_complex_filter():
    """Verify the filtered result comes back ordered by key, not by input.

    Mutation: dropping sorted() around groups.keys() in the filter
        branch, which would put id=2 first because it arrives first.
    Oracle: hand-computed order id=1 then id=2, against a fixture whose
        input order is the reverse.
    """
    ds = DataSet([
        {'id': 2, 'score': 75, 'date': Date(2024, 1, 1)},
        {'id': 1, 'score': 85, 'date': Date(2024, 1, 1)},
        {'id': 1, 'score': 90, 'date': Date(2024, 1, 2)},
        {'id': 1, 'score': 88, 'date': Date(2024, 1, 3)}
    ])
    result = ds.dedupe('id', lambda row: row['score'] >= 90)
    assert [row['id'] for row in result] == [1, 2]
    assert [row['score'] for row in result] == [90, 75]


def test_dedupe_filter_with_multiple_conditions():
    """Verify the filter sees the whole row, not just the key columns.

    Mutation: passing {k: row[k] for k in keys} to filter_fn, which
        hides the status and priority columns from it.
    Oracle: hand-picked rows chosen on two non-key columns.
    """
    ds = DataSet([
        {'id': 1, 'status': 'active', 'priority': 1},
        {'id': 1, 'status': 'active', 'priority': 2},
        {'id': 1, 'status': 'inactive', 'priority': 3},
        {'id': 2, 'status': 'active', 'priority': 1}
    ])
    result = ds.dedupe(
        'id',
        lambda row: row['status'] == 'active' and row['priority'] == 2)
    assert [row['priority'] for row in result] == [2, 1]


# --- Column Preservation Tests ---

def test_dedupe_preserves_column_types():
    """Verify declared column types carry into the deduped dataset.

    Mutation: dropping typs=self.typs from the rebuild, which re-infers
        the all-None note column as NoneType.
    Oracle: the schema declared on the source, compared whole.
    """
    ds = DataSet([
        {'date': Date(2024, 1, 1), 'note': None, 'value': 100},
        {'date': Date(2024, 1, 1), 'note': None, 'value': 200},
        {'date': Date(2024, 1, 2), 'note': None, 'value': 300}
    ])
    ds.columns = [('date', Date), ('note', str), ('value', int)]
    result = ds.dedupe('date')
    assert len(result) == 2
    assert result.colmap == {'date': Date, 'note': str, 'value': int}


def test_dedupe_preserves_column_order():
    """Verify column order comes from the schema, not from the row keys.

    Mutation: dropping cols=self.cols from the rebuild, which re-reads
        the order off the first row as z, a, m.
    Oracle: the declared order a, m, z, deliberately unlike the row
        key order.
    """
    ds = DataSet([
        {'z': 1, 'a': 2, 'm': 3},
        {'z': 1, 'a': 3, 'm': 4}
    ])
    ds.columns = [('a', int), ('m', int), ('z', int)]
    result = ds.dedupe('z')
    assert len(result) == 1
    assert result.cols == ['a', 'm', 'z']


def test_dedupe_multiple_keys_preserves_order():
    """Verify a composite-key result keeps first-seen rows in input order.

    Mutation: dropping the seen-guard so d[row] = i records the last
        index, which would keep sales 150 for East/A.
    Oracle: hand-computed first sale of each region-product pair.
    """
    ds = DataSet([
        {'region': 'East', 'product': 'A', 'sales': 100},
        {'region': 'East', 'product': 'B', 'sales': 200},
        {'region': 'East', 'product': 'A', 'sales': 150},
        {'region': 'West', 'product': 'A', 'sales': 300}
    ])
    result = ds.dedupe(['region', 'product'])
    assert [row['sales'] for row in result] == [100, 200, 300]


# --- Key Type Tests ---

def test_dedupe_string_keys():
    """Verify string keys match exactly, without case folding.

    Mutation: lowercasing a string key before the seen-test, which would
        merge name 'a' into name 'A'.
    Oracle: hand-computed 3 groups - 'A', 'a', 'B' - and the first city
        of each.
    """
    ds = DataSet([
        {'name': 'A', 'city': 'NYC', 'age': 30},
        {'name': 'A', 'city': 'LA', 'age': 31},
        {'name': 'a', 'city': 'SF', 'age': 26},
        {'name': 'B', 'city': 'NYC', 'age': 25}
    ])
    result = ds.dedupe('name')
    assert [row['name'] for row in result] == ['A', 'a', 'B']
    assert [row['city'] for row in result] == ['NYC', 'SF', 'NYC']


def test_dedupe_numeric_keys():
    """Verify each numeric key group keeps its first row.

    Mutation: dropping the seen-guard so d[row] = i records the last
        index, which would keep values b and d.
    Oracle: hand-computed rows a and c.
    """
    ds = DataSet([
        {'id': 1, 'value': 'a'},
        {'id': 1, 'value': 'b'},
        {'id': 2, 'value': 'c'},
        {'id': 2, 'value': 'd'}
    ])
    result = ds.dedupe('id')
    assert [dict(row) for row in result] == [
        {'id': 1, 'value': 'a'},
        {'id': 2, 'value': 'c'},
        ]


def test_dedupe_datetime_keys():
    """Verify a DateTime key keeps its time of day when converted.

    Mutation: Date.instance in place of DateTime.instance in
        convert_container_types, which truncates the key to its date and
        merges the 10:00 and 11:00 rows.
    Oracle: hand-computed 2 groups from 3 rows sharing one calendar day.
    """
    ds = DataSet([
        {'timestamp': datetime.datetime(2024, 1, 1, 10, 0, 0), 'event': 'first'},
        {'timestamp': datetime.datetime(2024, 1, 1, 10, 0, 0), 'event': 'duplicate'},
        {'timestamp': datetime.datetime(2024, 1, 1, 11, 0, 0), 'event': 'second'}
    ])
    ds.columns = [('timestamp', DateTime), ('event', str)]
    result = ds.dedupe('timestamp')
    assert [row['event'] for row in result] == ['first', 'second']
    assert [row['timestamp'].hour for row in result] == [10, 11]


def test_dedupe_date_keys():
    """Verify equal Date values match by value, not by object identity.

    Mutation: keying the seen-dict on id(row) rather than the key value,
        which would treat the two separately built Date(2024, 1, 15)
        instances as distinct.
    Oracle: hand-computed 2 groups, first value of each.
    """
    ds = DataSet([
        {'date': Date(2024, 1, 15), 'value': 'first'},
        {'date': Date(2024, 1, 15), 'value': 'duplicate'},
        {'date': Date(2024, 6, 30), 'value': 'second'}
    ])
    ds.columns = [('date', Date), ('value', str)]
    result = ds.dedupe('date')
    assert [row['value'] for row in result] == ['first', 'second']


def test_dedupe_float_keys():
    """Verify float keys compare at full precision, not truncated.

    Mutation: int(key) normalization before the seen-test, which would
        merge 1.5 and 1.9.
    Oracle: hand-computed 3 groups from keys 1.5, 1.9, 2.5.
    """
    ds = DataSet([
        {'key': 1.5, 'value': 'a'},
        {'key': 1.5, 'value': 'b'},
        {'key': 1.9, 'value': 'c'},
        {'key': 2.5, 'value': 'd'}
    ])
    ds.columns = [('key', float), ('value', str)]
    result = ds.dedupe('key')
    assert [row['value'] for row in result] == ['a', 'c', 'd']


def test_dedupe_mixed_type_keys():
    """Verify a key column no single type covers is not unified.

    The column holds int, float and str, so it infers `object` and
    nothing converts. 1 and 1.0 still share a group by `==`; '1' does
    not, because no conversion makes it numeric.

    Mutation: dropping the seen-guard so the last row of a group wins,
        which would surface value 'float' for the numeric key.
    Oracle: hand-computed - two groups, each keeping its first row.
    """
    ds = DataSet([
        {'key': 1, 'value': 'int'},
        {'key': 1.0, 'value': 'float'},
        {'key': '1', 'value': 'str'}
    ])
    assert ds.colmap['key'] is object

    result = ds.dedupe('key')
    assert len(result) == 2
    assert [row.value for row in result] == ['int', 'str']
    assert result[0] == {'key': 1.0, 'value': 'int'}


def test_dedupe_long_values():
    """Verify a long string key is compared whole, not by prefix.

    Mutation: truncating a string key to a fixed prefix before the
        seen-test, which would merge two 1000-char keys differing only
        in their last character.
    Oracle: hand-computed 2 groups, first value of each.
    """
    key_a = 'a' * 1000
    key_b = 'a' * 999 + 'b'
    ds = DataSet([
        {'key': key_a, 'value': 1},
        {'key': key_a, 'value': 2},
        {'key': key_b, 'value': 3}
    ])
    result = ds.dedupe('key')
    assert [row['value'] for row in result] == [1, 3]


# --- Empty and Single Row Tests ---

def test_dedupe_empty_dataset():
    """Verify an empty dataset keeps its declared schema.

    Mutation: dropping typs=self.typs from the rebuild, which leaves an
        empty schema once there is no row to infer from.
    Oracle: the schema declared on the source, compared whole.
    """
    ds = DataSet([], columns=[('key', int), ('value', str)])
    result = ds.dedupe('key')
    assert len(result) == 0
    assert result.colmap == {'key': int, 'value': str}


def test_dedupe_single_row():
    """Verify a one-row dataset comes back as a separate dataset.

    Mutation: an early `return self` shortcut for a dataset too short to
        hold a duplicate.
    Oracle: the result is a different object carrying the same row.
    """
    ds = DataSet([{'key': 1, 'value': 'only'}])
    result = ds.dedupe('key')
    assert result is not ds
    assert len(result) == 1
    assert result[0] == {'key': 1, 'value': 'only'}


# --- None Value Tests ---

def test_dedupe_with_none_values():
    """Verify None is a key value like any other, not a skipped row.

    Mutation: a `if row is None: continue` guard in the seen-loop, which
        would drop both None-keyed rows.
    Oracle: hand-computed 2 groups, first value of each.
    """
    ds = DataSet([
        {'key': None, 'value': 1},
        {'key': None, 'value': 2},
        {'key': 1, 'value': 3}
    ])
    result = ds.dedupe('key')
    assert [row['value'] for row in result] == [1, 3]


def test_dedupe_with_none_in_multiple_keys():
    """Verify a composite key separates rows differing only on key2.

    Mutation: keying on keys[0] alone, which would merge all four rows
        because key1 is None throughout.
    Oracle: hand-computed 3 groups from key2 values 1, None and 2.
    """
    ds = DataSet([
        {'key1': None, 'key2': 1, 'value': 'a'},
        {'key1': None, 'key2': 1, 'value': 'b'},
        {'key1': None, 'key2': None, 'value': 'c'},
        {'key1': None, 'key2': 2, 'value': 'd'}
    ])
    result = ds.dedupe(['key1', 'key2'])
    assert [row['value'] for row in result] == ['a', 'c', 'd']


def test_dedupe_empty_string_key():
    """Verify an empty string is a key value, not a missing one.

    Mutation: a falsy guard such as `if not row: continue` in the
        seen-loop, which would drop both empty-key rows.
    Oracle: hand-computed 2 groups, first value of each.
    """
    ds = DataSet([
        {'key': '', 'value': 1},
        {'key': '', 'value': 2},
        {'key': 'a', 'value': 3}
    ])
    result = ds.dedupe('key')
    assert [row['value'] for row in result] == [1, 3]


# --- Return Type Tests ---

def test_dedupe_returns_new_dataset():
    """Verify dedupe leaves the source dataset alone.

    Mutation: assigning the kept rows back to self.container before
        returning, which would shrink the source to one row.
    Oracle: the source still holds both input rows after the call.
    """
    ds = DataSet([
        {'key': 1, 'value': 'a'},
        {'key': 1, 'value': 'b'}
    ])
    result = ds.dedupe('key')
    assert isinstance(result, DataSet)
    assert result is not ds
    assert len(result) == 1
    assert [row['value'] for row in ds] == ['a', 'b']


# --- Edge Case Tests ---

def test_dedupe_filter_not_called_after_match():
    """Verify the filter stops being called once a row of a group matches.

    Mutation: dropping the break after a match, which reaches the value
        0 row and divides by zero.
    Oracle: a call-recording filter - rows 100 and 200 only, never 0.
    """
    seen_values = []

    def counting_filter(row):
        seen_values.append(row['value'])
        return 1000 / row['value'] > 5

    ds = DataSet([
        {'key': 1, 'value': 100},
        {'key': 1, 'value': 0},
        {'key': 2, 'value': 200}
    ])
    result = ds.dedupe('key', counting_filter)
    assert seen_values == [100, 200]
    assert [row['value'] for row in result] == [100, 200]


def test_dedupe_preserves_summary():
    """Verify the deduped dataset does not inherit the source's summary.

    Mutation: building the result with self.copy(), which carries the
        source's computed total of 600 and its GRAND label.
    Oracle: hand-computed 100 + 300 over the kept rows, under the
        default label.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 1, 'value': 200},
        {'id': 2, 'value': 300}
    ])
    ds.columns = [('id', int), ('value', int)]
    ds.add_summary_row(label='GRAND')
    assert ds.summary['value'] == 600

    result = ds.dedupe('id')

    assert len(result) == 2
    assert result.summary['value'] == 400
    assert result.summary['id'] == 'Total'


def test_dedupe_many_duplicates():
    """Verify a 100-row group collapses to its first row.

    Mutation: dropping the seen-guard so d[row] = i records the last
        index, which would keep value 99.
    Oracle: hand-computed value 0, the first of the hundred.
    """
    ds = DataSet([{'key': 1, 'value': i} for i in range(100)])
    result = ds.dedupe('key')
    assert len(result) == 1
    assert result[0]['value'] == 0


def test_dedupe_many_groups():
    """Verify 100 distinct keys all survive, each with its own row.

    Mutation: inverting the seen-guard to `if row in d`, which records
        only repeated keys and so empties this result.
    Oracle: hand-computed value = key * 10 across all 100 rows.
    """
    ds = DataSet([{'key': i, 'value': i * 10} for i in range(100)])
    result = ds.dedupe('key')
    assert [row['value'] for row in result] == [i * 10 for i in range(100)]


def test_dedupe_filter_returns_non_boolean():
    """Verify a truthy non-boolean filter result counts as a match.

    Mutation: `if filter_fn(row) is True` in place of the truth test,
        which would match nothing and fall back to the value 0 row.
    Oracle: hand-computed value 5, the first truthy row.
    """
    ds = DataSet([
        {'key': 1, 'value': 0},
        {'key': 1, 'value': 5},
        {'key': 1, 'value': 10}
    ])

    result = ds.dedupe('key', lambda row: row['value'])
    assert len(result) == 1
    assert result[0]['value'] == 5


def test_dedupe_single_column_dataset():
    """Verify a single-column dataset keeps first-seen input order.

    Mutation: sorting the no-filter output by key, as the filter branch
        does, which would return 1 before 2.
    Oracle: hand-computed input order 2, 1 from keys 2, 1, 2.
    """
    ds = DataSet([
        {'key': 2},
        {'key': 1},
        {'key': 2}
    ])
    result = ds.dedupe('key')
    assert [row['key'] for row in result] == [2, 1]
    assert result.cols == ['key']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
