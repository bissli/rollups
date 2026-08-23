"""Comprehensive tests for DataSet basic operations.

TODO: Define clear semantics for `dataset.columns = new_columns` assignment.
Currently this only updates schema metadata without synchronizing row data:
- Doesn't remove columns from rows that are no longer in the schema
- Doesn't add missing columns to rows
- Doesn't convert types for existing columns
- Creates schema/data divergence

Options:
A. Strict Synchronization: Make `columns =` remove unlisted columns from rows,
   add missing columns with None, and mark types as needing conversion
B. Deprecate direct assignment: Require using add_column, remove_column,
   rename_column helper methods for all modifications
C. Schema-only assignment: Document that direct assignment only updates
   metadata and requires manual row updates or convert_container_types()

Recommendation: Option A for consistency with existing helper methods.
"""
import logging
import operator
import pickle

import pytest
from rollups import DataSet, find

from libb import attrdict

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Standard dataset with 5 rows for list operations."""
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 200},
        {'id': 3, 'name': 'C', 'value': 300},
        {'id': 4, 'name': 'D', 'value': 400},
        {'id': 5, 'name': 'E', 'value': 500}])
    ds.columns = (('id', int), ('name', str), ('value', int))
    return ds


@pytest.fixture
def sort_dataset():
    """Sorting fixture whose row order no single column reproduces."""
    ds = DataSet([
        {'id': 3, 'name': 'C', 'category': 'X', 'value': 160},
        {'id': 1, 'name': 'A', 'category': 'X', 'value': 150},
        {'id': 5, 'name': 'E', 'category': 'Y', 'value': 195},
        {'id': 2, 'name': 'B', 'category': 'Y', 'value': 175},
        {'id': 4, 'name': 'D', 'category': 'X', 'value': 215}])
    ds.columns = (('id', int), ('name', str), ('category', str), ('value', int))
    return ds


# --- Serialization Tests ---

def test_basic_serialization():
    """Verify pickle round-trips rows, schema, and a declared summary.

    Mutation: a __getstate__/__setstate__ pair dropping _columns or
        _summary_args, so the copy re-guesses its schema or falls back
        to the default label column.
    Oracle: the pre-pickle dataset, plus the hand-computed total 300
        labeled in 'category' rather than the default 'name'.
    """
    ds = DataSet([
        {'name': 'A', 'category': 'X', 'value': 100},
        {'name': 'B', 'category': 'Y', 'value': 200},
        {'name': 'C', 'category': 'X', 'value': 300}])
    ds.columns = (('name', str), ('category', str), ('value', int))
    ds.filter_data(lambda row: row['value'] < 300)
    ds.add_summary_row(label_idx=1)

    ds2 = pickle.loads(pickle.dumps(ds))

    assert ds2.container == ds.container
    assert ds2.columns == ds.columns
    assert ds2.colmap == ds.colmap
    assert ds2.summary['value'] == 300
    assert ds2.summary['category'] == 'Total'
    assert ds2.summary['name'] is None


def test_serialization_with_nested_datasets():
    """Verify a nested DataSet is left as-is on read and survives pickling.

    Mutation: convert_container_types dropping the isinstance short
        circuit, so the first read rebuilds the nested DataSet.
    Oracle: identity against the inner dataset, and its own rows after
        the round trip.
    """
    inner = DataSet([{'x': 1}, {'x': 2}])
    ds = DataSet([{'name': 'A', 'nested': inner}])

    assert ds[0]['nested'] is inner

    ds2 = pickle.loads(pickle.dumps(ds))
    nested = ds2[0]['nested']

    assert len(ds2) == 1
    assert isinstance(nested, DataSet)
    assert nested.container == [{'x': 1}, {'x': 2}]


# --- List-like Operations Tests (split from monolithic test_list) ---

def test_len(basic_dataset):
    """Verify len() counts container rows, not the pagination total.

    Mutation: __len__ returning self.total, which pagination sets from
        its own argument.
    Oracle: 5 rows against a declared total of 99.
    """
    assert len(basic_dataset) == 5

    paged = DataSet(list(basic_dataset.container), total=99, per_page=10)
    assert len(paged) == 5
    assert paged.total == 99


def test_indexing(basic_dataset):
    """Verify indexing addresses rows and returns the row object itself.

    Mutation: an off-by-one in __getitem__, or returning dict(row) so a
        caller cannot write through the index.
    Oracle: hand-checked names at 0, 2 and -1, plus identity against
        the container.
    """
    assert basic_dataset[0]['name'] == 'A'
    assert basic_dataset[2]['name'] == 'C'
    assert basic_dataset[-1]['name'] == 'E'
    assert basic_dataset[2] is basic_dataset.container[2]


def test_append(basic_dataset):
    """Verify append lands the row at the end and validate converts it.

    Mutation: append inserting at position 0, or the validate branch
        skipping _convert_value.
    Oracle: hand-checked tail name 'F', and the string '700' typed int.
    """
    basic_dataset.append({'id': 6, 'name': 'F', 'value': 600})
    assert len(basic_dataset) == 6
    assert basic_dataset[-1]['name'] == 'F'

    basic_dataset.append({'id': 7, 'name': 'G', 'value': '700'}, validate=True)
    assert basic_dataset[-1]['value'] == 700
    assert isinstance(basic_dataset[-1]['value'], int)


def test_extend_with_list(basic_dataset):
    """Verify extend appends every row of a list, in order.

    Mutation: extend appending the sequence itself, or only its first
        row.
    Oracle: hand-computed name order A through H.
    """
    basic_dataset.extend([
        {'id': 6, 'name': 'F', 'value': 600},
        {'id': 7, 'name': 'G', 'value': 700},
        {'id': 8, 'name': 'H', 'value': 800}])
    assert len(basic_dataset) == 8
    assert [row['name'] for row in basic_dataset] == list('ABCDEFGH')


def test_extend_with_dataset(basic_dataset):
    """Verify extend over a DataSet adds its rows, not the DataSet.

    Mutation: extend appending the operand itself, so the tail holds a
        DataSet rather than a row.
    Oracle: hand-computed name order A through G.
    """
    other = DataSet([
        {'id': 6, 'name': 'F', 'value': 600},
        {'id': 7, 'name': 'G', 'value': 700}])
    basic_dataset.extend(other)
    assert len(basic_dataset) == 7
    assert [row['name'] for row in basic_dataset] == list('ABCDEFG')


def test_add_operator_datasets():
    """Verify + joins left then right, leaving both operands alone.

    Mutation: __add__ extending self in place, or building the result
        from the right operand first.
    Oracle: hand-computed name order A, B, C, D and unchanged operand
        lengths.
    """
    ds1 = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 200}])
    ds2 = DataSet([
        {'id': 3, 'name': 'C', 'value': 300},
        {'id': 4, 'name': 'D', 'value': 400}])

    result = ds1 + ds2

    assert [row['name'] for row in result] == ['A', 'B', 'C', 'D']
    assert len(ds1) == 2
    assert len(ds2) == 2


def test_add_operator_with_list():
    """Verify + takes a list operand and keeps the left-hand schema.

    Mutation: __add__ building a bare DataSet from the joined rows, so
        the left operand's declared types are re-guessed and lost.
    Oracle: hand-computed name order, and the schema declared on ds.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 200}])
    ds.columns = [('id', int), ('name', str), ('value', float)]

    result = ds + [{'id': 3, 'name': 'C', 'value': 300}]

    assert [row['name'] for row in result] == ['A', 'B', 'C']
    assert result.columns == [('id', int), ('name', str), ('value', float)]
    assert len(ds) == 2


def test_slice(basic_dataset):
    """Verify a slice yields a new DataSet over the requested rows only.

    Mutation: the slice branch handing back the container unsliced, or
        slicing self in place.
    Oracle: hand-computed names B, C, D and the untouched source length.
    """
    sl = basic_dataset[1:4]

    assert isinstance(sl, DataSet)
    assert [row['name'] for row in sl] == ['B', 'C', 'D']
    assert len(basic_dataset) == 5


def test_slicing_empty_dataset():
    """Verify slicing an empty dataset keeps the schema on a new DataSet.

    Mutation: the slice branch returning self.container[key], a bare
        list carrying no columns.
    Oracle: the declared schema [('a', int), ('b', str)].
    """
    ds = DataSet([], columns=[('a', int), ('b', str)])
    sl = ds[0:5]

    assert len(sl) == 0
    assert isinstance(sl, DataSet)
    assert sl.columns == [('a', int), ('b', str)]


def test_copy_independence(basic_dataset):
    """Verify copy() gives a new row list over the same row objects.

    Mutation: copy() aliasing self.container, or rebuilding each row the
        way shallowcopy does.
    Oracle: hand-computed orders after sorting only the copy, and a
        write through the copy showing up in the source.
    """
    cp = basic_dataset.copy()
    cp.sort_data('-value')

    assert [row['name'] for row in cp] == ['E', 'D', 'C', 'B', 'A']
    assert [row['name'] for row in basic_dataset] == ['A', 'B', 'C', 'D', 'E']

    cp[0]['value'] = 999
    assert basic_dataset[-1]['value'] == 999


# --- Sorting Tests ---

def test_sort_with_none_key(sort_dataset):
    """Verify sort_data with a None column leaves the row order alone.

    Mutation: sort_data mapping a None column name onto the first
        column before it delegates.
    Oracle: the fixture's own id order 3, 1, 5, 2, 4, which no column
        sort reproduces.
    """
    sort_dataset.sort_data(None)
    assert list(sort_dataset.unwind('id')) == [3, 1, 5, 2, 4]


def test_sort_with_tuple_keys(sort_dataset):
    """Verify sort() with a tuple key sorts by those columns in order.

    Mutation: reversing the key order to (value, category), which agrees
        on the first two rows and diverges at the third.
    Oracle: hand-sorted values 150, 160, 215, 175, 195.
    """
    sort_dataset.sort(('category', 'value'))
    assert list(sort_dataset.unwind('value')) == [150, 160, 215, 175, 195]


def test_sort_data_multiple_columns(sort_dataset):
    """Verify sort_data reads a leading '-' as descending, per column.

    Mutation: ignoring the '-' prefix, or applying it to every column.
    Oracle: hand-sorted values 215, 160, 150, 195, 175.
    """
    sort_dataset.sort_data('category', '-value')
    assert list(sort_dataset.unwind('value')) == [215, 160, 150, 195, 175]


@pytest.mark.parametrize(('key', 'expected'), [
    ('name', ['A', 'B', 'C', 'D', 'E']),
    ('id', [1, 2, 3, 4, 5]),
    ('value', [150, 160, 175, 195, 215]),
])
def test_sort_data_single_column(sort_dataset, key, expected):
    """Verify sort_data with one column orders the whole dataset by it.

    Mutation: sorting on the wrong column, or leaving the rows put.
    Oracle: the hand-sorted full sequence, against a fixture whose row
        order no column reproduces.
    """
    sort_dataset.sort_data(key)
    assert list(sort_dataset.unwind(key)) == expected


def test_sort_data_descending(sort_dataset):
    """Verify sort_data('-value') orders descending.

    Mutation: dropping the '-' so the sort runs ascending.
    Oracle: hand-sorted values 215, 195, 175, 160, 150.
    """
    sort_dataset.sort_data('-value')
    assert list(sort_dataset.unwind('value')) == [215, 195, 175, 160, 150]


def test_sort_with_itemgetter(sort_dataset):
    """Verify sort() takes an operator.itemgetter as its key.

    Mutation: sort() ignoring a callable key and sorting the rows
        themselves.
    Oracle: hand-sorted names A, B, C, D, E from an unsorted fixture.
    """
    sort_dataset.sort(operator.itemgetter('name'))
    assert list(sort_dataset.unwind('name')) == ['A', 'B', 'C', 'D', 'E']


def test_sort_with_lambda(sort_dataset):
    """Verify sort() orders by a lambda key, not by any column.

    Mutation: sort() falling back to attrgetter on the key, or ignoring
        the key entirely.
    Oracle: hand-computed order D, A, C, B, E from value modulo 100.
    """
    sort_dataset.sort(lambda row: row['value'] % 100)
    assert list(sort_dataset.unwind('name')) == ['D', 'A', 'C', 'B', 'E']


def test_sort_with_reverse(sort_dataset):
    """Verify sort() passes reverse through to the underlying sort.

    Mutation: dropping reverse from the list.sort call.
    Oracle: hand-sorted names E, D, C, B, A.
    """
    sort_dataset.sort(lambda row: row['name'], reverse=True)
    assert list(sort_dataset.unwind('name')) == ['E', 'D', 'C', 'B', 'A']


def test_sort_with_none_values():
    """Verify sort_data ranks None below every value, in both directions.

    Mutation: sort_data sorting a copy (inplace=False), or stripping
        the '-' prefix, which loses the descending arm.
    Oracle: hand-computed ids [1, 3, 2] ascending, [2, 3, 1] descending.
    """
    ds = DataSet([
        {'id': 1, 'value': None},
        {'id': 2, 'value': 100},
        {'id': 3, 'value': 50}])

    ds.sort_data('value')
    assert list(ds.unwind('id')) == [1, 3, 2]

    ds.sort_data('-value')
    assert list(ds.unwind('id')) == [2, 3, 1]


def test_sort_with_column_name(sort_dataset):
    """Verify sort() takes a bare column name as its key.

    Mutation: sort() dropping the (key,) wrap before attrgetter, which
        leaves a plain column name unusable.
    Oracle: hand-sorted names A, B, C, D, E, with the values 150, 175,
        160, 215, 195 carried along.
    """
    sort_dataset.sort('name')

    assert list(sort_dataset.unwind('name')) == ['A', 'B', 'C', 'D', 'E']
    assert list(sort_dataset.unwind('value')) == [150, 175, 160, 215, 195]


def test_sort_with_column_name_reverse(sort_dataset):
    """Verify a column-name key sorts descending on reverse=True.

    Mutation: dropping reverse from the attrgetter sort, which orders
        ascending whatever the caller asked for.
    Oracle: hand-sorted names E, D, C, B, A, the exact reverse of the
        ascending order, with values 195, 215, 160, 175, 150.
    """
    sort_dataset.sort('name', reverse=True)

    assert list(sort_dataset.unwind('name')) == ['E', 'D', 'C', 'B', 'A']
    assert list(sort_dataset.unwind('value')) == [195, 215, 160, 175, 150]


# --- Summary Row Tests ---

def test_summary_row(basic_dataset):
    """Verify the summary totals numeric columns and stamps the label.

    Mutation: an off-by-one on label_idx, which drops 'id' from the
        totaled columns and writes the label there instead.
    Oracle: hand-computed sums 15 and 1500, label in 'name'.
    """
    basic_dataset.add_summary_row(label_idx=1)

    assert basic_dataset.summary['id'] == 15
    assert basic_dataset.summary['value'] == 1500
    assert basic_dataset.summary['name'] == 'Total'


def test_summary_row_after_filter(basic_dataset):
    """Verify a summary declared after filtering totals kept rows only.

    Mutation: _match dropping the case-insensitive fold, so the
        lowercase pattern matches nothing, or the summary totaling the
        pre-filter container.
    Oracle: the single 'A' row, hand-computed totals 1 and 100.
    """
    basic_dataset.filter_data('a')
    basic_dataset.add_summary_row(label_idx=1)

    assert list(basic_dataset.unwind('name')) == ['A']
    assert basic_dataset.summary['id'] == 1
    assert basic_dataset.summary['value'] == 100


def test_summary_row_before_filter(basic_dataset):
    """Verify add_summary_row defers the total until the summary is read.

    Mutation: add_summary_row computing and caching the total at once,
        so a later filter cannot change it.
    Oracle: hand-computed 1 and 100, against the pre-filter 15 and 1500.
    """
    basic_dataset.add_summary_row(label_idx=1)
    basic_dataset.filter_data('A')

    assert basic_dataset.summary['id'] == 1
    assert basic_dataset.summary['value'] == 100


def test_summary_is_lazy_and_reflects_edits():
    """Verify the summary is built on read and re-totals the current rows.

    Mutation: __init__ priming the total eagerly, or caching the first
        result so an edit made after the first read is missed.
    Oracle: hand-computed 30 on the first read, then 130 after one row
        grows by 100.
    """
    z = DataSet([attrdict(y='a', x=10), attrdict(y='b', x=20)])

    assert z.summary.x == 30
    z[0]['x'] = 110
    assert z.summary.x == 130


# --- attrdict Construction Tests ---

def test_basic_operations_from_attrdict():
    """Verify copy shares rows and deepcopy does not, over attrdict rows.

    Mutation: copy() rebuilding rows, hiding the 99 write, or deepcopy()
        sharing them, leaking the 33 write back.
    Oracle: hand-computed x values 99 and 11, and the total 144.
    """
    a = attrdict(y='foo', x=10)
    b = attrdict(y='bar', x=11)
    c = attrdict(y='baz', x=12)
    z = DataSet((a, b, c))

    assert [_.x for _ in z] == [10, 11, 12]

    z.columns = [('y', str), ('x', int)]
    assert z[0].x == 10

    w = z.copy()
    w[0].x = 99
    assert z[0].x == 99

    ww = z.deepcopy()
    ww[1].x = 33
    assert z[1].x == 11

    ww.add_summary_row()
    assert ww.summary.x == 144


# --- Order, Itemize, Sample Tests ---

def test_order_method():
    """Verify order() rearranges whole rows into the value order given.

    Mutation: order() popping the wrong index, or leaving the unmatched
        rows in front of the reordered ones.
    Oracle: hand-specified order b, a, x over the input a, x, b, with
        the bar column carried along.
    """
    x = DataSet([
        {'foo': 'a', 'bar': 'c'},
        {'foo': 'x', 'bar': 'e'},
        {'foo': 'b', 'bar': 'd'}])

    x.order('foo', 'b', 'a', 'x')

    assert list(x.unwind('foo')) == ['b', 'a', 'x']
    assert list(x.unwind('bar')) == ['d', 'c', 'e']


def test_order_rejects_length_mismatch():
    """Verify order() refuses a value list that is not one per row.

    Mutation: garbling the length-guard message, or dropping the guard,
        so a short list silently reorders part of the dataset.
    Oracle: the documented AssertionError text, plus the input order
        a, x, b left untouched by the refusal.
    """
    x = DataSet([
        {'foo': 'a', 'bar': 'c'},
        {'foo': 'x', 'bar': 'e'},
        {'foo': 'b', 'bar': 'd'}])

    with pytest.raises(AssertionError) as refusal:
        x.order('foo', 'b', 'a')

    assert str(refusal.value) == 'container length != args length'
    assert list(x.unwind('foo')) == ['a', 'x', 'b']


def test_itemize():
    """Verify itemize splits into one single-row DataSet per row.

    Mutation: zipping the values against sorted(self.cols) rather than
        column order, which swaps the b and a values.
    Oracle: hand-written rows {'b': 1, 'a': 2} and {'b': 4, 'a': 3}.
    """
    ds = DataSet([{'b': 1, 'a': 2}, {'a': 3, 'b': 4}])
    result = ds.itemize()

    assert all(isinstance(item, DataSet) for item in result)
    assert [item.container for item in result] == [
        [{'b': 1, 'a': 2}],
        [{'b': 4, 'a': 3}]]
    assert result[0].cols == ['b', 'a']


def test_itemize_carries_declared_types():
    """Verify each single-row DataSet keeps the parent's declared types.

    Mutation: itemize letting the new DataSet infer its own types
        rather than passing typs, which reads an all-None column as
        NoneType.
    Oracle: the declared schema (int, str), against rows whose values
        alone infer (NoneType, str).
    """
    ds = DataSet([{'a': None, 'b': 'x'}, {'a': None, 'b': 'y'}],
                 columns=[('a', int), ('b', str)])

    items = ds.itemize()

    assert [item.typs for item in items] == [[int, str], [int, str]]
    assert [item.cols for item in items] == [['a', 'b'], ['a', 'b']]


def test_sample():
    """Verify sample clamps n to the row count and returns a deep copy.

    Mutation: dropping the min/max clamp, or copying shallowly so a
        write through the sample reaches the source rows.
    Oracle: boundary counts at n of 3, 7 and -1, and the untouched
        source values.
    """
    x = DataSet([{'a': i, 'b': i} for i in range(1, 7)])

    assert len(x.sample(3)) == 3
    assert len(x.sample(7)) == 6
    assert len(x.sample(-1)) == 0

    picked = x.sample(2)
    picked[0]['a'] = 999
    assert 999 not in list(x.unwind('a'))


# --- Calculation Tests ---

def test_pct_change():
    """Verify pct_change is forward, one period, with a leading None.

    Mutation: pct_change differencing a reversed series, or dividing
        by the current value rather than the previous one.
    Oracle: hand-computed [None, 1.00, 0.50, -0.67, 1.00, 0.50].
    """
    xx = DataSet([
        {'x': 1, 'y': 8., 'z': 'a'},
        {'x': 2, 'y': 21.8, 'z': 'a'},
        {'x': 3, 'y': 3.2, 'z': 'a'},
        {'x': 1, 'y': 0.1, 'z': 'b'},
        {'x': 2, 'y': 22., 'z': 'b'},
        {'x': 3, 'y': 3., 'z': 'b'}
    ])

    xx.pct_change('x', 'x_chg')
    result = [f'{_:.2f}' if _ else _ for _ in xx.unwind('x_chg')]
    assert result == [None, '1.00', '0.50', '-0.67', '1.00', '0.50']


# --- Pagination Tests ---

def test_pages_property():
    """Verify pages rounds a partly filled last page up.

    Mutation: int() truncation in place of math.ceil.
    Oracle: 1201 over 100 -> 13, straddling the boundary that 1200 over
        100 -> 12 pins from the other side.
    """
    container = [{'content': i} for i in range(600)]

    assert DataSet(container, page=1, per_page=100, total=1201).pages == 13
    assert DataSet(container, page=1, per_page=100, total=1200).pages == 12


@pytest.mark.parametrize(('page', 'expected'), [
    (3, [1, 2, 3, 4, 5, 6, 7, '...', 10, 11]),
    (1, [1, 2, 3, 4, 5, '...', 10, 11]),
    (11, [1, 2, '...', 9, 10, 11]),
])
def test_get_pages(page, expected):
    """Verify get_pages elides the gaps around the current page.

    Mutation: an off-by-one in the left_this or right_this window
        bounds, which moves where '...' lands.
    Oracle: hand-written page lists for pages 3, 1 and 11 of 11.
    """
    container = [{'content': i} for i in range(101)]
    dd = DataSet(container, page=page, per_page=10)
    assert list(dd.get_pages()) == expected


def test_get_pages_single_page():
    """Verify a part-filled single page still counts as one page.

    Mutation: int truncation in pages, giving 0 pages and an empty
        pager.
    Oracle: 1 row over per_page 10 -> pages 1 and [1].
    """
    container = [{'content': i} for i in range(1)]
    dd = DataSet(container, page=1, per_page=10)
    assert dd.pages == 1
    assert list(dd.get_pages()) == [1]


# --- Column Management Tests ---

def test_remove_column():
    """Verify remove_column is case-sensitive and clears the row keys.

    Mutation: matching the name case-insensitively, or dropping the
        column from the schema while leaving it in every row.
    Oracle: hand-tracked schema after each call, plus the row keys.
    """
    x = DataSet([
        {'a': 1, 'b': 1},
        {'a': 2, 'b': 2}])

    x.remove_column('A')
    assert x.cols == ['a', 'b']

    x.remove_column('a')
    assert x.cols == ['b']
    assert [set(row) for row in x.container] == [{'b'}, {'b'}]

    x.remove_column('c')
    assert x.cols == ['b']

    x.remove_column('b')
    assert x.cols == []
    assert [set(row) for row in x.container] == [set(), set()]


def test_rename_column():
    """Verify rename_column is case-sensitive and carries values across.

    Mutation: matching the source name case-insensitively, or renaming
        the schema entry without moving the row key.
    Oracle: hand-tracked schema and row contents after each call.
    """
    x = DataSet([
        {'a': 1, 'b': 1},
        {'a': 3, 'b': 4}])

    x.rename_column('A', 'b')
    assert x.cols == ['a', 'b']

    x.rename_column('a', 'c')
    assert x.cols == ['c', 'b']
    assert x.container[1] == {'c': 3, 'b': 4}

    x.rename_column('c', 'b')
    assert x.cols == ['b']
    assert x.container[1] == {'b': 3}

    x.rename_column('b', 'b')
    assert x.cols == ['b']

    x.rename_column('b', 'd')
    assert x.cols == ['d']
    assert x.container[1] == {'d': 3}


def test_rename_column_collision():
    """Verify renaming onto an existing column keeps the source values.

    Mutation: rename_column popping the source key without writing it
        under the new name.
    Oracle: hand-written merged row {'b': 1, 'c': 3}.
    """
    x = DataSet([
        {'a': 1, 'b': 2, 'c': 3}])
    x.columns = [('a', int), ('b', int), ('c', int)]

    x.rename_column('a', 'b')

    assert x.cols == ['b', 'c']
    assert x.container[0] == {'b': 1, 'c': 3}


# --- Type Conversion Tests ---

def test_convert_container_types():
    """Verify conversion coerces declared columns and leaves None alone.

    Mutation: rounding rather than truncating on the int cast, or
        coercing None to the type's empty value.
    Oracle: hand-written rows {'astr': '1', 'anint': 2, 'foo': 'bar'}
        and {'astr': None, 'anint': 2}, from the source value 2.7.
    """
    columns = [('astr', str), ('anint', int)]

    rows = [attrdict({'astr': 1, 'anint': 2.7, 'foo': 'bar'})]
    ds = DataSet(rows, columns)
    ds.convert_container_types()
    assert ds.container == [{'astr': '1', 'anint': 2, 'foo': 'bar'}]

    rows = [attrdict({'astr': None, 'anint': 2.7})]
    ds = DataSet(rows, columns)
    ds.convert_container_types()
    assert ds.container == [{'astr': None, 'anint': 2}]


# --- Extraction Tests ---

def test_unwind():
    """Verify unwind yields values in argument order, not schema order.

    Mutation: unwind falling back to self.cols and ignoring the column
        arguments, which reverses the pairs here.
    Oracle: hand-written tuples for ('a', 'b') and for ('b', 'a').
    """
    ds = DataSet([{'a': 1, 'b': 2}, {'a': 2, 'b': 3}])

    assert list(ds.unwind('a')) == [1, 2]
    assert list(ds.unwind('a', 'b')) == [(1, 2), (2, 3)]
    assert list(ds.unwind('b', 'a')) == [(2, 1), (3, 2)]

    a, b = zip(*ds.unwind('a', 'b'))
    assert (a, b) == ((1, 2), (2, 3))


def test_to_list():
    """Verify to_list transposes rows into one tuple per column.

    Mutation: returning the rows as tuples instead of transposing them.
    Oracle: hand-written column tuples for a 6-row, 2-column grid.
    """
    xx = DataSet([
        {'a': 1, 'b': 8.},
        {'a': 2, 'b': 21.8},
        {'a': 3, 'b': 3.2},
        {'a': 1, 'b': 0.1},
        {'a': 2, 'b': 22.},
        {'a': 3, 'b': 3.},
    ])
    result = xx.to_list()
    assert result == [(1, 2, 3, 1, 2, 3), (8.0, 21.8, 3.2, 0.1, 22.0, 3.0)]


# --- DataFrame Conversion Tests ---

def test_dataframe_conversion():
    """Verify dataframe picks columns, guards the index, and needs two.

    Mutation: dropping verify_integrity so a repeating index passes, or
        dropping the one-column guard.
    Oracle: hand-computed pivot column sums 33.0 and 25.1.
    """
    xx = DataSet([
        {'x': 1, 'y': 8., 'z': 'a'},
        {'x': 2, 'y': 21.8, 'z': 'a'},
        {'x': 3, 'y': 3.2, 'z': 'a'},
        {'x': 1, 'y': 0.1, 'z': 'b'},
        {'x': 2, 'y': 22., 'z': 'b'},
        {'x': 3, 'y': 3., 'z': 'b'}
    ])

    yy = xx.dataframe(None, 'x', 'y', 'z')
    assert list(yy.columns) == ['x', 'y', 'z']
    assert len(yy) == 6

    with pytest.raises(ValueError, match='Index has duplicate keys'):
        xx.dataframe('x', 'y', 'z')

    with pytest.raises(ValueError, match='Need to pass an index'):
        xx.dataframe(None, 'x')

    pivoted = xx.pivot('x', 'y', 'z')
    pp = pivoted.dataframe('x', 'x', 'a', 'b')
    assert pp.index.tolist() == [1, 2, 3]
    assert pp.a.sum() == pytest.approx(33.0)
    assert pp.b.sum() == pytest.approx(25.1)

    omit = pivoted.dataframe('x')
    assert list(omit.columns) == ['a', 'b']
    assert omit.a.tolist() == pp.a.tolist()


# --- Column Property Tests ---

def test_columns_mutable_supports_direct_mutation():
    """Verify the columns property hands back the live list, not a copy.

    Mutation: the columns getter returning list(self._columns), so an
        extend or append on it never reaches cols, colmap or typs.
    Oracle: cols and colmap read back after each in-place mutation.
    """
    ds = DataSet([{'a': 1, 'b': 2}])

    assert ds.cols == ['a', 'b']

    ds.columns.extend([('c', str)])
    assert ds.cols == ['a', 'b', 'c']
    assert set(ds.colmap.keys()) == {'a', 'b', 'c'}
    assert ds.colmap['c'] == str

    ds.columns.append(('d', float))
    assert ds.cols == ['a', 'b', 'c', 'd']
    assert ds.typs == [int, int, str, float]

    ds.columns = [('x', int), ('y', str)]
    assert ds.cols == ['x', 'y']


def test_cols_setter_preserves_types():
    """Verify the cols setter reorders, filters and dedupes the schema.

    Mutation: rebuilding entries with a default type rather than the
        current one, or dropping the unique() dedupe.
    Oracle: hand-written columns lists after each assignment.
    """
    ds = DataSet([{'a': 1, 'b': 2.5, 'c': 'text'}])
    ds.columns = [('a', int), ('b', float), ('c', str)]

    assert ds.cols == ['a', 'b', 'c']
    assert ds.colmap == {'a': int, 'b': float, 'c': str}

    ds.cols = ['c', 'a', 'b']
    assert ds.cols == ['c', 'a', 'b']
    assert ds.columns == [('c', str), ('a', int), ('b', float)]

    ds.cols = ['a', 'c']
    assert ds.columns == [('a', int), ('c', str)]

    ds.cols = ['a', 'c', 'd']
    assert ds.columns == [('a', int), ('c', str)]
    assert 'd' not in ds.colmap

    ds.cols = ['a', 'c', 'a', 'c', 'e']
    assert ds.columns == [('a', int), ('c', str)]


# --- Row Lookup Tests ---

def test_find_returns_first_match_not_last():
    """Verify find() returns the index of the first matching row.

    Mutation: recording the index and continuing the loop, so the last
        match wins.
    Oracle: hand-computed 1 where the value repeats at 1 and 2.
    """
    rows = [{'a': 1}, {'a': 2}, {'a': 2}]
    assert find(rows, 'a', 2) == 1


def test_find_missing_returns_minus_one():
    """Verify a miss returns -1 rather than 0 or None.

    Mutation: returning 0 or None on the no-match path, both of which a
        truthiness check at the call site would read as a hit.
    Oracle: hand-computed -1.
    """
    assert find([{'a': 1}], 'a', 9) == -1


def test_find_raise_err_raises_only_on_miss():
    """Verify raise_err raises on a miss and stays quiet on a hit.

    Mutation: raising before the return, so a found value raises too.
    Oracle: a hit returns index 0; a miss raises ValueError.
    """
    assert find([{'a': 1}], 'a', 1, raise_err=True) == 0
    with pytest.raises(ValueError):
        find([{'a': 1}], 'a', 9, raise_err=True)


def test_pop_removes_the_matched_row():
    """Verify pop returns the matched row and takes it out of the set.

    Mutation: returning the row without removing it, which a
        return-value-only assertion would miss.
    Oracle: hand-computed - the popped row is b='y' and exactly one row,
        b='x', is left behind.
    """
    ds = DataSet([{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}])

    got = ds.pop('a', 2)

    assert dict(got) == {'a': 2, 'b': 'y'}
    assert [dict(r) for r in ds] == [{'a': 1, 'b': 'x'}]


def test_pop_miss_returns_none_and_keeps_rows():
    """Verify a miss returns None without dropping a row.

    Mutation: popping the first row regardless of the value matching.
    Oracle: hand-computed - both rows survive a miss.
    """
    ds = DataSet([{'a': 1}, {'a': 2}])

    assert ds.pop('a', 99) is None
    assert len(ds) == 2


def test_add_empty_row_fills_none_not_blank():
    """Verify the appended row holds None in every column.

    Mutation: dict.fromkeys(self.cols, '') filling blanks instead of
        None, which reads as a value rather than a gap.
    Oracle: hand-computed - a second row of None against a first row of
        real values.
    """
    ds = DataSet([{'a': 1, 'b': 'x'}])

    ds.add_empty_row()

    assert [dict(r) for r in ds] == [{'a': 1, 'b': 'x'}, {'a': None, 'b': None}]


# --- Equality and Rendering Tests ---

def test_eq_compares_contents_not_identity():
    """Verify two separately built datasets with equal rows compare equal.

    Mutation: comparing by identity, which would make every distinct
        object unequal.
    Oracle: hand-built pairs - equal rows compare equal, a changed value
        compares unequal.
    """
    assert DataSet([{'a': 1}]) == DataSet([{'a': 1}])
    assert DataSet([{'a': 1}]) != DataSet([{'a': 2}])


def test_pp_renders_header_and_data():
    """Verify the pretty table carries the column names and the data.

    Mutation: rendering only the header row, dropping the data rows.
    Oracle: hand-read cell text - 'r1' and '2' from the data row.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}])

    out = str(ds.pp)

    assert 'a' in out
    assert 'b' in out
    assert 'r1' in out
    assert '2' in out


def test_pp_shows_total_only_once_a_summary_exists():
    """Verify the summary row reaches the table only after it is read.

    Mutation: rendering the summary unconditionally, which would print a
        Total on a dataset that never asked for one.
    Oracle: the same dataset before and after reading .summary - 'Total'
        is absent, then present.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}])

    assert 'Total' not in str(ds.pp)

    ds.summary

    assert 'Total' in str(ds.pp)


def test_pp_renders_a_column_less_dataset_that_declared_a_summary():
    """Verify pp answers an empty table rather than raising.

    A dataset that never had columns renders a zero-column table, and
    adding a row to one makes prettytable take max() over an empty
    sequence.

    Mutation: dropping the `cols and` guard on the summary row, so pp
        raises ValueError on a dataset with no columns.
    Oracle: the string the pre-cache implementation returned for the
        same input, which never rendered that row because an empty
        summary dict is falsy.
    """
    ds = DataSet()
    ds.add_summary_row()

    assert ds.pp == '\n++\n||\n++\n++'


# --- Constructor Tests ---

def test_new_dataset_pagination_defaults():
    """Verify a dataset built from rows starts unpaged.

    Mutation: pageable defaulting to '' rather than None, so a caller
        testing `is None` reads a fresh dataset as already paged.
    Oracle: hand-read defaults - no page, no page size, a total equal
        to the three rows given, and no pager state.
    """
    ds = DataSet([{'a': 1}, {'a': 2}, {'a': 3}])

    assert ds.page is None
    assert ds.per_page is None
    assert ds.total == 3
    assert ds.pageable is None


def test_copy_constructor_carries_summary_args():
    """Verify DataSet(other) keeps the summary the source declared.

    Mutation: the copy constructor dropping _summary_args, so the copy
        falls back to totaling every numeric column under 'Total'.
    Oracle: hand-computed row - label 'Sum' in name, x totaling 3, and
        y left None because it was never asked for.
    """
    src = DataSet([
        {'name': 'a', 'x': 1, 'y': 10},
        {'name': 'b', 'x': 2, 'y': 20}])
    src.add_summary_row(label='Sum', columns=['x'])

    summary = DataSet(src).summary

    assert summary['name'] == 'Sum'
    assert summary['x'] == 3
    assert summary['y'] is None


def test_copy_constructor_summary_totals_the_copys_own_rows():
    """Verify the copy's summary totals the copy's rows, not the source's.

    Mutation: the copy carrying a cached total, so a row added to the
        copy after construction is left out of its total.
    Oracle: hand-computed 3 over the source's two rows, then 103 once
        the copy holds a third row of 100.
    """
    src = DataSet([{'name': 'a', 'x': 1}, {'name': 'b', 'x': 2}])
    assert src.summary['x'] == 3

    copied = DataSet(src)
    copied.append({'name': 'c', 'x': 100})

    assert copied.summary['x'] == 103


def test_copy_constructor_carries_converted_flag(monkeypatch):
    """Verify a copy of a converted dataset does not convert again.

    Mutation: the copy constructor resetting _types_converted rather
        than carrying the source's, so every copy re-scans the rows.
    Oracle: a spy over convert_container_types, which must not fire
        while the copy is read.
    """
    src = DataSet([{'a': '1'}], columns=[('a', int)])
    assert list(src.unwind('a')) == [1]

    conversions = []
    convert = DataSet.convert_container_types

    def spy(self, *args, **kwargs):
        conversions.append(self)
        return convert(self, *args, **kwargs)

    monkeypatch.setattr(DataSet, 'convert_container_types', spy)
    copied = DataSet(src)

    assert list(copied.unwind('a')) == [1]
    assert conversions == []


def test_itemize_keeps_a_single_column_value_whole():
    """Verify a one-column dataset itemizes without truncating each value.

    Mutation: zipping cols against the scalar unwind yields (the pre-fix
        code), which raises on an int and clips 'yy' to 'y'.
    Oracle: hand-computed [1, 2] and ['x', 'yy'], each in its own
        one-row dataset.
    """
    ints = DataSet([{'a': 1}, {'a': 2}]).itemize()
    assert [d[0]['a'] for d in ints] == [1, 2]

    strings = DataSet([{'a': 'x'}, {'a': 'yy'}]).itemize()
    assert [d[0]['a'] for d in strings] == ['x', 'yy']


def test_sort_with_no_key_orders_by_column_values_and_returns_self():
    """Verify sort() with no key orders the rows and hands back self.

    Mutation: the no-key branch falling through with no return, which
        raises comparing bare rows; or dropping reverse so descending
        is ignored.
    Oracle: hand-computed ascending [1, 2, 3] and descending [3, 2, 1],
        plus identity of the returned dataset.
    """
    ds = DataSet([{'a': 3}, {'a': 1}, {'a': 2}])
    result = ds.sort()

    assert [row['a'] for row in ds] == [1, 2, 3]
    assert result is ds

    ds.sort(reverse=True)
    assert [row['a'] for row in ds] == [3, 2, 1]


def test_rename_column_to_itself_warns_without_dropping_it(caplog):
    """Verify a self-rename warns with the right spelling and is a no-op.

    Mutation: dropping the self-rename guard, so the column is removed
        then re-added; or misspelling 'Column' in the warning, which the
        exact-text check below no longer finds.
    Oracle: the column list unchanged at ['a', 'b'], and the exact
        warning text 'Column a matches rename column a'.
    """
    ds = DataSet([{'a': 1, 'b': 2}])
    with caplog.at_level(logging.WARNING):
        ds.rename_column('a', 'a')

    assert ds.cols == ['a', 'b']
    assert 'Column a matches rename column a' in caplog.text


if __name__ == '__main__':
    pytest.main([__file__])


# --- dump Tests ---


def test_dump_logs_the_given_label_and_the_state(caplog):
    """Verify dump emits the caller's label and the dataset state.

    Mutation: logging a hardcoded 'DataSet' in place of the label
        argument, or dropping the second call that logs __dict__.
    Oracle: two debug records - the exact label passed in, and a payload
        naming a column the dataset actually holds.
    """
    ds = DataSet([{'a': 1, 'b': 'x'}])

    with caplog.at_level(logging.DEBUG, logger='rollups.core'):
        ds.dump(label='MYLABEL')

    messages = [r.getMessage() for r in caplog.records]
    assert 'MYLABEL' in messages
    assert any('container' in m and "'a'" in m for m in messages)


def test_dump_label_defaults_to_dataset(caplog):
    """Verify the label defaults to 'DataSet' when none is given.

    Mutation: changing the default label, or making label a required
        argument.
    Oracle: the literal default named in the signature.
    """
    ds = DataSet([{'a': 1}])

    with caplog.at_level(logging.DEBUG, logger='rollups.core'):
        ds.dump()

    assert 'DataSet' in [r.getMessage() for r in caplog.records]
