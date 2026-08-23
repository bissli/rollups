import pytest
from opendate import Date, DateTime, Time
from rollups import DataSet

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Basic dataset for copy tests."""
    ds = DataSet([
        {'name': 'A', 'value': 100},
        {'name': 'B', 'value': 200}
    ])
    ds.columns = [('name', str), ('value', int)]
    return ds


@pytest.fixture
def dataset_with_list_values():
    """Dataset containing list values in rows."""
    return DataSet([
        {'items': ['a', 'b', 'c']},
        {'items': ['d', 'e']}
    ])


@pytest.fixture
def dataset_with_nested_structure():
    """Dataset with complex nested structures."""
    return DataSet([
        {'id': 1, 'data': {'nested': {'deep': [1, 2, 3]}}}
    ])


# --- Core Copy Behavior Tests - Verifying Different Semantics ---

class TestCopySemantics:
    """Tests verifying the different copy semantics."""

    def test_copy_creates_new_container(self, basic_dataset):
        """Verify copy() builds its own container list.

        Mutation: `ds.container = self.container` in copy(), aliasing
            the source list instead of `list(self.container)`.
        Oracle: container identity plus the hand-counted 2 rows.
        """
        copied = basic_dataset.copy()
        assert basic_dataset.container is not copied.container
        assert len(copied) == len(basic_dataset) == 2

    def test_copy_shares_row_objects(self, basic_dataset):
        """Verify copy() hands over the same row objects.

        Mutation: copy() rebuilding rows as `attrdict(row)`, which is
            shallowcopy's semantics.
        Oracle: the 999 written through the copy, read back from the
            original row.
        """
        copied = basic_dataset.copy()
        copied[0]['value'] = 999
        assert basic_dataset[0]['value'] == 999
        assert basic_dataset[0] is copied[0]

    def test_deepcopy_creates_independent_rows(self, basic_dataset):
        """Verify deepcopy() builds new row objects.

        Mutation: deepcopy() reusing the source rows
            (`ds.container = list(self.container)`).
        Oracle: hand-computed 100 still in the original after the copy
            is set to 999.
        """
        deep = basic_dataset.deepcopy()
        deep[0]['value'] = 999
        assert basic_dataset[0]['value'] == 100
        assert basic_dataset[0] is not deep[0]

    def test_shallowcopy_creates_new_row_objects(self, basic_dataset):
        """Verify shallowcopy() wraps every row in a new attrdict.

        Mutation: shallowcopy() reusing the source rows
            (`ds.container = list(self.container)`).
        Oracle: row identity checked on both rows.
        """
        rowcopied = basic_dataset.shallowcopy()
        assert len(rowcopied) == len(basic_dataset)
        assert basic_dataset[0] is not rowcopied[0]
        assert basic_dataset[1] is not rowcopied[1]


class TestMutableValueBehavior:
    """Tests for how copy methods handle mutable values in rows."""

    def test_copy_with_list_values_shares_reference(self, dataset_with_list_values):
        """Verify copy() leaves a list value shared.

        Mutation: copy() deep-copying each row
            (`attrdict(copy.deepcopy(dict(row)))`).
        Oracle: the 'x' appended through the copy, read from the
            original list.
        """
        copied = dataset_with_list_values.copy()
        copied[0]['items'].append('x')
        assert 'x' in dataset_with_list_values[0]['items']

    def test_deepcopy_with_list_values_creates_independent(
            self,
            dataset_with_list_values):
        """Verify deepcopy() copies a list value rather than sharing it.

        Mutation: `copy.copy(dict(row))` in place of
            `copy.deepcopy(dict(row))` in deepcopy().
        Oracle: hand-computed lengths 3 (original) and 4 (copy).
        """
        deep = dataset_with_list_values.deepcopy()
        deep[0]['items'].append('X')
        assert 'X' not in dataset_with_list_values[0]['items']
        assert len(dataset_with_list_values[0]['items']) == 3
        assert len(deep[0]['items']) == 4

    def test_shallowcopy_shares_mutable_values(self, dataset_with_list_values):
        """Verify shallowcopy() shares the values inside its new rows.

        Mutation: shallowcopy() deep-copying each row
            (`attrdict(copy.deepcopy(dict(row)))`).
        Oracle: list identity, plus the appended 'modified' read from
            the original.
        """
        rowcopied = dataset_with_list_values.shallowcopy()
        rowcopied[0]['items'].append('modified')
        assert 'modified' in dataset_with_list_values[0]['items']
        assert dataset_with_list_values[0]['items'] is rowcopied[0]['items']

    def test_shallowcopy_allows_independent_column_addition(self, basic_dataset):
        """Verify a key added to a shallowcopy row misses the original.

        Mutation: shallowcopy() reusing the source rows
            (`ds.container = list(self.container)`).
        Oracle: membership of 'new_column' in each row.
        """
        rowcopied = basic_dataset.shallowcopy()
        rowcopied[0]['new_column'] = 'test'
        assert 'new_column' in rowcopied[0]
        assert 'new_column' not in basic_dataset[0]

    def test_copy_with_dict_values_shares_reference(self):
        """Verify copy() leaves a dict value shared.

        Mutation: copy() deep-copying each row
            (`attrdict(copy.deepcopy(dict(row)))`).
        Oracle: dict identity, plus the hand-set 10 read from the
            original.
        """
        ds = DataSet([{'name': 'A', 'config': {'key': 'value', 'count': 5}}])
        copied = ds.copy()
        copied[0]['config']['count'] = 10
        assert ds[0]['config']['count'] == 10
        assert ds[0]['config'] is copied[0]['config']

    def test_shallowcopy_with_dict_values_shares_reference(self):
        """Verify shallowcopy() leaves a dict value shared.

        Mutation: shallowcopy() deep-copying each row
            (`attrdict(copy.deepcopy(dict(row)))`).
        Oracle: dict identity, plus the hand-set 10 read from the
            original.
        """
        ds = DataSet([{'name': 'A', 'config': {'key': 'value', 'count': 5}}])
        rowcopied = ds.shallowcopy()
        rowcopied[0]['config']['count'] = 10
        assert ds[0]['config']['count'] == 10
        assert ds[0]['config'] is rowcopied[0]['config']


class TestComplexNestedStructures:
    """Tests for copy behavior with complex nested structures."""

    def test_copy_with_nested_structure_shares(self, dataset_with_nested_structure):
        """Verify copy() shares a value nested three levels down.

        Mutation: copy() deep-copying each row
            (`attrdict(copy.deepcopy(dict(row)))`).
        Oracle: the 4 appended through the copy, read from the
            original's innermost list.
        """
        copied = dataset_with_nested_structure.copy()
        copied[0]['data']['nested']['deep'].append(4)
        assert 4 in dataset_with_nested_structure[0]['data']['nested']['deep']

    def test_deepcopy_with_nested_structure_independent(
            self,
            dataset_with_nested_structure):
        """Verify deepcopy() recurses to the innermost nested list.

        Mutation: `copy.copy(dict(row))` in place of
            `copy.deepcopy(dict(row))` in deepcopy().
        Oracle: hand-computed lengths 3 (original) and 4 (copy).
        """
        deep = dataset_with_nested_structure.deepcopy()
        deep[0]['data']['nested']['deep'].append(4)
        assert 4 not in dataset_with_nested_structure[0]['data']['nested']['deep']
        assert len(dataset_with_nested_structure[0]['data']['nested']['deep']) == 3
        assert len(deep[0]['data']['nested']['deep']) == 4

    def test_shallowcopy_with_nested_structure_shares(
            self,
            dataset_with_nested_structure):
        """Verify shallowcopy() shares a nested structure whole.

        Mutation: shallowcopy() deep-copying each row
            (`attrdict(copy.deepcopy(dict(row)))`).
        Oracle: identity of the nested dict, plus the appended 4 read
            from the original.
        """
        rowcopied = dataset_with_nested_structure.shallowcopy()
        rowcopied[0]['data']['nested']['deep'].append(4)
        assert 4 in dataset_with_nested_structure[0]['data']['nested']['deep']
        assert dataset_with_nested_structure[0]['data'] is rowcopied[0]['data']


class TestNestedDatasets:
    """Tests for copy behavior with nested DataSet objects."""

    def test_copy_with_nested_datasets_shares(self):
        """Verify copy() shares a DataSet held in a row.

        Mutation: copy() deep-copying each row
            (`attrdict(copy.deepcopy(dict(row)))`).
        Oracle: DataSet identity, plus the hand-counted 3 rows after
            one append through the copy.
        """
        inner_ds = DataSet([{'a': 1}, {'a': 2}])
        ds = DataSet([{'key': 'X', 'data': inner_ds}])
        copied = ds.copy()
        copied[0]['data'].append({'a': 3})
        assert len(ds[0]['data']) == 3
        assert ds[0]['data'] is copied[0]['data']

    def test_deepcopy_with_nested_datasets_independent(self):
        """Verify deepcopy() copies a DataSet held in a row.

        Mutation: `copy.copy(dict(row))` in place of
            `copy.deepcopy(dict(row))` in deepcopy().
        Oracle: hand-counted 2 rows inside the original against 3
            inside the copy.
        """
        inner_ds = DataSet([{'a': 1}, {'a': 2}])
        ds = DataSet([{'key': 'X', 'data': inner_ds}])
        deep = ds.deepcopy()
        deep[0]['data'].append({'a': 3})
        assert len(ds[0]['data']) == 2
        assert len(deep[0]['data']) == 3

    def test_shallowcopy_with_nested_datasets_shares(self):
        """Verify shallowcopy() shares a DataSet held in a row.

        Mutation: shallowcopy() deep-copying each row
            (`attrdict(copy.deepcopy(dict(row)))`).
        Oracle: DataSet identity, plus the hand-counted 3 rows after
            one append through the copy.
        """
        nested_ds = DataSet([{'x': 1}, {'x': 2}])
        ds = DataSet([{'name': 'A', 'nested': nested_ds}])
        rowcopied = ds.shallowcopy()
        rowcopied[0]['nested'].append({'x': 3})
        assert len(ds[0]['nested']) == 3
        assert ds[0]['nested'] is rowcopied[0]['nested']


# --- Parameterized Tests - Common Behavior Across All Copy Methods ---

@pytest.mark.parametrize('copy_method', ['copy', 'deepcopy', 'shallowcopy'])
class TestCommonCopyBehavior:
    """Tests for behavior common to all copy methods."""

    def test_preserves_columns(self, basic_dataset, copy_method):
        """Verify every copy method carries the column definitions over.

        Mutation: `_copy_structure` dropping `ds.columns`, leaving the
            fresh dataset's empty column list.
        Oracle: the fixture's hand-written [('name', str),
            ('value', int)].
        """
        method = getattr(basic_dataset, copy_method)
        result = method()
        assert result.columns == [('name', str), ('value', int)]
        assert result.cols == ['name', 'value']
        assert result.colmap == {'name': str, 'value': int}

    def test_allows_independent_appends(self, basic_dataset, copy_method):
        """Verify appending to a copy leaves the source row count alone.

        Mutation: a copy method aliasing the source list
            (`ds.container = self.container`), since append() writes
            into that list in place.
        Oracle: hand-counted 2 rows in the original against 3 in the
            copy.
        """
        method = getattr(basic_dataset, copy_method)
        result = method()
        result.append({'name': 'C', 'value': 300})
        assert len(basic_dataset) == 2
        assert len(result) == 3

    def test_allows_independent_filtering(self, copy_method):
        """Verify filtering a copy leaves the source rows in place.

        Mutation: a copy method returning `self` rather than a new
            dataset.
        Oracle: hand-computed [100, 200, 300] in the original against
            [200, 300] in the filtered copy.
        """
        ds = DataSet([
            {'name': 'A', 'value': 100},
            {'name': 'B', 'value': 200},
            {'name': 'C', 'value': 300}
        ])
        method = getattr(ds, copy_method)
        result = method()
        result.filter_data(lambda r: r['value'] > 150)
        assert list(ds.unwind('value')) == [100, 200, 300]
        assert list(result.unwind('value')) == [200, 300]

    def test_allows_independent_sorting(self, copy_method):
        """Verify sorting a copy leaves the source order alone.

        Mutation: a copy method aliasing the source list
            (`ds.container = self.container`), since sort_data() sorts
            that list in place.
        Oracle: hand-computed [3, 1, 2] in the original against
            [1, 2, 3] in the sorted copy.
        """
        ds = DataSet([
            {'value': 3},
            {'value': 1},
            {'value': 2}
        ])
        method = getattr(ds, copy_method)
        result = method()
        result.sort_data('value')
        assert list(ds.unwind('value')) == [3, 1, 2]
        assert list(result.unwind('value')) == [1, 2, 3]

    def test_independent_column_operations(self, basic_dataset, copy_method):
        """Verify a column added to a copy misses the source schema.

        Mutation: `_copy_structure` assigning `ds._columns =
            self._columns`, bypassing the setter that rebuilds the
            list, so add_column() edits the shared list in place.
        Oracle: membership of 'score' in each dataset's cols.
        """
        method = getattr(basic_dataset, copy_method)
        result = method()
        result.add_column('score', float, value=95.5)
        assert 'score' in result.cols
        assert 'score' not in basic_dataset.cols

    def test_empty_dataset(self, copy_method):
        """Verify a copy of an empty dataset keeps its declared columns.

        Mutation: `_copy_structure` re-guessing columns from the
            container (`DataSet.guess_columns(self.container)`), which
            has no row to guess from here.
        Oracle: the hand-written ['name', 'age'] schema.
        """
        ds = DataSet([], columns=[('name', str), ('age', int)])
        method = getattr(ds, copy_method)
        result = method()
        assert len(result) == 0
        assert result.cols == ['name', 'age']

    def test_single_row(self, copy_method):
        """Verify a one-row copy keeps the row as an attrdict.

        Mutation: rows rebuilt as plain dicts (`dict(row)` in place of
            `attrdict(...)`), which loses attribute access.
        Oracle: hand-computed 42 read by key and by attribute.
        """
        ds = DataSet([{'value': 42}])
        method = getattr(ds, copy_method)
        result = method()
        assert len(result) == 1
        assert result[0]['value'] == 42
        assert result[0].value == 42

    def test_preserves_summary_args(self, copy_method):
        """Verify a copy summarizes with the arguments it was given.

        Mutation: `_copy_structure` dropping `ds._summary_args`, so the
            copy falls back to the default 'Total' summary.
        Oracle: hand-computed label 'Sum' and total 300.
        """
        ds = DataSet([
            {'name': 'A', 'value': 100},
            {'name': 'B', 'value': 200}
        ])
        ds.add_summary_row(label='Sum', columns=['value'])
        method = getattr(ds, copy_method)
        result = method()
        assert result._summary_args == ds._summary_args
        assert result.summary['name'] == 'Sum'
        assert result.summary['value'] == 300

    def test_preserves_check_types_flag(self, copy_method):
        """Verify a copy honors check_types=False, while deepcopy converts.

        Mutation: `_ensure_types_converted` losing its `_check_types`
            guard, so a copy converts values it was told to leave.
        Oracle: the '123' string still unconverted in a copy, against
            the int 123 deepcopy forces.
        """
        ds = DataSet([
            {'value': '123'}
        ], columns=[('value', int)], check_types=False)
        method = getattr(ds, copy_method)
        result = method()
        assert result._check_types is False
        if copy_method == 'deepcopy':
            assert result[0]['value'] == 123
        else:
            assert result[0]['value'] == '123'

    def test_with_none_values(self, copy_method):
        """Verify a None value neither converts nor stops its neighbors.

        Mutation: `continue` -> `break` on the None guard in
            convert_container_types, leaving a later column
            unconverted.
        Oracle: hand-computed rows {'a': None, 'b': 7} and
            {'a': 3, 'b': None}.
        """
        ds = DataSet([
            {'a': None, 'b': '7'},
            {'a': '3', 'b': None}
        ], columns=[('a', int), ('b', int)])
        method = getattr(ds, copy_method)
        result = method()
        assert result[0]['a'] is None
        assert result[0]['b'] == 7
        assert result[1]['a'] == 3
        assert result[1]['b'] is None

    def test_preserves_pagination_info(self, copy_method):
        """Verify a copy keeps the declared page, per_page and total.

        Mutation: `_copy_structure` recomputing `ds.total =
            len(self.container)` instead of carrying `self.total`.
        Oracle: the hand-set 2, 5 and 100, against the 10 rows the
            container holds.
        """
        ds = DataSet(
            [{'x': i} for i in range(10)],
            page=2,
            per_page=5,
            total=100
        )
        method = getattr(ds, copy_method)
        result = method()
        assert result.page == 2
        assert result.per_page == 5
        assert result.total == 100

    def test_preserves_pageable(self, copy_method):
        """Verify a copy carries the pageable attribute over.

        Mutation: `_copy_structure` dropping `ds.pageable`, leaving the
            constructor's None.
        Oracle: the hand-set {'some': 'data'}.
        """
        ds = DataSet([{'x': 1}])
        ds.pageable = {'some': 'data'}
        method = getattr(ds, copy_method)
        result = method()
        assert result.pageable == {'some': 'data'}


@pytest.mark.parametrize('copy_method', ['copy', 'shallowcopy'])
def test_empty_option_creates_empty_dataset(basic_dataset, copy_method):
    """Verify empty=True drops the rows but keeps the schema.

    Mutation: `[] if empty else ...` reduced to the row-copying branch,
        so `empty` is ignored.
    Oracle: hand-counted 0 rows against the fixture's two-column
        schema.
    """
    method = getattr(basic_dataset, copy_method)
    empty_result = method(empty=True)
    assert len(empty_result) == 0
    assert empty_result.cols == ['name', 'value']
    assert empty_result.columns == [('name', str), ('value', int)]


@pytest.mark.parametrize(('copy_method', 'date_type', 'test_value'), [
    ('copy', Date, Date(2024, 1, 1)),
    ('copy', DateTime, DateTime(2024, 1, 1, 10, 30)),
    ('deepcopy', Date, Date(2024, 1, 1)),
    ('deepcopy', DateTime, DateTime(2024, 1, 1, 10, 30)),
    ('shallowcopy', Date, Date(2024, 1, 1)),
    ('shallowcopy', DateTime, DateTime(2024, 1, 1, 10, 30)),
])
def test_preserves_date_types(copy_method, date_type, test_value):
    """Verify a copy keeps a Date or DateTime column typed and valued.

    Mutation: `_copy_structure` dropping `ds.columns`, which loses the
        declared Date type and leaves the copy's colmap empty.
    Oracle: the hand-written column type and the date the row was
        built with.
    """
    ds = DataSet([{'dt': test_value, 'value': 100}])
    ds.columns = [('dt', date_type), ('value', int)]
    method = getattr(ds, copy_method)
    result = method()
    assert result.colmap['dt'] == date_type
    assert result[0]['dt'] == test_value


# --- Types Converted Flag Tests - Different Behavior Per Method ---

class TestTypesConvertedFlag:
    """Tests for _types_converted flag handling."""

    @pytest.mark.parametrize('copy_method', ['copy', 'shallowcopy'])
    def test_copy_preserves_types_converted_flag(self, copy_method):
        """Verify copy and shallowcopy do not re-run conversion.

        Mutation: `ds._types_converted = False` in place of the
            carried-over flag, which makes the copy convert on read.
        Oracle: the '123' string still unconverted in the copy.
        """
        ds = DataSet([
            {'value': '123'}
        ], columns=[('value', int)])
        ds._types_converted = True
        method = getattr(ds, copy_method)
        result = method()
        assert result._types_converted is True
        assert result[0]['value'] == '123'

    def test_deepcopy_ensures_types_converted(self):
        """Verify deepcopy converts first, check_types=False and all.

        Mutation: dropping the conversion block at the top of
            deepcopy(), which would set the flag over unconverted
            values.
        Oracle: hand-computed int 123 from the '123' string.
        """
        ds = DataSet([
            {'value': '123'}
        ], columns=[('value', int)], check_types=False)
        assert ds._types_converted is False
        deep = ds.deepcopy()
        assert deep._types_converted is True
        assert deep[0]['value'] == 123


# --- Deepcopy Summary Test ---


# --- Multiple Copy Tests ---

def test_copy_multiple_times():
    """Verify copies of copies all keep pointing at the same row.

    Mutation: copy() rebuilding rows as `attrdict(row)`, which would
        break the chain at the first copy.
    Oracle: the hand-set 10 read from the original and from both
        sibling copies.
    """
    ds = DataSet([{'value': 1}])
    copy1 = ds.copy()
    copy2 = ds.copy()
    copy3 = copy1.copy()
    copy1[0]['value'] = 10
    assert ds[0]['value'] == 10
    assert copy2[0]['value'] == 10
    assert copy3[0]['value'] == 10


def test_deepcopy_multiple_times():
    """Verify a deepcopy of a deepcopy stays independent of both.

    Mutation: deepcopy() reusing the source rows
        (`ds.container = list(self.container)`).
    Oracle: hand-set 1, 10, 20 and 30 across the four datasets.
    """
    ds = DataSet([{'value': 1}])
    deep1 = ds.deepcopy()
    deep2 = ds.deepcopy()
    deep3 = deep1.deepcopy()
    deep1[0]['value'] = 10
    deep2[0]['value'] = 20
    deep3[0]['value'] = 30
    assert ds[0]['value'] == 1
    assert deep1[0]['value'] == 10
    assert deep2[0]['value'] == 20
    assert deep3[0]['value'] == 30


def test_shallowcopy_multiple_times():
    """Verify chained shallowcopies split keys but share values.

    Mutation: shallowcopy() reusing the source rows, which would leak
        a new key into every dataset.
    Oracle: membership of 'new_col' per dataset, plus the 999 appended
        into the shared list.
    """
    ds = DataSet([{'value': 1, 'items': [1, 2, 3]}])
    rowcopy1 = ds.shallowcopy()
    rowcopy2 = ds.shallowcopy()
    rowcopy3 = rowcopy1.shallowcopy()
    rowcopy1[0]['new_col'] = 'test1'
    rowcopy2[0]['new_col'] = 'test2'
    assert 'new_col' not in ds[0]
    assert rowcopy1[0]['new_col'] == 'test1'
    assert rowcopy2[0]['new_col'] == 'test2'
    assert 'new_col' not in rowcopy3[0]
    rowcopy1[0]['items'].append(999)
    assert 999 in ds[0]['items']
    assert 999 in rowcopy2[0]['items']
    assert 999 in rowcopy3[0]['items']


def test_deepcopy_independence_after_modification():
    """Verify a deepcopy stays independent across row, value and length.

    Mutation: `copy.copy(dict(row))` in place of
        `copy.deepcopy(dict(row))` in deepcopy(), which leaves the
        scores list shared.
    Oracle: hand-computed 'A', 85 and 2 rows in the original.
    """
    ds = DataSet([
        {'name': 'A', 'scores': [85, 90, 95]},
        {'name': 'B', 'scores': [75, 80, 85]}
    ])
    deep = ds.deepcopy()
    deep[0]['name'] = 'AA'
    deep[0]['scores'][0] = 100
    deep.append({'name': 'C', 'scores': [70, 75, 80]})
    assert ds[0]['name'] == 'A'
    assert ds[0]['scores'][0] == 85
    assert len(ds) == 2


# --- Edge Case Tests ---

def test_copy_during_iteration():
    """Verify a copy taken mid-iteration holds every row, in order.

    Mutation: a copy method that truncates or empties the source
        container (`list(self.container)[:-1]`), which a copy taken
        part way through iteration would hide.
    Oracle: hand-computed [0, 1, 2, 3, 4] for the rows visited and for
        each of the three copies.
    """
    ds = DataSet([
        {'value': i} for i in range(5)
    ])
    copies = []
    visited = []
    for i, row in enumerate(ds):
        visited.append(row['value'])
        if i == 2:
            copies.extend((ds.copy(), ds.deepcopy(), ds.shallowcopy()))
    assert visited == [0, 1, 2, 3, 4]
    assert len(copies) == 3
    for c in copies:
        assert list(c.unwind('value')) == [0, 1, 2, 3, 4]


def test_copy_with_time_type():
    """Verify every copy method keeps a Time column typed and valued.

    Mutation: `_copy_structure` dropping `ds.columns`, which loses the
        declared Time type and leaves the copy's colmap empty.
    Oracle: the hand-written Time column type and Time(10, 30, 0).
    """
    ds = DataSet([
        {'time': Time(10, 30, 0), 'value': 100}
    ])
    ds.columns = [('time', Time), ('value', int)]

    for method_name in ['copy', 'deepcopy', 'shallowcopy']:
        method = getattr(ds, method_name)
        result = method()
        assert result.colmap['time'] == Time
        assert result[0]['time'] == Time(10, 30, 0)


def test_copy_after_filter_operation():
    """Verify copies taken after a filter hold only the kept rows.

    Mutation: a copy method aliasing the source list
        (`ds.container = self.container`), so a later append to the
        source shows up in the copies.
    Oracle: hand-computed [0, 2, 4, 6, 8] per copy, taken before the
        source gains a sixth row.
    """
    ds = DataSet([
        {'value': i} for i in range(10)
    ])
    ds.filter_data(lambda r: r['value'] % 2 == 0)
    assert len(ds) == 5

    results = [getattr(ds, name)()
               for name in ('copy', 'deepcopy', 'shallowcopy')]
    ds.append({'value': 100})

    for result in results:
        assert len(result) == 5
        assert list(result.unwind('value')) == [0, 2, 4, 6, 8]


def test_copy_after_join_operation():
    """Verify a copy of a joined dataset keeps the joined-in column.

    Mutation: `_copy_structure` dropping `ds.columns`, which loses the
        columns join() guessed from the merged rows.
    Oracle: the hand-written ['id', 'name', 'value'] schema and the
        100 and 200 paired to ids 1 and 2.
    """
    ds1 = DataSet([
        {'id': 1, 'name': 'A'},
        {'id': 2, 'name': 'B'}
    ])
    ds2 = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200}
    ])
    joined = DataSet.join(ds1, 'id', ds2, 'id')

    for method_name in ['copy', 'deepcopy', 'shallowcopy']:
        method = getattr(joined, method_name)
        result = method()
        assert result.cols == ['id', 'name', 'value']
        assert result[0]['value'] == 100
        assert result[1]['value'] == 200


def test_deepcopy_with_self_referencing_row():
    """Verify deepcopy rebuilds a cycle inside the copy, not across it.

    Mutation: `copy.copy(dict(row))` in place of
        `copy.deepcopy(dict(row))` in deepcopy(), which hands the copy
        the source's own cyclic dict.
    Oracle: identity of the copied cycle against both the copy and the
        original.
    """
    cyclic = {'key': 'value'}
    cyclic['self'] = cyclic
    ds = DataSet([
        {'name': 'A', 'data': cyclic},
    ])
    deep = ds.deepcopy()
    assert deep[0]['data']['self'] is deep[0]['data']
    assert deep[0]['data'] is not ds[0]['data']
    deep[0]['data']['key'] = 'modified'
    assert ds[0]['data']['key'] == 'value'


def test_copy_very_large_dataset():
    """Verify copies of a 1000-row dataset keep every row, in order.

    Mutation: an off-by-one slice in the container copy
        (`list(self.container)[:-1]` or `[1:]`).
    Oracle: hand-computed 1000 rows with 0 first and 9990 last, from
        the i * 10 rule.
    """
    ds = DataSet([{'id': i, 'value': i * 10} for i in range(1000)])

    copy_result = ds.copy()
    deep_result = ds.deepcopy()
    shallow_result = ds.shallowcopy()

    for result in (copy_result, deep_result, shallow_result):
        assert len(result) == 1000
        assert result[0]['value'] == 0
        assert result[999]['value'] == 9990

    deep_result[500]['value'] = -1
    assert ds[500]['value'] == 5000


# --- Source-State Tests - What deepcopy Leaves Behind ---

def test_deepcopy_does_not_reconvert_the_source():
    """Verify deepcopy flags the source so conversion runs only once.

    Mutation: `self._types_converted = False` (or None) after the
        conversion call in deepcopy(), leaving the source unflagged.
    Oracle: a spy counting convert_container_types calls, hand-counted
        at 1 across two deepcopies of the same dataset.
    """
    ds = DataSet([{'value': '123'}], columns=[('value', int)])
    convert_calls = []
    real_convert = ds.convert_container_types

    def counting_convert():
        convert_calls.append(1)
        real_convert()

    ds.convert_container_types = counting_convert

    first = ds.deepcopy()
    assert convert_calls == [1]
    second = ds.deepcopy()
    assert convert_calls == [1]
    assert first[0]['value'] == 123
    assert second[0]['value'] == 123


def test_deepcopy_leaves_source_ready_for_a_deep_filter():
    """Verify a filtered copy stays deep once the source was deepcopied.

    Mutation: `self._types_converted = False` (or None) after the
        conversion call in deepcopy(), which sends a later
        filter_data down its row-sharing branch.
    Oracle: the hand-written ['x'] still whole in the source after the
        filtered copy appends to its own list.
    """
    ds = DataSet([
        {'name': 'A', 'tags': ['x']}
    ], columns=[('name', str), ('tags', list)], check_types=False)
    ds.deepcopy()

    kept = ds.filter_data(lambda row: row['name'] == 'A', inplace=False)
    kept[0]['tags'].append('y')

    assert ds[0]['tags'] == ['x']
    assert kept[0]['tags'] == ['x', 'y']


def test_deepcopy_summary_is_independent_of_the_source():
    """Verify a deepcopy totals its own rows, unmoved by later source edits.

    Mutation: deepcopy sharing the source's row objects, so editing the
        source shifts the copy's recomputed total too.
    Oracle: hand-computed 300 for both, then 300 for the copy after the
        source's first value drops to 10 and its own total falls to 210.
    """
    ds = DataSet([
        {'name': 'A', 'value': 100},
        {'name': 'B', 'value': 200}])
    ds.columns = [('name', str), ('value', int)]
    ds.add_summary_row(label='Total', columns=['value'])

    deep = ds.deepcopy()
    assert ds.summary['value'] == 300
    assert deep.summary['value'] == 300

    ds[0]['value'] = 10
    assert deep.summary['value'] == 300
    assert ds.summary['value'] == 210


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
