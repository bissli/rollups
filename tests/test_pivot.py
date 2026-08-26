"""Tests for DataSet pivot operations.

Pivot transforms data from long format to wide format by creating new columns
based on unique values in a pivot column.
"""
import pytest
from opendate import UTC, Date, DateTime, Time
from rollups import DataSet

# --- Fixtures ---


@pytest.fixture
def basic_pivot_dataset():
    """Basic dataset for pivot tests with id, value, and category."""
    return DataSet([
        {'id': 1, 'value': 10., 'category': 'a'},
        {'id': 2, 'value': 20., 'category': 'a'},
        {'id': 3, 'value': 30., 'category': 'a'},
        {'id': 1, 'value': 100., 'category': 'b'},
        {'id': 2, 'value': 200., 'category': 'b'},
        {'id': 3, 'value': 300., 'category': 'b'}
    ])


@pytest.fixture
def multi_value_dataset():
    """Dataset with multiple value columns for pivot tests."""
    return DataSet([
        {'id': 1, 'value1': 10., 'value2': 100., 'category': 'a'},
        {'id': 2, 'value1': 20., 'value2': 200., 'category': 'a'},
        {'id': 3, 'value1': 30., 'value2': 300., 'category': 'a'},
        {'id': 1, 'value1': 1000., 'value2': 1100., 'category': 'b'},
        {'id': 2, 'value1': 2000., 'value2': 2200., 'category': 'b'},
        {'id': 3, 'value1': 3000., 'value2': 3300., 'category': 'b'}
    ])


@pytest.fixture
def aggregation_dataset():
    """Dataset with duplicate id/category combinations for aggregation tests."""
    return DataSet([
        {'id': 1, 'value': 10, 'category': 'x'},
        {'id': 1, 'value': 20, 'category': 'x'},
        {'id': 1, 'value': 5, 'category': 'y'},
        {'id': 2, 'value': 15, 'category': 'x'},
        {'id': 2, 'value': 8, 'category': 'y'}
    ])


# --- Basic Pivot Tests ---

def test_pivot_basic_single_column(basic_pivot_dataset):
    """Verify basic pivot with single data column.

    Mutation: flipping `row[pivot_col] == col` to `!=` in pivot's
        filterby.
    Oracle: hand-computed cross-tab, id 1 -> a=10, b=100; the flip
        swaps the two.
    """
    result = basic_pivot_dataset.pivot('id', 'value', 'category')

    assert len(result) == 3
    assert result.cols == ['id', 'a', 'b']
    result.sort_data('id')
    assert result[0].id == 1
    assert result[0].a == 10.0
    assert result[0].b == 100.0
    assert result[1].id == 2
    assert result[1].a == 20.0
    assert result[1].b == 200.0


def test_pivot_multiple_data_columns_with_alias(multi_value_dataset):
    """Verify pivot with multiple data columns using column aliases.

    Mutation: swapping the loop order in
        `[... for c in new_cols for d in data_cols]`.
    Oracle: hand-written column list, pivot value varying slowest.
    """
    result = multi_value_dataset.pivot(
        'id', ['value1', 'value2'], 'category',
        alias=lambda x, y: f'{x}:{y}'
    )

    assert len(result) == 3
    assert result.cols == ['id', 'a:value1', 'a:value2', 'b:value1', 'b:value2']
    result.sort_data('id')
    assert result[0]['a:value1'] == 10.0
    assert result[0]['a:value2'] == 100.0
    assert result[0]['b:value1'] == 1000.0
    assert result[0]['b:value2'] == 1100.0


def test_pivot_single_data_column_without_alias():
    """Verify pivot with single data column uses default column names.

    Mutation: default `alias=lambda x, _: x` returning the data column
        name in place of the pivot value.
    Oracle: hand-written column list ['id', 'type_x', 'type_y'].
    """
    ds = DataSet([
        {'id': 'A', 'value': 100, 'category': 'type_x'},
        {'id': 'A', 'value': 200, 'category': 'type_y'},
        {'id': 'B', 'value': 150, 'category': 'type_x'}
    ])

    result = ds.pivot('id', 'value', 'category')

    assert result.cols == ['id', 'type_x', 'type_y']
    result.sort_data('id')
    assert result[0].type_x == 100
    assert result[0].type_y == 200
    assert result[1].type_x == 150


def test_pivot_with_missing_combinations():
    """Verify pivot fills a missing index/pivot combination with 0.0.

    Mutation: dropping the `or [0.0]` fallback in pivot's filterby.
    Oracle: max over the absent (id 2, b) cell is 0.0; with no
        fallback max([]) raises and the cell becomes None.
    """
    ds = DataSet([
        {'id': 1, 'value': 10, 'category': 'a'},
        {'id': 1, 'value': 20, 'category': 'b'},
        {'id': 2, 'value': 30, 'category': 'a'}
    ])

    result = ds.pivot('id', 'value', 'category')

    result.sort_data('id')
    assert len(result) == 2
    assert result[0].a == 10
    assert result[0].b == 20
    assert result[1].a == 30
    assert result[1].b == 0.0

    by_max = ds.pivot('id', 'value', 'category', aggr=max)

    by_max.sort_data('id')
    assert by_max[1].a == 30
    assert by_max[1].b == 0.0


def test_pivot_with_string_index():
    """Verify pivot works with string index column.

    Mutation: filterby scanning every row of the dataset instead of
        the group's rows.
    Oracle: hand-computed 2x2 grid; the leak makes A.Math 85+75=160.
    """
    ds = DataSet([
        {'name': 'A', 'score': 85, 'subject': 'Math'},
        {'name': 'A', 'score': 90, 'subject': 'English'},
        {'name': 'B', 'score': 75, 'subject': 'Math'},
        {'name': 'B', 'score': 80, 'subject': 'English'}
    ])

    result = ds.pivot('name', 'score', 'subject')

    result.sort_data('name')
    assert len(result) == 2
    assert result[0].name == 'A'
    assert result[0].Math == 85
    assert result[0].English == 90
    assert result[1].name == 'B'
    assert result[1].Math == 75
    assert result[1].English == 80


def test_pivot_multiple_columns_simple_alias():
    """Verify pivot with multiple data columns and simple alias function.

    Mutation: calling `alias(d, c)` with the data column first.
    Oracle: hand-written names, pivot label then data column, plus the
        0.0 fill on the absent (id 2, second) cells.
    """
    ds = DataSet([
        {'id': 1, 'value1': 10, 'value2': 100, 'category': 'first'},
        {'id': 1, 'value1': 20, 'value2': 200, 'category': 'second'},
        {'id': 2, 'value1': 30, 'value2': 300, 'category': 'first'}
    ])

    result = ds.pivot('id', ['value1', 'value2'], 'category',
                      alias=lambda lbl, col: f'{lbl}_{col}')

    result.sort_data('id')
    assert 'first_value1' in result.cols
    assert 'first_value2' in result.cols
    assert 'second_value1' in result.cols
    assert 'second_value2' in result.cols
    assert result[0].first_value1 == 10
    assert result[0].first_value2 == 100
    assert result[0].second_value1 == 20
    assert result[0].second_value2 == 200
    assert result[1].first_value1 == 30
    assert result[1].second_value1 == 0.0
    assert result[1].second_value2 == 0.0


def test_pivot_single_row_per_index():
    """Verify pivot with exactly one row per index value.

    Mutation: dropping the `row[pivot_col] == col` guard so every cell
        takes the group's values.
    Oracle: hand-computed diagonal, every off-diagonal cell 0.0.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'category': 'A'},
        {'id': 2, 'value': 200, 'category': 'B'},
        {'id': 3, 'value': 300, 'category': 'C'}
    ])

    result = ds.pivot('id', 'value', 'category')

    assert len(result) == 3
    result.sort_data('id')
    assert result[0].A == 100
    assert result[0].B == 0.0
    assert result[0].C == 0.0
    assert result[1].A == 0.0
    assert result[1].B == 200
    assert result[1].C == 0.0
    assert result[2].A == 0.0
    assert result[2].B == 0.0
    assert result[2].C == 300


def test_pivot_numeric_pivot_column():
    """Verify pivot works when pivot column contains numeric values.

    Mutation: stringifying the pivot value for the column name,
        `alias(str(c), d)`.
    Oracle: int keys 2020 and 2021 index the result rows; hand-computed
        cells 10/20 and 30/40.
    """
    ds = DataSet([
        {'id': 'X', 'value': 10, 'year': 2020},
        {'id': 'X', 'value': 20, 'year': 2021},
        {'id': 'Y', 'value': 30, 'year': 2020},
        {'id': 'Y', 'value': 40, 'year': 2021}
    ])

    result = ds.pivot('id', 'value', 'year')

    result.sort_data('id')
    assert len(result) == 2
    assert result[0][2020] == 10
    assert result[0][2021] == 20
    assert result[1][2020] == 30
    assert result[1][2021] == 40


def test_pivot_three_way_categories():
    """Verify pivot with three distinct categories in pivot column.

    Mutation: flipping `row[pivot_col] == col` to `!=` in filterby.
    Oracle: hand-computed 2x3 grid; the flip makes cell (1, X) 500.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'category': 'X'},
        {'id': 1, 'value': 200, 'category': 'Y'},
        {'id': 1, 'value': 300, 'category': 'Z'},
        {'id': 2, 'value': 150, 'category': 'X'},
        {'id': 2, 'value': 250, 'category': 'Y'},
        {'id': 2, 'value': 350, 'category': 'Z'}
    ])

    result = ds.pivot('id', 'value', 'category')

    assert len(result) == 2
    assert set(result.cols) == {'id', 'X', 'Y', 'Z'}
    result.sort_data('id')
    assert result[0].X == 100
    assert result[0].Y == 200
    assert result[0].Z == 300
    assert result[1].X == 150
    assert result[1].Y == 250
    assert result[1].Z == 350


def test_pivot_alias_with_special_characters():
    """Verify pivot alias function can include special characters.

    Mutation: skipping the alias when there is a single data column,
        naming the column after the pivot value alone.
    Oracle: hand-written column list ['id', 'A.value', 'B.value'] and
        the cells behind those names.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'category': 'A'},
        {'id': 2, 'value': 200, 'category': 'B'}
    ])

    result = ds.pivot('id', 'value', 'category', alias=lambda x, y: f'{x}.{y}')

    assert result.cols == ['id', 'A.value', 'B.value']
    result.sort_data('id')
    assert result[0]['A.value'] == 100
    assert result[0]['B.value'] == 0.0
    assert result[1]['B.value'] == 200


# --- Aggregation Function Tests (Parameterized) ---

@pytest.mark.parametrize(('aggr_func', 'expected_x', 'expected_y'), [
    (max, 20, 5),
    (min, 10, 5),
    (sum, 30, 5),
    (len, 2, 1),
])
def test_pivot_with_aggregation_functions(
    aggregation_dataset,
    aggr_func,
    expected_x,
    expected_y):
    """Verify pivot with various aggregation functions.

    Mutation: ignoring `aggr` and hardcoding sum in the aggregation
        tuple pivot hands to bucket.
    Oracle: hand-computed over id 1, whose x cell holds [10, 20] and y
        cell holds [5].
    """
    result = aggregation_dataset.pivot('id', 'value', 'category', aggr=aggr_func)

    result.sort_data('id')
    assert result[0].x == expected_x
    assert result[0].y == expected_y


def test_pivot_with_average_aggregation():
    """Verify pivot with average aggregation function.

    Mutation: dropping filterby so the aggregation runs over the whole
        group rather than the cell's rows.
    Oracle: hand-computed avg(10, 20)=15.0 for A and 12.0 for B; the
        whole group averages to 14.0.
    """
    ds = DataSet([
        {'id': 1, 'value': 10, 'category': 'A'},
        {'id': 1, 'value': 20, 'category': 'A'},
        {'id': 1, 'value': 12, 'category': 'B'},
        {'id': 2, 'value': 30, 'category': 'A'}
    ])

    def avg(values):
        return sum(values) / len(values)

    result = ds.pivot('id', 'value', 'category', aggr=avg)

    result.sort_data('id')
    assert result[0].A == 15.0
    assert result[0].B == 12.0
    assert result[1].A == 30.0


def test_pivot_duplicate_aggregation():
    """Verify pivot aggregates duplicate index and pivot combinations.

    Mutation: default `aggr=sum` replaced by max, or filterby keeping
        only the first matching row.
    Oracle: hand-computed 10 + 20 + 30 = 60, which no single row and
        no extremum reaches.
    """
    ds = DataSet([
        {'id': 1, 'value': 10, 'category': 'A'},
        {'id': 1, 'value': 20, 'category': 'A'},
        {'id': 1, 'value': 30, 'category': 'A'},
        {'id': 2, 'value': 5, 'category': 'B'}
    ])

    result = ds.pivot('id', 'value', 'category')

    result.sort_data('id')
    assert result[0].A == 60
    assert result[1].B == 5


def test_pivot_list_aggregation():
    """Verify pivot with list aggregation to collect all values.

    Mutation: filterby sorting or deduping the matched values instead
        of keeping source-row order.
    Oracle: hand-computed ['banana', 'apple'], the order the rows were
        written in, which sorting would reverse.
    """
    ds = DataSet([
        {'group': 'X', 'item': 'banana', 'category': 'fruit'},
        {'group': 'X', 'item': 'apple', 'category': 'fruit'},
        {'group': 'X', 'item': 'carrot', 'category': 'vegetable'},
        {'group': 'Y', 'item': 'grape', 'category': 'fruit'}
    ])

    result = ds.pivot('group', 'item', 'category', aggr=list)

    result.sort_data('group')
    assert result[0].fruit == ['banana', 'apple']
    assert result[0].vegetable == ['carrot']
    assert result[1].fruit == ['grape']


# --- Type Preservation Tests ---

def test_pivot_column_types():
    """Verify pivot column types: declared index type, 0.0-filled cells.

    Mutation: filling an absent cell with int 0 rather than 0.0, which
        leaves the B column int.
    Oracle: hand-computed types - A is filled for every id and stays
        int, B is filled for id 2 only and is promoted to float.
    """
    ds = DataSet([
        {'id': 1, 'value': 10, 'category': 'A'},
        {'id': 1, 'value': 20, 'category': 'B'},
        {'id': 2, 'value': 30, 'category': 'A'}
    ])
    ds.columns = [('id', int), ('value', int), ('category', str)]

    result = ds.pivot('id', 'value', 'category')

    assert result.colmap['id'] == int
    assert result.colmap['A'] == int
    assert result.colmap['B'] == float
    for row in result:
        assert isinstance(row['id'], int)
        assert isinstance(row['A'], int)
        assert isinstance(row['B'], float)


# --- Empty Dataset Tests ---

def test_pivot_empty_dataset():
    """Verify pivot on empty dataset returns empty result.

    Mutation: inferring the index column type from the result rows
        instead of copying it from the source colmap.
    Oracle: no row survives to infer from, so only the declared int
        can produce colmap['id'] == int.
    """
    ds = DataSet([], columns=[('id', int), ('value', float), ('category', str)])

    result = ds.pivot('id', 'value', 'category')

    assert len(result) == 0
    assert result.cols == ['id']
    assert result.colmap['id'] == int


# --- Edge Case Tests ---

def test_pivot_with_none_in_pivot_column():
    """Verify a None pivot value becomes a column of its own.

    Mutation: skipping falsy pivot values when building new_cols.
    Oracle: hand-computed - the None column holds 200 for id 1 and the
        0.0 fill for id 2; skipping None drops the column entirely.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'category': 'A'},
        {'id': 1, 'value': 200, 'category': None},
        {'id': 2, 'value': 300, 'category': 'A'}
    ])

    result = ds.pivot('id', 'value', 'category')

    result.sort_data('id')
    assert len(result) == 2
    assert None in result.cols
    assert result[0].A == 100
    assert result[0][None] == 200
    assert result[1].A == 300
    assert result[1][None] == 0.0


def test_pivot_with_date_in_pivot_column():
    """Test pivot with Date values in pivot column.

    Mutation: stringifying the pivot value for the column name.
    Oracle: Date objects index the result row; hand-computed 100 and
        200 for id 1.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'date': Date(2024, 1, 1)},
        {'id': 1, 'value': 200, 'date': Date(2024, 1, 2)},
        {'id': 2, 'value': 300, 'date': Date(2024, 1, 1)}
    ])

    result = ds.pivot('id', 'value', 'date')

    assert len(result) == 2
    result.sort_data('id')
    assert result[0][Date(2024, 1, 1)] == 100
    assert result[0][Date(2024, 1, 2)] == 200
    assert result[1][Date(2024, 1, 1)] == 300


def test_pivot_very_many_categories():
    """Test pivot with many unique categories (stress test).

    Mutation: filterby scanning every row of the dataset instead of
        the group's rows.
    Oracle: id 0 owns rows 0, 10, ... 90, so cat_30 is 30 and cat_31 -
        another group's row - is the 0.0 fill, not 31.
    """
    ds = DataSet([
        {'id': i % 10, 'value': i, 'category': f'cat_{i}'} for i in range(100)
    ])

    result = ds.pivot('id', 'value', 'category')

    assert len(result) == 10
    assert len(result.cols) == 101
    result.sort_data('id')
    assert result[0].id == 0
    assert result[0]['cat_30'] == 30
    assert result[0]['cat_31'] == 0.0
    assert result[3]['cat_73'] == 73
    assert result[3]['cat_70'] == 0.0


def test_pivot_single_row_dataset():
    """Test pivot on single row dataset.

    Mutation: carrying the source data and pivot columns into the
        result row alongside the pivoted cell.
    Oracle: the whole row equals the hand-written {'id': 1, 'A': 100}.
    """
    ds = DataSet([{'id': 1, 'value': 100, 'category': 'A'}])

    result = ds.pivot('id', 'value', 'category')

    assert len(result) == 1
    assert result.cols == ['id', 'A']
    assert result[0] == {'id': 1, 'A': 100}


def test_pivot_all_same_category():
    """Test pivot where all rows have same category.

    Mutation: filterby scanning every row of the dataset instead of
        the group's rows.
    Oracle: hand-computed one cell per id; the leak makes every cell
        100+200+300=600.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'category': 'A'},
        {'id': 2, 'value': 200, 'category': 'A'},
        {'id': 3, 'value': 300, 'category': 'A'}
    ])

    result = ds.pivot('id', 'value', 'category')

    assert len(result) == 3
    assert result.cols == ['id', 'A']
    result.sort_data('id')
    assert result[0].A == 100
    assert result[1].A == 200
    assert result[2].A == 300


def test_pivot_empty_string_category():
    """Test pivot with empty string as category.

    Mutation: skipping falsy pivot values when building new_cols.
    Oracle: hand-computed - the '' column holds 100 for id 1 and 300
        for id 2; skipping falsy values drops the column entirely.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'category': ''},
        {'id': 1, 'value': 200, 'category': 'A'},
        {'id': 2, 'value': 300, 'category': ''}
    ])

    result = ds.pivot('id', 'value', 'category')

    assert len(result) == 2
    assert '' in result.cols
    result.sort_data('id')
    assert result[0][''] == 100
    assert result[0].A == 200
    assert result[1][''] == 300
    assert result[1].A == 0.0


def test_pivot_with_boolean_pivot_column():
    """Test pivot with boolean values in pivot column.

    Mutation: stringifying the pivot value for the column name, which
        gives 'True' and 'False' keys.
    Oracle: the bool objects index the result row; hand-computed 100
        for True and 200 for False on id 1.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'is_active': True},
        {'id': 1, 'value': 200, 'is_active': False},
        {'id': 2, 'value': 300, 'is_active': True}
    ])
    ds.columns = [('id', int), ('value', int), ('is_active', bool)]

    result = ds.pivot('id', 'value', 'is_active')

    assert len(result) == 2
    result.sort_data('id')
    assert result[0][True] == 100
    assert result[0][False] == 200
    assert result[1][True] == 300


def test_pivot_with_float_pivot_column():
    """Test pivot with float values in pivot column.

    Mutation: rounding the pivot value for the column name, which
        collapses 1.5 and 2.5 onto one column.
    Oracle: the floats index the result row; hand-computed 100 and 200
        held apart on id 1.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'rate': 1.5},
        {'id': 1, 'value': 200, 'rate': 2.5},
        {'id': 2, 'value': 300, 'rate': 1.5}
    ])

    result = ds.pivot('id', 'value', 'rate')

    assert len(result) == 2
    result.sort_data('id')
    assert result[0][1.5] == 100
    assert result[0][2.5] == 200
    assert result[1][1.5] == 300


def test_pivot_aggregation_raises_exception():
    """Test pivot behavior when aggregation function raises exception.

    Mutation: broadening bucket's `except (ValueError, TypeError)` to
        `except Exception`, which swallows the error into None.
    Oracle: ZeroDivisionError reaches the caller.
    """
    ds = DataSet([
        {'id': 1, 'value': 10, 'category': 'A'},
        {'id': 1, 'value': 0, 'category': 'A'},
    ])

    def bad_agg(values):
        return sum(100 / v for v in values)

    with pytest.raises(ZeroDivisionError):
        ds.pivot('id', 'value', 'category', aggr=bad_agg)


def test_pivot_orders_rows_by_index():
    """Verify pivot rows come out sorted by index, None last.

    Mutation: dropping the sort before groupby, or dropping the
        `row[col] is None` term from bucket's sort key.
    Oracle: hand-computed [1, 3, None] from source order [3, None, 1];
        unsorted keeps 3 first, and the missing None term raises on
        comparing None with int.
    """
    ds = DataSet([
        {'id': 3, 'value': 300, 'category': 'A'},
        {'id': None, 'value': 50, 'category': 'A'},
        {'id': 1, 'value': 100, 'category': 'A'}
    ])

    result = ds.pivot('id', 'value', 'category')

    assert len(result) == 3
    assert [row.id for row in result] == [1, 3, None]
    assert [row.A for row in result] == [100, 300, 50]


def test_pivot_with_time_in_pivot_column():
    """Test pivot with Time values in pivot column.

    Mutation: stringifying the pivot value for the column name.
    Oracle: Time objects index the result row; hand-computed 100 at
        09:00 and 200 at 14:00 for id 1.
    """
    ds = DataSet([
        {'id': 1, 'value': 100, 'time': Time(9, 0, 0, tzinfo=UTC)},
        {'id': 1, 'value': 200, 'time': Time(14, 0, 0, tzinfo=UTC)},
        {'id': 2, 'value': 300, 'time': Time(9, 0, 0, tzinfo=UTC)}
    ])

    result = ds.pivot('id', 'value', 'time')

    assert len(result) == 2
    result.sort_data('id')
    assert result[0][Time(9, 0, 0, tzinfo=UTC)] == 100
    assert result[0][Time(14, 0, 0, tzinfo=UTC)] == 200
    assert result[1][Time(9, 0, 0, tzinfo=UTC)] == 300


def test_pivot_with_datetime_in_pivot_column():
    """Test pivot with DateTime values in pivot column.

    Mutation: truncating the pivot value to its date for the column
        name, which collapses the two timestamps onto one column.
    Oracle: DateTime objects index the result row; hand-computed 100
        and 200 held apart on id 1.
    """
    dt1 = DateTime(2024, 1, 1, 9, 0, 0, tzinfo=UTC)
    dt2 = DateTime(2024, 1, 1, 14, 0, 0, tzinfo=UTC)
    ds = DataSet([
        {'id': 1, 'value': 100, 'timestamp': dt1},
        {'id': 1, 'value': 200, 'timestamp': dt2},
        {'id': 2, 'value': 300, 'timestamp': dt1}
    ])

    result = ds.pivot('id', 'value', 'timestamp')

    assert len(result) == 2
    result.sort_data('id')
    assert result[0][dt1] == 100
    assert result[0][dt2] == 200
    assert result[1][dt1] == 300


def test_pivot_set_aggregation():
    """Test pivot with set aggregation to collect unique values.

    Mutation: filterby scanning every row of the dataset instead of
        the group's rows.
    Oracle: hand-computed {'apple', 'banana'} for X and {'grape'} for
        Y; the leak gives Y all three.
    """
    ds = DataSet([
        {'group': 'X', 'item': 'apple', 'category': 'fruit'},
        {'group': 'X', 'item': 'apple', 'category': 'fruit'},
        {'group': 'X', 'item': 'banana', 'category': 'fruit'},
        {'group': 'Y', 'item': 'grape', 'category': 'fruit'}
    ])

    result = ds.pivot('group', 'item', 'category', aggr=set)

    result.sort_data('group')
    assert result[0].fruit == {'apple', 'banana'}
    assert result[1].fruit == {'grape'}


def test_pivot_with_negative_values():
    """Test pivot keeps the sign of values it aggregates.

    Mutation: summing absolute values, or keeping only the first
        matching row.
    Oracle: hand-computed -100 + 30 = -70, which the absolute sum
        (130) and the first row (-100) both miss.
    """
    ds = DataSet([
        {'id': 1, 'value': -100, 'category': 'A'},
        {'id': 1, 'value': 30, 'category': 'A'},
        {'id': 1, 'value': 200, 'category': 'B'},
        {'id': 2, 'value': -50, 'category': 'A'}
    ])

    result = ds.pivot('id', 'value', 'category')

    result.sort_data('id')
    assert result[0].A == -70
    assert result[0].B == 200
    assert result[1].A == -50


# --- Transpose Tests ---

def test_transpose_swaps_rows_and_columns():
    """Verify one column's values become the new column names.

    Mutation: an off-by-one in cols[:pivot_index] + cols[pivot_index+1:],
        which either drops a column or repeats the pivot column.
    Oracle: hand-computed transpose of a 2x3 grid.
    """
    ds = DataSet([
        {'name': 'r1', 'q1': 1, 'q2': 2},
        {'name': 'r2', 'q1': 3, 'q2': 4},
    ])

    got = ds.transpose('metric')

    assert got.cols == ['metric', 'r1', 'r2']
    assert [tuple(r[c] for c in got.cols) for r in got] == [
        ('q1', 1, 3),
        ('q2', 2, 4),
    ]


def test_transpose_honors_pivot_index():
    """Verify pivot_index picks which column supplies the new names.

    Mutation: ignoring pivot_index and always pivoting on column 0.
    Oracle: pivoting on column 1 must name the columns from q1's values,
        which are 1 and 3, not from name's values r1 and r2.
    """
    ds = DataSet([
        {'name': 'r1', 'q1': 1},
        {'name': 'r2', 'q1': 3},
    ])

    got = ds.transpose('metric', pivot_index=1)

    assert got.cols == ['metric', 1, 3]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class _Sub(DataSet):
    """Subclass carrying a marker a plain DataSet cannot fake."""

    marker = 'sub'


def test_transpose_and_pivot_return_the_receivers_class():
    """Verify transpose and pivot both build through self.__class__.

    Mutation: transpose returning a bare `DataSet(new_rows, ...)` rather
    than building through the receiver's class.
    Oracle: the exact type of both results, and the transposed column
    names, which are the old rows' pivot values.
    """
    ds = _Sub([{'k': 'a', 'v': 1}, {'k': 'b', 'v': 2}])
    ds.columns = [('k', str), ('v', int)]

    flipped = ds.transpose('name')
    assert type(flipped) is _Sub
    assert flipped.cols == ['name', 'a', 'b']

    pivoted = ds.pivot('k', 'v', 'k')
    assert type(pivoted) is _Sub
