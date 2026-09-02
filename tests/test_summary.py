import math

import pytest
from opendate import UTC, Date, DateTime
from rollups import DataSet

from libb import lazydict

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Basic dataset for summary tests."""
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 200},
        {'id': 3, 'name': 'C', 'value': 300}])
    ds.columns = (('id', int), ('name', str), ('value', int))
    return ds


@pytest.fixture
def multi_column_dataset():
    """Dataset with multiple column types."""
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100, 'count': 5},
        {'id': 2, 'name': 'B', 'value': 200, 'count': 10},
        {'id': 3, 'name': 'C', 'value': 300, 'count': 15}])
    ds.columns = (('id', int), ('name', str), ('value', int), ('count', int))
    return ds


# --- Basic Summary Row Tests ---

def test_summary_row_default_excludes_label_column(basic_dataset):
    """Verify the default totals every numeric column but the label one.

    Mutation: `col != label_col` flipped to `col == label_col` in
        add_summary_row, which drops 'value' from the total.
    Oracle: hand-computed 100+200+300, and 'Total' where the id total
        of 6 would otherwise sit.
    """
    basic_dataset.add_summary_row()

    assert basic_dataset.summary['id'] == 'Total'
    assert basic_dataset.summary['value'] == 600
    assert basic_dataset.summary['name'] is None


def test_summary_row_computed_lazily_after_append():
    """Verify the total counts rows appended after the row was declared.

    Mutation: add_summary_row computing the row eagerly and caching it,
        which would total only the two rows present at declare time.
    Oracle: hand-computed 5+10+15 against the eager 5+10.
    """
    ds = DataSet([
        {'amount': 100, 'category': 'A', 'count': 5},
        {'amount': 200, 'category': 'B', 'count': 10}])
    ds.columns = (('amount', int), ('category', str), ('count', int))

    ds.add_summary_row()
    ds.append({'amount': 300, 'category': 'C', 'count': 15})

    assert ds.summary['amount'] == 'Total'
    assert ds.summary['count'] == 30


def test_summary_row_explicit_columns_overrides_exclusion(basic_dataset):
    """Verify the label wins over a label column the caller aggregated.

    Mutation: guarding the label write with `label_col not in columns`,
        which would leave the id total of 6 in the label column.
    Oracle: 'Total' against the 6 that aggregating id produces.
    """
    basic_dataset.add_summary_row(columns=['id', 'value'])

    assert basic_dataset.summary['id'] == 'Total'
    assert basic_dataset.summary['value'] == 600


def test_summary_row_label_idx_middle_column(multi_column_dataset):
    """Verify a middle label column still totals the columns before it.

    Mutation: excluding by position (every column up to label_idx) or an
        off-by-one on label_idx, which moves 'Total' onto 'value'.
    Oracle: hand-computed 6 / 'Total' / 600 / 30 across the four
        columns.
    """
    multi_column_dataset.add_summary_row(label_idx=1)

    assert multi_column_dataset.summary['id'] == 6
    assert multi_column_dataset.summary['name'] == 'Total'
    assert multi_column_dataset.summary['value'] == 600
    assert multi_column_dataset.summary['count'] == 30


# --- Label Index Boundary Tests (Parameterized) ---

@pytest.mark.parametrize('label_idx', [100, -1, -100])
def test_summary_row_label_idx_boundary(label_idx):
    """Verify an out-of-range label_idx writes no label and totals all.

    Mutation: dropping either half of the
        `0 <= label_idx < len(self.columns)` guard, which makes -1 label
        the last column and 100 or -100 raise IndexError.
    Oracle: hand-computed 6 and 600, with 'Total' in no column.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
        {'id': 3, 'value': 300}])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row(label_idx=label_idx)

    assert ds.summary['id'] == 6
    assert ds.summary['value'] == 600
    assert 'Total' not in ds.summary.values()


# --- Column Type Tests ---

def test_summary_row_all_numeric_columns():
    """Verify each numeric column totals its own values, not a sibling's.

    Mutation: bucket reusing one source column for every alias, which
        would put 60 in both 'b' and 'c'.
    Oracle: hand-computed 60 and 600, an order of magnitude apart.
    """
    ds = DataSet([
        {'a': 1, 'b': 10, 'c': 100},
        {'a': 2, 'b': 20, 'c': 200},
        {'a': 3, 'b': 30, 'c': 300}])
    ds.columns = (('a', int), ('b', int), ('c', int))

    ds.add_summary_row()

    assert ds.summary['a'] == 'Total'
    assert ds.summary['b'] == 60
    assert ds.summary['c'] == 600


def test_summary_row_no_numeric_columns():
    """Verify the label still lands when no column can be totaled.

    Mutation: returning the all-None summary whenever nothing was
        aggregated, which would drop the 'Total' label.
    Oracle: hand-computed 'Total' in name, None in city.
    """
    ds = DataSet([
        {'name': 'Alice', 'city': 'NYC'},
        {'name': 'Bob', 'city': 'LA'},
        {'name': 'Carol', 'city': 'SF'}])
    ds.columns = (('name', str), ('city', str))

    ds.add_summary_row()

    assert ds.summary['name'] == 'Total'
    assert ds.summary['city'] is None


def test_summary_row_empty_columns_list():
    """Verify an empty columns list totals nothing, unlike None.

    Mutation: `if columns is None` relaxed to `if not columns`, which
        auto-selects the numeric columns and totals value to 300.
    Oracle: hand-computed None against the 300 auto-selection gives.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 200}])
    ds.columns = (('id', int), ('name', str), ('value', int))

    ds.add_summary_row(columns=[])

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] is None


def test_summary_row_float_columns():
    """Verify a float column totals without losing the fraction.

    Mutation: coercing the total with int(), which turns 601.5 into 601.
    Oracle: hand-computed 100.5 + 200.25 + 300.75.
    """
    ds = DataSet([
        {'id': 1, 'value': 100.5},
        {'id': 2, 'value': 200.25},
        {'id': 3, 'value': 300.75}])
    ds.columns = (('id', int), ('value', float))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] == 601.5


def test_summary_row_flagged_as_summary():
    """Verify the summary row is marked as one and a data row is not.

    Mutation: dropping `summary['__is_summary__'] = True`, or
        is_summary_row defaulting the missing flag to True.
    Oracle: a data row from the same dataset as the negative control.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200}])
    ds.columns = (('id', int), ('value', int))

    assert DataSet.is_summary_row(ds.summary)
    assert not DataSet.is_summary_row(ds[0])


# --- Edge Case Tests ---

def test_summary_row_empty_dataset():
    """Verify an empty dataset summarizes to None, with no label.

    Mutation: `if total:` weakened to `if total is not None:`, which
        indexes row 0 of an empty bucket result and raises IndexError.
    Oracle: hand-computed None in every column, 'Total' nowhere.
    """
    ds = DataSet([], columns=[('id', int), ('value', int)])
    ds.add_summary_row()

    assert ds.summary['id'] is None
    assert ds.summary['value'] is None


def test_summary_row_single_row():
    """Verify a one-row dataset totals to that row's own value.

    Mutation: an off-by-one that skips a group's first row, which leaves
        a one-row dataset with nothing to total.
    Oracle: hand-computed 100, the only value present.
    """
    ds = DataSet([{'id': 1, 'value': 100}])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] == 100


def test_summary_row_with_none_values():
    """Verify None values are skipped rather than poisoning the total.

    Mutation: dropping bucket's non-None filter, so sum() raises
        TypeError and the total falls back to None.
    Oracle: hand-computed 100+300 with the None row skipped.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': None},
        {'id': 3, 'value': 300}])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] == 400


def test_summary_row_all_none_values():
    """Verify an all-None column totals to None rather than zero.

    Mutation: bucket's `result or [None]` changed to `result or []`,
        which makes the empty sum return 0.
    Oracle: None against the 0 an empty sum gives.
    """
    ds = DataSet([
        {'id': 1, 'value': None},
        {'id': 2, 'value': None},
        {'id': 3, 'value': None}])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] is None


def test_summary_row_with_infinity_values():
    """Verify an infinite value carries through to the total.

    Mutation: filtering non-finite values out alongside None, which
        would give 400.
    Oracle: IEEE addition of 100.0 + inf + 300.0.
    """
    ds = DataSet([
        {'id': 1, 'value': 100.0},
        {'id': 2, 'value': float('inf')},
        {'id': 3, 'value': 300.0}])
    ds.columns = (('id', int), ('value', float))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] == float('inf')


def test_summary_row_with_nan_values():
    """Verify a NaN value propagates instead of being dropped.

    Mutation: treating NaN as missing the way None is treated, which
        would give 400.
    Oracle: IEEE addition, where NaN poisons the sum.
    """
    ds = DataSet([
        {'id': 1, 'value': 100.0},
        {'id': 2, 'value': float('nan')},
        {'id': 3, 'value': 300.0}])
    ds.columns = (('id', int), ('value', float))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert math.isnan(ds.summary['value'])


def test_summary_row_specific_columns():
    """Verify only the named columns are totaled.

    Mutation: ignoring the caller's columns list and auto-selecting
        every numeric column, which would total val2 to 60.
    Oracle: hand-computed 600 in val1, None in val2.
    """
    ds = DataSet([
        {'id': 1, 'val1': 100, 'val2': 10},
        {'id': 2, 'val1': 200, 'val2': 20},
        {'id': 3, 'val1': 300, 'val2': 30}])
    ds.columns = (('id', int), ('val1', int), ('val2', int))

    ds.add_summary_row(columns=['val1'])

    assert ds.summary['id'] == 'Total'
    assert ds.summary['val1'] == 600
    assert ds.summary['val2'] is None


def test_summary_row_very_large_dataset():
    """Verify every row reaches the total on a 10000-row dataset.

    Mutation: aggregating a truncated slice of the container, or a
        chunked accumulator that drops its last partial chunk.
    Oracle: hand-computed 9999*10000/2 = 49995000.
    """
    ds = DataSet([{'id': i, 'value': i} for i in range(10000)])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] == 49995000


def test_summary_row_date_column():
    """Verify a Date column is left out of the default total.

    Mutation: dropping the fill-empty-values pass in calc_summary_row,
        which leaves the un-totaled Date column out of the row.
    Oracle: hand-computed 'Total' in id, None in date, 300 in value.
    """
    ds = DataSet([
        {'id': 1, 'date': Date(2024, 1, 1), 'value': 100},
        {'id': 2, 'date': Date(2024, 1, 2), 'value': 200}])
    ds.columns = (('id', int), ('date', Date), ('value', int))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['date'] is None
    assert ds.summary['value'] == 300


def test_summary_row_datetime_column_with_cols_funcs():
    """Verify cols_funcs aggregates a DateTime column the default skips.

    Mutation: ignoring cols_funcs and falling back to the numeric
        columns, which leaves 'dt' None.
    Oracle: hand-computed latest DateTime 2024-01-02 11:45 and 300.
    """
    ds = DataSet([
        {'id': 1, 'dt': DateTime(2024, 1, 1, 10, 30, tzinfo=UTC), 'value': 100},
        {'id': 2, 'dt': DateTime(2024, 1, 2, 11, 45, tzinfo=UTC), 'value': 200}])
    ds.columns = (('id', int), ('dt', DateTime), ('value', int))

    ds.add_summary_row(cols_funcs=[('dt', max), ('value', sum)])

    assert ds.summary['id'] == 'Total'
    assert ds.summary['dt'] == DateTime(2024, 1, 2, 11, 45, tzinfo=UTC)
    assert ds.summary['value'] == 300


def test_summary_row_custom_label():
    """Verify the caller's label reaches the label column.

    Mutation: hardcoding 'Total' in place of the label argument.
    Oracle: the caller's 'Grand Total', which no default produces.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200}])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row(label='Grand Total')

    assert ds.summary['id'] == 'Grand Total'
    assert ds.summary['value'] == 300


def test_summary_row_boolean_columns():
    """Verify a bool column counts as non-numeric and is not totaled.

    Mutation: `libb.isnumeric(typ)` swapped for
        `issubclass(typ, (int, float))`, which counts bool as numeric
        and totals 'flag' to 2.
    Oracle: hand-computed None in flag against the 2 that
        True+False+True gives.
    """
    ds = DataSet([
        {'id': 1, 'flag': True, 'value': 100},
        {'id': 2, 'flag': False, 'value': 200},
        {'id': 3, 'flag': True, 'value': 300}])
    ds.columns = (('id', int), ('flag', bool), ('value', int))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['flag'] is None
    assert ds.summary['value'] == 600


def test_summary_redeclaration_takes_effect_on_next_read():
    """Verify a second declaration is honored on the next read.

    Mutation: caching the first summary row, so the relabel is ignored
        and the row keeps saying 'Total'.
    Oracle: 'Total' from the first read, then 'Updated' after the
        second declaration, with the hand-computed total 300 throughout.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200}])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row()
    assert ds.summary['id'] == 'Total'

    ds.add_summary_row(label='Updated')

    assert ds.summary['id'] == 'Updated'
    assert ds.summary['value'] == 300


def test_summary_row_mixed_int_float():
    """Verify int and float columns each keep their own total type.

    Mutation: coercing every total to the label column's int type,
        which truncates 7.5 to 7.
    Oracle: hand-computed 600 and 7.5.
    """
    ds = DataSet([
        {'id': 1, 'int_val': 100, 'float_val': 1.5},
        {'id': 2, 'int_val': 200, 'float_val': 2.5},
        {'id': 3, 'int_val': 300, 'float_val': 3.5}])
    ds.columns = (('id', int), ('int_val', int), ('float_val', float))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['int_val'] == 600
    assert ds.summary['float_val'] == 7.5


def test_summary_row_negative_values():
    """Verify negative values subtract from the total.

    Mutation: taking abs() of each value before totaling, which gives
        350.
    Oracle: hand-computed -100 + 200 - 50.
    """
    ds = DataSet([
        {'id': 1, 'value': -100},
        {'id': 2, 'value': 200},
        {'id': 3, 'value': -50}])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] == 50


# --- Lazy Summary, Caching, and Summary Column Tests ---

def test_summary_is_lazy_until_read():
    """Verify the total reflects an edit made after declaring it.

    Mutation: computing the total inside add_summary_row instead of
        storing the arguments for later.
    Oracle: hand-computed 104 = 100 + 4 after the edit, against the 6
        an eager total would have frozen.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}, {'a': 'r2', 'b': 4}])
    ds.add_summary_row()

    ds[0]['b'] = 100

    assert ds.summary['b'] == 104


def test_summary_recomputes_on_each_read():
    """Verify a read after the first re-totals the current rows.

    Mutation: caching the first result in _summary and returning it on
        later reads, so an edit made between two reads is missed.
    Oracle: hand-computed 6 on the first read, then 104 on the second
        after one row grows by 98.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}, {'a': 'r2', 'b': 4}])

    assert ds.summary['b'] == 6
    ds[0]['b'] = 100
    assert ds.summary['b'] == 104


def test_summary_without_declaration_totals_numeric_columns():
    """Verify reading .summary with nothing declared still totals.

    Mutation: returning None when _summary_args is empty rather than
        calling add_summary_row.
    Oracle: hand-computed 6 for b, with the label written into a.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}, {'a': 'r2', 'b': 4}])

    assert ds.summary['b'] == 6
    assert ds.summary['a'] == 'Total'


def test_summary_cols_funcs_replaces_sum():
    """Verify cols_funcs aggregates with the given function.

    Mutation: ignoring cols_funcs and always summing.
    Oracle: hand-computed - max of (2, 9) is 9, where the sum is 11.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}, {'a': 'r2', 'b': 9}])
    ds.add_summary_row(cols_funcs=[('b', max)])

    assert ds.summary['b'] == 9


def test_summary_row_is_not_appended_to_the_container():
    """Verify declaring a summary leaves the row count alone.

    Mutation: appending the summary row in add_summary_row, which would
        make it show up in iteration and in len().
    Oracle: hand-computed 2 rows before and after the declaration.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}, {'a': 'r2', 'b': 4}])

    ds.add_summary_row()

    assert len(ds) == 2
    assert not any(DataSet.is_summary_row(r) for r in ds)


def test_is_summary_row_defaults_false():
    """Verify a plain row is not read as a summary row.

    Mutation: defaulting the __is_summary__ lookup to True, which would
        make every ordinary row look like a total.
    Oracle: the flagged summary row is True, a plain row is False.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}])

    assert DataSet.is_summary_row(ds.summary)
    assert not DataSet.is_summary_row(ds[0])


def test_add_summary_column_skips_falsy_values():
    """Verify falsy cells are left out of the row function.

    Mutation: dropping the `if row.get(_)` guard, which feeds 0 and None
        to row_func - counting the zero and raising on the None.
    Oracle: hand-counted - row one has two truthy cells, row two has one.
    """
    ds = DataSet([{'a': 1, 'b': 2}, {'a': 0, 'b': 4}, {'a': None, 'b': 5}])

    ds.add_summary_column('cnt', row_func=lambda vals: len(list(vals)))

    assert [r['cnt'] for r in ds] == [2, 1, 1]


def test_add_summary_column_types_the_new_column_float():
    """Verify the added column is declared float.

    Mutation: adding the column as int, which would truncate a
        fractional total on the next type conversion.
    Oracle: the declared type read back from colmap.
    """
    ds = DataSet([{'a': 1, 'b': 2}])

    ds.add_summary_column('tot')

    assert ds.cols == ['a', 'b', 'tot']
    assert ds.colmap['tot'] is float


def test_add_summary_column_honors_selected_columns():
    """Verify only the named columns reach row_func.

    Mutation: ignoring the columns argument and totaling every column.
    Oracle: hand-computed 1 from column a alone, against 3 for a + b.
    """
    ds = DataSet([{'a': 1, 'b': 2}])

    ds.add_summary_column('tot', columns=['a'])

    assert ds[0]['tot'] == 1


# --- Direct Computation, Index Guard, and Predicate Tests ---

def test_calc_summary_row_defaults_label_the_first_column(basic_dataset):
    """Verify calc_summary_row's own defaults label column 0 'Total'.

    Mutation: the calc_summary_row signature defaulting label_idx to 1,
        or label to 'total' / 'TOTAL', which no caller of the summary
        property would notice.
    Oracle: hand-computed 'Total' in id, None in name, 600 in value.
    """
    summary = basic_dataset.calc_summary_row(columns=['value'])

    assert summary['id'] == 'Total'
    assert summary['name'] is None
    assert summary['value'] == 600


def test_summary_row_label_idx_one_past_last_column(basic_dataset):
    """Verify a label_idx of len(columns) writes no label and totals all.

    Mutation: `label_idx < len(self.columns)` relaxed to `<=` in
        add_summary_row or in calc_summary_row, which indexes one off
        the end and raises IndexError.
    Oracle: hand-computed 1+2+3 and 100+200+300, with 'Total' in no
        column.
    """
    basic_dataset.add_summary_row(label_idx=3)

    assert basic_dataset.summary['id'] == 6
    assert basic_dataset.summary['value'] == 600
    assert basic_dataset.summary['name'] is None
    assert 'Total' not in basic_dataset.summary.values()


def test_summary_row_never_aggregates_the_label_column():
    """Verify the label column's values are never fed to the total.

    Mutation: add_summary_row losing the label column from the
        exclusion - label_col set to None, to the column type rather
        than its name, or dropped by an off-by-one in the index guard.
    Oracle: a counting int in the label column - zero additions,
        against the two that totaling its two rows would make.
    """
    class CountingInt(int):
        """Int that counts how often it is added to something."""

        adds = 0

        def __radd__(self, other):
            CountingInt.adds += 1
            return int(self) + other

    ds = DataSet([
        {'id': CountingInt(1), 'value': 100},
        {'id': CountingInt(2), 'value': 200}])
    ds.columns = (('id', int), ('value', int))

    ds.add_summary_row()

    assert ds.summary['id'] == 'Total'
    assert ds.summary['value'] == 300
    assert CountingInt.adds == 0


def test_is_summary_row_answers_with_a_bool():
    """Verify the predicate answers False for a data row, not None.

    Mutation: the `__is_summary__` lookup defaulting to None, so a
        caller comparing against False, or writing the answer out as
        JSON, reads null where it expects false.
    Oracle: the literal False and True, against the None a missing
        default returns.
    """
    ds = DataSet([{'a': 'r1', 'b': 2}])

    assert DataSet.is_summary_row(ds[0]) is False
    assert DataSet.is_summary_row(ds.summary) is True


def test_calc_summary_row_with_defaults_totals_numeric_columns():
    """Verify calc_summary_row() with its documented defaults does not raise.

    Mutation: passing columns=None straight to bucket, which maps over
        None and raises TypeError.
    Oracle: hand-computed 3 = 1 + 2 for the numeric column b.
    """
    ds = DataSet([{'a': 'x', 'b': 1}, {'a': 'y', 'b': 2}])

    row = ds.calc_summary_row()

    assert row['b'] == 3


def test_calc_summary_row_returns_a_resolving_row_on_both_branches():
    """Verify the summary row computes a stored callable, rows or no rows.

    The declared return type is the row class, and the two branches build
    it differently: one copies a bucket result, the other constructs one.
    A copy that erased the class would leave a caller a summary row that
    hands a computed column back as a function.

    Mutation: building the populated branch through a copy that returns
        the base mapping class instead of the row class.
    Oracle: hand-computed 3 = 1 + 2 through the stored callable, on the
        row from each branch, plus the declared type of both.
    """
    populated = DataSet([{'g': 'x', 'v': 1.0}, {'g': 'x', 'v': 2.0}],
                        columns=[('g', str), ('v', float)]).calc_summary_row()
    empty = DataSet([], columns=[('g', str), ('v', float)]).calc_summary_row()

    for row in (populated, empty):
        assert isinstance(row, lazydict)
        row['x'] = 1
        row['y'] = 2
        row['computed'] = lambda r: r.x + r.y
        assert row.computed == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
