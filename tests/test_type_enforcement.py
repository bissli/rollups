"""Comprehensive tests for DataSet type enforcement improvements.

Tests for enhanced type inference and validation across entry points:
- guess_columns(): Multi-row scanning for robust type inference
- append(): Optional immediate type validation
- extend(): Optional immediate type validation
"""
import datetime
import logging

import pytest
from opendate import Date, DateTime, Time
from rollups import DataSet, force_type, infer_numeric_type
from rollups import is_dynamic_date_code, islistoftuples, smart_type
from rollups.types import _cached_date_parse, _cached_datetime_parse

# --- Fixtures ---


@pytest.fixture
def sparse_rows():
    """Rows with None values in first rows, types later."""
    return [
        {'a': None, 'b': 'text', 'c': None},
        {'a': 100, 'b': 'more', 'c': Date(2024, 1, 1)},
        {'a': 200, 'b': 'text', 'c': Date(2024, 1, 2)},
    ]


# --- TestGuessColumnsScanning - Multi-row scanning tests ---

class TestGuessColumnsScanning:
    """Tests for improved guess_columns() with multi-row scanning."""

    def test_guess_columns_scans_past_none_in_first_row(self, sparse_rows):
        """Scanning continues past a None until a typed value shows up.

        Mutation: scanning only the exemplar row instead of the window.
        Oracle: hand-computed pairs; 'a' and 'c' are None until row 1.
        """
        columns = DataSet.guess_columns(sparse_rows)

        assert columns == [('a', int), ('b', str), ('c', Date)]

    def test_guess_columns_all_none_values(self):
        """A column with no non-None value keeps the NoneType seed.

        Mutation: seeding `typ` with str instead of type(None).
        Oracle: hand-computed; 'b' holds None in every row, 'a' does not.
        """
        rows = [
            {'a': 100, 'b': None},
            {'a': 200, 'b': None},
            {'a': 300, 'b': None},
        ]
        colmap = dict(DataSet.guess_columns(rows))

        assert colmap['a'] is int
        assert colmap['b'] is type(None)

    def test_guess_columns_finds_type_in_last_scanned_row(self):
        """The default window covers row 99 but stops before row 100.

        Mutation: off-by-one in `scan_start + scan_limit`.
        Oracle: rows 99 and 100 straddle the 100-row default window.
        """
        inside = [{'val': None} for _ in range(99)]
        inside.append({'val': 42.5})
        inside.extend([{'val': None} for _ in range(100)])

        assert dict(DataSet.guess_columns(inside))['val'] is float

        outside = [{'val': None} for _ in range(100)]
        outside.append({'val': 42.5})

        assert dict(DataSet.guess_columns(outside))['val'] is type(None)

    @pytest.mark.parametrize(('scan_limit', 'expected_type'), [
        (50, type(None)),
        (51, int),
    ])
    def test_guess_columns_respects_scan_limit(self, scan_limit, expected_type):
        """A custom scan_limit sets the last row inference may read.

        Mutation: off-by-one in `min(len(rows), scan_start + scan_limit)`.
        Oracle: the int sits at index 50, straddling the two limits.
        """
        rows = [{'val': None} for _ in range(50)]
        rows.append({'val': 123})

        colmap = dict(DataSet.guess_columns(rows, scan_limit=scan_limit))

        assert colmap['val'] is expected_type

    def test_guess_columns_empty_dataset(self):
        """No rows yields no columns rather than an index error.

        Mutation: dropping the `if rows` guard before rows[exemplar].
        Oracle: an empty row list has no exemplar to read names from.
        """
        columns = DataSet.guess_columns([])

        assert columns == []

    def test_guess_columns_single_row_no_none(self):
        """Each column takes the class of its own value, in order.

        Mutation: zip(cols, typs) misaligned by a shifted type list.
        Oracle: hand-computed triple; the three types are all different.
        """
        rows = [{'a': 1, 'b': 'text', 'c': 2.5}]
        columns = DataSet.guess_columns(rows)

        assert columns == [('a', int), ('b', str), ('c', float)]

    @pytest.mark.parametrize(('value', 'col_type'), [
        (Date(2024, 1, 1), Date),
        (DateTime(2024, 1, 1, 10, 30), DateTime),
        (Time(14, 45), Time),
        (datetime.datetime(2024, 1, 1, 0, 0), Date),
        (datetime.datetime(2024, 1, 1, 10, 30), datetime.datetime),
    ])
    def test_guess_columns_preserves_datetime_types(self, value, col_type):
        """Date/DateTime/Time stay apart, and only midnight demotes.

        Mutation: dropping smart_type's midnight guard, so a 10:30
            datetime.datetime also types as Date.
        Oracle: the two plain datetimes straddle midnight.
        """
        rows = [
            {'val': None},
            {'val': value},
        ]
        colmap = dict(DataSet.guess_columns(rows))

        assert colmap['val'] is col_type

    def test_guess_columns_with_explicit_exemplar(self):
        """The exemplar row supplies both the names and the scan start.

        Mutation: reading names from rows[0] rather than rows[exemplar].
        Oracle: row 0 carries different names, 'wrong' and 'columns'.
        """
        rows = [
            {'wrong': 1, 'columns': 2},
            {'a': 100, 'b': 'text'},
            {'a': 200, 'b': 'more'},
        ]
        columns = DataSet.guess_columns(rows, exemplar=1)

        assert columns == [('a', int), ('b', str)]

    def test_guess_columns_non_numeric_type_wins_after_int(self):
        """A later str never overrides an int already settled on.

        Mutation: `if typ is int and row_typ is float` losing its second
            term, so any later type replaces int.
        Oracle: hand-computed; only float promotes an int column.
        """
        rows = [
            {'val': None},
            {'val': 100},
            {'val': 'text'},
        ]
        colmap = dict(DataSet.guess_columns(rows))

        assert colmap['val'] is int

    def test_guess_columns_with_provided_cols(self):
        """Explicit cols pick the columns; types still come from scanning.

        Mutation: `if not cols` inverted, so the caller's names are lost.
        Oracle: 'c' is present in both rows but was not asked for.
        """
        rows = [
            {'a': None, 'b': None, 'c': 'ignore'},
            {'a': 1, 'b': 2.5, 'c': 'ignored'},
        ]
        columns = DataSet.guess_columns(rows, cols=['a', 'b'])

        assert columns == [('a', int), ('b', float)]

    def test_guess_columns_with_provided_typs(self):
        """Explicit typs are used as given, with no inference.

        Mutation: dropping the `if not typs` guard so the rows are scanned.
        Oracle: the row holds strings, so inference would answer str.
        """
        rows = [
            {'a': '100', 'b': '200.5'},
        ]
        columns = DataSet.guess_columns(rows, typs=[int, float])

        assert columns == [('a', int), ('b', float)]

    def test_guess_columns_scan_limit_one(self):
        """scan_limit=1 reads the exemplar row and nothing after it.

        Mutation: off-by-one widening the window to rows[0:2].
        Oracle: the int sits in row 1, one row past the window.
        """
        rows = [
            {'val': None},
            {'val': 100},
        ]
        colmap = dict(DataSet.guess_columns(rows, scan_limit=1))

        assert colmap['val'] is type(None)

    def test_guess_columns_boolean_inference(self):
        """A bool column types as bool, not as int.

        Mutation: smart_type returning int for any int instance, which
            True is.
        Oracle: hand-computed; type(True) is bool.
        """
        rows = [
            {'flag': None},
            {'flag': True},
            {'flag': False},
        ]
        colmap = dict(DataSet.guess_columns(rows))

        assert colmap['flag'] is bool

    def test_guess_columns_promotes_int_to_float(self):
        """An int column promotes to float when a float turns up.

        Mutation: dropping the `typ = float` promotion.
        Oracle: hand-computed; 10 then 10.055 must widen to float.
        """
        rows = [
            {'val': 10},
            {'val': 10.055},
            {'val': 11.1},
        ]
        colmap = dict(DataSet.guess_columns(rows))
        assert colmap['val'] is float

    def test_guess_columns_pure_int_stays_int(self):
        """An all-int column is never widened.

        Mutation: promoting on `typ is int` alone, without checking that
            the new value is a float.
        Oracle: hand-computed; all three values are int.
        """
        rows = [
            {'val': 10},
            {'val': 20},
            {'val': 30},
        ]
        colmap = dict(DataSet.guess_columns(rows))
        assert colmap['val'] is int

    def test_guess_columns_promotes_int_to_float_sparse(self):
        """Promotion still happens when a None sits between the values.

        Mutation: ending the row scan at the first None instead of
            skipping it.
        Oracle: hand-computed; the float sits behind a None.
        """
        rows = [
            {'val': 10},
            {'val': None},
            {'val': 10.055},
        ]
        colmap = dict(DataSet.guess_columns(rows))
        assert colmap['val'] is float

    def test_guess_columns_int_to_float_beyond_scan_limit(self):
        """Promotion sees only floats inside the scan window.

        Mutation: the scan window off by one, pulling the float at index
            5 inside the limit-5 window.
        Oracle: the same rows at limits 5 and 6, straddling that float.
        """
        rows = [{'val': 10}] * 5 + [{'val': 10.055}]
        columns_limited = DataSet.guess_columns(rows, scan_limit=5)
        assert dict(columns_limited)['val'] is int

        columns_full = DataSet.guess_columns(rows, scan_limit=6)
        assert dict(columns_full)['val'] is float

    def test_guess_columns_bool_not_promoted_to_float(self):
        """A bool column is not promoted by a later float.

        Mutation: the two `typ is int` identity checks rewritten with
            issubclass, which bool satisfies.
        Oracle: hand-computed; bool stays bool even beside 1.5.
        """
        rows = [
            {'flag': True},
            {'flag': 1.5},
        ]
        colmap = dict(DataSet.guess_columns(rows))
        assert colmap['flag'] is bool

    def test_guess_columns_promotes_only_mixed_column(self):
        """Promotion is per column, not shared across the row.

        Mutation: hoisting `typ` out of the per-column loop.
        Oracle: hand-computed triple; only 'a' mixes int and float.
        """
        rows = [
            {'a': 1, 'b': 'text', 'c': 10},
            {'a': 2.5, 'b': 'more', 'c': 20},
        ]
        columns = DataSet.guess_columns(rows)
        assert columns == [('a', float), ('b', str), ('c', int)]

    def test_guess_columns_promotes_int_zero_to_float(self):
        """A zero counts as a value, so it settles the column type.

        Mutation: `if val is not None` written as `if val`, skipping 0.
        Oracle: hand-computed; a skipped 0 would let the later str win.
        """
        assert dict(DataSet.guess_columns([
            {'val': 0},
            {'val': 'text'},
        ]))['val'] is int

        assert dict(DataSet.guess_columns([
            {'val': 0},
            {'val': 0.5},
        ]))['val'] is float


# --- TestAppendValidation - append() with validation parameter ---

class TestAppendValidation:
    """Tests for append() with validation parameter."""

    def test_append_validate_false_default(self):
        """append() defers conversion until the first read.

        Mutation: append ignoring `validate` and always converting.
        Oracle: the raw strings before the read, the numbers after.
        """
        ds = DataSet([], columns=[('a', int), ('b', float)])
        ds.append({'a': '123', 'b': '45.6'})

        assert ds.container[0]['a'] == '123'
        assert ds.container[0]['b'] == '45.6'

        _ = ds[0]

        assert ds.container[0]['a'] == 123
        assert ds.container[0]['b'] == 45.6

    def test_append_validate_true_immediate_conversion(self):
        """append(validate=True) converts each value to its column type.

        Mutation: converting every column with int rather than its own
            declared type.
        Oracle: hand-computed 45.6, which int conversion truncates to 45.
        """
        ds = DataSet([], columns=[('a', int), ('b', float)])
        ds.append({'a': '123', 'b': '45.6'}, validate=True)

        assert ds.container[0]['a'] == 123
        assert ds.container[0]['b'] == 45.6
        assert isinstance(ds.container[0]['a'], int)
        assert isinstance(ds.container[0]['b'], float)

    @pytest.mark.parametrize(('col_type', 'string_value', 'expected_parts'), [
        (Date, '2024-01-15', (2024, 1, 15, 0, 0)),
        (DateTime, '2024-01-15 10:30:00', (2024, 1, 15, 10, 30)),
    ])
    def test_append_validate_converts_date_strings(self, col_type, string_value,
                                                   expected_parts):
        """A date string lands as the column's own temporal type.

        Mutation: the Date branch of _convert_value swapping day and
            month, or falling through to the DateTime parser.
        Oracle: the components read off the input string by hand.
        """
        ds = DataSet([], columns=[('val', col_type)])
        ds.append({'val': string_value}, validate=True)

        val = ds.container[0]['val']
        parts = (val.year, val.month, val.day,
                 getattr(val, 'hour', 0), getattr(val, 'minute', 0))

        assert isinstance(val, col_type)
        assert parts == expected_parts

    def test_append_validate_preserves_none(self):
        """None survives validation instead of being coerced.

        Mutation: dropping _convert_value's `val is None` short-circuit,
            so a str column stores the text 'None'.
        Oracle: None in, None out, for an int and a str column alike.
        """
        ds = DataSet([], columns=[('a', int), ('b', str)])
        ds.append({'a': None, 'b': None}, validate=True)

        assert ds.container[0]['a'] is None
        assert ds.container[0]['b'] is None

    def test_append_validate_without_columns(self):
        """With no columns declared, the row is stored untouched.

        Mutation: dropping `and self.columns`, so the row is rebuilt from
            an empty column list and loses every key.
        Oracle: both keys survive, still holding their strings.
        """
        ds = DataSet([])
        ds.append({'a': '123', 'b': '45.6'}, validate=True)

        assert set(ds.container[0]) == {'a', 'b'}
        assert ds.container[0]['a'] == '123'
        assert ds.container[0]['b'] == '45.6'

    def test_append_validate_with_time_type(self):
        """A datetime.time is converted to the Time column type.

        Mutation: dropping the Time branch of _convert_value, leaving the
            plain datetime.time in place.
        Oracle: hand-read 14:30:00 on an opendate Time.
        """
        ds = DataSet([], columns=[('time', Time)])
        ds.append({'time': datetime.time(14, 30, 0)}, validate=True)

        val = ds.container[0]['time']

        assert isinstance(val, Time)
        assert (val.hour, val.minute, val.second) == (14, 30, 0)

    def test_append_validate_false_stores_the_caller_row(self):
        """The lazy path stores the caller's dict; validation copies it.

        Mutation: the validate branch converting in place and appending
            the caller's own dict.
        Oracle: object identity, plus the caller's dict still holding the
            raw string after a validated append.
        """
        lazy_row = {'a': '1'}
        ds_lazy = DataSet([], columns=[('a', int)])
        ds_lazy.append(lazy_row, validate=False)

        assert ds_lazy.container[0] is lazy_row

        validated_row = {'a': '1'}
        ds_validated = DataSet([], columns=[('a', int)])
        ds_validated.append(validated_row, validate=True)

        assert ds_validated.container[0] is not validated_row
        assert validated_row['a'] == '1'
        assert ds_validated.container[0]['a'] == 1

    def test_append_validate_with_missing_columns(self):
        """A column the row omits is filled with None.

        Mutation: `obj.get(name)` written as `obj[name]`, raising on an
            absent column.
        Oracle: hand-computed; only 'a' is supplied.
        """
        ds = DataSet([], columns=[('a', int), ('b', str), ('c', float)])
        ds.append({'a': '123'}, validate=True)

        assert ds.container[0]['a'] == 123
        assert ds.container[0]['b'] is None
        assert ds.container[0]['c'] is None


# --- TestExtendValidation - extend() with validation parameter ---

class TestExtendValidation:
    """Tests for extend() with validation parameter."""

    def test_extend_validate_false_default(self):
        """extend() defers conversion until the first read.

        Mutation: extend ignoring `validate` and always converting.
        Oracle: the raw strings before the read, the numbers after.
        """
        ds = DataSet([], columns=[('a', int), ('b', float)])
        ds.extend([
            {'a': '100', 'b': '10.5'},
            {'a': '200', 'b': '20.5'},
        ])

        assert ds.container[0]['a'] == '100'
        assert ds.container[1]['b'] == '20.5'

        _ = ds[0]

        assert ds.container[0]['a'] == 100
        assert ds.container[1]['b'] == 20.5

    def test_extend_validate_true_immediate_conversion(self):
        """extend(validate=True) converts every row as it is added.

        Mutation: the extend loop skipping a row of the sequence.
        Oracle: hand-computed 100, 200, 300 across three rows.
        """
        ds = DataSet([], columns=[('a', int), ('b', float)])
        ds.extend([
            {'a': '100', 'b': '10.5'},
            {'a': '200', 'b': '20.5'},
            {'a': '300', 'b': '30.5'},
        ], validate=True)

        assert [row['a'] for row in ds.container] == [100, 200, 300]
        assert all(isinstance(row['a'], int) for row in ds.container)
        assert all(isinstance(row['b'], float) for row in ds.container)

    @pytest.mark.parametrize(('typ', 'strings', 'expected_parts'), [
        (Date, ['2024-01-01', '2024-01-02'],
         [(2024, 1, 1, 0, 0), (2024, 1, 2, 0, 0)]),
        (DateTime, ['2024-01-01 09:30:00', '2024-01-01 14:45:00'],
         [(2024, 1, 1, 9, 30), (2024, 1, 1, 14, 45)]),
    ])
    def test_extend_validate_converts_temporal_strings(self, typ, strings,
                                                       expected_parts):
        """Each row parses its own date string, not the first row's.

        Mutation: the parse cache keyed on the column instead of the
            string, so every row takes row 0's value.
        Oracle: components read off each string by hand; the two differ.
        """
        ds = DataSet([], columns=[('when', typ), ('value', int)])
        ds.extend([{'when': text, 'value': str(i)}
                   for i, text in enumerate(strings)], validate=True)

        values = [row['when'] for row in ds.container]
        parts = [(val.year, val.month, val.day,
                  getattr(val, 'hour', 0), getattr(val, 'minute', 0))
                 for val in values]

        assert all(isinstance(val, typ) for val in values)
        assert parts == expected_parts
        assert [row['value'] for row in ds.container] == [0, 1]

    def test_extend_validate_preserves_none_values(self):
        """None survives validation in whichever row and column it sits.

        Mutation: dropping _convert_value's `val is None` short-circuit,
            so the str column stores the text 'None'.
        Oracle: hand-computed grid of two rows, one None in each.
        """
        ds = DataSet([], columns=[('a', int), ('b', str)])
        ds.extend([
            {'a': None, 'b': 'text'},
            {'a': '100', 'b': None},
        ], validate=True)

        assert ds.container[0]['a'] is None
        assert ds.container[0]['b'] == 'text'
        assert ds.container[1]['a'] == 100
        assert ds.container[1]['b'] is None

    def test_extend_validate_without_columns_defined(self):
        """With no columns declared, rows are stored untouched.

        Mutation: dropping `and self.columns`, so each row is rebuilt from
            an empty column list and loses every key.
        Oracle: both rows keep their key and their string.
        """
        ds = DataSet([])
        ds.extend([
            {'a': '100'},
            {'a': '200'},
        ], validate=True)

        assert [set(row) for row in ds.container] == [{'a'}, {'a'}]
        assert [row['a'] for row in ds.container] == ['100', '200']

    def test_extend_validate_large_batch(self):
        """Every row of a long batch is converted, in order.

        Mutation: the loop skipping the first or last row of the sequence.
        Oracle: ids 0 to 99 as a hand-computed range.
        """
        ds = DataSet([], columns=[('id', int), ('value', float)])
        rows = [{'id': str(i), 'value': str(i * 1.5)} for i in range(100)]
        ds.extend(rows, validate=True)

        assert [row['id'] for row in ds.container] == list(range(100))
        assert [row['value'] for row in ds.container] == [i * 1.5 for i in range(100)]
        assert all(isinstance(row['value'], float) for row in ds.container)

    def test_extend_validate_from_another_dataset(self):
        """Validated rows are copies, so the source dataset is untouched.

        Mutation: the validate branch converting the row in place and
            appending the caller's own dict.
        Oracle: the source row still holds 1 after the copy is set to 99.
        """
        source = DataSet([
            {'a': 1, 'b': 2.5},
            {'a': 3, 'b': 4.5},
        ])
        source.columns = [('a', int), ('b', float)]

        target = DataSet([], columns=[('a', int), ('b', float)])
        target.extend(source.container, validate=True)

        assert len(target) == 2
        assert target.container[0] is not source.container[0]

        target.container[0]['a'] = 99

        assert source.container[0]['a'] == 1
        assert target.container[1]['b'] == 4.5

    def test_extend_validate_with_time_type(self):
        """Each datetime.time is converted to the Time column type.

        Mutation: dropping the Time branch of _convert_value, leaving the
            plain datetime.time in place.
        Oracle: hand-read 09:30 and 14:45 on opendate Times.
        """
        ds = DataSet([], columns=[('time', Time)])
        ds.extend([
            {'time': datetime.time(9, 30, 0)},
            {'time': datetime.time(14, 45, 0)},
        ], validate=True)

        values = [row['time'] for row in ds.container]

        assert all(isinstance(val, Time) for val in values)
        assert [(val.hour, val.minute) for val in values] == [(9, 30), (14, 45)]

    def test_extend_validate_with_mixed_valid_invalid(self):
        """A value that will not convert is kept, not blanked or raised.

        Mutation: `return result if result is not None else val` reduced
            to `return result`, wiping the unconvertible value to None.
        Oracle: hand-computed; 'invalid' survives between two conversions.
        """
        ds = DataSet([], columns=[('a', int)])
        ds.extend([
            {'a': '100'},
            {'a': 'invalid'},
            {'a': '300'},
        ], validate=True)

        assert [row['a'] for row in ds.container] == [100, 'invalid', 300]

    def test_extend_validate_empty_sequence(self):
        """An empty validated extend adds nothing and converts nothing.

        Mutation: extend marking `_types_converted` True because it ran
            the validate branch.
        Oracle: the row stored earlier still converts on the first read.
        """
        ds = DataSet([{'a': '1'}], columns=[('a', int)])

        ds.extend([], validate=True)

        assert len(ds) == 1
        assert ds._types_converted is False
        assert ds[0]['a'] == 1

    def test_extend_validate_with_missing_columns(self):
        """Every declared column appears in every row, None where absent.

        Mutation: building the converted row from the keys the row has
            rather than from the declared columns.
        Oracle: hand-computed grid; each row supplies one of three columns.
        """
        ds = DataSet([], columns=[('a', int), ('b', str), ('c', float)])
        ds.extend([
            {'a': '100'},
            {'b': 'text'},
            {'c': '3.14'},
        ], validate=True)

        assert ds.container[0]['a'] == 100
        assert ds.container[0]['b'] is None
        assert ds.container[0]['c'] is None
        assert ds.container[1]['a'] is None
        assert ds.container[1]['b'] == 'text'
        assert ds.container[2]['c'] == 3.14  # noqa

    def test_extend_validate_preserves_existing_data(self):
        """Validated rows are appended after the rows already held.

        Mutation: extend replacing the container instead of appending.
        Oracle: hand-computed; row 0 keeps its 1, the new row lands at 1.
        """
        ds = DataSet([{'a': 1, 'b': 2.5}], columns=[('a', int), ('b', float)])
        _ = ds[0]

        ds.extend([{'a': '100', 'b': '20.5'}], validate=True)

        assert len(ds) == 2
        assert ds.container[0]['a'] == 1
        assert ds.container[1]['a'] == 100


# --- Parameterized Tests - Common append/extend validation behavior ---

@pytest.mark.parametrize(('method_name', 'data'), [
    ('append', {'a': '123', 'b': '45.6'}),
    ('extend', [{'a': '123', 'b': '45.6'}]),
])
class TestCommonValidationBehavior:
    """Tests for behavior common to both append and extend validation."""

    def test_check_types_false_ignores_validate(self, method_name, data):
        """check_types=False blocks conversion on write and on read.

        Mutation: _ensure_types_converted dropping its `_check_types`
            term, so the first read converts anyway.
        Oracle: the raw strings, still raw after a read.
        """
        ds = DataSet([], columns=[('a', int), ('b', float)], check_types=False)
        method = getattr(ds, method_name)
        method(data, validate=True)

        assert ds.container[0]['a'] == '123'
        assert ds.container[0]['b'] == '45.6'

        _ = ds[0]

        assert ds.container[0]['a'] == '123'
        assert ds.container[0]['b'] == '45.6'


# --- TestIntegrationTypeEnforcement - Integration tests ---

class TestIntegrationTypeEnforcement:
    """Integration tests combining improvements across entry points."""

    def test_creation_with_sparse_data_uses_scanning(self):
        """The constructor infers types by scanning, not from row 0 alone.

        Mutation: the constructor passing scan_limit=1 to guess_columns.
        Oracle: hand-computed triple; the types only show up in row 2.
        """
        rows = [
            {'a': None, 'b': 'text1', 'c': None},
            {'a': None, 'b': 'text2', 'c': None},
            {'a': 100, 'b': 'text3', 'c': Date(2024, 1, 1)},
        ]
        ds = DataSet(rows)

        assert ds.columns == [('a', int), ('b', str), ('c', Date)]

    def test_append_then_extend_both_validated(self):
        """Rows keep insertion order across a validated append and extend.

        Mutation: extend inserting at the front instead of appending.
        Oracle: hand-computed 10, 20, 30 in the order written.
        """
        ds = DataSet([], columns=[('a', int), ('b', float)])

        ds.append({'a': '10', 'b': '1.5'}, validate=True)
        ds.extend([
            {'a': '20', 'b': '2.5'},
            {'a': '30', 'b': '3.5'},
        ], validate=True)

        assert [row['a'] for row in ds.container] == [10, 20, 30]
        assert [row['b'] for row in ds.container] == [1.5, 2.5, 3.5]

    def test_mixed_lazy_and_validated_appends(self):
        """The first read converts every lazily appended row, not just one.

        Mutation: _ensure_types_converted converting only the row asked
            for rather than the whole container.
        Oracle: rows 0 and 2, raw until a read of row 0 converts both.
        """
        ds = DataSet([], columns=[('a', int)])

        ds.append({'a': '100'}, validate=False)
        ds.append({'a': '200'}, validate=True)
        ds.append({'a': '300'}, validate=False)

        assert [row['a'] for row in ds.container] == ['100', 200, '300']

        _ = ds[0]

        assert [row['a'] for row in ds.container] == [100, 200, 300]

    def test_validated_append_after_lazy_extend(self):
        """A validated append converts its own row only.

        Mutation: append(validate=True) converting the whole container.
        Oracle: the two lazily added strings, still strings.
        """
        ds = DataSet([], columns=[('a', int)])

        ds.extend([{'a': '100'}, {'a': '200'}], validate=False)
        ds.append({'a': '300'}, validate=True)

        assert [row['a'] for row in ds.container] == ['100', '200', 300]

    def test_dataset_creation_scans_default_row_limit(self):
        """Creation inherits guess_columns' 100-row default window.

        Mutation: the constructor passing a scan_limit of its own.
        Oracle: a typed value at index 99 and at index 100, straddling
            the default window.
        """
        inside = [{'val': None} for _ in range(99)]
        inside.append({'val': 42})

        assert DataSet(inside).colmap['val'] is int

        outside = [{'val': None} for _ in range(100)]
        outside.append({'val': 42})

        assert DataSet(outside).colmap['val'] is type(None)

    def test_type_enforcement_after_bucket_operation(self):
        """bucket() aggregates the converted values, not the raw strings.

        Mutation: bucket losing its @ensure_types_converted decorator, so
            sum() adds the raw strings and yields None.
        Oracle: hand-computed sums, 300 for A and 400 for B.
        """
        ds = DataSet([
            {'category': 'A', 'value': '100'},
            {'category': 'A', 'value': '200'},
            {'category': 'B', 'value': '400'},
        ], columns=[('category', str), ('value', int)])

        result = ds.bucket(['category'], ['value'])

        assert result.colmap['value'] is int
        assert {row['category']: row['value'] for row in result} == {'A': 300, 'B': 400}
        assert all(isinstance(row['value'], int) for row in result)

    def test_validated_operations_before_bucket(self):
        """Rows added with validate=True aggregate as numbers.

        Mutation: the extend loop skipping a row, so group A loses its
            200.
        Oracle: hand-computed sums, 300 for A and 400 for B.
        """
        ds = DataSet([], columns=[('category', str), ('value', int)])

        ds.append({'category': 'A', 'value': '100'}, validate=True)
        ds.extend([
            {'category': 'A', 'value': '200'},
            {'category': 'B', 'value': '400'},
        ], validate=True)

        result = ds.bucket(['category'], ['value'])

        assert result.colmap['value'] is int
        assert {row['category']: row['value'] for row in result} == {'A': 300, 'B': 400}

    def test_type_consistency_across_operations(self):
        """Inferred types hold through validated writes and a bucket.

        Mutation: _convert_value numifying every column with int, so the
            appended 20.5 lands as 20.
        Oracle: hand-computed rows, ids 1 to 3 with their own values plus
            the all-None group.
        """
        ds = DataSet([
            {'id': None, 'val': None},
            {'id': None, 'val': None},
            {'id': 1, 'val': 10.5},
        ])

        assert ds.columns == [('id', int), ('val', float)]

        ds.append({'id': '2', 'val': '20.5'}, validate=True)
        ds.extend([{'id': '3', 'val': '30.5'}], validate=True)

        assert [row['id'] for row in ds.container] == [None, None, 1, 2, 3]

        result = ds.bucket(['id'], ['val'])

        assert result.colmap['val'] is float
        assert [(row['id'], row['val']) for row in result] == [
            (1, 10.5),
            (2, 20.5),
            (3, 30.5),
            (None, None),
        ]


# --- TestGuessColumnsEdgeCases - Edge cases for scanning ---

class TestGuessColumnsEdgeCases:
    """Edge case tests for guess_columns() scanning."""

    def test_guess_columns_float_vs_int_preference(self):
        """A value's own class decides int against float.

        Mutation: smart_type normalizing whole floats to int, or every
            number to float.
        Oracle: two rows of equal numeric value, differing only in class.
        """
        assert dict(DataSet.guess_columns([
            {'val': None},
            {'val': 100.0},
        ]))['val'] is float

        assert dict(DataSet.guess_columns([
            {'val': None},
            {'val': 100},
        ]))['val'] is int

    def test_guess_columns_scan_window_starts_at_exemplar(self):
        """The scan window is scan_limit rows counted from the exemplar.

        Mutation: `min(len(rows), scan_start + scan_limit)` losing the
            scan_start term.
        Oracle: with exemplar=2 and scan_limit=2 the window is rows 2-3;
            counted from row 0 it would be empty.
        """
        rows = [
            {'val': 'a'},
            {'val': 'b'},
            {'val': None},
            {'val': 7},
        ]
        colmap = dict(DataSet.guess_columns(rows, exemplar=2, scan_limit=2))

        assert colmap['val'] is int

    def test_guess_columns_scan_limit_zero(self):
        """scan_limit=0 reads no rows at all.

        Mutation: `rows[scan_start:scan_end]` widened by one row.
        Oracle: an empty window leaves the NoneType seed, though row 0
            holds an int.
        """
        rows = [
            {'val': 100},
            {'val': 200},
        ]
        colmap = dict(DataSet.guess_columns(rows, scan_limit=0))

        assert colmap['val'] is type(None)

    def test_guess_columns_mixed_columns_scanning(self):
        """Each column scans on until it finds its own first value.

        Mutation: the scan window cut back to the exemplar row alone.
        Oracle: hand-computed triple; the three first values sit in rows
            0, 2 and 3.
        """
        rows = [
            {'a': None, 'b': 1, 'c': None},
            {'a': None, 'b': 2, 'c': None},
            {'a': 'text', 'b': 3, 'c': None},
            {'a': 'more', 'b': 4, 'c': Date(2024, 1, 1)},
        ]
        columns = DataSet.guess_columns(rows)

        assert columns == [('a', str), ('b', int), ('c', Date)]

    def test_guess_columns_default_exemplar_is_first_row(self):
        """The default exemplar is row 0.

        Mutation: the `exemplar: int = 0` default changed to another row.
        Oracle: the two rows carry different names, so each exemplar
            gives a different answer.
        """
        rows = [
            {'a': 1},
            {'b': 'text'},
        ]

        assert DataSet.guess_columns(rows) == [('a', int)]
        assert DataSet.guess_columns(rows, exemplar=1) == [('b', str)]


# --- TestBulkLoadValidation - lazy against validated bulk loads ---

class TestBulkLoadValidation:
    """Tests for the lazy and validated paths on bulk loads."""

    def test_bulk_load_lazy_then_convert(self):
        """A lazy bulk load converts on the first read, once.

        Mutation: extend marking `_types_converted` True on the lazy path.
        Oracle: the flag before and after the first read, and the ints
            that read produces.
        """
        rows = [{'id': str(i), 'value': str(i * 1.5)} for i in range(1000)]

        ds = DataSet([], columns=[('id', int), ('value', float)])
        ds.extend(rows, validate=False)

        assert len(ds) == 1000
        assert ds._types_converted is False

        _ = ds[0]

        assert ds._types_converted is True
        assert all(isinstance(row['id'], int) for row in ds.container)

    def test_validated_append_for_critical_data(self):
        """A validated append round-trips numbers without losing digits.

        Mutation: _convert_value numifying with int for every numeric
            column, truncating the amounts.
        Oracle: the ids and amounts recomputed independently as numbers.
        """
        ds = DataSet([], columns=[('transaction_id', int), ('amount', float)])

        for i in range(10):
            ds.append({
                'transaction_id': str(1000 + i),
                'amount': str(100.50 * (i + 1))
            }, validate=True)

        ids = [row['transaction_id'] for row in ds.container]

        assert ids == list(range(1000, 1010))
        assert [row['amount'] for row in ds.container] == [
            100.50 * (i + 1) for i in range(10)]


# --- TestAppendExtendCombinations - Combined operations tests ---

class TestAppendExtendCombinations:
    """Tests combining append and extend operations."""

    def test_alternating_validated_lazy_operations(self):
        """Each call's validate flag governs only that call's rows.

        Mutation: validate remembered on the instance, so one call's
            setting leaks into the next.
        Oracle: hand-computed alternating list of ints and strings.
        """
        ds = DataSet([], columns=[('a', int)])

        ds.append({'a': '1'}, validate=True)
        ds.append({'a': '2'}, validate=False)
        ds.extend([{'a': '3'}, {'a': '4'}], validate=True)
        ds.extend([{'a': '5'}, {'a': '6'}], validate=False)
        ds.append({'a': '7'}, validate=True)

        assert [row['a'] for row in ds.container] == [1, '2', 3, 4, '5', '6', 7]


# --- Additional Edge Case Tests ---

def test_guess_columns_negative_exemplar():
    """A negative exemplar counts back from the last row.

    Mutation: the exemplar clamped to 0 for the names or the scan start.
    Oracle: hand-computed; row 0 would answer str, rows 1 onward int.
    """
    rows = [
        {'a': 'text'},
        {'a': 1},
        {'a': 2},
    ]

    assert DataSet.guess_columns(rows, exemplar=-2) == [('a', int)]


def test_guess_columns_inconsistent_row_keys():
    """Column names come from the exemplar row alone.

    Mutation: names collected as the union of every row's keys, or
        `row.get(col)` written as `row[col]` and raising on a short row.
    Oracle: hand-computed pair; 'c' and 'd' appear only in later rows.
    """
    rows = [
        {'a': None, 'b': 'x'},
        {'a': 3, 'c': 1},
        {'b': 'y', 'd': 2},
    ]
    columns = DataSet.guess_columns(rows)

    assert columns == [('a', int), ('b', str)]


def test_validate_with_nested_dict_values():
    """A dict column keeps its contents through validation.

    Mutation: _convert_value's fallback losing its argument, `typ()` for
        `typ(val)`, so the row stores an empty dict.
    Oracle: the nested literal written out by hand.
    """
    ds = DataSet([], columns=[('data', dict)])
    ds.append({'data': {'nested': {'value': 123}}}, validate=True)

    assert ds.container[0]['data'] == {'nested': {'value': 123}}


def test_validate_with_list_values():
    """A list column keeps its items, and a value it cannot take survives.

    Mutation: dropping _convert_value's suppress, so list(42) raises
        instead of leaving the value alone.
    Oracle: the hand-written list, and the 42 that list() cannot take.
    """
    ds = DataSet([], columns=[('items', list)])
    ds.append({'items': [1, 2, 3]}, validate=True)
    ds.append({'items': 42}, validate=True)

    assert ds.container[0]['items'] == [1, 2, 3]
    assert ds.container[1]['items'] == 42


def test_guess_columns_with_empty_string_values():
    """An empty string is a value, and settles the column as str.

    Mutation: inference skipping '' as if it were missing.
    Oracle: hand-computed; a skipped '' would let the later int win.
    """
    rows = [
        {'val': ''},
        {'val': 42},
    ]
    colmap = dict(DataSet.guess_columns(rows))

    assert colmap['val'] is str


def test_infer_numeric_strings_edge_cases():
    """infer_numeric_strings types numeric text and leaves codes as str.

    Mutation: dropping infer_numeric_type's leading-zero guard, so '007'
        types as int, or its percent branch answering int.
    Oracle: hand-classified row of strings, each a different case.
    """
    rows = [{
        'plain': '42',
        'decimal': '42.5',
        'padded': '  42  ',
        'code': '007',
        'pct': '15%',
        'ident': 'a_1',
    }]

    assert DataSet.guess_columns(rows, infer_numeric_strings=True) == [
        ('plain', int),
        ('decimal', float),
        ('padded', int),
        ('code', str),
        ('pct', float),
        ('ident', str),
    ]
    assert [typ for _, typ in DataSet.guess_columns(rows)] == [str] * 6


def test_validate_coerces_value_to_str_column():
    """A str column stringifies whatever it is given.

    Mutation: _convert_value returning the value untouched once no
        temporal branch matched.
    Oracle: hand-computed '123', a str where an int went in.
    """
    ds = DataSet([], columns=[('text', str)])
    ds.append({'text': 123}, validate=True)

    assert ds.container[0]['text'] == '123'
    assert isinstance(ds.container[0]['text'], str)


def test_guess_columns_all_rows_empty_dicts():
    """An empty exemplar row yields no columns, whatever later rows hold.

    Mutation: names collected as the union of every row's keys.
    Oracle: row 1 carries 'a', which must not reach the result.
    """
    rows = [
        {},
        {'a': 1},
        {},
    ]
    columns = DataSet.guess_columns(rows)

    assert columns == []


# --- force_type and islistoftuples coercion helpers ---

def test_force_type_number():
    """Verify a numeric string converts to float, not str.

    Mutation: returning somestr unchanged instead of float(neg_val).
    Oracle: hand-computed 1.5, checked for float type not '1.5'.
    """
    got = force_type('1.5')
    assert got == 1.5
    assert isinstance(got, float)


def test_force_type_parenthesized_is_negative():
    """Verify '(1.5)' reads as -1.5, the accounting negative.

    Mutation: dropping the paren_re substitution, so float('(1.5)')
        raises and the value falls through unchanged.
    Oracle: hand-computed -1.5; the sign is what separates the two.
    """
    assert force_type('(1.5)') == -1.5
    assert force_type('(2)') == -2.0


def test_force_type_date_then_passthrough():
    """Verify a date string parses and an unparsable one is returned.

    Mutation: swapping the two except branches, so an unparsable
        string raises instead of coming back unchanged.
    Oracle: hand-computed DateTime(2020, 1, 1) and the literal 'abc'.
    """
    parsed = force_type('01Jan20')
    assert isinstance(parsed, DateTime)
    assert (parsed.year, parsed.month, parsed.day) == (2020, 1, 1)
    assert force_type('abc') == 'abc'


def test_force_type_honors_date_fmt():
    """Verify date_fmt drives the parse, so a mismatched format falls
    through to the string.

    Mutation: hardcoding '%d%b%y' and ignoring the date_fmt argument.
    Oracle: '2020-01-01' parses only under the passed format; under the
        default it must come back as the untouched string.
    """
    assert force_type('2020-01-01', date_fmt='%Y-%m-%d').year == 2020
    assert force_type('2020-01-01') == '2020-01-01'


def test_islistoftuples_requires_pairs():
    """Verify only sequences of two-item sequences pass.

    Mutation: len(y) == 2 relaxed to len(y) >= 2, which admits triples.
    Oracle: hand-classified - a pair passes, a triple does not.
    """
    assert islistoftuples([('a', int)])
    assert islistoftuples([['a', int]])
    assert not islistoftuples([('a', int, 3)])
    assert not islistoftuples([('a',)])


def test_islistoftuples_rejects_non_sequence():
    """Verify a non-sequence is rejected rather than raising.

    Mutation: dropping the libb.issequence guard, so None raises
        TypeError instead of returning False.
    Oracle: hand-classified False for None.
    """
    assert not islistoftuples(None)


# --- Type conversion and inference helpers ---

def test_is_dynamic_date_code_grammar():
    """Verify only the uppercase codes, with optional offset and 'b'.

    Mutation: the isinstance guard inverted, the regex lowercased, or
        the 'b' business-day suffix uppercased.
    Oracle: hand-classified codes; each accepted form has a rejected
        twin that differs by one character.
    """
    for code in ['N', 'T', 'Y', 'P', 'M', 'T-3', 'P+2', 'P+2b', 'N-10b']:
        assert is_dynamic_date_code(code), code

    for other in ['t', 'p+2', 'P+2B', 'X', 'TT', '', 'T-', 'T-3d',
                  '2024-01-01', 'March']:
        assert not is_dynamic_date_code(other), other

    assert not is_dynamic_date_code(None)
    assert not is_dynamic_date_code(datetime.date(2024, 3, 5))
    assert not is_dynamic_date_code(20240305)


def test_infer_numeric_type_zero_padded_code_with_separator():
    """Verify a zero-padded code stays str even carrying a separator.

    Mutation: the leading-zero guard reading check_str[2] instead of
        check_str[1], so '01,234' reads as the number 1234.
    Oracle: hand-classified - '01,234' is padded like '007', while
        '1,234' has no leading zero and numifies to 1234.
    """
    assert infer_numeric_type('01,234') is None
    assert infer_numeric_type('007') is None
    assert infer_numeric_type('1,234') is int


def test_smart_type_warns_when_midnight_datetime_becomes_date(caplog):
    """Verify the midnight-datetime downgrade to Date is announced.

    Mutation: the warning text blanked or rewritten, leaving no notice
        that a DateTime column was silently typed as Date.
    Oracle: the documented sentence; a non-midnight datetime types as
        datetime.datetime and logs nothing.
    """
    with caplog.at_level(logging.WARNING, logger='rollups.types'):
        midnight_type = smart_type(datetime.datetime(2024, 3, 5, 0, 0))

    assert midnight_type is Date
    assert [r.getMessage() for r in caplog.records] == [
        ('Converting midnight datetime.datetime to Date type'
         ' - incorrectly typing DateTime columns')]

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='rollups.types'):
        morning_type = smart_type(datetime.datetime(2024, 3, 5, 9, 30))

    assert morning_type is datetime.datetime
    assert caplog.records == []


def test_convert_value_instances_temporal_values():
    """Verify a date, a datetime and a time string reach their column type.

    Mutation: Date.instance / DateTime.instance / Time.parse called on
        None, which opendate answers with None rather than raising.
    Oracle: hand-computed 2024-03-05 midnight, 2024-03-05 and 14:30.
    """
    row_in = {
        'dt': datetime.date(2024, 3, 5),
        'd': datetime.datetime(2024, 3, 5, 13, 45),
        't': '14:30',
        }
    ds = DataSet([], columns=[('dt', DateTime), ('d', Date), ('t', Time)])
    ds.append(row_in, validate=True)
    row = ds.container[0]

    assert isinstance(row['dt'], DateTime)
    assert (row['dt'].year, row['dt'].month, row['dt'].day) == (2024, 3, 5)
    assert (row['dt'].hour, row['dt'].minute) == (0, 0)

    assert isinstance(row['d'], Date)
    assert row['d'] == datetime.date(2024, 3, 5)

    assert isinstance(row['t'], Time)
    assert (row['t'].hour, row['t'].minute) == (14, 30)


def test_convert_value_parses_dynamic_date_code():
    """Verify 'T' resolves to today for a Date and a DateTime column.

    Mutation: Date.parse / DateTime.parse called on None, which opendate
        answers with None, so the code silently empties the cell.
    Oracle: datetime.date.today() from the standard library, and
        midnight for the DateTime column.
    """
    today = datetime.date.today()
    ds = DataSet([], columns=[('d', Date), ('dt', DateTime)])
    ds.append({'d': 'T', 'dt': 'T'}, validate=True)
    row = ds.container[0]
    stamp = row['dt']

    assert isinstance(row['d'], Date)
    assert row['d'] == today
    assert (stamp.year, stamp.month, stamp.day) == \
        (today.year, today.month, today.day)
    assert (stamp.hour, stamp.minute, stamp.second) == (0, 0, 0)


def test_dynamic_date_code_never_enters_the_parse_cache():
    """Verify a dynamic code bypasses the lru cache a fixed date uses.

    Mutation: the is_dynamic_date_code test fed a constant, so 'T' is
        cached and keeps answering today long after today has passed.
    Oracle: lru_cache.cache_info() as a spy - the fixed date adds one
        lookup to each cache, the dynamic code must add none.
    """
    ds = DataSet([], columns=[('d', Date), ('dt', DateTime)])
    date_start = _cached_date_parse.cache_info()
    dt_start = _cached_datetime_parse.cache_info()

    ds.append({'d': '1993-07-19', 'dt': '1993-07-19'}, validate=True)
    date_fixed = _cached_date_parse.cache_info()
    dt_fixed = _cached_datetime_parse.cache_info()
    date_lookups = date_fixed.hits + date_fixed.misses
    dt_lookups = dt_fixed.hits + dt_fixed.misses

    assert date_lookups == date_start.hits + date_start.misses + 1
    assert dt_lookups == dt_start.hits + dt_start.misses + 1

    ds.append({'d': 'T', 'dt': 'T'}, validate=True)
    date_code = _cached_date_parse.cache_info()
    dt_code = _cached_datetime_parse.cache_info()

    assert date_code.hits + date_code.misses == date_lookups
    assert dt_code.hits + dt_code.misses == dt_lookups


def test_convert_container_types_converts_every_column_of_a_row():
    """Verify a temporal column does not end conversion of its row.

    Mutation: `continue` turned into `break` in the Date, DateTime or
        Time branch, abandoning every later column of that row.
    Oracle: hand-computed row - each of the four columns holds a value
        of the wrong class going in, and the right one coming out.
    """
    row_in = {
        'd': datetime.date(2024, 3, 5),
        'dt': datetime.datetime(2024, 3, 5, 13, 45),
        't': datetime.time(14, 30),
        'n': '42',
        }
    columns = [('d', Date), ('dt', DateTime), ('t', Time), ('n', int)]
    ds = DataSet([row_in], columns=columns)

    ds.convert_container_types()
    row = ds.container[0]

    assert isinstance(row['d'], Date)
    assert isinstance(row['dt'], DateTime)
    assert isinstance(row['t'], Time)
    assert row['n'] == 42
    assert isinstance(row['n'], int)


def test_append_after_first_read_still_converts_the_new_row():
    """Verify a row appended after the first read is type-converted too.

    Mutation: _ensure_types_converted short-circuiting on the flag, so a
        row added after the first read keeps its raw string.
    Oracle: hand-computed int 2 for the appended '2', against the str it
        keeps when conversion is skipped.
    """
    ds = DataSet([{'id': '1'}], columns=[('id', int)])
    assert ds[0]['id'] == 1

    ds.append({'id': '2'})

    assert ds[1]['id'] == 2
    assert isinstance(ds[1]['id'], int)


def test_extend_after_first_read_still_converts_the_new_rows():
    """Verify rows extended after the first read are type-converted too.

    Mutation: leaving _types_converted set on extend, so the added rows
        keep their raw strings.
    Oracle: hand-computed int 3 for the extended '3'.
    """
    ds = DataSet([{'id': '1'}], columns=[('id', int)])
    assert ds[0]['id'] == 1

    ds.extend([{'id': '3'}])

    assert ds[1]['id'] == 3
    assert isinstance(ds[1]['id'], int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
