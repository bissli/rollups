"""Comprehensive tests for numeric string type inference.

Tests for infer_numeric_type() function that detects whether strings
represent integers or floats with high confidence, enabling automatic
type inference during DataSet creation.
"""
import datetime

import pytest
from opendate import Date, DateTime
from rollups import DataSet, infer_numeric_type, smart_type

# --- Basic Numeric String Detection Tests (Parameterized) ---


class TestInferNumericTypeBasics:
    """Tests for basic numeric string detection."""

    @pytest.mark.parametrize(('value', 'expected'), [
        ('123', int),
        ('0', int),
        ('999', int),
        ('-123', int),
        ('-1', int),
        ('+123', int),
    ])
    def test_integer_strings(self, value, expected):
        """Verify a plain integer string infers as int, never float.

        Mutation: probing libb.numify(val, float) before the int probe,
          which types every one of these float.
        Oracle: hand-typed int per string; a sign is the only decoration
          and none of them carries a decimal point.
        """
        assert infer_numeric_type(value) == expected

    @pytest.mark.parametrize(('value', 'expected'), [
        ('123.45', float),
        ('0.5', float),
        ('3.14159', float),
        ('-45.67', float),
        ('-0.5', float),
        ('+45.67', float),
        ('0.123', float),
        ('00.5', float),
        ('123.0', float),
        ('1.50', float),
    ])
    def test_float_strings(self, value, expected):
        """Verify a decimal string infers as float, never int.

        Mutation: dropping the "'.' not in check_str" clause from the
          leading-zero guard, which rejects '00.5' as an identifier.
        Oracle: hand-typed float per string; '123.0' and '1.50' hold
          whole values and still read float.
        """
        assert infer_numeric_type(value) == expected

    @pytest.mark.parametrize('value', ['0123', '007', '-007', '+0123'])
    def test_leading_zero_integer_rejected(self, value):
        """Verify digits behind a leading zero read as a code.

        Mutation: relaxing len(check_str) >= 3 to > 3, or dropping the
          lstrip('+-') so the signed '-007' skips the guard.
        Oracle: libb.numify('007', int) returns 7, so None here is the
          guard's own work.
        """
        assert infer_numeric_type(value) is None


# --- Formatted Numeric String Tests (Parameterized) ---

class TestInferNumericTypeFormatted:
    """Tests for formatted numeric strings."""

    @pytest.mark.parametrize(('value', 'expected'), [
        ('(100)', int),
        ('(45.67)', float),
        ('(1,234)', int),
        ('( 100 )', int),
        ('(  45.67  )', float),
    ])
    def test_parentheses_negative(self, value, expected):
        """Verify parentheses notation keeps the int/float distinction.

        Mutation: pre-validating with a plain-decimal regex before the
          numify probes, which rejects every parenthesized form.
        Oracle: hand-typed int/float, matching the same digits written
          without the parentheses.
        """
        assert infer_numeric_type(value) == expected

    @pytest.mark.parametrize(('value', 'expected'), [
        ('5.5%', float),
        ('100%', float),
        ('0.5%', float),
        ('  5.5%  ', float),
        ('100 %', float),
        ('(5.5%)', float),
    ])
    def test_percentage_values(self, value, expected):
        """Verify a percentage always infers as float, never int.

        Mutation: dropping the '%' early return so '100%' reaches the
          int probe first.
        Oracle: libb.numify('100%', int) returns 100, which is exactly
          what the early return keeps out.
        """
        assert infer_numeric_type(value) == expected


# --- Scientific Notation Tests (Parameterized) ---

class TestInferNumericTypeScientific:
    """Tests for scientific notation."""

    @pytest.mark.parametrize('value', [
        '1e6', '1.5e3', '2.5e-10',  # lowercase e
        '1E6', '1.5E3', '2.5E-10',  # uppercase E
        '1e-6', '5.5e-3',          # negative exponent
        '1e2', '1e10',             # integer result (still float)
        '1.5e6', '2.5e-3',         # decimal mantissa
        '1e+6', '2.5e+10',         # explicit positive exponent
        '007e2',                   # exponent behind a leading zero
        '(1e6)', '1e-2%',          # combined with other formatting
    ])
    def test_scientific_notation(self, value):
        """Verify scientific notation always returns float.

        Mutation: dropping the "'e' not in check_str.lower()" clause of
          the leading-zero guard, which rejects '007e2'.
        Oracle: hand-typed float for every mantissa and exponent shape,
          the integer-valued '1e2' included.
        """
        assert infer_numeric_type(value) == float


# --- Whitespace Handling Tests (Parameterized) ---

class TestInferNumericTypeWhitespace:
    """Tests for whitespace handling."""

    @pytest.mark.parametrize(('value', 'expected'), [
        ('  123', int),
        ('\t45.67', float),
        ('   100   ', int),
        ('123  ', int),
        ('45.67\t', float),
        ('  \t 123 \n ', int),
        ('\t123\t', int),
        ('\t\t45.67\t\t', float),
        ('\n123\n', int),
        ('45.67\n', float),
    ])
    def test_whitespace_handling(self, value, expected):
        """Verify padding around a number does not change its type.

        Mutation: pre-validating with a regex anchored at the string
          ends, which rejects every padded number.
        Oracle: the same digits without padding infer identically.
        """
        assert infer_numeric_type(value) == expected

    def test_leading_zero_guard_sees_stripped_value(self):
        """Verify the leading-zero guard runs on the stripped string.

        Mutation: dropping the val.strip() before the guard, so the
          padded '  007  ' starts with a space and slips past it.
        Oracle: '007' unpadded is already rejected, while
          libb.numify('  007  ', int) returns 7.
        """
        assert infer_numeric_type('007') is None
        assert infer_numeric_type('  007  ') is None

    @pytest.mark.parametrize('value', ['1 23', '45 .67'])
    def test_internal_whitespace_rejected(self, value):
        """Verify whitespace inside a number is not stripped.

        Mutation: val.replace(' ', '') before the probes, which turns
          '1 23' into int 123.
        Oracle: '  123' carries the same characters at its ends and is
          int, so the pair fixes where stripping stops.
        """
        assert infer_numeric_type(value) is None


# --- Non-Numeric String Rejection Tests (Parameterized) ---

class TestInferNumericTypeNonNumeric:
    """Tests for rejecting non-numeric strings."""

    @pytest.mark.parametrize('value', [
        'hello', 'test', 'abc123',  # pure text
        '123abc', '1,200m', '$100',  # mixed alphanumeric
        '100#', '@123', '45&67',     # special characters
        '', '   ',                    # empty
        '2024-01-15', '2024/01/15', '01-15-2024',  # dates
        '10:30:00', '14:45',         # times
    ])
    def test_non_numeric_rejected(self, value):
        """Verify text, symbols, dates and times are not numbers.

        Mutation: stripping non-numeric characters (a currency sign, a
          unit suffix) before the probes, which types '$100' as int.
        Oracle: '100' on its own is int, so each of these differs from
          a number only by characters the guard must not discard.
        """
        assert infer_numeric_type(value) is None

    @pytest.mark.parametrize('value', [123, 45.67, None, []])
    def test_non_string_input(self, value):
        """Verify a non-string value is never inferred.

        Mutation: dropping the isinstance(val, str) guard, which lets
          libb.numify answer for an int or float input.
        Oracle: libb.numify(123, int) returns 123, so the guard is the
          only thing returning None here.
        """
        assert infer_numeric_type(value) is None


# --- Edge Cases Tests (Parameterized) ---

class TestInferNumericTypeEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.parametrize('value', [
        '.',     # only decimal point
        '-',     # only minus sign
        '1.2.3',  # multiple decimal points
        '--123',  # multiple minus signs
        '()',    # empty parentheses
        '(  )',  # whitespace-only parentheses
        '%',     # percentage without number
    ])
    def test_invalid_patterns_rejected(self, value):
        """Verify a malformed number pattern returns None.

        Mutation: returning float unconditionally from the '%' branch
          instead of testing the numify result, which types '%' float.
        Oracle: libb.numify returns None for every one of these.
        """
        assert infer_numeric_type(value) is None

    @pytest.mark.parametrize(('value', 'expected'), [
        ('.5', float),
        ('.123', float),
        ('123.', float),
    ])
    def test_partial_decimal_notation(self, value, expected):
        """Verify a point with digits on one side only is float.

        Mutation: requiring a digit on both sides of the point before
          the float probe, which rejects '.5' and '123.'.
        Oracle: float('.5') and float('123.') parse where int() raises,
          so float is the only reading.
        """
        assert infer_numeric_type(value) == expected

    @pytest.mark.parametrize(('value', 'expected'), [
        ('0', int),
        ('0.0', float),
        ('00', int),
        ('0.00', float),
    ])
    def test_zero_variations(self, value, expected):
        """Verify every spelling of zero infers, none of them dropped.

        Mutation: `if int_result:` in place of `if int_result is not
          None:`, which sends falsy 0 down the float probe and out as
          None.
        Oracle: hand-typed int/float; '00' sits at the leading-zero
          guard's three-character boundary and stays int.
        """
        assert infer_numeric_type(value) == expected

    @pytest.mark.parametrize(('value', 'expected'), [
        ('999999999999999', int),
        ('999999999999999.99', float),
        ('0.00000001', float),
        ('3.141592653589793', float),
        ('0.00000000001', float),
    ])
    def test_large_and_small_numbers(self, value, expected):
        """Verify magnitude alone does not change the inferred type.

        Mutation: clamping the int probe to values that fit an int32
          and falling back to float, which types '999999999999999' as
          float.
        Oracle: hand-typed int/float; Python's int is arbitrary
          precision, so the digit count carries no signal.
        """
        assert infer_numeric_type(value) == expected


# --- Int vs Float Distinction Tests ---

class TestInferNumericTypeIntVsFloat:
    """Tests for distinguishing int from float."""

    @pytest.mark.parametrize(('value', 'expected'), [
        ('100.0', float),
        ('100', int),
        ('1e2', float),
        ('1,000', int),
        ('1,000.0', float),
    ])
    def test_int_vs_float_distinction(self, value, expected):
        """Verify the int/float line holds across formatting.

        Mutation: probing float before int, which collapses '100' and
          '1,000' onto float.
        Oracle: hand-typed pairs differing only by a trailing '.0'.
        """
        assert infer_numeric_type(value) == expected


# --- Invalid Number Tests (Parameterized) ---

class TestInferNumericTypeInvalidNumbers:
    """Tests for strings that look numeric but aren't valid."""

    @pytest.mark.parametrize('value', [
        '+-123', '-+123',   # multiple signs
        '12-34', '12+34',   # sign in middle
        '123e', 'e123',     # e without valid exponent
    ])
    def test_invalid_numbers_rejected(self, value):
        """Verify a sign in the wrong place is not a number.

        Mutation: probing libb.numify(check_str, int) rather than val,
          which hands the probe a string already stripped of '+-' and
          types '+-123' as int.
        Oracle: libb.numify('+-123', int) is None while '123' is 123.
        """
        assert infer_numeric_type(value) is None


# --- Comma Handling Tests ---

class TestInferNumericTypeCommas:
    """Tests for comma handling in numeric strings."""

    @pytest.mark.parametrize(('value', 'expected'), [
        ('1,000', int),
        ('1,200', int),
        ('1,000,000', int),
        ('1,200,000', int),
        ('+1,234', int),
        ('1,234,567.89', float),
        ('1,234.56', float),
        ('12,34', int),  # unusual positions still processes
        ('1,2,3,4', int),
    ])
    def test_comma_handling(self, value, expected):
        """Verify commas are dropped wherever they sit.

        Mutation: validating comma placement with a groups-of-three
          regex before the probes, which rejects '12,34' and '1,2,3,4'.
        Oracle: hand-typed int/float; libb.numify drops the commas, so
          '12,34' reads 1234.
        """
        assert infer_numeric_type(value) == expected


# --- Boundary Condition Tests ---

class TestInferNumericTypeBoundaryConditions:
    """Tests for boundary conditions and limits."""

    @pytest.mark.parametrize(('value', 'expected'), [
        ('9' * 100, int),
        ('9' * 50 + '.' + '9' * 50, float),
        ('0.' + '1' * 200, float),
        ('1.7976931348623157e+308', float),
        ('2.2250738585072014e-308', float),
    ])
    def test_no_length_or_magnitude_cap(self, value, expected):
        """Verify neither length nor magnitude blocks inference.

        Mutation: a length cap (len(stripped) > 32 returns None) ahead
          of the probes, which drops all five.
        Oracle: hand-typed int/float; Python parses each one, and the
          two exponents sit just inside the float range.
        """
        assert infer_numeric_type(value) == expected


# --- Underscore Rejection Tests (Parameterized) ---

class TestInferNumericTypeUnderscoreRejection:
    """Tests for rejecting strings with underscores as non-numeric.

    An underscore marks an identifier or code, not a number.
    """

    @pytest.mark.parametrize('value', [
        '123_456', 'test_value',           # single underscore
        '12345_67890_11111', 'a_b_c',      # multiple underscores
        '1_000_000', '123_456.78',         # numeric-looking
        '_123', '_test',                    # underscore at start
        '123_', 'test_',                    # underscore at end
        '123__456', 'test__value',         # consecutive underscores
    ])
    def test_underscore_strings_rejected(self, value):
        """Verify an underscore marks an identifier, not a number.

        Mutation: dropping the "'_' in val" guard, which types
          '1_000_000' int through Python's own digit separators.
        Oracle: libb.numify('1_000_000', int) returns 1000000, so the
          guard is the only thing returning None.
        """
        assert infer_numeric_type(value) is None


# --- smart_type Integration Tests (Parameterized) ---

class TestSmartTypeIntegration:
    """Tests for smart_type() integration with infer_numeric_type()."""

    @pytest.mark.parametrize(('value', 'expected'), [
        ('123', int),
        ('-456', int),
        ('1,200', int),
        ('123.45', float),
        ('-67.89', float),
        ('1.5e3', float),
        ('(100)', int),
        ('5.5%', float),
        ('1,234.56', float),
    ])
    def test_smart_type_with_inference(self, value, expected):
        """Verify smart_type hands a numeric string to the inference.

        Mutation: dropping the infer_numeric_strings branch, which
          types every one of these str.
        Oracle: infer_numeric_type returns the same int/float for each
          string on its own.
        """
        assert smart_type(value, infer_numeric_strings=True) == expected

    @pytest.mark.parametrize('value', ['hello', 'test123', 'abc'])
    def test_smart_type_non_numeric_string(self, value):
        """Verify a text string still types str with inference on.

        Mutation: returning infer_numeric_type(val) without the None
          check, which types 'hello' as None.
        Oracle: infer_numeric_type('hello') is None, so the None check
          is what keeps str.
        """
        assert smart_type(value, infer_numeric_strings=True) == str

    @pytest.mark.parametrize('value', ['123', '123.45', '1,200'])
    def test_smart_type_disabled_by_default(self, value):
        """Verify smart_type leaves numeric strings alone by default.

        Mutation: flipping smart_type's infer_numeric_strings default
          to True.
        Oracle: the same values with the flag on return int and float.
        """
        assert smart_type(value) == str

    @pytest.mark.parametrize(('value', 'expected'), [
        (123, int),
        (45.67, float),
        ('text', str),
        (True, bool),
    ])
    def test_smart_type_preserves_native_types(self, value, expected):
        """Verify smart_type reads the exact class of a native value.

        Mutation: an isinstance ladder in place of val.__class__, which
          types True as int.
        Oracle: bool subclasses int, so only the exact class gives bool.
        """
        assert smart_type(value) == expected

    def test_smart_type_preserves_date_types(self):
        """Verify the date branches run in order, midnight included.

        Mutation: hoisting the midnight datetime.datetime check above
          the DateTime check, or writing it as an isinstance test,
          which demotes a midnight DateTime to Date.
        Oracle: DateTime subclasses datetime.datetime, so only the
          branch order keeps a midnight DateTime a DateTime; a plain
          midnight datetime.datetime is the one that becomes Date.
        """
        assert smart_type(Date(2024, 1, 1)) == Date
        assert smart_type(DateTime(2024, 1, 1, 10, 30)) == DateTime
        assert smart_type(DateTime(2024, 1, 1, 0, 0)) == DateTime
        assert smart_type(datetime.datetime(2024, 1, 1)) == Date
        assert smart_type(datetime.datetime(2024, 1, 1, 10, 30)) \
            == datetime.datetime


# --- DataSet Creation Integration Tests ---

class TestDataSetCreationWithNumericStrings:
    """Integration tests for DataSet creation with numeric strings."""

    def test_guess_columns_integer_strings(self):
        """Verify guess_columns types a column of integer strings int.

        Mutation: initializing typ to str instead of type(None) in the
          scan loop, so the first non-None value never lands.
        Oracle: infer_numeric_type('123') is int.
        """
        rows = [{'value': '123'}, {'value': '456'}]
        columns = DataSet.guess_columns(rows, infer_numeric_strings=True)
        assert dict(columns)['value'] == int

    def test_guess_columns_float_strings(self):
        """Verify guess_columns types a column of decimals float.

        Mutation: guess_columns calling smart_type without forwarding
          infer_numeric_strings, which leaves the column str.
        Oracle: infer_numeric_type('123.45') is float.
        """
        rows = [{'value': '123.45'}, {'value': '678.90'}]
        columns = DataSet.guess_columns(rows, infer_numeric_strings=True)
        assert dict(columns)['value'] == float

    def test_guess_columns_mixed_numeric_strings(self):
        """Verify each column takes the type its own format implies.

        Mutation: dropping the '%' early return in infer_numeric_type,
          which types the '100%' column int.
        Oracle: libb.numify('100%', int) returns 100, while '1,200' and
          '(100)' have no float reading.
        """
        rows = [{'a': '1,200', 'b': '(100)', 'c': '100%'}]
        columns = DataSet.guess_columns(rows, infer_numeric_strings=True)
        colmap = dict(columns)
        assert colmap['a'] == int
        assert colmap['b'] == int
        assert colmap['c'] == float

    def test_guess_columns_text_vs_numeric(self):
        """Verify a text column and a numeric column type apart.

        Mutation: returning infer_numeric_type(val) from smart_type
          without the None check, which types 'hello' None.
        Oracle: libb.numify rejects 'hello' and 'abc123' and accepts
          '123'.
        """
        rows = [{'numeric': '123', 'text': 'hello', 'mixed': 'abc123'}]
        columns = DataSet.guess_columns(rows, infer_numeric_strings=True)
        colmap = dict(columns)
        assert colmap['numeric'] == int
        assert colmap['text'] == str
        assert colmap['mixed'] == str

    def test_guess_columns_disabled_by_default(self):
        """Verify guess_columns leaves numeric strings str by default.

        Mutation: flipping guess_columns' infer_numeric_strings default
          to True.
        Oracle: the same rows with the flag on type int.
        """
        rows = [{'value': '123'}, {'value': '456'}]
        columns = DataSet.guess_columns(rows)
        assert dict(columns)['value'] == str

    def test_dataset_creation_auto_converts_numeric_strings(self):
        """Verify the constructor infers and then converts the values.

        Mutation: the constructor not forwarding infer_numeric_strings
          to guess_columns, which leaves every column str.
        Oracle: hand-computed 123 and 45.67 against the raw strings.
        """
        rows = [
            {'int_col': '123', 'float_col': '45.67', 'str_col': 'text'},
            {'int_col': '456', 'float_col': '89.01', 'str_col': 'more'},
        ]
        ds = DataSet(rows, infer_numeric_strings=True)

        assert ds.colmap['int_col'] == int
        assert ds.colmap['float_col'] == float
        assert ds.colmap['str_col'] == str
        assert ds[0]['int_col'] == 123
        assert ds[0]['float_col'] == 45.67

    def test_dataset_creation_no_inference_by_default(self):
        """Verify the constructor infers nothing by default.

        Mutation: flipping the constructor's infer_numeric_strings
          default to True.
        Oracle: the same rows with the flag on type int and float.
        """
        rows = [{'int_col': '123', 'float_col': '45.67'}]
        ds = DataSet(rows)

        assert ds.colmap['int_col'] == str
        assert ds.colmap['float_col'] == str
        assert ds[0]['int_col'] == '123'

    def test_dataset_creation_sparse_numeric_strings(self):
        """Verify the scan skips None and types on the first value.

        Mutation: dropping the "val is not None" guard in the scan
          loop, which settles the column as NoneType on row 0.
        Oracle: the same rows without the leading Nones type int.
        """
        rows = [{'val': None}, {'val': None}, {'val': '123'}]
        ds = DataSet(rows, infer_numeric_strings=True)
        assert ds.colmap['val'] == int

    def test_column_mixing_int_and_float_strings_is_float(self):
        """Verify a column holding both string forms lands on float.

        Mutation: reversing the promotion to "typ is float and row_typ
          is int", which types the int-first column int and truncates
          45.67 out of it.
        Oracle: hand-computed 100.0 and 45.67, the same in both row
          orders.
        """
        rows = [{'val': None}, {'val': '45.67'}, {'val': '100'}]
        ds = DataSet(rows, infer_numeric_strings=True)
        assert ds.colmap['val'] == float
        assert ds[2]['val'] == 100.0

        rows = [{'val': None}, {'val': '100'}, {'val': '45.67'}]
        ds = DataSet(rows, infer_numeric_strings=True)
        assert ds.colmap['val'] == float
        assert ds[1]['val'] == 100.0
        assert ds[2]['val'] == 45.67


# --- Multi-Row Scanning Tests ---

class TestDataSetNumericStringScanning:
    """Tests for multi-row scanning with numeric strings."""

    def test_numeric_string_in_row_99(self):
        """Verify the last row inside the default scan limit is read.

        Mutation: dropping the default scan_limit to 99, or slicing
          rows[scan_start:scan_end - 1].
        Oracle: the value sits at index 99, the last of the 100 rows
          the default limit admits.
        """
        rows = [{'val': None} for _ in range(99)]
        rows.append({'val': '42.5'})
        ds = DataSet(rows, infer_numeric_strings=True)
        assert ds.colmap['val'] == float

    def test_respects_scan_limit_with_numeric_strings(self):
        """Verify scan_limit cuts the scan exactly at its row count.

        Mutation: scan_end = scan_start + scan_limit + 1, which reads
          one row past the limit.
        Oracle: the value sits at index 50, so a limit of 50 must miss
          it and 51 must find it.
        """
        rows = [{'val': None} for _ in range(50)]
        rows.append({'val': '123'})

        columns = DataSet.guess_columns(
            rows, scan_limit=50, infer_numeric_strings=True)
        assert dict(columns)['val'] is type(None)

        columns = DataSet.guess_columns(
            rows, scan_limit=51, infer_numeric_strings=True)
        assert dict(columns)['val'] == int

    def test_mixed_numeric_and_text_strings_give_object(self):
        """Verify a text value beside a numeric string gives object.

        `infer_numeric_strings` reads '123' as int, so the column holds
        one int and one str - a cross-family mix no single type covers.

        Mutation: infer_numeric_strings dropped on the way into
          smart_type, which types '123' str and makes the column str.
        Oracle: the same rows without the trailing 'text' type int, and
          the pair of strings without the flag types str.
        """
        rows = [{'val': None}, {'val': '123'}, {'val': 'text'}]
        columns = DataSet.guess_columns(rows, infer_numeric_strings=True)
        assert dict(columns)['val'] is object

        assert dict(DataSet.guess_columns(
            rows[:2], infer_numeric_strings=True))['val'] is int
        assert dict(DataSet.guess_columns(rows))['val'] is str

    def test_leading_zero_code_column_keeps_digits(self):
        """Verify a leading-zero code column stays str, digits intact.

        Mutation: relaxing the leading-zero guard's length test from
          >= 3 to > 3, which types the three-character '007' int and
          rewrites the value as 7.
        Oracle: libb.numify('007', int) returns 7, so only the guard
          keeps the string whole.
        """
        ds = DataSet([{'val': '007'}, {'val': '012'}],
                     infer_numeric_strings=True)
        assert ds.colmap['val'] == str
        assert ds[0]['val'] == '007'
        assert ds[1]['val'] == '012'


# --- Value Conversion Tests ---

class TestDataSetNumericStringConversion:
    """Tests for actual value conversion with numeric strings."""

    def test_conversion_integer_strings(self):
        """Verify an int column converts its strings, zero included.

        Mutation: `return result if result else val` in _convert_value,
          which leaves the falsy 0 as the string '0'.
        Oracle: hand-computed 123 and 0, each an int instance.
        """
        ds = DataSet([{'value': '123'}, {'value': '0'}],
                     infer_numeric_strings=True)
        assert ds[0]['value'] == 123
        assert ds[1]['value'] == 0
        assert isinstance(ds[0]['value'], int)
        assert isinstance(ds[1]['value'], int)

    def test_conversion_float_strings(self):
        """Verify a float column converts through the float probe.

        Mutation: _convert_value calling libb.numify(val, int) with the
          type hardcoded, which returns None for '123.45' and leaves
          the raw string.
        Oracle: hand-computed 123.45 and 0.0, each a float instance.
        """
        ds = DataSet([{'value': '123.45'}, {'value': '0.00'}],
                     infer_numeric_strings=True)
        assert ds[0]['value'] == 123.45
        assert ds[1]['value'] == 0.0
        assert isinstance(ds[0]['value'], float)
        assert isinstance(ds[1]['value'], float)

    def test_conversion_formatted_strings(self):
        """Verify comma, percent and parentheses forms convert.

        Mutation: _convert_value calling typ(val) instead of
          libb.numify, which raises and leaves the raw strings.
        Oracle: hand-computed 1200, 5.5 and -100; the parentheses carry
          the minus sign.
        """
        ds = DataSet([
            {'amount': '1,200', 'rate': '5.5%', 'loss': '(100)'}
        ], infer_numeric_strings=True)

        assert ds[0]['amount'] == 1200
        assert isinstance(ds[0]['amount'], int)
        assert ds[0]['rate'] == 5.5
        assert isinstance(ds[0]['rate'], float)
        assert ds[0]['loss'] == -100
        assert isinstance(ds[0]['loss'], int)

    def test_conversion_preserves_none(self):
        """Verify a None hole and a missing key both stay None.

        Mutation: dropping the "if name not in row" fill in
          convert_container_types, which raises KeyError on the last
          row.
        Oracle: hand-typed None for the hole; the last row carries no
          'value' key at all.
        """
        ds = DataSet([
            {'value': '123'}, {'value': None}, {'value': '456'}, {},
        ], infer_numeric_strings=True)

        assert ds[0]['value'] == 123
        assert ds[1]['value'] is None
        assert ds[2]['value'] == 456
        assert ds[3]['value'] is None

    def test_conversion_scientific_notation(self):
        """Verify exponent strings convert to their float values.

        Mutation: extending the identifier guard to reject any string
          holding 'e', which leaves '1e6' a str column.
        Oracle: hand-computed 1000000.0 and 0.0025.
        """
        ds = DataSet([{'value': '1e6'}, {'value': '2.5e-3'}],
                     infer_numeric_strings=True)
        assert ds[0]['value'] == 1000000.0
        assert ds[1]['value'] == 0.0025
        assert isinstance(ds[0]['value'], float)


# --- Backward Compatibility Tests ---

class TestDataSetNumericStringBackwardsCompatibility:
    """Tests ensuring backward compatibility with existing behavior."""

    def test_explicit_type_overrides_inference(self):
        """Verify an explicit column type beats the inference.

        Mutation: the constructor calling guess_columns even when
          columns is given.
        Oracle: the same row without columns types int, so only the
          explicit pair holds it str.
        """
        ds = DataSet([{'value': '123'}], columns=[('value', str)],
                     infer_numeric_strings=True)
        assert ds.colmap['value'] == str
        assert ds[0]['value'] == '123'

    def test_actual_numbers_not_affected(self):
        """Verify real numbers keep their types and their promotion.

        Mutation: dropping the int-to-float promotion in the scan loop,
          which types the mixed column int and truncates 2.5 to 2.
        Oracle: hand-computed 1.0 and 2.5 in the promoted column.
        """
        ds = DataSet([{'value': 123}])
        assert ds.colmap['value'] == int
        assert ds[0]['value'] == 123

        ds = DataSet([{'value': 1}, {'value': 2.5}])
        assert ds.colmap['value'] == float
        assert ds[0]['value'] == 1.0
        assert isinstance(ds[0]['value'], float)
        assert ds[1]['value'] == 2.5

    def test_mixed_actual_and_string_numbers(self):
        """Verify a real number and a numeric string share a column.

        Mutation: dropping the int-to-float promotion in the scan loop,
          which types the second column int and leaves '45.67' a raw
          string.
        Oracle: hand-computed 456 in the int column, 123.0 and 45.67 in
          the promoted one.
        """
        rows = [{'val': None}, {'val': 123}, {'val': '456'}]
        ds = DataSet(rows, infer_numeric_strings=True)
        assert ds.colmap['val'] == int
        assert ds[2]['val'] == 456

        rows = [{'val': None}, {'val': 123}, {'val': '45.67'}]
        ds = DataSet(rows, infer_numeric_strings=True)
        assert ds.colmap['val'] == float
        assert ds[1]['val'] == 123.0
        assert ds[2]['val'] == 45.67


# --- DataSet Operations Tests ---

class TestDataSetNumericStringOperations:
    """Tests for DataSet operations with numeric strings."""

    def test_bucket_with_numeric_strings(self):
        """Verify bucket sums an inferred int column per key.

        Mutation: bucket defaulting a bare column name to max instead
          of sum, which gives group A 200.
        Oracle: hand-computed 300 for A (100 + 200) and 450 for B.
        """
        ds = DataSet([
            {'category': 'A', 'value': '100'},
            {'category': 'A', 'value': '200'},
            {'category': 'B', 'value': '450'},
        ], infer_numeric_strings=True)

        result = ds.bucket(['category'], ['value'])
        result.sort_data('category')

        assert result.colmap['value'] == int
        assert result[0]['value'] == 300
        assert result[1]['value'] == 450

    def test_join_with_numeric_strings(self):
        """Verify join matches on the inferred int keys.

        Mutation: pairing rows positionally instead of by key, which
          hands id 1 the extra 75 from the right side's first row.
        Oracle: hand-paired rows; the right side is in reverse order
          and carries an id 3 the left side lacks.
        """
        ds1 = DataSet([
            {'id': '1', 'value': '100'},
            {'id': '2', 'value': '200'},
        ], infer_numeric_strings=True)
        ds2 = DataSet([
            {'id': '2', 'extra': '75'},
            {'id': '3', 'extra': '90'},
            {'id': '1', 'extra': '50'},
        ], infer_numeric_strings=True)

        result = DataSet.join(ds1, 'id', ds2, 'id')
        result.sort_data('id')

        assert len(result) == 2
        assert result.colmap['id'] == int
        assert result[0]['id'] == 1
        assert result[0]['value'] == 100
        assert result[0]['extra'] == 50
        assert result[1]['id'] == 2
        assert result[1]['extra'] == 75


# --- Real-World Data Pattern Tests ---

class TestInferNumericTypeRealWorldData:
    """Tests with real-world data patterns."""

    @pytest.mark.parametrize(('value', 'expected'), [
        ('1,234.56', float),
        ('(1,234.56)', float),
        ('999,999.99', float),
        ('100%', float),
        ('5.5%', float),
        ('0.05%', float),
        ('(100.00)', float),
        ('(1,234,567.89)', float),
        ('1,000,000', int),
        ('1,234,567,890', int),
        ('0.0001', float),
        ('0.25', float),
    ])
    def test_real_world_patterns(self, value, expected):
        """Verify percent and grouped forms infer as their numeric types.

        Mutation: pre-validating with a plain-decimal regex, which
          rejects every parenthesized and comma-grouped form.
        Oracle: hand-typed int/float per string.
        """
        assert infer_numeric_type(value) == expected


# --- Underscore String Preservation in DataSet Operations ---

class TestDataSetUnderscorePreservation:
    """Tests that DataSet operations keep underscore-delimited strings."""

    def test_underscore_strings_not_inferred_as_numeric(self):
        """Verify an underscore id column stays str, text intact.

        Mutation: dropping the "'_' in val" guard, which types
          '12345_67890' int and rewrites the value as 1234567890.
        Oracle: libb.numify('12345_67890', int) returns 1234567890.
        """
        rows = [
            {'id': '12345_67890', 'value': 100},
            {'id': '11111_22222', 'value': 200},
        ]
        ds = DataSet(rows, infer_numeric_strings=True)

        assert ds.colmap['id'] == str
        assert ds[0]['id'] == '12345_67890'
        assert ds[1]['id'] == '11111_22222'

    def test_bucket_preserves_underscore_strings(self):
        """Verify bucket groups on every key column, ids intact.

        Mutation: grouping on keycols[0] alone, which merges category
          A's two codes into one row of 340.
        Oracle: hand-computed 300, 40 and 150 for the three key pairs.
        """
        ds = DataSet([
            {'category': 'A', 'code': '12345_67890', 'value': 100},
            {'category': 'A', 'code': '12345_67890', 'value': 200},
            {'category': 'A', 'code': '99999_88888', 'value': 40},
            {'category': 'B', 'code': '11111_22222', 'value': 150},
        ], infer_numeric_strings=True)

        result = ds.bucket(['category', 'code'], ['value'])
        result.sort_data('category', 'code')

        assert len(result) == 3
        assert result[0]['code'] == '12345_67890'
        assert result[0]['value'] == 300
        assert result[1]['code'] == '99999_88888'
        assert result[1]['value'] == 40
        assert result[2]['code'] == '11111_22222'
        assert result[2]['value'] == 150

    def test_underscore_string_as_bucket_key(self):
        """Verify one id under two groups stays two rows.

        Mutation: grouping on the id alone rather than the whole key
          tuple, which merges group A's and group B's shared id.
        Oracle: hand-computed 300, 40 and 150 per (group, id).
        """
        ds = DataSet([
            {'group': 'A', 'id': '12345_67890_11111', 'category': 'X',
             'value': 100},
            {'group': 'A', 'id': '12345_67890_11111', 'category': 'X',
             'value': 200},
            {'group': 'A', 'id': '22222_33333_44444', 'category': 'X',
             'value': 40},
            {'group': 'B', 'id': '22222_33333_44444', 'category': 'Y',
             'value': 150},
        ])
        ds.columns = [
            ('group', str), ('id', str), ('category', str), ('value', int),
        ]

        result = ds.bucket(['group', 'id', 'category'], [('value', sum)])
        result.sort_data('group', 'id')

        assert len(result) == 3
        for row in result:
            assert isinstance(row['id'], str)
            assert '_' in row['id']

        assert [row['value'] for row in result] == [300, 40, 150]
        assert result[0]['id'] == '12345_67890_11111'
        assert result[2]['group'] == 'B'

    def test_underscore_strings_survive_multiple_buckets(self):
        """Verify a bucket of a bucket keeps the ids and the maxima.

        Mutation: the second bucket applying sum in place of the max it
          is handed, which makes group A 450.
        Oracle: hand-computed max(300, 150) for A and 30 for B, with
          the four ids flattened out of the nested lists.
        """
        ds = DataSet([
            {'group': 'A', 'subgroup': 'X', 'code': '12345_12345_12345',
             'value': 100},
            {'group': 'A', 'subgroup': 'X', 'code': '67890_67890_67890',
             'value': 200},
            {'group': 'A', 'subgroup': 'Y', 'code': '11111_22222_33333',
             'value': 150},
            {'group': 'B', 'subgroup': 'Z', 'code': '44444_55555_66666',
             'value': 30},
        ])
        ds.columns = [('group', str), ('subgroup', str), ('code', str),
                      ('value', int)]

        first_bucket = ds.bucket(['group', 'subgroup'],
                                 [('code', list), ('value', sum)])
        first_bucket.sort_data('group', 'subgroup')

        assert [row['value'] for row in first_bucket] == [300, 150, 30]
        assert first_bucket[0].code == ['12345_12345_12345',
                                        '67890_67890_67890']

        second_bucket = first_bucket.bucket(
            ['group'], [('code', list), ('subgroup', list), ('value', max)]
        )
        second_bucket.sort_data('group')

        assert len(second_bucket) == 2
        assert [row['value'] for row in second_bucket] == [300, 30]

        flat_codes = []
        for row in second_bucket:
            for item in row.code:
                if isinstance(item, list):
                    flat_codes.extend(item)
                else:
                    flat_codes.append(item)
        assert flat_codes == [
            '12345_12345_12345', '67890_67890_67890',
            '11111_22222_33333', '44444_55555_66666',
        ]

    def test_mixed_underscore_and_numeric_strings(self):
        """Verify an underscore column and a numeric column type apart.

        Mutation: dropping the "'_' in val" guard, which types the id
          column int and rewrites '12345_67890' as 1234567890.
        Oracle: libb.numify('12345_67890', int) returns 1234567890,
          while 'ABC_123' has no numeric reading either way.
        """
        rows = [
            {'id': '12345_67890', 'amount': '1000', 'code': 'ABC_123'},
            {'id': '11111_22222', 'amount': '2000', 'code': 'DEF_456'},
        ]
        ds = DataSet(rows, infer_numeric_strings=True)

        assert ds.colmap['id'] == str
        assert ds.colmap['amount'] == int
        assert ds.colmap['code'] == str
        assert ds[0]['id'] == '12345_67890'
        assert ds[0]['amount'] == 1000


# --- Edge Case Tests ---

class TestInferNumericTypeAdditionalEdgeCases:
    """Additional edge case tests for numeric inference."""

    def test_unicode_minus_sign(self):
        """Verify a unicode minus sign is not read as a sign.

        Mutation: normalizing U+2212 to an ASCII hyphen before the
          probes, which types the string int.
        Oracle: int('\u2212123') raises, so Python itself refuses it.
        """
        assert infer_numeric_type('\u2212123') is None

    def test_unicode_digits(self):
        """Verify non-ASCII digits parse as Python parses them.

        Mutation: an ASCII-digit-only pre-filter, which rejects the
          Arabic-Indic spelling of 123.
        Oracle: int('\u0661\u0662\u0663') is 123.
        """
        assert infer_numeric_type('\u0661\u0662\u0663') == int

    def test_multiple_percentage_signs(self):
        """Verify a doubled percent sign is not a number.

        Mutation: returning float unconditionally from the '%' branch
          instead of testing the numify result.
        Oracle: libb.numify('5%%', float) is None.
        """
        assert infer_numeric_type('5%%') is None

    def test_empty_after_formatting(self):
        """Verify separators with no digits are not a number.

        Mutation: `libb.numify(val, int) or 0` in place of the None
          check, which types ',' as int 0.
        Oracle: libb.numify(',', int) is None; nothing is left once the
          commas go.
        """
        assert infer_numeric_type(',') is None
        assert infer_numeric_type(',,') is None


class TestDataSetNumericInferenceEdgeCases:
    """Edge cases for DataSet numeric string inference."""

    def test_all_none_column(self):
        """Verify an all-None column keeps the NoneType placeholder.

        Mutation: initializing typ to str in the scan loop, which types
          the column str and starts converting its None values.
        Oracle: type(None) is what the loop starts with, and no row
          here replaces it.
        """
        rows = [{'val': None}, {'val': None}, {'val': None}]
        ds = DataSet(rows, infer_numeric_strings=True)
        assert ds.colmap['val'] is type(None)

    def test_empty_dataset_with_inference(self):
        """Verify an empty container yields no columns and no crash.

        Mutation: dropping the "if rows" guard on rows[exemplar], which
          raises IndexError for an empty container.
        Oracle: an empty row list has no column names to read.
        """
        ds = DataSet([], infer_numeric_strings=True)
        assert len(ds) == 0
        assert ds.columns == []

    def test_single_row_numeric_string(self):
        """Verify a one-row dataset still infers and converts.

        Mutation: starting the scan at exemplar + 1, which leaves a
          one-row dataset with nothing to read.
        Oracle: infer_numeric_type('42') is int.
        """
        ds = DataSet([{'val': '42'}], infer_numeric_strings=True)
        assert ds.colmap['val'] == int
        assert ds[0]['val'] == 42

    def test_mixed_formats_in_column(self):
        """Verify a percent among int forms promotes the column.

        Mutation: dropping the int-to-float promotion in the scan loop,
          which stops at int and leaves '5.5%' a raw string.
        Oracle: hand-computed 100.0, 1200.0, -50.0 and 5.5 in the
          promoted column.
        """
        rows = [
            {'val': '100'},
            {'val': '1,200'},
            {'val': '(50)'},
            {'val': '5.5%'},
        ]
        ds = DataSet(rows, infer_numeric_strings=True)
        assert ds.colmap['val'] == float
        assert [row['val'] for row in ds] == [100.0, 1200.0, -50.0, 5.5]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
