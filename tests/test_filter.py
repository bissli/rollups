import logging
import re

import pytest
from opendate import Date, DateTime
from rollups import DataSet

# --- Fixtures ---


@pytest.fixture
def basic_dataset():
    """Basic dataset for filter tests."""
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 200},
        {'id': 3, 'name': 'C', 'value': 300},
        {'id': 4, 'name': 'D', 'value': 400},
        {'id': 5, 'name': 'E', 'value': 500}])
    ds.columns = (('id', int), ('name', str), ('value', int))
    return ds


@pytest.fixture
def fruit_dataset():
    """Dataset with fruit names for pattern matching tests."""
    ds = DataSet([
        {'id': 1, 'name': 'Apple', 'value': 100},
        {'id': 2, 'name': 'Banana', 'value': 200},
        {'id': 3, 'name': 'Apricot', 'value': 300},
        {'id': 4, 'name': 'Cherry', 'value': 400}])
    ds.columns = (('id', int), ('name', str), ('value', int))
    return ds


# --- Basic filter operations ---

def test_filter_string_pattern(basic_dataset):
    """Verify a string pattern keeps only the rows carrying it.

    Mutation: `if i != -1` narrowed to `if i > 0` in DataSet._match,
        dropping a match that starts at offset zero.
    Oracle: hand-computed single survivor 'A', whose match sits at
        offset zero.
    """
    basic_dataset.filter_data('A')

    assert [r['name'] for r in basic_dataset] == ['A']


def test_filter_with_replacement():
    """Verify replace rewrites only the matched span of the field.

    Mutation: `row[key] = replace(fld)` handing the whole field to the
        callback instead of splicing fld[:i] + replace(fld[i:j]) +
        fld[j:].
    Oracle: hand-computed 'A_pp_le' from pattern 'pp' on 'Apple'.
    """
    ds = DataSet([
        {'id': 1, 'name': 'Apple', 'value': 100},
        {'id': 2, 'name': 'Banana', 'value': 200},
        {'id': 3, 'name': 'Cherry', 'value': 300}])
    ds.columns = (('id', int), ('name', str), ('value', int))

    ds.filter_data('pp', lambda s: f'_{s}_')

    assert [r['name'] for r in ds] == ['A_pp_le']


def test_filter_callable_predicate(basic_dataset):
    """Verify a callable predicate is used directly, not as a pattern.

    Mutation: the `callable(pattern_or_predicate)` branch dropped, so a
        function falls through to _match and hits `pattern.lower()`.
    Oracle: hand-computed survivors B and D, the values divisible by
        200.
    """
    basic_dataset.filter_data(lambda r: r['value'] % 200 == 0)

    assert [r['name'] for r in basic_dataset] == ['B', 'D']


def test_filter_with_None_predicate(basic_dataset):
    """Verify a None pattern keeps every row and still honors inplace.

    Mutation: `return None if inplace else self.deepcopy()` reduced to
        `return None`, or to a row-sharing `self.copy()`.
    Oracle: writing to the returned copy leaves the original row at its
        hand-set value 100.
    """
    assert basic_dataset.filter_data(None) is None
    assert len(basic_dataset) == 5

    copied = basic_dataset.filter_data(None, inplace=False)

    assert [r['id'] for r in copied] == [1, 2, 3, 4, 5]
    copied[0]['value'] = -1
    assert basic_dataset[0]['value'] == 100


# --- Inplace parameter ---

@pytest.mark.parametrize('inplace', [True, False])
def test_filter_inplace_parameter(basic_dataset, inplace):
    """Verify inplace picks between mutating self and returning a copy.

    Mutation: the inplace arm returning self rather than None, or
        `self.copy(empty=True)` reduced to a bare DataSet() that drops
        the column schema.
    Oracle: hand-computed single survivor 'A', and the declared
        three-column schema on the returned copy.
    """
    schema = list(basic_dataset.columns)

    result = basic_dataset.filter_data('A', inplace=inplace)

    if inplace:
        assert result is None
        assert [r['name'] for r in basic_dataset] == ['A']
    else:
        assert [r['name'] for r in basic_dataset] == ['A', 'B', 'C', 'D', 'E']
        assert [r['name'] for r in result] == ['A']
        assert list(result.columns) == schema


@pytest.mark.parametrize('convert_first', [False, True])
def test_filter_not_inplace(convert_first):
    """Verify inplace=False returns rows independent of the original.

    Mutation: `result.container = filtered_rows` in either arm of the
        inplace=False branch, sharing row objects with the original.
    Oracle: writing to the copy's first row leaves the original row at
        its hand-set 'A'.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 200},
        {'id': 3, 'name': 'C', 'value': 300},
        {'id': 4, 'name': 'AB', 'value': 400},
        {'id': 5, 'name': 'E', 'value': 500}])
    ds.columns = (('id', int), ('name', str), ('value', int))
    if convert_first:
        assert ds[0]['id'] == 1

    filtered = ds.filter_data('A', inplace=False)

    assert len(ds) == 5
    assert [r['name'] for r in filtered] == ['A', 'AB']

    filtered[0]['name'] = 'MUTATED'

    assert ds[0]['name'] == 'A'


def test_filter_chained_not_inplace(fruit_dataset):
    """Verify chained inplace=False filters leave the source untouched.

    Mutation: `.lower()` dropped from the field side of _match's find,
        so pattern 'a' misses 'Apple' and 'Apricot'.
    Oracle: hand-computed three then one survivors, against the
        source's four unchanged names.
    """
    source_names = ['Apple', 'Banana', 'Apricot', 'Cherry']

    first = fruit_dataset.filter_data('a', inplace=False)
    second = first.filter_data('pp', inplace=False)

    assert [r['name'] for r in fruit_dataset] == source_names
    assert [r['name'] for r in first] == ['Apple', 'Banana', 'Apricot']
    assert [r['name'] for r in second] == ['Apple']


# --- String pattern matching ---

@pytest.mark.parametrize(('pattern', 'expected_names'), [
    ('ALPHA', ['Alpha']),
    ('alpha', ['Alpha']),
    ('lph', ['Alpha']),
    ('a', ['Alpha', 'Beta', 'Gamma']),
])
def test_filter_pattern_matching(pattern, expected_names):
    """Verify pattern matching folds case and matches mid-word.

    Mutation: `fld.lower().find(pattern.lower())` reduced to
        `fld.find(pattern)`, or to a startswith test.
    Oracle: hand-computed name lists; 'lph' sits mid-word and 'ALPHA'
        differs from the data only in case.
    """
    ds = DataSet([
        {'id': 1, 'name': 'Alpha', 'value': 100},
        {'id': 2, 'name': 'Beta', 'value': 200},
        {'id': 3, 'name': 'Gamma', 'value': 300}])
    ds.columns = (('id', int), ('name', str), ('value', int))

    ds.filter_data(pattern)

    assert [r['name'] for r in ds] == expected_names


def test_filter_pattern_is_literal_not_regex():
    """Verify a string pattern matches literally, never as a regex.

    Mutation: `re.search(pattern, fld)` in place of `fld.find(pattern)`
        in _match.
    Oracle: 'testXvalue', which the regex 't.v' matches and a literal
        find does not.
    """
    ds = DataSet([
        {'name': 'test.value'},
        {'name': 'testXvalue'},
        {'name': 'other'}])

    ds.filter_data('t.v')

    assert [r['name'] for r in ds] == ['test.value']
    assert re.search('t.v', 'testXvalue') is not None


# --- Empty results and schema survival ---

def test_filter_empty_dataset():
    """Verify filtering an empty dataset keeps its declared schema.

    Mutation: `self.copy(empty=True)` reduced to a bare DataSet(), so
        the returned copy loses the column schema.
    Oracle: the hand-declared three-column schema, on the dataset
        filtered in place and on the returned copy alike.
    """
    ds = DataSet([])
    ds.columns = (('id', int), ('name', str), ('value', int))
    schema = [('id', int), ('name', str), ('value', int)]

    copied = ds.filter_data('test', inplace=False)
    ds.filter_data('test')

    assert len(ds) == 0
    assert len(copied) == 0
    assert list(ds.columns) == schema
    assert list(copied.columns) == schema


def test_filter_no_matches(basic_dataset):
    """Verify a pattern absent from every field empties the dataset.

    Mutation: `match = False` initialized True in _match, so a row with
        no matching field is kept.
    Oracle: 'XYZ' appears in no id, name, or value, so zero rows
        survive.
    """
    basic_dataset.filter_data('XYZ')

    assert len(basic_dataset) == 0
    assert basic_dataset.container == []


def test_filter_preserves_columns(basic_dataset):
    """Verify the column schema survives filtering down to zero rows.

    Mutation: the inplace arm re-inferring columns from the surviving
        rows, which yields an empty schema once no row is left.
    Oracle: the hand-declared three-column schema, checked after 'XYZ'
        drops every row.
    """
    schema = [('id', int), ('name', str), ('value', int)]
    assert list(basic_dataset.columns) == schema

    basic_dataset.filter_data('XYZ')

    assert len(basic_dataset) == 0
    assert list(basic_dataset.columns) == schema


def test_filter_all_match(basic_dataset):
    """Verify an always-true predicate keeps the same row objects.

    Mutation: the inplace arm rebuilding rows as new attrdicts instead
        of reusing them, breaking aliasing a caller relies on.
    Oracle: identity of the five row objects captured before the call.
    """
    rows_before = list(basic_dataset.container)

    basic_dataset.filter_data(lambda r: r['value'] > 0)

    assert len(basic_dataset) == 5
    assert all(a is b for a, b in zip(basic_dataset.container, rows_before))


# --- Sequential and complex filtering ---

def test_filter_multiple_patterns(fruit_dataset):
    """Verify a second in-place filter sees only the first pass's rows.

    Mutation: the inplace arm not assigning filtered_rows to
        self.container, leaving the dataset unfiltered between calls.
    Oracle: 'Cherry' is the only row matching 'err' and the first pass
        already dropped it, so the second pass must return zero rows.
    """
    fruit_dataset.filter_data('a')
    assert [r['name'] for r in fruit_dataset] == ['Apple', 'Banana', 'Apricot']

    fruit_dataset.filter_data('err')

    assert len(fruit_dataset) == 0


def test_filter_predicate_with_multiple_conditions():
    """Verify the logkey and plain paths keep exactly the same rows.

    Mutation: `if predicate(row)` inverted in filter_data's logkey
        loop, so the two branches disagree.
    Oracle: the plain listcomp path as a differential re-implementation,
        plus hand-computed surviving ids [2, 3, 5].
    """
    rows = [
        {'id': 1, 'value': 100, 'score': 85},
        {'id': 2, 'value': 200, 'score': 75},
        {'id': 3, 'value': 150, 'score': 90},
        {'id': 4, 'value': 250, 'score': 65},
        {'id': 5, 'value': 175, 'score': 95}]

    def predicate(row):
        return row['value'] > 120 and row['score'] > 70

    plain = DataSet([dict(r) for r in rows])
    plain.filter_data(predicate)
    logged = DataSet([dict(r) for r in rows])
    logged.filter_data(predicate, logkey='id')

    assert [r['id'] for r in plain] == [2, 3, 5]
    assert [r['id'] for r in logged] == [r['id'] for r in plain]


def test_filter_by_state_attribute():
    """Verify a callable predicate ignores the replace argument.

    Mutation: `if callable(pattern_or_predicate)` narrowed with
        `and replace is None`, sending a callable into _match.
    Oracle: the surviving category values, still upper-case 'X', with
        no 'category~orig' key added.
    """
    ds = DataSet([
        {'id': 1, 'category': 'X', 'value': 100},
        {'id': 2, 'category': 'Y', 'value': 200},
        {'id': 3, 'category': 'X', 'value': 300},
        {'id': 4, 'category': 'X', 'value': 400},
        {'id': 5, 'category': 'Y', 'value': 500}])
    ds.columns = (('id', int), ('category', str), ('value', int))

    ds.filter_data(
        lambda r: r['category'] == 'X',
        replace=lambda s: s.lower())

    assert [r['id'] for r in ds] == [1, 3, 4]
    assert [r['category'] for r in ds] == ['X', 'X', 'X']
    assert all('category~orig' not in r for r in ds)


# --- Field types and replacement ---

def test_filter_numeric_fields_not_matched():
    """Verify a string pattern never matches a numeric field.

    Mutation: `if not isinstance(fld, str): continue` in _match
        replaced by `fld = str(fld)`, so '456' matches the int column.
    Oracle: '456' occurs only in the int value of row two, so the
        correct result is zero rows.
    """
    ds = DataSet([
        {'name': 'test123', 'value': 123},
        {'name': 'other', 'value': 456}])

    ds.filter_data('456')

    assert len(ds) == 0


def test_filter_with_replacement_preserves_original():
    """Verify the pre-replacement text is saved under '{col}~orig'.

    Mutation: `row[key + '~orig'] = row[key]` moved below the
        overwrite, so the saved value is the replaced text.
    Oracle: hand-computed 'test' under 'name~orig' against 'TEST' in
        'name'.
    """
    ds = DataSet([{'name': 'test', 'value': 1}])

    ds.filter_data('test', lambda s: s.upper())

    assert ds[0]['name'] == 'TEST'
    assert ds[0]['name~orig'] == 'test'


def test_filter_replace_every_matching_column():
    """Verify replace rewrites the first hit in every matching column.

    Mutation: `fld.replace(...)` in place of the index splice, or a
        break once the first matching column is found.
    Oracle: hand-computed 'bANana' and 'bANdana', both keeping their
        second 'an' untouched.
    """
    ds = DataSet([{'id': 1, 'name': 'banana', 'note': 'bandana'}])

    ds.filter_data('an', lambda s: s.upper())

    assert ds[0]['name'] == 'bANana'
    assert ds[0]['note'] == 'bANdana'
    assert ds[0]['name~orig'] == 'banana'
    assert ds[0]['note~orig'] == 'bandana'


def test_filter_with_none_values_in_data():
    """Verify a None field is skipped rather than matched as text.

    Mutation: `if not isinstance(fld, str): continue` in _match
        replaced by `fld = str(fld)`, so 'one' matches str(None).
    Oracle: 'one' occurs in 'None' and in no real value, so the correct
        result is zero rows.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': None, 'value': 200},
        {'id': 3, 'name': 'C', 'value': None}])
    ds.columns = (('id', int), ('name', str), ('value', int))

    kept = ds.filter_data('A', inplace=False)
    ds.filter_data('one')

    assert [r['name'] for r in kept] == ['A']
    assert len(ds) == 0


@pytest.mark.parametrize(('col', 'values', 'attr', 'threshold'), [
    ('date',
     [Date(2024, 1, 15), Date(2024, 6, 15), Date(2024, 12, 31)],
     'month', 3),
    ('dt',
     [DateTime(2024, 1, 15, 10, 30), DateTime(2024, 6, 15, 14, 45),
      DateTime(2024, 12, 31, 23, 59)],
     'hour', 13),
])
def test_filter_temporal_values(col, values, attr, threshold):
    """Verify a temporal column filters by value, never by its text.

    Mutation: `if not isinstance(fld, str): continue` in _match
        replaced by `fld = str(fld)`, so '2024' matches every temporal
        column.
    Oracle: '2024' appears in every rendered date and in no string
        column, so the string pass must keep zero rows; the callable
        pass keeps the hand-picked last two values.
    """
    rows = [{'id': i + 1, col: value} for i, value in enumerate(values)]

    by_text = DataSet([dict(r) for r in rows])
    by_text.filter_data('2024')

    by_value = DataSet([dict(r) for r in rows])
    by_value.filter_data(lambda r: getattr(r[col], attr) > threshold)

    assert len(by_text) == 0
    assert [r[col] for r in by_value] == values[1:]


# --- Pattern edge cases ---

def test_filter_empty_string_pattern():
    """Verify an empty pattern short-circuits instead of matching.

    Mutation: `elif pattern_or_predicate:` widened to
        `elif pattern_or_predicate is not None:`, routing '' into
        _match, which cannot match a row holding no string field.
    Oracle: row three carries only ints, so _match would drop it while
        the short-circuit keeps it.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A'},
        {'id': 2, 'name': 'B'},
        {'id': 3, 'count': 7}])

    ds.filter_data('')

    assert [r['id'] for r in ds] == [1, 2, 3]


def test_filter_with_unicode_characters():
    """Verify matching compares codepoints, without accent folding.

    Mutation: _match normalizing with NFKD and stripping to ASCII
        before comparing, which reduces the pattern to an empty string.
    Oracle: the diaeresis and CJK rows, which accent-stripping would
        wrongly keep.
    """
    ds = DataSet([
        {'id': 1, 'name': 'Caf\u00e9'},
        {'id': 2, 'name': 'Na\u00efve'},
        {'id': 3, 'name': '\u65e5\u672c\u8a9e'}])

    ds.filter_data('\u00e9')

    assert [r['name'] for r in ds] == ['Caf\u00e9']


def test_filter_very_long_pattern():
    """Verify a long pattern is compared whole, not by a prefix.

    Mutation: `pattern.lower()[:100]` truncating the needle inside
        _match.
    Oracle: a 999-character decoy sharing its whole text with the
        1000-character target's prefix.
    """
    target = 'A' * 1000
    decoy = 'A' * 999
    ds = DataSet([
        {'id': 1, 'name': target},
        {'id': 2, 'name': decoy},
        {'id': 3, 'name': 'B'}])

    ds.filter_data(target)

    assert [r['id'] for r in ds] == [1]


def test_filter_with_newline_in_string():
    """Verify a newline in the pattern is matched, not normalized away.

    Mutation: _match collapsing whitespace with `' '.join(s.split())`
        on both sides before comparing.
    Oracle: a decoy row differing from the target only by a space where
        the target has a newline.
    """
    ds = DataSet([
        {'id': 1, 'text': 'line1\nline2'},
        {'id': 2, 'text': 'line1 line2'},
        {'id': 3, 'text': 'single line'}])

    ds.filter_data('line1\nline2')

    assert [r['id'] for r in ds] == [1]


def test_filter_predicate_returns_non_boolean():
    """Verify a truthy non-boolean return keeps the row.

    Mutation: `if predicate(row)` tightened to
        `if predicate(row) is True`, dropping every row.
    Oracle: hand-computed survivors 5 and 10, with 0 the only falsy
        return.
    """
    ds = DataSet([
        {'id': 1, 'value': 0},
        {'id': 2, 'value': 5},
        {'id': 3, 'value': 10}])

    ds.filter_data(lambda r: r['value'])

    assert [r['value'] for r in ds] == [5, 10]


def test_filter_predicate_raises_exception():
    """Verify a predicate error propagates and leaves the rows alone.

    Mutation: `predicate(row)` wrapped in try/except returning False,
        swallowing the error and silently dropping the row.
    Oracle: ZeroDivisionError reaching the caller, with all three rows
        still in place afterwards.
    """
    ds = DataSet([
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 0},
        {'id': 3, 'value': 300}])

    def bad_predicate(row):
        return 1000 / row['value'] > 5

    with pytest.raises(ZeroDivisionError):
        ds.filter_data(bad_predicate)

    assert [r['id'] for r in ds] == [1, 2, 3]


# --- Logging ---

def test_filter_with_logkey(caplog):
    """Verify logkey names the dropped rows, not the kept ones.

    Mutation: the logger.debug call moved into the `if predicate(row)`
        arm, logging survivors instead of drops.
    Oracle: 'Cherry' is the only row pattern 'a' drops.
    """
    ds = DataSet([
        {'id': 1, 'name': 'Apple', 'value': 100},
        {'id': 2, 'name': 'Banana', 'value': 200},
        {'id': 3, 'name': 'Cherry', 'value': 300}])
    ds.columns = (('id', int), ('name', str), ('value', int))

    with caplog.at_level(logging.DEBUG):
        ds.filter_data('a', logkey='name')

    assert 'Filtered name Cherry from dataset' in caplog.text
    assert 'Apple' not in caplog.text
    assert 'Banana' not in caplog.text


def test_filter_callable_with_logkey(caplog):
    """Verify the logkey path evaluates the predicate once per row.

    Mutation: a second `predicate(row)` call to decide the log line in
        filter_data's logkey loop.
    Oracle: a call-counting spy, five calls for five rows.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 50},
        {'id': 3, 'name': 'C', 'value': 150},
        {'id': 4, 'name': 'D', 'value': 75},
        {'id': 5, 'name': 'E', 'value': 200}])
    ds.columns = (('id', int), ('name', str), ('value', int))
    calls = []

    def spy(row):
        calls.append(row['id'])
        return row['value'] > 80

    with caplog.at_level(logging.DEBUG):
        ds.filter_data(spy, logkey='name')

    assert calls == [1, 2, 3, 4, 5]
    assert [r['name'] for r in ds] == ['A', 'C', 'E']
    assert 'Filtered name B from dataset' in caplog.text
    assert 'Filtered name D from dataset' in caplog.text


def test_filter_logkey_column_absent(caplog):
    """Verify a logkey naming an absent column logs None, not KeyError.

    Mutation: `row.get(logkey)` replaced by `row[logkey]` in
        filter_data's debug line.
    Oracle: a logkey column no row carries; the correct run drops the
        row and writes None into the log line.
    """
    ds = DataSet([{'id': 1, 'name': 'A', 'value': 100}])

    with caplog.at_level(logging.DEBUG):
        ds.filter_data('XYZ', logkey='missing')

    assert len(ds) == 0
    assert 'Filtered missing None from dataset' in caplog.text


# --- Summary interaction ---

def test_filter_preserves_summary_row():
    """Verify a declared summary totals the rows left after filtering.

    Mutation: add_summary_row computing the total eagerly rather than
        only storing its arguments, so the total still reads 600.
    Oracle: hand-computed 100, the value of the one row pattern 'A'
        keeps, against 600 for the unfiltered three.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100},
        {'id': 2, 'name': 'B', 'value': 200},
        {'id': 3, 'name': 'C', 'value': 300}])
    ds.columns = (('id', int), ('name', str), ('value', int))
    ds.add_summary_row()

    ds.filter_data('A')

    assert ds.summary['value'] == 100


# --- Converted-copy independence ---

def test_filter_not_inplace_deep_copies_nested_values():
    """Verify a converted dataset's filtered copy owns its nested values.

    Mutation: copy.deepcopy(dict(row)) weakened to copy.copy, leaving
        the copy's rows pointing at the source row's own list objects.
    Oracle: the source's hand-set ['red'], unchanged after appending
        'green' through the copy.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A', 'tags': ['red']},
        {'id': 2, 'name': 'B', 'tags': ['blue']}])
    ds.columns = (('id', int), ('name', str), ('tags', list))
    assert ds[0]['id'] == 1

    filtered = ds.filter_data(lambda row: row['id'] == 1, inplace=False)
    filtered[0]['tags'].append('green')

    assert filtered[0]['tags'] == ['red', 'green']
    assert ds[0]['tags'] == ['red']
    assert ds[1]['tags'] == ['blue']


def test_filter_not_inplace_copy_needs_no_reconversion():
    """Verify the copy of a converted dataset is itself marked converted.

    Mutation: result._types_converted set to False or None, so the
        first read of the copy re-runs convert_container_types over
        rows the deep copy already carried in converted form.
    Oracle: a spy standing in for the copy's convert_container_types,
        which records no call.
    """
    ds = DataSet([
        {'id': '1', 'name': 'A'},
        {'id': '2', 'name': 'B'}])
    ds.columns = (('id', int), ('name', str))
    assert ds[0]['id'] == 1

    filtered = ds.filter_data(lambda row: row['id'] == 1, inplace=False)
    conversions = []
    filtered.convert_container_types = lambda: conversions.append('ran')

    assert [row['id'] for row in filtered] == [1]
    assert conversions == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
