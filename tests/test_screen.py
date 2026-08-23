"""Screen query parsing and dataset screening.

Covers the screen helpers that live in screen.py: query-string
parsing, column-reference resolution, comparison, and the screening
pass that applies a filter dict to a DataSet.
"""
import json
import operator

import pytest
from rollups import DataSet, apply_screen, get_or_val, interpret_screen
from rollups import matches

import libb


def test_interpret_screen_comparison_operators():
    """Verify each comparison token maps to its operator and numeric value.

    Mutation: swapping gt for lt, or mapping '<>' to eq instead of ne.
    Oracle: hand-built (operator, value) pairs per token.
    """
    assert interpret_screen('>3.5') == [(operator.gt, 3.5, libb.safe_mult, 1)]
    assert interpret_screen('<=-2') == [(operator.le, -2.0, libb.safe_mult, 1)]
    assert interpret_screen('=3.2') == [(operator.eq, 3.2, libb.safe_mult, 1)]
    assert interpret_screen('<>Magenta') == [
        (operator.ne, 'Magenta', libb.safe_mult, 1)]


def test_interpret_screen_splits_on_comma():
    """Verify a comma-separated query yields one condition per clause.

    Mutation: parsing only the first clause, or splitting on whitespace.
    Oracle: hand-counted three clauses with their own operators.
    """
    result = interpret_screen('  >3.5  ,\t<=-2 ,=3.2  , XX')
    assert len(result) == 4
    assert [cmp for cmp, _, _, _ in result] == [
        operator.gt, operator.le, operator.eq, None]


def test_interpret_screen_bare_term_is_regex():
    """Verify a term with no comparison token parses with a None operator.

    Mutation: defaulting a bare term to eq, which would turn the regex
        path into an equality test and break alternation like 'US|GR'.
    Oracle: hand-computed None operator with the term left intact.
    """
    assert interpret_screen('XX') == [(None, 'XX', libb.safe_mult, 1)]
    assert interpret_screen('US|GR') == [(None, 'US|GR', libb.safe_mult, 1)]


def test_interpret_screen_arithmetic_suffix():
    """Verify a trailing arithmetic term sets the operator and operand.

    Mutation: dropping the operand so the default multiplier 1 is used,
        which silently turns '*0.03' into a no-op comparison.
    Oracle: hand-computed safe_mult/0.03 and safe_add/10 pairs.
    """
    assert interpret_screen('>=_capacity*0.03') == [
        (operator.ge, '_capacity', libb.safe_mult, 0.03)]
    assert interpret_screen('>=_baseline+10') == [
        (operator.ge, '_baseline', libb.safe_add, 10)]


def test_interpret_screen_preserves_non_numeric_value():
    """Verify a value that only looks numeric stays a string.

    Mutation: coercing every value with float(), which turns the
        grade 'B-' into a parse failure or a bare number.
    Oracle: hand-computed 'B-' retained as str, and an embedded-space
        category kept whole after stripping.
    """
    assert interpret_screen('=B-') == [(operator.eq, 'B-', libb.safe_mult, 1)]
    result = interpret_screen('<> Alpha Beta Gamma (ABG) ')
    assert result == [
        (operator.ne, 'Alpha Beta Gamma (ABG)', libb.safe_mult, 1)]


def test_interpret_screen_empty_input():
    """Verify an empty or non-string query parses to None, not a clause.

    Mutation: returning an empty list, which apply_screen would iterate
        as zero conditions and silently keep every row, or dropping the
        isinstance guard so a numeric query reaches .strip().
    Oracle: hand-computed None for each input the parser cannot use.
    """
    assert interpret_screen('') is None
    assert interpret_screen(None) is None
    assert interpret_screen(3.5) is None


def test_get_or_val_resolves_leading_underscore_to_column():
    """Verify a leading underscore reads the named column from the row.

    Mutation: returning the token unchanged, which would compare against
        the literal string '_score' instead of the row's value.
    Oracle: hand-built row whose score is 24.0.
    """
    row = {'score': 24., 'score_limit': 33.5}
    assert get_or_val('_score', row) == 24.0


def test_get_or_val_passes_through_plain_value():
    """Verify a value with no leading underscore is returned as-is.

    Mutation: stripping the first character unconditionally, which would
        turn the threshold 30.0 into a column lookup and yield None.
    Oracle: hand-computed identity for a float and a plain string.
    """
    assert get_or_val(30.0, {'a': 1}) == 30.0
    assert get_or_val('Magenta', {'a': 1}) == 'Magenta'


def test_matches_regex_path_when_operator_is_none():
    """Verify a None operator searches the value as a regex, not equality.

    Mutation: falling back to equality, which would fail the partial
        match 'fo' against 'foo'.
    Oracle: hand-computed partial hit and miss.
    """
    assert matches(None, 'foo', 'fo') is True
    assert matches(None, 'foo', 'zz') is False


def test_matches_operator_path():
    """Verify a supplied operator compares directly.

    Mutation: ignoring the operator and regex-searching instead.
    Oracle: hand-computed pairs, including a prefix case that the two
        paths disagree on -- 'fo' is equal to 'foo' under neither operator
        but does match it as a regex.
    """
    assert matches(operator.eq, 2, 2) is True
    assert matches(operator.eq, 2, 4) is False
    assert matches(operator.eq, 'foo', 'fo') is False


def test_matches_coerces_numeric_string():
    """Verify a numeric string is parsed before an operator comparison.

    Mutation: dropping the libb.parse coercion, which makes the int 2
        and the string '2' unequal and silently drops matching rows.
    Oracle: hand-computed 2 == '2' true, 2 == 'foo' false.
    """
    assert matches(None, 2, '2') is True
    assert matches(operator.eq, 2, '2') is True
    assert matches(operator.eq, 2, 'foo') is False


def test_matches_renders_bool_as_yes_no():
    """Verify a bool is compared as its Yes/No rendering.

    Mutation: comparing the raw bool, under which 'Yes' never matches
        True and every boolean screen returns nothing.
    Oracle: hand-computed True -> 'Yes', False -> 'No'.
    """
    assert matches(operator.eq, True, 'Yes') is True
    assert matches(operator.eq, False, 'No') is True
    assert matches(operator.eq, True, 'No') is False


@pytest.mark.parametrize(('screen', 'expected_ids'), [
    ('{"x": "<>bar", "y": "=1"}', [1]),
    ('{"x": "<>foo", "y": ">=2", "z": "<>0"}', [2, 5, 6]),
    ('{"x": "<>foo", "y": "<4,>2", "z": "<>None"}', [2, 6]),
    ('{"x": "<>foo", "y": "<=4", "z": "<>None, <=_y-1"}', [2]),
    ])
def test_apply_screen_filters_rows(screen, expected_ids):
    """Verify every column and every clause narrows the dataset.

    Mutation: applying only the first column of the filter dict, OR-ing
        the clauses of one column instead of AND-ing them, or dropping the
        arithmetic suffix from the '<=_y-1' clause.
    Oracle: hand-computed surviving ids over rows built so that each
        clause drops a row no other clause drops.
    """
    ds = DataSet([
        {'id': 1, 'x': 'foo', 'y': 1, 'z': None},
        {'id': 2, 'x': 'bar', 'y': 3, 'z': 2},
        {'id': 3, 'x': 'bar', 'y': 1, 'z': 2},
        {'id': 4, 'x': 'baz', 'y': 5, 'z': 0},
        {'id': 5, 'x': 'bar', 'y': 3, 'z': None},
        {'id': 6, 'x': 'bar', 'y': 3, 'z': 3},
        ])
    apply_screen(ds, json.loads(screen))
    assert [row['id'] for row in ds] == expected_ids


def test_apply_screen_skips_unknown_column():
    """Verify a screen naming a missing column leaves the dataset intact.

    Mutation: raising KeyError, or filtering every row out, when a saved
        screen references a column the current dataset does not carry.
    Oracle: hand-computed full 2-row dataset retained.
    """
    ds = DataSet([{'x': 'foo', 'y': 1}, {'x': 'bar', 'y': 3}])
    apply_screen(ds, {'nosuchcol': '>0'})
    assert [dict(row) for row in ds] == [
        {'x': 'foo', 'y': 1}, {'x': 'bar', 'y': 3}]


def test_apply_screen_skips_empty_screen():
    """Verify an empty screen value is a no-op rather than a match-none.

    Mutation: treating '' as a failed parse and dropping every row.
    Oracle: hand-computed full 2-row dataset retained.
    """
    ds = DataSet([{'x': 'foo', 'y': 1}, {'x': 'bar', 'y': 3}])
    apply_screen(ds, {'x': ''})
    assert [dict(row) for row in ds] == [
        {'x': 'foo', 'y': 1}, {'x': 'bar', 'y': 3}]


def test_interpret_screen_percent_value_drops_the_sign():
    """Verify a percent value parses to the bare number, sign removed.

    Mutation: slicing the wrong end of the value ('50%'[:1]) or the
        wrong width ('50%'[:-2]), each of which yields 5.0 for 50 percent,
        or dropping the percent branch so the value stays a string.
    Oracle: hand-computed 50.0 and 12.5 from the written percentages.
    """
    assert interpret_screen('>50%') == [(operator.gt, 50.0, libb.safe_mult, 1)]
    assert interpret_screen('<=12.5%') == [
        (operator.le, 12.5, libb.safe_mult, 1)]


def test_interpret_screen_null_token_becomes_none():
    """Verify the 'null' spelling parses to None, whatever its case.

    Mutation: dropping 'null' from the null-token set, which leaves the
        literal string 'null' as the compared value so a null screen never
        reaches an empty cell.
    Oracle: hand-computed None, cross-checked against the 'none'
        spelling that stands for the same thing.
    """
    assert interpret_screen('=null') == [
        (operator.eq, None, libb.safe_mult, 1)]
    assert interpret_screen('<>NULL') == [
        (operator.ne, None, libb.safe_mult, 1)]
    assert interpret_screen('=none') == [
        (operator.eq, None, libb.safe_mult, 1)]


def test_interpret_screen_division_suffix():
    """Verify a '/' suffix parses to safe_divide and divides on apply.

    Mutation: dropping '/' from the operator table, which leaves a None
        operator that apply_screen then tries to call.
    Oracle: hand-computed (safe_divide, 2) pair, plus the ids of the
        rows whose value clears half the limit.
    """
    assert interpret_screen('>=_limit/2') == [
        (operator.ge, '_limit', libb.safe_divide, 2)]
    ds = DataSet([
        {'id': 1, 'value': 60, 'limit': 100},
        {'id': 2, 'value': 40, 'limit': 100},
        {'id': 3, 'value': 30, 'limit': 50},
        ])
    apply_screen(ds, {'value': '>=_limit/2'})
    assert [row['id'] for row in ds] == [1, 3]


def test_matches_regex_path_ignores_case():
    """Verify the regex path matches whatever the case on either side.

    Mutation: dropping the re.IGNORECASE flag, under which a screen
        typed in lower case misses every capitalized group name.
    Oracle: hand-computed hits for both casings, and a miss for a
        group sharing no prefix.
    """
    assert matches(None, 'Magenta', 'mage') is True
    assert matches(None, 'magenta', 'MAGE') is True
    assert matches(None, 'Crimson', 'mage') is False


def test_matches_renders_screen_side_bool_as_yes_no():
    """Verify a bool on the screen side is rendered before comparing.

    Mutation: leaving the screen-side bool alone, or rendering True as
        'yes' rather than 'Yes', so a stored 'Yes' never matches a screen
        that points at a boolean column.
    Oracle: hand-computed True -> 'Yes' and False -> 'No', with a
        crossed pair proving the rendering is not a constant.
    """
    assert matches(operator.eq, 'Yes', True) is True
    assert matches(operator.eq, 'No', False) is True
    assert matches(operator.eq, 'Yes', False) is False


def test_matches_coerces_numeric_row_value():
    """Verify a numeric string on the row side is parsed before compare.

    Mutation: skipping the parse of the row value, or bailing out once
        it does parse, either of which drops every row whose number arrived
        as text.
    Oracle: hand-computed 2 == '2' and 2 < 10, with the row side
        written as a string.
    """
    assert matches(operator.eq, '2', 2) is True
    assert matches(operator.lt, '2', 10) is True


def test_matches_unparsable_row_value_fails_the_screen():
    """Verify a row value that will not parse loses a numeric screen.

    Mutation: returning True when the row side will not parse, which
        keeps every 'n/a' cell in a numeric screen.
    Oracle: hand-computed False for 'n/a' under both an equality and an
        ordering comparison.
    """
    assert matches(operator.eq, 'n/a', 2) is False
    assert matches(operator.lt, 'n/a', 2) is False


def test_get_or_val_strips_only_the_underscore():
    """Verify only the leading underscore is cut from a reference.

    Mutation: widening the strip set past '_', which eats the leading
        letters of a column such as AVAL and resolves to the wrong column.
    Oracle: hand-built row carrying both AVAL and the VAL the wider
        strip would land on.
    """
    assert get_or_val('_AVAL', {'AVAL': 0.12, 'VAL': 0.09}) == 0.12


def test_apply_screen_empty_screen_does_not_stop_later_columns():
    """Verify a skipped empty screen still lets later columns filter.

    Mutation: breaking out of the filter loop on an empty screen rather
        than moving to the next column, which ignores every screen typed
        after a blank one.
    Oracle: hand-computed ids of the rows with y > 2.
    """
    ds = DataSet([
        {'id': 1, 'x': 'foo', 'y': 1},
        {'id': 2, 'x': 'bar', 'y': 3},
        {'id': 3, 'x': 'baz', 'y': 5},
        ])
    apply_screen(ds, {'x': '', 'y': '>2'})
    assert [row['id'] for row in ds] == [2, 3]


def test_apply_screen_unknown_column_does_not_stop_later_columns():
    """Verify a screen on a missing column still lets later ones run.

    Mutation: breaking out of the filter loop on an unmatched column
        rather than moving to the next, which drops every later screen once
        a saved query names a column the dataset no longer carries.
    Oracle: hand-computed ids of the rows with y > 2.
    """
    ds = DataSet([
        {'id': 1, 'x': 'foo', 'y': 1},
        {'id': 2, 'x': 'bar', 'y': 3},
        {'id': 3, 'x': 'baz', 'y': 5},
        ])
    apply_screen(ds, {'nosuchcol': '>0', 'y': '>2'})
    assert [row['id'] for row in ds] == [2, 3]


def test_interpret_screen_parses_not_equal():
    """Verify '!=' parses to the ne operator, matching '<>'.

    Mutation: leaving '!' out of the comparison charset, so '!=' parses
        to a None operator with an empty pattern.
    Oracle: the ne operator and pattern 'Magenta', the same '<>' gives.
    """
    assert interpret_screen('!=Magenta')[0][:2] == (operator.ne, 'Magenta')
    assert interpret_screen('<>Magenta')[0][:2] == (operator.ne, 'Magenta')


def test_apply_screen_not_equal_drops_matching_rows():
    """Verify a '!=' screen drops the rows that equal the value.

    Mutation: '!=' parsing to a None operator, which raises downstream
        rather than filtering.
    Oracle: hand-computed survivors Crimson and Indigo, Magenta gone.
    """
    ds = DataSet([
        {'group': 'Crimson'},
        {'group': 'Magenta'},
        {'group': 'Indigo'},
        ])

    apply_screen(ds, {'group': '!=Magenta'})

    assert [row['group'] for row in ds] == ['Crimson', 'Indigo']


if __name__ == '__main__':
    pytest.main([__file__])
