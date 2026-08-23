"""Coverage for the csv writer and the retry decorator it carries.
"""
import io

import pytest
from rollups import DataSet, on_error_randomize


@pytest.fixture
def two_rows():
    """Two rows keyed in the reverse of the declared column order."""
    ds = DataSet([
        {'amount': 1.5, 'group': None, 'label': 'a'},
        {'amount': None, 'group': 'g', 'label': 'b'},
    ])
    ds.columns = [('label', str), ('group', str), ('amount', float)]
    return ds


def test_write_csv_orders_fields_by_columns_not_by_row_key_order(two_rows):
    """Verify each field is placed by self.columns, not by row key order.

    Mutation: the row loop writing `row.values()` rather than
    `[_format(row, c, t) for c, t in self.columns]`, which agrees with
    the header only while a row happens to be keyed in column order.
    Oracle: hand-written csv text, whose rows are the reverse of the
    order the fixture's dicts are keyed in.
    """
    buf = io.StringIO()
    two_rows.write_csv(buf)
    assert buf.getvalue() == 'label,group,amount\r\na,,1.5\r\nb,g,\r\n'


def test_write_csv_header_false_omits_only_the_header(two_rows):
    """Verify header=False drops the header and keeps every data row.

    Mutation: the `if header:` guard dropped, so the header is always
    written; or the guard inverted, so the rows are lost with it.
    Oracle: the same hand-written csv text less its first line.
    """
    buf = io.StringIO()
    two_rows.write_csv(buf, header=False)
    assert buf.getvalue() == 'a,,1.5\r\nb,g,\r\n'


def test_write_csv_format_hooks_rewrite_values_and_labels(two_rows):
    """Verify the format and format_label kwargs replace the defaults.

    Mutation: kwargs.pop('format') ignored, so the default formatter
    runs and the hook is silently dead.
    Oracle: hand-written csv text under an upper-casing label hook and
    a value hook that answers a constant.
    """
    buf = io.StringIO()
    two_rows.write_csv(
        buf,
        format=lambda row, col, typ: 'X',
        format_label=lambda col: col.upper())
    assert buf.getvalue() == 'LABEL,GROUP,AMOUNT\r\nX,X,X\r\nX,X,X\r\n'


def test_write_csv_to_a_path_matches_the_buffer(two_rows, tmp_path):
    r"""Verify the path branch writes the same bytes as the buffer branch.

    Mutation: the path branch opening in text mode without newline='',
    or writing nothing at all because the branch tests the wrong type.
    Oracle: hand-written bytes, which pin the \r\n line terminator
    directly rather than against the other branch.
    """
    target = tmp_path / 'out.csv'
    two_rows.write_csv(str(target))

    assert target.read_bytes() == b'label,group,amount\r\na,,1.5\r\nb,g,\r\n'


def test_write_csv_raises_and_names_the_row_it_could_not_write():
    """Verify a formatter that raises propagates rather than truncating.

    Mutation: the except block swallowing the error instead of
    re-raising, leaving a silently short file.
    Oracle: the raised exception itself, and a buffer holding only the
    header.
    """
    ds = DataSet([{'label': 'a'}])
    ds.columns = [('label', str)]
    buf = io.StringIO()

    def explode(row, col, typ):
        raise ValueError('no')

    with pytest.raises(ValueError):
        ds.write_csv(buf, format=explode)
    assert buf.getvalue() == 'label\r\n'


def test_on_error_randomize_retries_the_named_argument_only():
    """Verify the retry randomizes the path argument, keeping its suffix.

    Mutation: arg=1 read as arg=0, so the retry rewrites the dataset
    argument and the path is passed through unchanged.
    Oracle: two recorded calls -- the first the original path, the
    second a different name under the same directory and extension.
    """
    seen = []

    @on_error_randomize(arg=1)
    def write(dataset, path):
        seen.append((dataset, path))
        if len(seen) == 1:
            raise OSError('locked')
        return path

    result = write('DS', '/tmp/report.csv')

    assert len(seen) == 2
    assert seen[0] == ('DS', '/tmp/report.csv')
    assert seen[1][0] == 'DS'
    assert seen[1][1] != '/tmp/report.csv'
    assert seen[1][1].startswith('/tmp/report')
    assert seen[1][1].endswith('.csv')
    assert result == seen[1][1]


def test_on_error_randomize_reraises_when_there_is_no_filename():
    """Verify a call with no path argument re-raises instead of retrying.

    Notes
    -----
    - A real buffer target does NOT reach this branch: `args[1]` is the
      buffer, which is truthy, so the decorator stringifies it and
      retries against nonsense. That is pre-existing behavior, not
      something this test pins.

    Mutation: the `if not filename: raise` guard dropped, so the retry
    runs against a randomized `None` and raises IndexError rather than
    surfacing the original OSError.
    Oracle: the OSError escaping, and exactly one recorded call.
    """
    seen = []

    @on_error_randomize(arg=1)
    def write(dataset):
        seen.append(dataset)
        raise OSError('locked')

    with pytest.raises(OSError):
        write('DS')
    assert len(seen) == 1
