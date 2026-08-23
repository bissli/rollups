"""Coverage for the workbook seam: registration, routing, and errors."""
import io as _io

import pytest
from opendate import Date
from rollups import DataSet, register_excel_backend
from rollups.io import excel_backend
from rollups.io import register_excel_backend as _register


class SpyBackend:
    """Records every call the excel functions route to a backend."""

    def __init__(self, sheets=None):
        self.calls = []
        self.sheets = sheets or {}

    def parse(self, *args, **kwargs):
        self.calls.append(('parse', args, kwargs))
        return self.sheets

    def dataset_to_excel(self, dataset, file_or_name, **kwargs):
        self.calls.append(('dataset_to_excel', dataset, file_or_name))


@pytest.fixture
def spy(monkeypatch):
    """Register a spy backend and restore whatever was there before."""
    import rollups.io as dio
    previous = dio._EXCEL_BACKEND
    backend = SpyBackend({'Sheet1': [{'label': 'a', 'amount': 1}]})
    _register(backend)
    yield backend
    dio._EXCEL_BACKEND = previous


def test_no_backend_registered_raises_a_named_error(monkeypatch):
    """Verify an unregistered backend raises, naming the way to fix it.

    Mutation: excel_backend() answering None on no registration, so the
    caller dies later with AttributeError on NoneType instead.
    Oracle: the exception type, and the registration function named in
    the message.
    """
    import rollups.io as dio
    monkeypatch.setattr(dio, '_EXCEL_BACKEND', None)

    with pytest.raises(RuntimeError) as exc:
        excel_backend()
    assert 'register_excel_backend' in str(exc.value)


def test_from_excel_routes_to_the_backends_parse(spy):
    """Verify from_excel reads through parse and builds a DataSet.

    Mutation: from_excel calling dataset_to_excel, or reaching a
    module-global `excel` name rather than the registered backend.
    Oracle: a hand-written expected call list against the spy, plus the
    row the stub sheet declares.
    """
    ds = DataSet.from_excel('book.xlsx')

    assert [c[0] for c in spy.calls] == ['parse']
    assert ds[0].label == 'a'
    assert ds[0].amount == 1


def test_parse_is_asked_for_a_read_only_workbook(spy):
    """Verify read_only is forced on for every workbook read.

    Mutation: the kwargs.update({'read_only': True}) dropped, so a read
    can write back to the caller's workbook.
    Oracle: the kwargs the spy recorded for its parse call.
    """
    DataSet.from_excel('book.xlsx')

    _, _, kwargs = spy.calls[0]
    assert kwargs['read_only'] is True


def test_from_excel_sheets_yields_one_dataset_per_sheet(spy):
    """Verify each sheet becomes its own named DataSet.

    Mutation: the generator yielding only the first sheet, or yielding
    the raw rows rather than a DataSet.
    Oracle: hand-written sheet names and per-sheet row counts.
    """
    spy.sheets = {'One': [{'a': 1}], 'Two': [{'a': 2}, {'a': 3}]}

    out = dict(DataSet.from_excel_sheets('book.xlsx'))

    assert sorted(out) == ['One', 'Two']
    assert [len(out['One']), len(out['Two'])] == [1, 2]
    assert isinstance(out['One'], DataSet)


def test_write_excel_routes_to_dataset_to_excel_not_parse(spy):
    """Verify the writer calls the write side of the backend.

    Mutation: write_excel calling the backend's parse, which a stub
    that answers a dict would let pass unnoticed.
    Oracle: the recorded call name, and the dataset and target handed
    through unchanged.
    """
    ds = DataSet([{'label': 'a'}])
    ds.columns = [('label', str)]
    target = _io.BytesIO()

    ds.write_excel(target)

    assert [c[0] for c in spy.calls] == ['dataset_to_excel']
    assert spy.calls[0][1] is ds
    assert spy.calls[0][2] is target


def test_register_excel_backend_is_reachable_from_the_package_root(spy):
    """Verify the seam is public, since a caller registers at import.

    Mutation: register_excel_backend left out of __all__, so a caller
    has to reach into rollups.io for it.
    Oracle: identity against the function the io module defines.
    """
    import rollups
    assert 'register_excel_backend' in rollups.__all__
    assert register_excel_backend is _register


def test_write_excel_hands_the_backend_a_working_cell_converter(spy):
    """Verify the default convert_value runs against real cell values.

    Mutation: io.py importing only `Date`, so the isinstance check
    against `Date | DateTime | Time` raises NameError the moment a
    backend calls the converter -- invisible to a spy that never does.
    Oracle: hand-computed renderings -- a date passes through, an
    iterable joins on commas, a builtin is untouched.
    """
    seen = {}

    def capture(dataset, file_or_name, **kwargs):
        convert = kwargs['convert_value']
        row = dataset[0]
        seen['when'] = convert(row, 'when')
        seen['tags'] = convert(row, 'tags')
        seen['amount'] = convert(row, 'amount')

    spy.dataset_to_excel = capture

    ds = DataSet([{'when': Date(2024, 3, 5), 'tags': ['a', 'b'], 'amount': 2}])
    ds.columns = [('when', Date), ('tags', list), ('amount', int)]
    ds.write_excel(_io.BytesIO())

    assert seen['when'] == Date(2024, 3, 5)
    assert seen['tags'] == 'a,b'
    assert seen['amount'] == 2


def test_write_excel_retries_exactly_once_on_an_unwritable_path(spy):
    """Verify the write-retry fires once, not once per decorator layer.

    The retry decorator belongs on the free function only. Leaving it on
    the delegating method too stacks two layers, so a failing write
    makes four attempts and the last one carries a name randomized
    twice.

    Mutation: @on_error_randomize(arg=1) restored on DataSet.write_excel
    while write_excel_file still carries it.
    Oracle: a hand-counted two attempts -- the original path, then one
    randomized sibling that still ends in .xlsx and has no second
    random run spliced into it.
    """
    attempts = []

    def always_fails(dataset, file_or_name, **kwargs):
        attempts.append(str(file_or_name))
        raise OSError('locked')

    spy.dataset_to_excel = always_fails

    ds = DataSet([{'a': 1}])
    ds.columns = [('a', int)]
    with pytest.raises(OSError):
        ds.write_excel('/tmp/dataset-retry-probe.xlsx')

    assert len(attempts) == 2
    assert attempts[0] == '/tmp/dataset-retry-probe.xlsx'
    assert attempts[1] != attempts[0]
    assert attempts[1].endswith('.xlsx')
    assert attempts[1].startswith('/tmp/dataset-retry-probe')


def test_failed_read_logs_the_filename_not_the_class(spy, caplog):
    """Verify the error log names the workbook, not the DataSet class.

    from_excel is a classmethod, so args[0] is the class, not the file.

    Mutation: the wrapper reading args[0] rather than skipping a leading
    class, so every failure logs the same class repr whatever file was
    asked for.
    Oracle: two reads of differently named files -- each message has to
    carry its own name and neither may carry the class name.
    """
    def always_fails(*args, **kwargs):
        raise OSError('locked')

    spy.parse = always_fails

    with caplog.at_level('ERROR', logger='rollups.io'):
        for name in ('alpha.xlsx', 'beta.xlsx'):
            with pytest.raises(OSError):
                DataSet.from_excel(name)

    headers = [line
               for record in caplog.records
               for line in record.message.splitlines()
               if line.startswith('Excel parsing failed for file:')]
    assert headers == ['Excel parsing failed for file: alpha.xlsx',
                       'Excel parsing failed for file: beta.xlsx']
