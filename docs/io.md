# Reading and writing

Three formats: csv and json are built in, excel goes through a backend
a caller registers.

## csv

```python
rows = DataSet.read('input.csv')      # from_csv is the same method
rows.write_csv('output.csv')
```

Both take a path or an already-open handle, so a `StringIO` works
wherever a filename does.

### Type suffixes in the header

A header field may carry a type suffix, which is how a csv file
declares its own column types:

| Suffix | Type    |
| ------ | ------- |
| `:s`   | `str`   |
| `:i`   | `int`   |
| `:f`   | `float` |
| `:b`   | `bool`  |
| `:d`   | `Date`  |

```
name:s,age:i,score:f,ok:b,when:d
ana,30,1.5,True,2024-03-05
```

A field with no suffix reads as `str`. The suffix is stripped from the
column name, so `age:i` becomes the column `age`.

### What reading tolerates

- An empty field becomes `None`, not `''`.
- A blank line is skipped.
- **A row that will not decode is logged and skipped rather than
  raising**, so one bad line does not lose the file. Check the row count
  where that matters.

### Reader options

Anything else is passed to `csv.reader`, so `delimiter='\t'` and the
rest work. Two options are handled here instead:

```python
DataSet.read('input.csv', skips=1)
DataSet.read('input.csv',
             rename_fields=lambda fields: [f.lower() for f in fields])
```

`skips` drops that many rows before the header is read - for a file
carrying a title line above its column names.

`rename_fields` takes the whole header row as a list and returns the
list of names to use - not one field at a time, so `str.lower` will not
serve.
It sees the header **before** the type suffix is split off, so a renamer
has to carry the suffix through or the column falls back to `str`. It is
also the answer to a source with repeated header names, which would
otherwise collide with the last one winning.

### Writer options

```python
rows.write_csv('out.csv', header=False)
rows.write_csv('out.csv', format=my_value_fn, format_label=str.upper)
```

`header` writes the column names, on by default. `format` and
`format_label` override how a value and a column name are rendered. A
`None` value is written as an empty field.

Where the path cannot be written, a randomized name is tried once
before giving up.

## json

```python
rows.json(raw=True)     # '[{"a": 1, "b": "x"}]'
rows.json()             # '{"order": [...], "types": [...], "data": [...]}'
```

`raw=True` emits a bare array of row objects. The default emits an
object carrying `order`, `types` and `data`, which is what lets the
column names and types survive a round trip.

```python
rows.json(columns=['a'])                 # only these columns
rows.json(format_value=fn)               # fn(row, column, type)
rows.json(generated_at='2024-03-05')     # extra keys, ignored when raw
```

Dates, datetimes and times are written in ISO form.

Reading is the mirror. By default `from_json` reads whichever of the two
shapes it is handed and answers a DataSet, so the obvious round trip
works:

```python
DataSet.from_json(rows.json())                    # a DataSet, types intact
DataSet.from_json('[{"a": 1}]')                   # a DataSet
rows, extra = DataSet.from_json(text, raw=False)  # a pair
```

Name `raw` to fix the shape instead: `raw=True` reads a bare array,
`raw=False` reads the object and answers a pair whose second item holds
every key but `data`, `order` and `types` - whatever was folded in on
the way out.

A declared type wins over the parsed one, so `types` of `['int']` reads
`2.0` back as `2`. Where the payload declares no types, an ISO date
string reads back as a date; where it declares them, the declared type
is the only converter, so a column declared `str` keeps its strings
whatever they look like.

## excel

This package never imports a workbook library. Register a backend and
the excel methods route to it:

```python
import rollups

rollups.register_excel_backend(my_excel_module)
```

```python
rows = DataSet.from_excel('book.xlsx')
for name, sheet in DataSet.from_excel_sheets('book.xlsx'):
    ...
rows.write_excel('out.xlsx')
```

Until a backend is registered, all three raise `RuntimeError` naming the
call that fixes it. Everything else in the package works without one.

The backend is any object carrying a `parse` and a `dataset_to_excel`
attribute - a module of two functions does as well as an instance.
[Extending](extending.md) carries the signatures.

`from_excel_sheets` is a generator, so it raises on iteration, not on
the call.

### Cell conversion

The writer hands the backend a `convert_value` that renders each cell:
a date, time or builtin passes through unchanged, an iterable is joined
with commas, and anything else is parsed as a number where it can be,
else rendered as a `str`.

Give a class a true `render_as_str` attribute to skip that number parse,
which is how a class whose text form only looks numeric keeps its own
spelling. Pass a different `convert_value` to replace the rule.

See [Extending](extending.md) for writing a backend.

## Next

- [Summaries and output](summaries.md) - rendering what was read
- [Extending](extending.md) - writing a workbook backend
- [Architecture](architecture.md) - where the IO layer sits
