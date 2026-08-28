# rollups

An iterable of dictionaries, with typed columns.

`DataSet` is a list of dictionary rows. Each row is an ordinary dict
addressed by field name, and each column declares a Python type that its
values convert to on first read. The container adds filtering, sorting,
grouping, joining, pivoting, and summarizing.

A DataFrame stores each column as an array, so a row is a view built on
demand and operations are written as vectors and masks. A `DataSet`
stores each row as a dict, so `for row in rows` reads the stored object.

Use a DataFrame for rolling windows, matrix operations, and large row
counts. Use a `DataSet` for records that arrive as dictionaries - a
database cursor, a json payload, a csv - and for SQL-shaped operations
on them. `dataframe()` and `from_dataframe()` convert between the two.

```python
from rollups import DataSet

rows = DataSet([
    {'name': 'ana', 'group': 'a', 'amount': 120.5},
    {'name': 'bo',  'group': 'b', 'amount':  80.0},
    {'name': 'cy',  'group': 'a', 'amount':  45.25},
    ])
rows.columns = [('name', str), ('group', str), ('amount', float)]

for row in rows:
    if row.amount > 100:                       # a float, parsed on read
        print(f'{row.name} is over by {row.amount - 100:.2f}')

rows[0].name                       # 'ana'
rows[0]['name']                    # 'ana' - attribute or key
rows.sort_data('-amount')          # SQL order by, descending
rows.bucket('group', ['amount'])   # a: 165.75, b: 80.0
```

## Install

From [PyPI](https://pypi.org/project/rollups/):

```
pip install rollups
```

Python 3.11 or later. It depends on `pandas`, `libb-util`, `opendate`,
and `prettytable`.

Source: [github.com/bissli/rollups](https://github.com/bissli/rollups).

To work on it:

```
git clone https://github.com/bissli/rollups
cd rollups
poetry install --extras test
python -m pytest -q
```

## Typed columns

A column declares the type its values convert to, so a `'120.5'` read
from a csv comes back as a `float`. The rows remain dicts. The types are
metadata on the container.

A column is a `(name, type)` pair, declared or inferred:

```python
records = [{'name': 'ana', 'amount': '120.5'}]

DataSet(records, columns=[('name', str), ('amount', float)])
DataSet(records)                             # types inferred from the rows
DataSet(records, infer_numeric_strings=True)   # '120.5' infers as float
```

Values convert as they enter - at construction, and on `append` and
`extend` - so a value that does not match its column raises at the line
that built the dataset rather than at some later reader. Any class
taking one argument serves as a column type.

[Getting started](docs/getting-started.md) covers the type system, the
column operations, and the three kinds of copy.

## What it does

| Area           | Calls                                                                              |
| -------------- | ---------------------------------------------------------------------------------- |
| Build          | `DataSet(...)`, `from_list`, `from_empty`, `from_dataframe`                        |
| Read and write | `read` / `from_csv`, `write_csv`, `json`, `from_json`, `from_excel`, `write_excel` |
| Columns        | `add_column`, `remove_column`, `rename_column`, `cols`, `typs`, `colmap`           |
| Rows           | `append`, `extend`, `pop`, `filter_data`, `dedupe`, `sample`, `partition`          |
| Order          | `sort`, `sort_data`, `order`, `reverse`                                            |
| Reshape        | `bucket`, `pivot`, `flatten`, `transpose`, `unwind`                                |
| Combine        | `join`, `diff`, `meld_datasets`, `match_rows`                                      |
| Series         | `shift`, `backfill`, `pct_change`                                                  |
| Present        | `summary`, `add_summary_row`, `add_summary_column`, `to_string`             |
| Screen         | `apply_screen`                                                                     |

Every one of these is documented under [docs/](docs/README.md).

## Grouping

`bucket` is the SQL `GROUP BY`, and takes any callable as the
aggregation:

```python
rows.bucket('group', ['amount'])                 # sum, skipping None
rows.bucket('group', [('amount', max)])          # any callable
rows.bucket('group', [('amount', sum, 'total')]) # name the result
rows.bucket([], ['amount'])                      # one row, everything
```

`pivot` turns a column's values into columns of their own, `flatten`
reverses that, and `transpose` swaps rows for columns. See
[Aggregating](docs/aggregation.md).

## Joining

```python
DataSet.join(left, 'key', right, 'key', 'left', bcol=['amount'])
```

Four join types, key columns that may be named differently on each
side, per-side column selection and renaming, and a `first` flag for
one-to-one matching. Where both sides carry a column, the left value
wins unless it is `None`. See [Joining](docs/joins.md).

## Screening

A screen is a query language with one query per column. A screen is
text, so it can come from a config file or a web form:

```python
from rollups import apply_screen

apply_screen(rows, {'group': 'a|b', 'amount': '>50,<200'})
# rows is filtered in place
```

Comparison operators, regex search, `None` handling, and references to
another column in the same row. See [Screening](docs/screening.md).

## Reading and writing

```python
rows = DataSet.read('input.csv')      # type suffixes: name:s, age:i
rows.write_csv('output.csv')
rows.json(raw=True)                   # '[{"name": "ana"}]'
```

A csv header field may carry a type suffix - `name:s`, `age:i`,
`score:f`, `on:b`, `when:d` - and a field without one reads as `str`. A
line that will not decode is logged and skipped, and the read
continues.

Excel goes through a backend the caller registers. This package imports
no workbook library:

```python
import rollups

rollups.register_excel_backend(my_excel_module)
```

See [Reading and writing](docs/io.md).

## Presenting

```python
totals = DataSet([
    {'name': 'ana', 'amount': 120.5},
    {'name': 'bo',  'amount':  80.0},
    ], columns=[('name', str), ('amount', float)])
totals.add_summary_row(label='Total')
print(totals.to_string())
```

```
+-------+--------+
|  name | amount |
+-------+--------+
|  ana  | 120.50 |
|   bo  | 80.00  |
+-------+--------+
| Total | 200.50 |
+-------+--------+
```

The summary recomputes on every read, so filtering the rows and reading
again gives the current total. See [Summaries and output](docs/summaries.md).

## Working with pandas

`DataSet.dataframe()` and `DataSet.from_dataframe()` convert between the
two representations, so a columnar step can run inside row-shaped work.

`rollups.frame` holds four functions that take and return
`pandas.DataFrame` and do not reference `DataSet`:

```python
from rollups import bucket_dataframe, join_dataframes
```

They answer differently from the `DataSet` versions on the same inputs:
a malformed key raises where `DataSet` matches nothing silently, `inner`
keeps left row order, and no result column is coerced to a
reconstructed type.
[DataFrame-native join and group-by](docs/dataframe-native.md) tabulates
every difference.

## Extending

Subclass `DataSet`, and every operation returns the subclass:

```python
class Report(DataSet):

    def totals(self):
        return self.bucket([], [c for c, t in self.columns if t is float])
```

Every algorithm behind a method is also an exported function, so
`bucket_dataset(rows, ...)` and `rows.bucket(...)` do the same thing.
See [Extending](docs/extending.md) for the class-propagation rule, the
workbook backend, and custom column types.

## Documentation

| Guide                                        | Covers                                                       |
| -------------------------------------------- | ------------------------------------------------------------ |
| [Getting started](docs/getting-started.md)   | building a dataset, columns, the type system, copying             |
| [Rows](docs/rows.md)                         | adding, removing, ordering, filtering, and series operations |
| [Screening](docs/screening.md)               | the query language for filtering by column                   |
| [Joining](docs/joins.md)                     | the four join types, `first`, diffing and melding              |
| [Aggregating](docs/aggregation.md)           | `bucket`, `pivot`, `flatten`, `transpose`                    |
| [Reading and writing](docs/io.md)            | csv, json, and the excel backend                             |
| [Summaries and output](docs/summaries.md)    | summary rows, table rendering, paging                        |
| [DataFrame-native](docs/dataframe-native.md) | the four functions that take and return frames               |
| [Architecture](docs/architecture.md)         | the modules, the layering, where a new function goes         |
| [Extending](docs/extending.md)               | subclassing, the backend registry, custom types              |

## Development

```
python -m pytest -q          # the suite
ruff check src/rollups/     # lint
bump2version patch           # never hand-edit the version
```

## License

MIT. See [LICENSE](LICENSE).
