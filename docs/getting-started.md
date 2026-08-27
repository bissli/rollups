# Getting started

## Building a dataset

The constructor takes rows as dictionaries:

```python
from rollups import DataSet

rows = DataSet([
    {'name': 'ana', 'group': 'a', 'amount': 120.5},
    {'name': 'bo',  'group': 'b', 'amount':  80.0},
    ])
```

Five ways in, each for a different shape of input:

| Call                                  | Takes                                         |
| ------------------------------------- | --------------------------------------------- |
| `DataSet(container)`                  | dictionaries, or another `DataSet` to copy    |
| `DataSet.from_list(rows, cols, typs)` | tuples plus the names and types to give them  |
| `DataSet.from_empty(columns)`         | nothing - one row of empty values per column  |
| `DataSet.from_dataframe(df)`          | a `pandas.DataFrame`                          |
| `DataSet.read(path)`                  | a csv file - see [Reading and writing](io.md) |

```python
DataSet.from_list([('ana', 120.5)], ['name', 'amount'], [str, float])
DataSet.from_empty([('name', str), ('amount', float)])   # '' and 0.0
```

`from_empty` gives `''` for str, `0` for int, `0.0` for float, and
`None` for anything else - so a bool column starts at `None`, not
`False`.

## Columns

A column is a `(name, type)` pair. Declare them, or let the constructor
infer them by scanning the rows:

```python
records = [{'name': 'ana', 'amount': '120.5'}]

DataSet(records, columns=[('name', str), ('amount', float)])
DataSet(records)                              # amount infers as str
DataSet(records, infer_numeric_strings=True)  # amount infers as float
```

Inference reads the type set of each column, not just its first value.
A column takes the one type its values share, or that family's widest
member where they span one - `int` beside `float` gives `float`, and
`datetime.date` beside `Date` gives `Date`. Any other mix gives
`object`:

```python
DataSet([{'x': 1}, {'x': 2.5}])     # x infers as float
DataSet([{'x': 1}, {'x': 'n/a'}])   # x infers as object, both kept
```

`scan_limit` caps how many rows that reads, at 1000 by default. It is a
sample, so a disagreeing value below the cap still raises
`ConversionError` on the way in; pass `scan_limit=None` to read every
row where that matters more than the cost of the pass.

`cols` and `typs` are an alternative to `columns`, taking the two halves
separately. All three read back:

```python
rows.columns    # [('name', str), ('amount', float)]
rows.cols       # ['name', 'amount']
rows.typs       # [str, float]
rows.colmap     # {'name': str, 'amount': float}
```

`exemplar` picks which row supplies the column names and starts type
inference. It defaults to the first.

### Changing the columns

```python
rows.add_column('doubled', float, value=lambda row: row['amount'] * 2)
rows.add_column('tag', str, value='x')          # the same value everywhere
rows.add_column('seq', int, values=[1, 2])      # one value per row
rows.rename_column('amount', 'total')
rows.remove_column('tag')
```

`value` takes a callable applied per row, or a plain value written into
every row. `values` takes one value per row, positionally.

`index` sets the position. Without it the column is appended - and that
holds for a column already present, which is dropped from its old
position and re-added at the end. Pass `index` to keep it where it was.

`rename_column` and `remove_column` are no-ops on a name that is not
there, logged at debug rather than raised.

## Types

Values convert to their column's type **as they enter** - in the
constructor, in `append` and in `extend`. A value that does not convert
raises `ConversionError` at the line that supplied it, rather than at
whatever reads the dataset next.

```python
rows = DataSet([{'n': '42'}], columns=[('n', int)])
rows.container[0]['n']    # 42, an int - already converted
```

`ensure_types()` is what a reader calls to be sure. It is a no-op for
data that entered through those three, and earns its keep after a
column is RE-DECLARED, which no entry point sees. It compares the
declared columns against the ones the last pass ran on, so it catches a
re-declaration whichever writer made it.
`convert_container_types()` is the pass it runs, and converts
unconditionally - the call to re-run conversion after rows have been
edited in place.

`DataSet.guess_columns(rows)` is the inference the constructor uses. Call
it directly to see what a set of rows infers to before committing to
it.

Declare a column `object` where its type is unknown, or where the
values under it are of several types - which is what `flatten` does for
its `val` column, and what inference answers for a column no single
type covers. Every value is an instance of `object`, so such a column
converts nothing and rejects nothing.

### What converts

`int` and `float` go through a permissive numeric parse, so `'1,200'`
reads as `1200` and the accounting negative `'(1.5)'` reads as `-1.5`.
`Date`, `DateTime` and `Time` parse from strings and from each other.

Anything else falls through to `typ(val)`. That means **any class taking
one argument works as a column type, with nothing to register**:

```python
class Money:

    def __init__(self, value):
        self.amount = float(value)


DataSet([{'x': '1.50'}], columns=[('x', Money)])[0].x   # Money
```

A constructor that raises makes the read raise `ConversionError`,
naming the column, its declared type and the value. A declared type is a
guarantee, not a claim: nothing silently keeps a value of the wrong
class. Declare `object` where a column genuinely holds anything.

Inference works the same way: a value of an unrecognized class infers as
its own class, and a column mixing it with another class infers as
`object`.

[Extending](extending.md#declaring-a-custom-column-type) carries the
rest of the contract - what such a class should implement for joining,
diffing, and the excel writer.

## Reading rows

A row is a `libb.lazydict`, so a column reads either way:

```python
rows[0].name       # 'ana'
rows[0]['name']    # 'ana'
```

It subclasses `libb.attrdict` and is not the same class - `lazydict`
additionally calls a stored callable on access.

### Computed columns

Store a callable and the row gains a computed column. It is called
with the row, so it reads whatever the row holds at the time:

```python
rows = DataSet([{'a': 1, 'b': 2, 'total': lambda row: row.a + row.b}])
rows[0].total          # 3
rows[0]['a'] = 10
rows[0].total          # 12
```

- Only attribute access calls it. `row['total']` and
  `row.get('total')` hand back the callable itself, which is what lets
  the writers and type conversion see what they are dealing with.
- Type conversion leaves a callable alone whatever the column
  declares, so a computed column keeps working in a typed dataset.
- The writers render it, they do not resolve it: a csv or workbook
  cell gets the callable's `repr`. Add a real column with
  `add_column(name, typ, value=fn)` for output, which calls `fn` once
  per row and stores the result.

`rows` iterates its rows, `len(rows)` counts them, and `rows[2]` and
`rows[1:3]` index and slice as a list does.

## Copying

Three copies, sharing progressively less:

| Call            | New container | New rows | New values |
| --------------- | ------------- | -------- | ---------- |
| `copy()`        | yes           | no       | no         |
| `shallowcopy()` | yes           | yes      | no         |
| `deepcopy()`    | yes           | yes      | yes        |

**`copy()` shares its rows**, so adding a column to the copy adds it to
the original too - both hold the same dictionaries:

```python
base = DataSet([{'v': 1}], columns=[('v', int)])
base.copy().add_column('tag', str, value='x')
base[0]     # {'v': 1, 'tag': 'x'} - the original changed
```

`shallowcopy()` gives each dataset its own rows, so adding or removing a
column on one leaves the other alone. The values inside the rows are
still shared, so mutating a list or dict held in a cell still shows up
on both.

`deepcopy()` shares nothing. Use it whenever the copy will be modified
and neither of the two above is clearly enough.

`copy()` and `shallowcopy()` take `empty=True` for the column structure
with no rows.

## Next

- [Rows](rows.md) - adding, removing, ordering, filtering
- [Screening](screening.md) - filtering by a text query
- [Joining](joins.md) - combining two datasets
- [Aggregating](aggregation.md) - `bucket`, `pivot`, `flatten`
- [Reading and writing](io.md) - csv, json, excel
