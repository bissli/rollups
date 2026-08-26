# Extending

Four ways to build on this package without forking it: subclass
`DataSet`, call the free functions directly, register a workbook
backend, or declare a custom column type.

## Subclassing DataSet

Add methods to a subclass, and every operation returns the subclass
rather than the base class:

```python
from rollups import DataSet


class Report(DataSet):

    def totals_row(self):
        return self.bucket([], [c for c, t in self.columns if t is float])


rows = Report([{'group': 'a', 'amount': 1.0}])
type(rows.bucket('group', ['amount']))   # Report
```

`bucket`, `flatten`, `pivot` and `transpose` build their result
through the receiver's class, and so do `copy`, `shallowcopy`,
`deepcopy` and `dedupe`. `itemize` hands back a plain list, but each
one-row dataset in it takes the subclass. `apply_screen` filters in place and
builds nothing.

### The class-propagation rule for join

`join` is a classmethod taking both datasets as arguments, so the
receiver is the class, not a row container. **The result takes the class
the method was reached through:**

```python
Report.join(a, 'key', b, 'key')      # a Report, whatever a and b are
DataSet.join(a, 'key', b, 'key')     # a DataSet, even if a is a Report
```

Naming the base class is therefore a way to ask for the base class,
which is what a caller writing `DataSet.join(...)` over a subclassed
input already means. Reach the subclass through `self.__class__.join`,
or name it.

### Reading rows in a subclass method

A method that reads row values calls `self.ensure_types()` first.
Conversion is lazy - a value is still the string it was read from until
something triggers it:

```python
class Report(DataSet):

    def largest(self, col):
        self.ensure_types()
        return max(row[col] for row in self.container)
```

Reading through `self` rather than `self.container` converts as a side
effect, but relying on that makes a later switch to `self.container`
silently change the answer. Call it.

`ensure_types` is idempotent and reads one flag on the second call, so
calling it in every method costs nothing.

## Calling the free functions

Every algorithm behind a method is also an exported function taking a
dataset:

```python
from rollups import bucket_dataset, flatten_dataset, join_datasets
from rollups import pivot_dataset, transpose_dataset

bucket_dataset(rows, 'group', ['amount'])
```

The function is what the method delegates to, so the two do the same
thing. The function suits a pipeline, or anywhere the operation has to
be passed as a callable.

`join_datasets` takes a `cls` keyword the method fills in; left unset,
the result takes the left input's class.

The DataFrame-native pair - `join_dataframes` and `bucket_dataframe` -
take and return frames and know nothing about `DataSet`. See
[DataFrame-native join and group-by](dataframe-native.md).

## Registering a workbook backend

This package reads and writes excel through a registered backend. It
never imports one itself, so nothing here depends on a workbook library:

```python
import rollups

rollups.register_excel_backend(my_excel_module)
```

The backend is any object carrying two attributes:

| Attribute          | Signature                                                        |
| ------------------ | ---------------------------------------------------------------- |
| `parse`            | `(*args, **kwargs) -> dict[str, list[dict]]`, sheet name to rows |
| `dataset_to_excel` | `(dataset, file_or_name, **kwargs) -> None`                      |

A module satisfies that as readily as an instance, so a module of two
functions is the usual shape. Register once at import time; a second
call replaces the first.

Until a backend is registered, `DataSet.from_excel`, `from_excel_sheets` and
`write_excel` raise `RuntimeError` naming the fix. Everything else in
the package works without a backend.

## Declaring a custom column type

A column type is any class taking one argument. No registration:

```python
class Money:

    def __init__(self, value):
        self.amount = float(value)


rows = DataSet([{'x': '1.50'}], columns=[('x', Money)])
rows[0].x            # Money, built from the string
```

Conversion tries the date and numeric types it knows, then falls through
to `typ(val)`. A constructor that raises makes the read raise
`ConversionError`, naming the column, its declared type and the value -
so a column that cannot hold what it declares says so rather than
quietly keeping the wrong class.

Inference works the same way round: `smart_type` answers
`val.__class__` for anything it does not recognize, so a dataset built
from rows already holding `Money` infers a `Money` column.

Two things a custom type should carry:

- `__eq__`, if rows will be joined or diffed on that column.
- `render_as_str = True`, if the excel writer should spell the value
  out rather than parse it as a number. A class whose text form only
  looks numeric needs this to keep its own spelling.

## What this package deliberately does not offer

Evidence rejected each of these. Reopening one needs a use case, not a
preference.

- **A type-converter registry.** The `typ(val)` fallthrough above
  already converts any one-argument class, and `smart_type` already
  infers one. A registry would add a second way to do what already
  works.
- **A screen-operator registry.** The screen language is a fixed
  vocabulary of comparison operators. A caller wanting different
  matching writes a predicate and filters, with no help from this
  package.
- **An aggregation-op registry.** `bucket` already takes any callable,
  named or not. `frame.FAST_OPS` is a pandas performance map, not an
  extension point: an op missing from it still runs, once per group
  rather than vectorized.
- **`register_method`, or plugin entry points.** A function taking a
  dataset needs no registration to be called, and a subclass is the
  supported way to add a method. Neither buys anything a plain import
  does not.

## Next

- [Architecture](architecture.md) - the module layout, the layering
  invariant, and the rule for where a new function goes
- [Reading and writing](io.md) - the excel backend from the calling side
