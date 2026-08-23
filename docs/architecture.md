# Architecture

The package is eight modules. Each holds one concern, and the import
graph between them runs one way.

| Module         | Holds                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------- |
| `types.py`     | inference and conversion: `smart_type`, `force_type`, `infer_numeric_type`                      |
| `screen.py`    | the screen query language: `interpret_screen`, `apply_screen`                                   |
| `join.py`      | two-dataset operations: `join_datasets`, `diff_datasets`, `meld_datasets`, `match_rows`         |
| `aggregate.py` | one-dataset reshapes: `bucket_dataset`, `flatten_dataset`, `pivot_dataset`, `transpose_dataset` |
| `io.py`        | csv, json, excel, and the workbook backend seam                                                 |
| `frame.py`     | the DataFrame-native functions, which never touch `DataSet`                                     |
| `core.py`      | the `DataSet` class and the methods that delegate to the above                                  |
| `__init__.py`  | the public surface                                                                              |

## The layering

```
layer 0   types  screen  io  join  frame     import nothing from rollups
layer 1   aggregate                          imports types only
layer 2   core                               imports aggregate, io, join
layer 3   __init__                           imports everything
```

**Nothing imports `core`; `core` imports everything.** That is the one
rule to check after any edit here, and it is the reason the split works
at all.

It holds because no free function needs the class:

- A function handed a dataset builds its result through
  `dataset.__class__(...)`, so it never names `DataSet`.
- `read_csv_rows`, which has no input dataset, returns
  `(rows, columns)` and lets the classmethod construct. **The IO layer
  never builds a `DataSet`.**
- Where a free function needs another operation it calls the *method* on
  the dataset it was handed. `pivot_dataset` calls `dataset.bucket(...)`,
  which is why `aggregate.py` imports no sibling and a subclass
  override takes effect.

Type annotations reach the class through a `TYPE_CHECKING` import, so
`from .core import DataSet` appears in `aggregate.py`, `io.py` and
`join.py` but never runs.

`tests/test_architecture.py` enforces this. Run it after any edit here:

```
python -m pytest tests/test_architecture.py -q
```

It parses each module and reports the file and line of any runtime
import of `core`, in every spelling. That matters because the spellings
do not fail alike: `from .core import DataSet` raises at import, but
`from . import core` - the idiom `core.py` itself uses for its own
delegates - completes the cycle in silence.

## The delegation pattern

A `DataSet` method whose whole body is a call to one free function
carries a summary and a `See Also`, never a second copy of the contract:

```python
@classmethod
def join(cls, adataset, akey, bdataset, bkey, jointype='inner', ...):
    """Join two datasets on their key columns.

    See Also
    --------
    rollups.join.join_datasets : the contract, the join-type table,
        and the worked examples

    Notes
    -----
    - The result takes the class this method was reached through.
    """
    return join.join_datasets(adataset, akey, bdataset, bkey, ...,
                              cls=cls)
```

Six methods are nothing but the call: `join`, `flatten`, `transpose`,
`json`, `write_csv` and `write_excel`. Two more wrap it in the `inplace`
swap - `bucket` and `pivot` overwrite the receiver's container and
columns when asked, and hand back the receiver.

`read`, `from_excel` and `from_excel_sheets` sit a step further out.
Each calls a free function and then constructs, because the IO layer
never builds a `DataSet` itself. `read` also sets `filename` on the
result, which a free function could not do for it.

The contract lives on the free function. The method documents only what
it adds - an `inplace` flag, or an attribute it sets on the result.
Two copies of a `Parameters` block drift: fix a bug in the body and
only one copy says so.

`inplace` stays on the method and never reaches the free function. It
replaces the receiver, and a free function has no receiver.

## Where a new function goes

| The function                                                | Goes in                                        |
| ----------------------------------------------------------- | ---------------------------------------------- |
| takes a dataset and returns one                             | `join.py` (two inputs) or `aggregate.py` (one) |
| crosses the process boundary - a file, a socket, a workbook | `io.py`                                        |
| decides or applies a type                                   | `types.py`                                     |
| takes and returns `pandas.DataFrame`                        | `frame.py`                                     |
| mutates the receiver, in under 15 lines                     | a method on `core.py`                          |

Two placements look wrong and are not:

- `guess_dataframe_dataset_columns` sits in `core.py`, not `types.py`,
  because it calls `pd.api.types.infer_dtype`. `types.py` is the leaf
  every other module rests on, and it imports nothing beyond the
  standard library, `libb` and `opendate`.
- `dataframe()` and `from_dataframe()` are methods on `core.py`, not
  functions in `frame.py`. They bridge the two representations, and
  `frame.py` promises it does not depend on `DataSet`.

## Conversion is per-function

Values convert to their column's type on first read, not at
construction. A free function that reads rows calls
`dataset.ensure_types()` first rather than assuming a caller did. It is
idempotent and reads one flag on the second call.

`join_datasets` (on both sides), `bucket_dataset`, `pivot_dataset`,
`write_csv_file`, `write_excel_file` and `to_json` call it.
`flatten_dataset` and `transpose_dataset` deliberately do not: `flatten`
never converted, and `transpose` converts only as a side effect of
iterating the dataset rather than its container.

## Next

- [Extending](extending.md) - subclassing, the backend registry, and
  what this package deliberately does not offer
- [Getting started](getting-started.md) - the surface these modules
  add up to
