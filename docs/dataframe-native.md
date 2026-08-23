# DataFrame-native join and group-by

`rollups.frame` holds four functions that take and return
`pandas.DataFrame` and know nothing about `DataSet`. Two of them,
`join_dataframes` and `bucket_dataframe`, do what `DataSet.join` and
`DataSet.bucket` do. The other two mirror a `DataSet` constructor but
hand back a frame.

```python
from rollups import bucket_dataframe, dataframe_from_list
from rollups import empty_dataframe, join_dataframes
```

Use them where the data already is a DataFrame, or where the next
step is itself pandas work. A report still wants a `DataSet`, for the
column types, formatting, and summary rows a bare frame does not carry.

## join_dataframes

```python
join_dataframes(left, lkey, right, rkey, how='inner', *,
                lsuffix='', rsuffix='', lcols=None, rcols=None,
                first=False, prefer='left')
```

`lkey` and `rkey` are a column name or a list of them, paired by
position, and may be named differently on each side. `how` is `inner`,
`outer`, `left`, `right`, or `cross`.

What it adds over `pandas.merge`:

- **A column both sides carry stays one column.** `merge` would spell it
  `v_x` and `v_y`; here the result has a single `v`, taking the left
  value wherever that value is not null and the right value where it is.
  `prefer='right'` reverses which side leads. The coalesce runs per row,
  so one column can take the left value on one row and the right on the
  next.
- **Null keys match each other.** A null-keyed row on one side pairs with
  the null-keyed rows on the other, and an object key column matches a
  numeric one instead of raising.
- **Fixed row and column order.** Columns are the left frame's, then the
  right frame's newcomers, each in its own frame's order. `inner`,
  `left`, and `outer` run in left row order, each row followed by its
  matches in right row order; `outer` then appends the right-only rows.
  `right` runs in right row order.
- **`lsuffix` / `rsuffix` rename a whole side**, key columns included,
  which keeps two same-named columns apart rather than coalescing
  them.
- **`lcols` / `rcols` pick what each side contributes.** A key column
  left out still drives the match and is then dropped.

`first=True` pairs the nth row of a key group with the nth row of the
same group on the other side, rather than taking the cartesian product.
Per key it keeps `min(nleft, nright)` rows under `inner`, `nleft` under
`left`, `nright` under `right`, and `max(nleft, nright)` under `outer`.
It drops the surplus on the longer side and logs one warning counting
the rows it dropped. Use it only where each key is known to match
once.

## bucket_dataframe

```python
bucket_dataframe(df, keycols, aggregations)
```

The SQL `GROUP BY`: one row per distinct key combination. `keycols` is a
column name or a list of them; empty or `None` totals every row into one
row carrying no key column. Groups come back in sorted key order with
the null key last.

`aggregations` takes the same five shapes `bucket` does:

| Form                        | Meaning                          |
| --------------------------- | -------------------------------- |
| `'sales'`                   | sum the column, skipping nulls   |
| `('sales',)`                | the same                         |
| `('sales', max)`            | apply `max` to the column        |
| `('sales', min, 'lowest')`  | apply `min`, name the result     |
| `('sales', sum, fn)`        | apply `sum` to what `fn` selects |
| `('sales', sum, fn, 'net')` | filter and name                  |

A three-item tuple reads its third item by type: a string is an alias, a
callable is a filter. An empty `aggregations` returns the distinct keys.

`sum`, `max`, and `min` over a numeric column run vectorized through
pandas. Any other callable, and any aggregation carrying a filter, runs
once per group. Either way the result is one row per group: an op
returning several values puts them all in one cell rather than fanning
out into several rows.

The two paths agree on every ordinary value. They part at the extremes,
because the vectorized reduction is pandas' and the per-group one is
Python's: pandas compensates a float sum and wraps an integer one at
int64, where a Python `sum` neither compensates nor overflows. A
per-group result also arrives in the numpy dtype nearest its values, so
a nullable `Int64` column comes back `float64`. Adding a filter to an
aggregation moves it from one path to the other.

A filter receives the group as a DataFrame, key columns included, and
returns the values to aggregate:

```python
fn = lambda group: group['c'].loc[group['c'] != 7]
bucket_dataframe(df, 'key', [('c', max, fn)])
```

Return an empty selection freely - an empty group aggregates to null
rather than raising. An op that raises `TypeError` or `ValueError`
aggregates to null too, except over uniform sets, frozensets, lists, or
tuples, which are merged into one of their own kind.

## Nulls

Every gap `join_dataframes` and `bucket_dataframe` introduce is a pandas
null - `NaN`, `NaT`, or `pd.NA`, whichever the column's dtype carries.
Neither hands back a Python `None` of its own. An int column holding a
gap widens to float, as it does anywhere else in pandas.

The two constructors are the exception, and only partly. Both hand a
real Python `None` to pandas, which then decides by dtype: an object
column keeps the `None`, and every other column turns it into that
dtype's null. So `empty_dataframe([('flag', bool)])` gives an object
column holding `None`, while `dataframe_from_list` with a missing int
gives a `float64` column holding `NaN`.

So read a gap with `isna()`, never `is None`, and count populated fields
with `notna()` rather than a truth test. `isna()` is the one test that
holds across all four.

## empty_dataframe and dataframe_from_list

Two constructors mirroring a `DataSet` one, returning a frame instead:

| Function              | Mirrors             |
| --------------------- | ------------------- |
| `empty_dataframe`     | `DataSet.from_empty` |
| `dataframe_from_list` | `DataSet.from_list`  |

They take the same arguments as the methods they mirror, so swapping one
in is a one-word edit. That is the point: they exist to move a call site
onto vectorized pandas one at a time.

```python
empty_dataframe([('name', str), ('count', int)])
dataframe_from_list([('a', 1), ('b', 2)], ['name', 'count'], [str, int])
```

`empty_dataframe` carries the same defaults `from_empty` produces: `''`
for str, `0` for int, `0.0` for float, `None` otherwise. A bool column
therefore starts at `None`, not `False`.

`dataframe_from_list` coerces int, float, str and bool columns to the
type given. A gap in a coerced column stays a gap rather than raising.

These two are for migration, not for reports. A report wants the
formatting and summary rows a `DataSet` carries and a bare frame does
not. Use them where the next step is itself pandas work.

## Differences from DataSet.join and DataSet.bucket

Same inputs, different answers. Each of these is deliberate.

| Behavior                              | `DataSet`                           | `rollups.frame`                   |
| ------------------------------------- | ----------------------------------- | --------------------------------- |
| Missing value                         | `None`                              | a pandas null                     |
| `inner` row order                     | the RIGHT side's key order          | left row order                    |
| `first=True` pairing                  | pops from the END, last row wins    | pairs by position, first row wins |
| `first=True` under `inner`            | one row per key, whatever the size  | `min(nleft, nright)` rows per key |
| Cross join                            | `None` as the key on both sides     | `how='cross'`                     |
| One-sided `None` key                  | matches nothing, silently           | raises                            |
| Keys of unequal arity                 | matches nothing, silently           | raises                            |
| A key or column that is absent        | `KeyError`, or a silent null column | raises `ValueError` naming it     |
| An all-null group, `sum`/`max`/`min`  | null                                | null                              |
| An all-null group, `list`/`set`/`len` | `[None]`, `{None}`, `1`             | null                              |
| Result column types                   | reconstructed and coerced           | whatever pandas infers            |

Check these two before migrating a call site.

The **all-null group** is the one place a value changes rather than a
null changing spelling. `bucket`'s default filter hands the operation a
one-item `[None]` so that `max()` has something to work on, which makes
`len` report 1 and `list` report `[None]` for a group that held no data
at all. The frame path returns a null, which is what pandas returns for
an empty selection anywhere else.

`DataSet.join` **coerces the result** to the column types it worked out,
so a float landing in an int column comes back truncated. Nothing here
coerces: a value arrives as pandas read it.

One bug does not carry over. `DataSet.join(a, None, b, None, first=True)`
empties both source datasets, because the `None`-key bucket holds the
source container itself and `first` pops from it. Neither function here
touches the frames it receives.

## Next

- [Joining](joins.md) - the `DataSet` join this one mirrors
- [Aggregating](aggregation.md) - the `DataSet` group-by this one mirrors
- [Getting started](getting-started.md) - moving between the two
  representations
