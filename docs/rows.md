# Rows

Adding, removing, ordering, and reshaping rows. For filtering by a text
query instead, see [Screening](screening.md).

## Adding and removing

```python
rows.append({'name': 'ana', 'amount': 120.5})
rows.extend(other_rows)          # a list of rows, or another DataSet
rows.add_empty_row()             # None in every column
rows.pop('name', 'ana')          # remove and return the first match
```

`append` and `extend` take `validate=True` to convert types now rather
than on the first read. That costs time up front and surfaces a bad
value immediately, while the default defers both.

`pop` answers the row it removed, or `None` where nothing matched.

## Ordering

```python
rows.sort_data('group', '-amount')   # SQL order by, `-` for descending
rows.sort(key=lambda row: row['amount'])
rows.sort('amount')                  # by column name
rows.reverse()
```

`sort_data` is the usual choice: it takes column names, most
significant first, and a `-` prefix sorts that column descending.

`sort` is `list.sort`. It takes a callable, a column name, a list of
names, or nothing at all - which sorts by every column value, left to
right.

All four sort in place and return the dataset, so they chain.

`order` is different: it rearranges rows into the order of the values
supplied, rather than sorting them.

```python
rows.order('group', 'c', 'a', 'b')   # rows come back in that order
```

It needs one value per row and raises `AssertionError` on a mismatch, so
it is for imposing a known order, not for reordering a subset.

## Filtering

```python
rows.filter_data(lambda row: row['amount'] > 50)   # keep what matches
rows.filter_data('foo')                            # case-insensitive
```

A callable keeps the rows it returns true for. A string is matched
against every `str` field in the row, case-insensitively, and keeps a
row where any of them matches.

It filters **in place** and returns `None`. Pass `inplace=False` for a
new dataset instead. A string pattern also takes `replace`, a callable
that rewrites the matched text as it goes.

## Deduplicating

```python
rows.dedupe('group')                 # one row per group, the first kept
rows.dedupe(['group', 'region'])     # a composite key
rows.dedupe('group', filter_fn=pick) # choose which row survives
```

`dedupe` answers a new dataset rather than filtering in place.
`filter_fn` receives the rows sharing a key and chooses one; without it
the first occurrence wins.

## Splitting and sampling

```python
by_group = rows.partition(lambda row: row['group'])
by_group['a']          # a DataSet holding the 'a' rows
by_group['zzz']        # an empty DataSet with the same columns

rows.sample(10)        # a deep copy holding 10 random rows
```

`partition` answers a `defaultdict`, so reading a key that never came up
gives an empty dataset carrying this one's columns rather than raising.

`sample` returns everything where `n` exceeds the row count, and nothing
where `n` is negative.

## Series operations

These three treat a column as an ordered series, so they depend on the
current row order. Sort first.

```python
rows.shift('amount', 1, 'previous')      # each row sees the one before
rows.shift('amount', -1, 'next')         # ... the one after
rows.backfill('amount')                  # fill None from the neighbors
rows.pct_change('amount', 'change')      # fractional change per step
```

`shift` moves values by `periods` positions. A positive value moves
them forward, so row *n* receives row *n-1*'s value and the first row
gets `None`. A negative value moves them backward, and the last row
gets `None`.

`backfill` fills each `None` from the value before it. A run of leading
`None`s takes the first value that follows instead, so
`[1, None, None, 4]` becomes `[1, 1, 1, 4]` and `[None, 2]` becomes
`[2, 2]`.

`pct_change` writes the fractional change between consecutive values -
`100.0` then `110.0` gives `0.1`. The first row is `None`, matching
numpy. The source column has to be `int` or `float`.

`shift` and `backfill` write back over the source column when given no
new name. `pct_change` always needs one.

## Getting the values out

```python
rows.to_list()                  # one tuple per COLUMN, in row order
rows.to_array(['amount'])       # a numpy array
list(rows.unwind('amount'))     # one column's values
list(rows.unwind('g', 'v'))     # a tuple per row
[dict(row) for row in rows]     # plain dicts
```

`to_list` transposes: it answers one tuple per column, not per row.

`unwind` yields a column's values in row order, or a tuple per row for
several columns. Pair it with `zip` to split several columns into their
own sequences:

```python
groups, amounts = zip(*rows.unwind('group', 'amount'))
```

`to_array` takes the named columns, in that order, and a `numpy_type`.
Without one it uses the first column's type.

## Next

- [Screening](screening.md) - filtering by a text query
- [Joining](joins.md) - combining two datasets
- [Aggregating](aggregation.md) - grouping and reshaping
- [Summaries and output](summaries.md) - totals and table rendering
