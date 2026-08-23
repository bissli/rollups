# Aggregating

## bucket

`bucket` is the SQL `GROUP BY`: it groups rows by key columns and
applies an aggregation to each group.

```python
ds.bucket(keycols, aggregations)
bucket_dataset(ds, keycols, aggregations)   # the same, as a function
```

`keycols` is a column name or a list of them. An empty list totals the
whole dataset into one row.

### Aggregation formats

`aggregations` accepts five shapes, from terse to fully specified:

| Form                        | Meaning                              |
| --------------------------- | ------------------------------------ |
| `'sales'`                   | sum the column, skipping `None`      |
| `('sales',)`                | the same                             |
| `('sales', max)`            | apply `max` to the column            |
| `('sales', min, 'lowest')`  | apply `min`, name the result         |
| `('sales', sum, fn)`        | apply `sum` to what `fn` selects     |
| `('sales', sum, fn, 'net')` | filter and name                      |

A three-item tuple reads its third item by type: a string is an alias, a
callable is a filter.

```python
ds = DataSet([
    {'key': 1, 'b': 2, 'c': 4},
    {'key': 1, 'b': 3, 'c': 5},
    {'key': 2, 'b': 4, 'c': 7},
    ])

ds.bucket('key', ['b', 'c'])        # b: 5 and 4, c: 9 and 7
ds.bucket('key', [('b', list)])     # b: [2, 3] and [4]
ds.bucket('key', [('b', sum, 'total')])
```

A filter receives the group's rows and returns the values to aggregate.
Return a one-item fallback list rather than an empty one, or the
aggregation has nothing to work on:

```python
fn = lambda rows: [r.c for r in rows if r.c != 7] or [None]
ds.bucket('key', [('c', max, fn)])
```

Aggregate the same column twice by giving each result its own alias.
Without an alias the second overwrites the first.

### Types

The result column keeps the source column's type unless the operation
changed it. An `int` column summed with a float stays `float`; a `list`
or `set` aggregation takes the type of what it collected. Where the
source type is unknown, the type is inferred from the first non-`None`
result.

`bucket_dataframe` groups the same way over a `pandas.DataFrame`, taking
the same aggregation shapes. See
[DataFrame-native join and group-by](dataframe-native.md).

## pivot

`pivot` turns the values of one column into columns of their own.

```python
ds.pivot(index_col, data_cols, pivot_col, aggr=sum)
pivot_dataset(ds, index_col, data_cols, pivot_col, aggr=sum)
```

Given rows keyed by `idx` with a category in `col`:

```
+-----+-------+-----+          +-----+-------+-------+
| idx |   d1  | col |          | idx |   a   |   b   |
+-----+-------+-----+          +-----+-------+-------+
|  1  |  8.00 |  a  |    ->    |  1  |  8.00 |  0.10 |
|  2  | 21.80 |  a  |          |  2  | 21.80 | 22.00 |
|  1  |  0.10 |  b  |          |  3  |  3.20 |  3.00 |
|  2  | 22.00 |  b  |          +-----+-------+-------+
+-----+-------+-----+
```

Pivoting several data columns at once needs an alias function, or every
data column collides on the same generated name:

```python
ds.pivot('idx', ['d1', 'd2'], 'col', alias=lambda x, y: x + ':' + y)
```

That yields `a:d1`, `a:d2`, `b:d1`, `b:d2`.

## flatten

`flatten` is the reverse pivot: it moves each named column into its own
row, recording the column name and its value.

```python
ds.flatten(kept, flattened, key='key', val='val')
flatten_dataset(ds, kept, flattened, key='key', val='val')
```

`kept` names the columns to carry through unchanged. `flattened` names
the columns to fold into rows. Flattening two columns over three rows
gives six rows.

## transpose

`transpose` swaps rows for columns: one column's values become the
column names, and the remaining column names become a column of their
own.

```python
ds.transpose(new_index_name, pivot_index=0)
transpose_dataset(ds, new_index_name, pivot_index=0)
```

`pivot_index` is the position of the column whose values supply the new
names; `new_index_name` names the column that will hold the old names.

```
+-------+----+----+          +--------+---+---+
| label | q1 | q2 |          | metric | a | b |
+-------+----+----+    ->    +--------+---+---+
|   a   |  1 |  2 |          |   q1   | 1 | 3 |
|   b   |  3 |  4 |          |   q2   | 2 | 4 |
+-------+----+----+          +--------+---+---+
```

It uses every column and keeps the rows in the order it found them, so
sort and select before calling it. Two rows sharing a value in the
pivot column collide on one output column, and the later row wins.

## Next

- [Joining](joins.md) - combining two datasets
- [Summaries and output](summaries.md) - totals across the whole dataset
- [DataFrame-native join and group-by](dataframe-native.md) - the same,
  over frames
