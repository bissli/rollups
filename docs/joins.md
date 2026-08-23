# Joining

`DataSet.join` matches rows from two datasets on one or more key
columns.

```python
DataSet.join(adataset, akey, bdataset, bkey, jointype='inner')
join_datasets(adataset, akey, bdataset, bkey, jointype='inner')
```

The method and the function do the same thing; the method fills in the
class the result should take. See
[Extending](extending.md#the-class-propagation-rule-for-join) for what
that means under a subclass.

## Join types

Take four datasets. `a` and `b` have unique keys; `c` repeats key 1;
`d` repeats key 2:

```
      a                b                c                d
+-----+---+---+  +-----+---+---+  +-----+---+---+  +-----+---+---+
| key | b | c |  | key | d | e |  | key | d | e |  | key | x | y |
+-----+---+---+  +-----+---+---+  +-----+---+---+  +-----+---+---+
|  1  | 1 | 1 |  |  1  | 2 | 2 |  |  1  | 3 | 3 |  |  2  | 3 | 3 |
|  2  | 1 | 1 |  |  2  | 2 | 2 |  |  1  | 4 | 4 |  |  2  | 4 | 4 |
+-----+---+---+  +-----+---+---+  +-----+---+---+  +-----+---+---+
```

| Type    | Rows returned                                          |
| ------- | ------------------------------------------------------ |
| `inner` | only keys present in both (the default)                |
| `outer` | every key from either side, missing columns left empty |
| `left`  | every key in `adataset`                                |
| `right` | every key in `bdataset`                                |

Where a key repeats, the join is a cartesian product on that key.
`DataSet.join(a, 'key', c, 'key')` returns two rows, because a's single
key-1 row pairs with both of c's.

## Selecting and renaming columns

`acol` and `bcol` restrict which columns each side contributes. `amod`
and `bmod` append a suffix to each side's column names, which keeps two
same-named columns apart.

```python
DataSet.join(a, 'key', b, 'key', 'left', bcol=['e'])
```

Where both sides carry the same column, the left value wins unless it is
`None`, in which case the right fills in. Pass `bfirst=True` to reverse
that precedence.

## The `first` flag

By default a repeated key fans out into every pairing. `first=True`
takes one row from each side instead. How many pairs survive depends on
the join type, so "stop at the shorter side" is wrong:

| Type    | Rows kept per key                |
| ------- | -------------------------------- |
| `inner` | one, whatever either side holds   |
| `left`  | as many as `adataset` holds      |
| `right` | as many as `bdataset` holds      |
| `outer` | as many as the longer side holds |

Pairing runs from the END of each key's rows, so the last row wins, not
the first. Where one side runs out, the columns it would have supplied
come through as `None`.

This drops rows, and logs a warning naming the key it dropped them from.
`DataSet.join(a, 'key', c, 'key', first=True)` returns one row where the
default returns two. Use it only where each key is known to match once
and the extras are meant to be discarded.

`join_dataframes` spells this differently: it pairs from the front and
keeps `min(nleft, nright)` under `inner`. Check the comparison table in
[DataFrame-native join and group-by](dataframe-native.md) before
migrating a `first=True` call site.

## Cartesian keys

Passing `None` as a key joins every row against every row. Joining on
mismatched key columns has the same effect for the non-matching rows:
they come through with `None` in the columns the other side would have
supplied.

## Joining frames

`join_dataframes` does all of this over `pandas.DataFrame` on both
sides, and fixes a few things this one gets wrong. See
[DataFrame-native join and group-by](dataframe-native.md).

## Comparing instead of joining

`diff_datasets` answers what changed between two datasets rather than
merging them:

```python
same, diff, only_in_first, only_in_second = diff_datasets(
    ds1, ds2, keycols, comparecols)
```

Rows are matched on `keycols` and compared on `comparecols`. `same` and
the two `only` lists hold complete rows from their source. A `diff` row
holds the key columns plus one entry per compared column: a differing
column carries the pair `(first_value, second_value)`, a matching one
carries `None`.

Matching walks keys in sorted order. Where a key repeats within one
dataset, the last row wins.

`match_rows` does the same matching over two plain lists, taking a key
function per side instead of column names, and returns
`(only_in_first, only_in_second, pairs)`.

## Melding by position

`meld_datasets` widens one dataset with columns taken from others,
matching rows by **position** rather than by key:

```python
meld_datasets(base, [other], ['other'], [['amount']])
```

Each melder contributes the columns named for it, renamed
`{prefix}_{col}`. Every dataset must hold the same rows in the same
order, and a length mismatch raises `ValueError`.

It modifies `base` unless given `inplace=False`, which works on a deep
copy instead. Use it only where the row order is guaranteed. Where it is
not, join on a key.

## Next

- [Aggregating](aggregation.md) - grouping one dataset
- [DataFrame-native join and group-by](dataframe-native.md) - the same,
  over frames
- [Screening](screening.md) - filtering before the join
