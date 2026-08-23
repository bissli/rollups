# Summaries and output

Totaling a dataset, rendering it as a table, and paging it.

## The summary row

```python
rows.summary        # a row of totals, computed on the spot
```

Declaring one is optional. Reading `summary` on a dataset with no
declared summary totals every numeric column and labels the first
column `Total`.

**Nothing is cached.** The summary recomputes on every read, so
filtering the rows and reading again gives the total of what is left:

```python
rows.summary['amount']              # 4.0
rows.filter_data(lambda r: r['amount'] > 2)
rows.summary['amount']              # 3.0
```

### Declaring one

```python
rows.add_summary_row(label='Total', columns=['amount'])
rows.add_summary_row(label='Max', cols_funcs=[('amount', max)])
rows.add_summary_row(label_idx=1, label='Sum')
```

`columns` narrows which columns are totaled - the rest come back `None`
rather than being left out. `cols_funcs` pairs a column with the
function to apply, so a column can be maxed where its neighbor is
summed. `label_idx` picks which column the label is written into.

`add_summary_row` declares; it does not compute. `calc_summary_row`
takes the same arguments and answers the row immediately, without
declaring anything.

`DataSet.is_summary_row(row)` tells a summary row from a data row, which
is what a caller needs when iterating a rendered table.

### A per-row total

`add_summary_column` goes the other way, adding a column that combines
each row's own values:

```python
rows.add_summary_column('total')                       # sum of every column
rows.add_summary_column('total', columns=['a', 'b'])
rows.add_summary_column('spread', columns=['a', 'b'], row_func=max)
```

## Rendering a table

```python
print(rows.pp)
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

`pp` renders through `prettytable`, and includes the summary row where
one was declared. Where none was, it renders the plain table and runs no
totaling at all, so `pp` on a large dataset costs nothing extra.

For a log line rather than a table, `dump()` writes the dataset's
attributes at debug level.

## Paging

Paging is carried, not performed. Give the dataset the page it holds
and the page size, and it answers the questions a pager asks. It does
not slice the rows.

```python
page = DataSet(rows_for_this_page,
               columns=columns,
               page=2, per_page=10, total=25)

page.pages       # 3
page.has_prev    # True
page.has_next    # True
```

`total` is the row count across all pages. Without it, the container is
counted, which is only right when the dataset holds every row.

`get_pages()` gives the page numbers a pager should render, eliding the
middle with `'...'`:

```python
list(page.get_pages())
# [1, 2, 3]

# on page 20 of 100:
# [1, 2, '...', 18, 19, 20, 21, 22, 23, 24, '...', 99, 100]
```

The four arguments set the shape of that window - how many pages to
always show at each end, and how many either side of the current one:

```python
page.get_pages(start_max=2, left_this=2, right_this=5, end_max=2)
```

## Next

- [Rows](rows.md) - getting the values out as lists and arrays
- [Aggregating](aggregation.md) - totals per group rather than overall
- [Reading and writing](io.md) - csv, json, excel
