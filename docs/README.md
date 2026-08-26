# Documentation

`DataSet` is a list of dictionary rows, each column declaring a Python
type that its values convert to as they enter. Start at
[Getting started](getting-started.md); the project [README](../README.md)
carries the overview and the install.

## Using it

| Guide                                 | Covers                                                   |
| ------------------------------------- | -------------------------------------------------------- |
| [Getting started](getting-started.md) | building a dataset, columns, the type system, copying    |
| [Rows](rows.md)                       | adding, removing, ordering, filtering, series operations |
| [Screening](screening.md)             | the text query language for filtering by column          |
| [Joining](joins.md)                   | the four join types, `first`, diffing and melding        |
| [Aggregating](aggregation.md)         | `bucket`, `pivot`, `flatten`, `transpose`                |
| [Reading and writing](io.md)          | csv, json, and the excel backend                         |
| [Summaries and output](summaries.md)  | summary rows, table rendering, paging                    |

## Working with pandas

| Guide                                   | Covers                                         |
| --------------------------------------- | ---------------------------------------------- |
| [DataFrame-native](dataframe-native.md) | the four functions that take and return frames |

## Working on it

| Guide                           | Covers                                                 |
| ------------------------------- | ------------------------------------------------------ |
| [Architecture](architecture.md) | the modules, the layering, where a new function goes   |
| [Extending](extending.md)       | subclassing, the backend registry, custom column types |

## Conventions in these pages

- Every code example runs against the working tree before it ships. An
  example that shows output shows what the code printed.
- A method and the free function behind it do the same thing.
  `rows.bucket(...)` and `bucket_dataset(rows, ...)` are interchangeable.
- Where a call mutates its receiver rather than answering a new dataset,
  the page says so at that call.
