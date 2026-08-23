# Screening

A screen filters a dataset with a small query language, one query per
column. It exists so a saved filter can travel as text, from a config
file or a web form, rather than as code.

```python
apply_screen(ds, {'group': '<>Magenta', 'score': '>30'})
```

The dataset is filtered in place. A query naming a column the dataset
does not carry is skipped with a warning, and an empty query is skipped
outright, so a partly filled form does not empty the result.

## Operators

| Token        | Meaning                  |
| ------------ | ------------------------ |
| `>`          | greater than             |
| `>=`         | greater than or equal    |
| `<`          | less than                |
| `<=`         | less than or equal       |
| `=`          | equal                    |
| `<>` or `!=` | not equal                |
| none         | regex search             |

A term with no operator is searched as a regex, case-insensitive, so
`US|GR` matches either and `fo` matches `foo`. A term with an operator
is compared directly.

Comma-separated clauses all have to hold:

```python
{'y': '<4,>2'}      # 2 < y < 4
```

## Values

A value that parses as a number is compared as one, so `'2'` and `2`
match. A value that does not parse stays a string, which is what keeps a
grade like `B-` from being mangled into arithmetic. The words
`none` and `null` mean `None`. A trailing `%` is stripped and the number
kept.

A boolean column is compared against `Yes` and `No`.

## Column references

A value beginning with an underscore names another column in the same
row, so one column can be screened against another:

```python
{'score': '<_score_limit'}
```

A reference can carry an arithmetic suffix, applied before the
comparison:

```python
{'size': '>=_capacity*0.03'}
{'value': '>=_baseline+10'}
```

The four operators are `+`, `-`, `*`, and `/`, each null-safe: a `None`
on either side yields no match rather than an error.

## Next

- [Rows](rows.md) - filtering with a predicate instead of a query
- [Joining](joins.md) - matching rows across two datasets
- [Getting started](getting-started.md) - columns and the type system
