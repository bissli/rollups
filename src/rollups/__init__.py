"""An iterable of dictionaries, with typed columns.

`DataSet` is a list of dictionary rows. Each column declares a Python
type, and a value converts to it on first read rather than at
construction. The row is the primary object, and remains an ordinary
dict.

Notes
-----
- `attrdict`, `oset` and `emptydict` are libb's names, not this
  package's. Import them from libb: `from libb import lazydict as
  attrdict`, `OrderedSet as oset`, and `emptydict`. Note that a row is
  a `lazydict`, which is not the same class as `libb.attrdict`.
- See docs/README.md for the guides.
"""
