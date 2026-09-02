"""Pins the package shape and its one behavioral dependency.

Every other test exercises a function. These exercise what no function
test can see: what may import what, that no module hardcodes the base
class, and that the row class this package is built on still resolves a
stored callable. All of them survive any rewrite of the bodies they
guard.
"""
import ast
import pathlib

import pytest

from libb import lazydict

LAYER_ZERO = ['types', 'screen', 'join', 'aggregate', 'io', 'frame']

SRC = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'rollups'


@pytest.mark.parametrize('module', LAYER_ZERO)
def test_no_module_imports_core_at_runtime(module):
    """Verify nothing outside __init__ imports core when it runs.

    core imports the others, so a runtime import back into core is a
    cycle. A TYPE_CHECKING import is fine and is what the annotations
    use.

    Mutation: `from . import core` added at module level to any of these
    files -- the idiom core.py itself uses for its own delegates, so it
    is the one a maintainer copies in by hand. Python tolerates that
    spelling of the cycle silently, where `from .core import DataSet`
    raises at import.
    Oracle: the AST of each module, walked independently of the import
    machinery, so the tolerated spelling is caught alongside the
    fatal one and is reported by file and line rather than as a suite
    that will not start.
    """
    tree = ast.parse((SRC / f'{module}.py').read_text())
    guarded = {id(node)
               for branch in ast.walk(tree)
               if isinstance(branch, ast.If)
               and ast.unparse(branch.test) == 'TYPE_CHECKING'
               for node in ast.walk(branch)}

    # Every spelling, not just the one that raises: `from . import core`
    # and `import rollups.core` reach core as surely as
    # `from .core import X`, and neither trips the import machinery.
    offenders = []
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module in {'core', 'rollups.core'} or (
                    node.module is None
                    and any(alias.name == 'core' for alias in node.names)):
                offenders.append(node.lineno)
        elif isinstance(node, ast.Import) and any(
                alias.name == 'rollups.core' for alias in node.names):
            offenders.append(node.lineno)

    assert offenders == [], f'{module}.py imports core at {sorted(offenders)}'


@pytest.mark.parametrize('module', LAYER_ZERO + ['core'])
def test_no_module_hardcodes_the_base_class(module):
    """Verify a derived result is built through the caller's class.

    Naming DataSet(...) in a body silently downgrades a subclass. The
    supported spellings are cls(...), self.__class__(...) and
    dataset.__class__(...).

    Mutation: dataset.__class__(buckets) in aggregate.bucket_dataset, or
    self.__class__() in core._copy_structure, rewritten to DataSet(...).
    Oracle: the AST call nodes, which see through the string form that a
    grep for 'DataSet(' cannot distinguish from an annotation.
    """
    tree = ast.parse((SRC / f'{module}.py').read_text())
    offenders = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == 'DataSet']
    assert offenders == [], f'{module}.py constructs DataSet at {offenders}'


def test_row_class_resolves_a_stored_callable_through_copy_and_merge():
    """Verify the row class still computes, and keeps doing so once derived.

    A stored callable is this package's computed column: `_convert_value`
    guards such a value rather than converting it, so the row class is
    what makes it a value at all. Nothing but the dependency floor pins
    that, and a row reaches a caller through `copy` and `|` as often as
    it does directly.

    Mutation: `copy` or `__or__` in the row's base class rebuilt as
        `attrdict(...)` rather than `type(self)(...)`, which downgrades a
        derived row to a mapping that hands the function back instead.
    Oracle: hand-computed 3 from 1 + 2 on the row, and 21 after the
        merge replaces the operand with 20.
    """
    row = lazydict(a=1, b=2, c=lambda r: r.a + r.b)

    assert row.c == 3
    assert row.copy().c == 3
    assert (row | {'b': 20}).c == 21
