"""Reading and writing a dataset at the process boundary.

These functions cross to a file or a buffer. None of them constructs a
`DataSet`: the reader answers rows and columns, and the caller builds
the container. That is what keeps this module free of any dependency on
the class.

Notes
-----
- A csv header field may carry a type suffix: `name:s`, `age:i`,
  `score:f`, `on:b`, `when:d`. A field without one reads as str.
"""


import logging
import os
import random
from collections.abc import Callable
from functools import wraps


logger = logging.getLogger(__name__)


def on_error_randomize(arg: int | None = None, kwarg: str | None = None) -> Callable:
    """Randomize specified kw/arg if there is an IO/EnvironmentError"""

    def wrapper(io_fn):
        @wraps(io_fn)
        def new_io_fn(*args, **kwargs):
            if arg is not None and len(args) > arg:
                filename = args[arg]
            elif kwarg and kwarg in kwargs:
                filename = kwargs[kwarg]
            else:
                filename = None  # StringIO, BytesIO
            try:
                return io_fn(*args, **kwargs)
            except OSError:
                if not filename:
                    raise
                name, ext = os.path.splitext(str(filename))
                new_filename = f'{name}{str(random.getrandbits(128))}{ext}'
                logger.warning(f'Could not write to file {filename}, retrying with {new_filename}')
                if arg is not None:
                    listargs = list(args)
                    listargs[arg] = new_filename
                    args = tuple(listargs)
                elif kwarg:
                    kwargs[kwarg] = new_filename
                return io_fn(*args, **kwargs)

        return new_io_fn

    return wrapper
