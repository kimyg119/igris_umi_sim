"""Avoid retained-log collection pauses during one bounded control run."""
from contextlib import contextmanager
import gc

@contextmanager
def defer_cyclic_gc(*,before_restore=None):
    enabled=gc.isenabled()
    if enabled:gc.collect()
    gc.disable()
    try:
        yield
    finally:
        # Close command ownership before restoring automatic collection: a
        # retained trace can trigger a long collection on the next allocation.
        try:
            if before_restore is not None:before_restore()
        finally:
            if enabled:gc.enable()
