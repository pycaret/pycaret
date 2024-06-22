"""
Module containing tweaks to joblib.hashing to improve
performance.

Changes include:
1. Using the xxhash algorithm by default instead of md5.
2. Using the highest available pickle protocol by default.
3. Using O(1) hashing (https://github.com/joblib/joblib/pull/1011).
4. Special support for pandas/numpy object arrays.
5. Caching the function output only if the call takes more than 0.1 seconds.
6. Avoiding hashing the function signature twice in Memory class.
"""

import hashlib
import pickle
import struct
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Union

from joblib.hashing import Hasher, Pickler
from joblib.memory import (
    MemorizedFunc,
    MemorizedResult,
    Memory,
    NotMemorizedResult,
    filter_args,
    format_call,
    format_signature,
    format_time,
    get_func_name,
)
from xxhash import xxh128 as xxh

try:
    from math import prod
except ImportError:
    import operator
    from functools import reduce

    def prod(iterable) -> float:
        return reduce(operator.mul, iterable, 1)


if TYPE_CHECKING:
    import numpy as np

DEFAULT_MIN_TIME_TO_CACHE = 0.1
DEFAULT_BYTES_LIMIT = 1024 * 1024 * 1024 * 10  # 10 GB
DEFAULT_CALLS_BETWEEN_REDUCE = 20


# From https://github.com/joblib/joblib/pull/1011
class _FileWriteToHash:
    """For Pickler, a file-like api that translates file.write(bytes) to
    hash.update(bytes)

    From the Pickler docs:
    - https://docs.python.org/3/library/pickle.html#pickle.Pickler

    > The file argument must have a write() method that accepts a single
    > bytes argument. It can thus be an on-disk file opened for binary
    > writing, an io.BytesIO instance, or any other custom object that meets
    > this interface.
    """

    closed = False

    def __init__(self, hash):
        self.hash = hash

    def write(self, bytes):
        self.hash.update(bytes)


class FastHasher(Hasher):
    def __init__(self, hash_name="xxhash", protocol=None):
        # Initialise the hash obj
        if not isinstance(hash_name, str):
            self._hash = hash_name
        elif hash_name == "xxhash":
            self._hash = xxh()
        else:
            self._hash = hashlib.new(hash_name)
        self.stream = _FileWriteToHash(self._hash)
        protocol = protocol or pickle.HIGHEST_PROTOCOL
        Pickler.__init__(self, self.stream, protocol=protocol)

    def hash(self, obj, return_digest=True):
        try:
            # Pickler.dump will trigger a sequence of self.stream.write(bytes)
            # calls, which will in turn relay to self._hash.update(bytes)
            self.dump(obj)
        except pickle.PicklingError as e:
            e.args += ("PicklingError while hashing %r: %r" % (obj, e),)
            raise
        if return_digest:
            # Read the resulting hash
            return self._hash.hexdigest()

    def save_global(self, obj, name=None, pack=struct.pack):
        # Fixes joblib issue. In the except block,
        # Pickler.save_global has been moved to bottom so it
        # can actually work.
        kwargs = dict(name=name, pack=pack)
        del kwargs["pack"]
        try:
            Pickler.save_global(self, obj, **kwargs)
        except pickle.PicklingError:
            module = getattr(obj, "__module__", None)
            if module == "__main__":
                my_name = name
                if my_name is None:
                    my_name = obj.__name__
                mod = sys.modules[module]
                if not hasattr(mod, my_name):
                    # IPython doesn't inject the variables define
                    # interactively in __main__
                    setattr(mod, my_name, obj)
            Pickler.save_global(self, obj, **kwargs)

    dispatch = Hasher.dispatch.copy()
    for key in dispatch:
        if dispatch[key] == Hasher.save_global:
            dispatch[key] = save_global


# Based on joblib.hashing.NumpyHasher
class FastNumpyHasher(FastHasher):
    def __init__(self, hash_name="xxhash", coerce_mmap=False, protocol=None):
        self.coerce_mmap = coerce_mmap
        super().__init__(hash_name=hash_name, protocol=protocol)
        # delayed import of numpy, to avoid tight coupling
        import numpy as np

        self.np = np

    def _make_array_contiguous_if_needed(self, arr: "np.ndarray") -> "np.ndarray":
        if not arr.flags.c_contiguous:
            obj_c_contiguous = self.np.ascontiguousarray(arr)
        else:
            obj_c_contiguous = arr
        return obj_c_contiguous

    def _get_coerced_mmap_class(self, arr: "np.ndarray") -> type:
        # We store the class, to be able to distinguish between
        # Objects with the same binary content, but different
        # classes.
        if self.coerce_mmap and isinstance(arr, self.np.memmap):
            # We don't make the difference between memmap and
            # normal ndarrays, to be able to reload previously
            # computed results with memmap.
            klass = self.np.ndarray
        else:
            klass = arr.__class__
        return klass

    def _get_numpy_metadata_tuple(self, arr: "np.ndarray"):
        klass = self._get_coerced_mmap_class(arr)
        return (klass, ("HASHED", arr.dtype, arr.shape, arr.strides))

    def save(self, obj):
        """Subclass the save method, to hash ndarray subclass, rather
        than pickling them. Off course, this is a total abuse of
        the Pickler class.
        """
        if isinstance(obj, self.np.ndarray) and not obj.dtype.hasobject:
            # Compute a hash of the object
            # The update function of the hash requires a c_contiguous buffer.
            obj_c_contiguous = self._make_array_contiguous_if_needed(obj)

            try:
                self.stream.write(memoryview(obj_c_contiguous))
            except ValueError:
                self.stream.write(memoryview(obj_c_contiguous.view(self.np.uint8)))

            # We also return the dtype and the shape, to distinguish
            # different views on the same data with different dtypes.

            # The object will be pickled by the pickler hashed at the end.
            obj = self._get_numpy_metadata_tuple(obj)
        elif isinstance(obj, self.np.dtype):
            # numpy.dtype consistent hashing is tricky to get right. This comes
            # from the fact that atomic np.dtype objects are interned:
            # ``np.dtype('f4') is np.dtype('f4')``. The situation is
            # complicated by the fact that this interning does not resist a
            # simple pickle.load/dump roundtrip:
            # ``pickle.loads(pickle.dumps(np.dtype('f4'))) is not
            # np.dtype('f4') Because pickle relies on memoization during
            # pickling, it is easy to
            # produce different hashes for seemingly identical objects, such as
            # ``[np.dtype('f4'), np.dtype('f4')]``
            # and ``[np.dtype('f4'), pickle.loads(pickle.dumps('f4'))]``.
            # To prevent memoization from interfering with hashing, we isolate
            # the serialization (and thus the pickle memoization) of each dtype
            # using each time a different ``pickle.dumps`` call unrelated to
            # the current Hasher instance.
            self.stream.write("_HASHED_DTYPE".encode("utf-8"))
            pickle.dump(obj, file=self.stream, protocol=self.proto)
            return
        super().save(obj)


class FastPandasHasher(FastNumpyHasher):
    """
    Hasher with special logic for handling Pandas and numpy object dtype arrays.
    """

    def __init__(self, hash_name="xxhash", coerce_mmap=False, protocol=None):
        super().__init__(
            hash_name=hash_name, coerce_mmap=coerce_mmap, protocol=protocol
        )
        # delayed import of numpy, to avoid tight coupling
        import pandas as pd

        self.pd = pd

    def save(self, obj):
        if (
            isinstance(obj, self.np.ndarray)
            and obj.dtype.hasobject
            and obj.ndim >= 1
            and prod(obj.shape) > 1000
        ):
            try:
                meta = self._get_numpy_metadata_tuple(obj)
                super().save(self.pd.util.hash_array(self.np.ravel(obj)))
                obj = meta
            except TypeError:
                pass
        super().save(obj)


def fast_hash(obj, hash_name="xxhash", coerce_mmap=False, protocol=None):
    """Quick calculation of a hash to identify uniquely Python objects
    containing numpy arrays.

    Compared to default joblib, this function uses the xxhash algorithm,
    hashes in O(1) space and has special handling for object arrays.

    Parameters
    -----------
    coerce_mmap: boolean
        Make no difference between np.memmap and np.ndarray
    protocol: int
        Pickle protocol version to use
    """
    valid_hash_names = ("xxhash", "md5", "sha1")
    if hash_name not in valid_hash_names:
        raise ValueError(
            "Valid options for 'hash_name' are {}. "
            "Got hash_name={!r} instead.".format(valid_hash_names, hash_name)
        )
    if "pandas" in sys.modules:
        hasher = FastPandasHasher(
            hash_name=hash_name, coerce_mmap=coerce_mmap, protocol=protocol
        )
    elif "numpy" in sys.modules:
        hasher = FastNumpyHasher(
            hash_name=hash_name, coerce_mmap=coerce_mmap, protocol=protocol
        )
    else:
        hasher = FastHasher(hash_name=hash_name, protocol=protocol)
    return hasher.hash(obj)


class FastMemorizedFunc(MemorizedFunc):
    # Will only cache if function took longer than min_time_to_cache
    # seconds to run.
    def __init__(self, *args, min_time_to_cache=DEFAULT_MIN_TIME_TO_CACHE, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_output_identifiers = None
        self.min_time_to_cache = min_time_to_cache

    def _get_argument_hash(self, *args, **kwargs):
        return fast_hash(
            filter_args(self.func, self.ignore, args, kwargs),
            coerce_mmap=(self.mmap_mode is not None),
        )

    # Changes here include:
    # 1. _cached_call calls _get_output_identifiers and then calls call,
    #    which also calls _get_output_identifiers. Here, we cache the
    #    output identifiers to avoid calculating them twice.
    # 2. min_time_to_cache logic.

    def call(self, *args, **kwargs):
        """Force the execution of the function with the given arguments and
        persist the output values.
        """
        start_time = time.time()
        # PYCARET CHANGES
        # This will be set if call is called from _cached_call
        if self._cached_output_identifiers:
            func_id, args_id = self._cached_output_identifiers
            self._cached_output_identifiers = None
        else:
            func_id, args_id = self._get_output_identifiers(*args, **kwargs)
        # PYCARET CHANGES END
        if self._verbose > 0:
            print(format_call(self.func, args, kwargs))

        # PYCARET CHANGES
        func_start_time = time.monotonic()
        output = self.func(*args, **kwargs)
        func_duration = time.monotonic() - func_start_time
        if func_duration >= self.min_time_to_cache:
            self.store_backend.dump_item(
                [func_id, args_id], output, verbose=self._verbose
            )

            duration = time.time() - start_time
            metadata = self._persist_input(duration, args, kwargs)
        else:
            metadata = None
        # PYCARET CHANGES END

        if self._verbose > 0:
            _, name = get_func_name(self.func)
            # PYCARET CHANGES
            if metadata is not None:
                msg = "%s - %s" % (name, format_time(duration))
            else:
                msg = "%s - not caching as it took %s" % (
                    name,
                    format_time(func_duration),
                )
            print(max(0, (80 - len(msg))) * "_" + msg)
        # PYCARET CHANGES END
        return output, metadata

    # The code here is identical as in joblib, except for
    # clearly marked parts
    def _cached_call(self, args, kwargs, shelving=False):
        func_id, args_id = self._get_output_identifiers(*args, **kwargs)
        metadata = None
        msg = None

        # Whether or not the memorized function must be called
        must_call = False

        # FIXME: The statements below should be try/excepted
        # Compare the function code with the previous to see if the
        # function code has changed
        if not (
            self._check_previous_func_code(stacklevel=4)
            and self.store_backend.contains_item([func_id, args_id])
        ):
            if self._verbose > 10:
                _, name = get_func_name(self.func)
                self.warn(
                    "Computing func {0}, argument hash {1} "
                    "in location {2}".format(
                        name,
                        args_id,
                        self.store_backend.get_cached_func_info([func_id])["location"],
                    )
                )
            must_call = True
        else:
            try:
                t0 = time.time()
                if not shelving:
                    # When shelving, we do not need to load the output
                    out = self.store_backend.load_item(
                        [func_id, args_id], msg=msg, verbose=self._verbose
                    )
                else:
                    out = None

                if self._verbose > 4:
                    t = time.time() - t0
                    _, name = get_func_name(self.func)
                    msg = "%s cache loaded - %s" % (name, format_time(t))
                    print(max(0, (80 - len(msg))) * "_" + msg)
            except Exception:
                # XXX: Should use an exception logger
                _, signature = format_signature(self.func, *args, **kwargs)
                self.warn(
                    "Exception while loading results for "
                    "{}\n {}".format(signature, traceback.format_exc())
                )

                must_call = True

        if must_call:
            # PYCARET CHANGES
            self._cached_output_identifiers = func_id, args_id
            out, metadata = self.call(*args, **kwargs)
            if self.mmap_mode is not None and metadata is not None:
                # PYCARET CHANGES END
                # Memmap the output at the first call to be consistent with
                # later calls
                out = self.store_backend.load_item(
                    [func_id, args_id], msg=msg, verbose=self._verbose
                )

        return (out, args_id, metadata)

    def call_and_shelve(self, *args, **kwargs):
        # PYCARET CHANGES
        out, args_id, metadata = self._cached_call(args, kwargs, shelving=True)
        if metadata is None:
            return NotMemorizedResult(out)
        # PYCARET CHANGES END
        return MemorizedResult(
            self.store_backend,
            self.func,
            args_id,
            metadata=metadata,
            verbose=self._verbose - 1,
            timestamp=self.timestamp,
        )


class FastMemory(Memory):
    def __init__(
        self,
        *args,
        min_time_to_cache=DEFAULT_MIN_TIME_TO_CACHE,
        caches_between_reduce=DEFAULT_CALLS_BETWEEN_REDUCE,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_time_to_cache = min_time_to_cache
        self.caches_between_reduce = caches_between_reduce
        self.reduce_size()

    def reduce_size(self):
        self._cache_counter = 0
        return super().reduce_size()

    def cache(self, func=None, ignore=None, verbose=None, mmap_mode=False):
        ret = super().cache(func, ignore, verbose, mmap_mode)
        if isinstance(ret, MemorizedFunc):
            ret.__class__ = FastMemorizedFunc
            ret.min_time_to_cache = self.min_time_to_cache
            ret._cached_output_identifiers = None
            self._cache_counter += 1
            if self._cache_counter >= self.caches_between_reduce:
                self.reduce_size()
        return ret

    def __del__(self):
        self.reduce_size()


def get_memory(memory: Union[bool, str, Path, Memory]) -> Memory:
    if memory is None or isinstance(memory, Memory):
        return memory
    if isinstance(memory, (str, Path, bool)):
        if not memory:
            return None
        if memory:
            tmpdir = tempfile.gettempdir() if isinstance(memory, bool) else str(memory)
            return FastMemory(tmpdir, verbose=0, bytes_limit=DEFAULT_BYTES_LIMIT)
    raise TypeError(
        f"memory must be a bool, str or joblib.Memory object, got {type(memory)}"
    )
