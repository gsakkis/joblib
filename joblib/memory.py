"""
A context object for caching a function's return value each time it
is called with the same input arguments.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.


from __future__ import with_statement
import os
import time
import pydoc
import re
import functools
import traceback
import warnings
import inspect
import sys
import weakref

# Local imports
from . import hashing
from .func_inspect import get_func_code, get_func_name, filter_args
from .func_inspect import format_call
from .func_inspect import format_signature
from ._memory_helpers import open_py_source
from .logger import Logger, format_time, pformat
from ._compat import _basestring, PY3_OR_LATER
from ._store_backends import StoreBackendBase, FileSystemStoreBackend

if sys.version_info[:2] >= (3, 4):
    import pathlib
else:
    pathlib = None


FIRST_LINE_TEXT = "# first line:"

# TODO: The following object should have a data store object as a sub
# object, and the interface to persist and query should be separated in
# the data store.
#
# This would enable creating 'Memory' objects with a different logic for
# pickling that would simply span a MemorizedFunc with the same
# store (or do we want to copy it to avoid cross-talks?), for instance to
# implement HDF5 pickling.

# TODO: Same remark for the logger, and probably use the Python logging
# mechanism.


def extract_first_line(func_code):
    """ Extract the first line information from the function code
        text if available.
    """
    if func_code.startswith(FIRST_LINE_TEXT):
        func_code = func_code.split('\n')
        first_line = int(func_code[0][len(FIRST_LINE_TEXT):])
        func_code = '\n'.join(func_code[1:])
    else:
        first_line = -1
    return func_code, first_line


class JobLibCollisionWarning(UserWarning):
    """ Warn that there might be a collision between names of functions.
    """


_STORE_BACKENDS = {'local': FileSystemStoreBackend}


def register_store_backend(backend_name, backend):
    """Extend available store backends.

    The Memory, MemorizeResult and MemorizeFunc objects are designed to be
    agnostic to the type of store used behind. By default, the local file
    system is used but this function gives the possibility to extend joblib's
    memory pattern with other types of storage such as cloud storage (S3, GCS,
    OpenStack, HadoopFS, etc) or blob DBs.

    Parameters
    ----------
    backend_name: str
        The name identifying the store backend being registered. For example,
        'local' is used with FileSystemStoreBackend.
    backend: StoreBackendBase subclass
        The name of a class that implements the StoreBackendBase interface.

    """
    if not isinstance(backend_name, _basestring):
        raise ValueError("Store backend name should be a string, "
                         "'{0}' given.".format(backend_name))
    if backend is None or not issubclass(backend, StoreBackendBase):
        raise ValueError("Store backend should inherit "
                         "StoreBackendBase, "
                         "'{0}' given.".format(backend))

    _STORE_BACKENDS[backend_name] = backend


def _store_backend_factory(backend, location, verbose=0, backend_options=None):
    """Return the correct store object for the given location."""
    if backend_options is None:
        backend_options = {}

    if pathlib is not None and isinstance(location, pathlib.Path):
        location = str(location)

    if isinstance(location, StoreBackendBase):
        return location
    elif isinstance(location, _basestring):
        obj = None
        location = os.path.expanduser(location)
        # The location is not a local file system, we look in the
        # registered backends if there's one matching the given backend
        # name.
        for backend_key, backend_obj in _STORE_BACKENDS.items():
            if backend == backend_key:
                obj = backend_obj()

        # By default, we assume the FileSystemStoreBackend can be used if no
        # matching backend could be found.
        if obj is None:
            raise TypeError('Unknown location {0} or backend {1}'.format(
                            location, backend))

        # The store backend is configured with the extra named parameters,
        # some of them are specific to the underlying store backend.
        obj.configure(location, verbose=verbose,
                      backend_options=backend_options)
        return obj
    elif location is not None:
        warnings.warn(
            "Instanciating a backend using a {} as a location is not "
            "supported by joblib. Returning None instead.".format(
                location.__class__.__name__), UserWarning)

    return None


def _build_func_identifier(func):
    if not isinstance(func, _basestring):
        modules, funcname = get_func_name(func)
        modules.append(funcname)
        func = os.path.join(*modules)
    return func


# An in-memory store to avoid looking at the disk-based function
# source code to check if a function definition has changed
_FUNCTION_HASHES = weakref.WeakKeyDictionary()


###############################################################################
# class `MemorizedResult`
###############################################################################
class MemorizedResult(Logger):
    """Object representing a cached value.

    Attributes
    ----------
    location: str
        The location of joblib cache. Depends on the store backend used.

    func: function or str
        function whose output is cached. The string case is intended only for
        instanciation based on the output of repr() on another instance.
        (namely eval(repr(memorized_instance)) works).

    argument_hash: str
        hash of the function arguments.

    backend: str
        Type of store backend for reading/writing cache files.
        Default is 'local'.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        The memmapping mode used when loading from cache numpy arrays. See
        numpy.load for the meaning of the different values.

    verbose: int
        verbosity level (0 means no message).

    timestamp, metadata: string
        for internal use only.
    """
    def __init__(self, location, func_id, args_id, backend='local',
                 mmap_mode=None, verbose=0, timestamp=None, metadata=None):
        Logger.__init__(self)
        self.path = (func_id, args_id)
        self.store_backend = _store_backend_factory(backend, location,
                                                    verbose=verbose)
        self.mmap_mode = mmap_mode

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = self.store_backend.get_metadata(self.path)

        self.duration = self.metadata.get('duration', None)
        self.verbose = verbose
        self.timestamp = timestamp

    @property
    def func_id(self):
        return self.path[0]

    @property
    def args_id(self):
        return self.path[1]

    @property
    def func(self):
        warnings.warn(
            "The 'func' attribute has been deprecated.\n"
            "Use `func_id` attribute instead.",
            DeprecationWarning, stacklevel=2)
        return self.func_id

    @property
    def argument_hash(self):
        warnings.warn(
            "The 'argument_hash' attribute has been deprecated in version "
            "0.12 and will be removed in version 0.14.\n"
            "Use `args_id` attribute instead.",
            DeprecationWarning, stacklevel=2)
        return self.args_id

    def get(self):
        """Read value from cache and return it."""
        try:
            return self.store_backend.load_item(self.path,
                                                timestamp=self.timestamp,
                                                metadata=self.metadata,
                                                verbose=self.verbose)
        except (ValueError, KeyError) as exc:
            # KeyError is expected under Python 2.7, ValueError under Python 3
            new_exc = KeyError(
                "Error while trying to load a MemorizedResult's value. "
                "It seems that this folder is corrupted : {}".format(
                    os.path.join(self.store_backend.location, *self.path)))
            new_exc.__cause__ = exc
            raise new_exc

    def clear(self):
        """Clear value from cache"""
        self.store_backend.clear_item(self.path)

    def __repr__(self):
        return '{}(location="{}", func="{}", args_id="{}")'.format(
            self.__class__.__name__, self.store_backend.location,
            self.func_id, self.args_id)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['timestamp'] = None
        return state


class NotMemorizedResult(object):
    """Class representing an arbitrary value.

    This class is a replacement for MemorizedResult when there is no cache.
    """
    __slots__ = ('value', 'valid')

    def __init__(self, value):
        self.value = value
        self.valid = True

    def get(self):
        if self.valid:
            return self.value
        else:
            raise KeyError("No value stored.")

    def clear(self):
        self.valid = False
        self.value = None

    def __repr__(self):
        if self.valid:
            return ('{class_name}({value})'
                    .format(class_name=self.__class__.__name__,
                            value=pformat(self.value)))
        else:
            return self.__class__.__name__ + ' with no value'

    # __getstate__ and __setstate__ are required because of __slots__
    def __getstate__(self):
        return {"valid": self.valid, "value": self.value}

    def __setstate__(self, state):
        self.valid = state["valid"]
        self.value = state["value"]


###############################################################################
# class `NotMemorizedFunc`
###############################################################################
class NotMemorizedFunc(object):
    """No-op object decorating a function.

    This class replaces MemorizedFunc when there is no cache. It provides an
    identical API but does not write anything on disk.

    Attributes
    ----------
    func: callable
        Original undecorated function.
    """
    # Should be a light as possible (for speed)
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def call_and_shelve(self, *args, **kwargs):
        return NotMemorizedResult(self.func(*args, **kwargs))

    def __repr__(self):
        return '{0}(func={1})'.format(self.__class__.__name__, self.func)

    def clear(self, warn=True):
        # Argument "warn" is for compatibility with MemorizedFunc.clear
        pass


###############################################################################
# class `MemorizedFunc`
###############################################################################
class MemorizedFunc(Logger):
    """Callable object decorating a function for caching its return value
    each time it is called.

    Methods are provided to inspect the cache or clean it.

    Attributes
    ----------
    func: callable
        The original, undecorated, function.

    location: string
        The location of joblib cache. Depends on the store backend used.

    backend: str
        Type of store backend for reading/writing cache files.
        Default is 'local', in which case the location is the path to a
        disk storage.

    ignore: list or None
        List of variable names to ignore when choosing whether to
        recompute.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        The memmapping mode used when loading from cache
        numpy arrays. See numpy.load for the meaning of the different
        values.

    compress: boolean, or integer
        Whether to zip the stored data on disk. If an integer is
        given, it should be between 1 and 9, and sets the amount
        of compression. Note that compressed arrays cannot be
        read by memmapping.

    verbose: int, optional
        The verbosity flag, controls messages that are issued as
        the function is evaluated.
    """
    # ------------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------------

    def __init__(self, func, location, backend='local', ignore=None,
                 mmap_mode=None, compress=False, verbose=1, timestamp=None):
        Logger.__init__(self)
        self.mmap_mode = mmap_mode
        self.compress = compress
        self.func = func
        self.func_id = _build_func_identifier(func)
        self.ignore = ignore if ignore is not None else []
        self._verbose = verbose

        # retrieve store object from backend type and location.
        self.store_backend = _store_backend_factory(backend, location,
                                                    verbose=verbose,
                                                    backend_options=dict(
                                                        compress=compress,
                                                        mmap_mode=mmap_mode),
                                                    )
        if self.store_backend is not None:
            # Create func directory on demand.
            self.store_backend.store_cached_func_code([self.func_id])

        self.timestamp = timestamp if timestamp is not None else time.time()
        try:
            functools.update_wrapper(self, func)
        except:
            " Objects like ufunc don't like that "
        if inspect.isfunction(func):
            doc = pydoc.TextDoc().document(func)
            # Remove blank line
            doc = doc.replace('\n', '\n\n', 1)
            # Strip backspace-overprints for compatibility with autodoc
            doc = re.sub('\x08.', '', doc)
        else:
            # Pydoc does a poor job on other objects
            doc = func.__doc__
        self.__doc__ = 'Memoized version of %s' % doc

    def _cached_call(self, args, kwargs, shelving=False):
        """Call wrapped function and cache result, or read cache if available.

        This function returns the wrapped function output and some metadata.

        Arguments:
        ----------

        args, kwargs: list and dict
            input arguments for wrapped function

        shelving: bool
            True when called via the call_and_shelve function.


        Returns
        -------
        output: value or tuple or None
            Output of the wrapped function.
            If shelving is True and the call has been already cached,
            output is None.

        argument_hash: string
            Hash of function arguments.

        metadata: dict
            Some metadata about wrapped function call (see _persist_input()).
        """
        out = metadata = None
        func_id, args_id = path = self._get_output_identifiers(*args, **kwargs)
        # FIXME: The statements below should be try/excepted
        # Compare the function code with the previous to see if the
        # function code has changed
        if not (self._check_previous_func_code(stacklevel=4) and
                self.store_backend.contains_item(path)):
            if self._verbose > 10:
                _, func_name = get_func_name(self.func)
                func_info = self.store_backend.get_cached_func_info([func_id])
                self.warn('Computing func {}, argument hash {} in location {}'
                          .format(func_name, args_id, func_info['location']))
        elif shelving:
            return out, args_id, metadata
        else:
            try:
                start_time = time.time()
                out = self._load_item(path)
                if self._verbose > 4:
                    self._print_duration(time.time() - start_time,
                                         context='cache loaded ')
                return out, args_id, metadata
            except Exception:
                # XXX: Should use an exception logger
                _, signature = format_signature(self.func, *args, **kwargs)
                self.warn('Exception while loading results for '
                          '{}\n {}'.format(signature, traceback.format_exc()))

        if self._verbose > 0:
            print(format_call(self.func, args, kwargs))
        start_time = time.time()
        out = self.call(path, args, kwargs)
        duration = time.time() - start_time
        if self._verbose > 0:
            self._print_duration(duration)

        metadata = self._persist_input(duration, path, args, kwargs)
        if self.mmap_mode is not None:
            # Memmap the output at the first call to be consistent with
            # later calls
            out = self._load_item(path, metadata)

        return out, args_id, metadata

    def call_and_shelve(self, *args, **kwargs):
        """Call wrapped function, cache result and return a reference.

        This method returns a reference to the cached result instead of the
        result itself. The reference object is small and pickeable, allowing
        to send or store it easily. Call .get() on reference object to get
        result.

        Returns
        -------
        cached_result: MemorizedResult or NotMemorizedResult
            reference to the value returned by the wrapped function. The
            class "NotMemorizedResult" is used when there is no cache
            activated (e.g. location=None in Memory).
        """
        _, args_id, metadata = self._cached_call(args, kwargs, shelving=True)
        return MemorizedResult(self.store_backend, self.func_id, args_id,
                               metadata=metadata, verbose=self._verbose - 1,
                               timestamp=self.timestamp)

    def __call__(self, *args, **kwargs):
        return self._cached_call(args, kwargs)[0]

    def __getstate__(self):
        """ We don't store the timestamp when pickling, to avoid the hash
            depending from it.
        """
        state = self.__dict__.copy()
        state['timestamp'] = None
        return state

    # ------------------------------------------------------------------------
    # Private interface
    # ------------------------------------------------------------------------

    def _get_output_identifiers(self, *args, **kwargs):
        """Return the func identifier and input parameter hash of a result."""
        args_id = hashing.hash(
            filter_args(self.func, self.ignore, args, kwargs),
            coerce_mmap=self.mmap_mode is not None)
        return self.func_id, args_id

    def _hash_func(self):
        """Hash a function to key the online cache"""
        func_code_h = hash(getattr(self.func, '__code__', None))
        return id(self.func), hash(self.func), func_code_h

    def _write_func_code(self, func_code, first_line):
        """ Write the function code and the filename to a file.
        """
        # We store the first line because the filename and the function
        # name is not always enough to identify a function: people
        # sometimes have several functions named the same way in a
        # file. This is bad practice, but joblib should be robust to bad
        # practice.
        func_code = u'%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
        self.store_backend.store_cached_func_code([self.func_id], func_code)

        # Also store in the in-memory store of function hashes
        if PY3_OR_LATER:
            is_named_callable = (hasattr(self.func, '__name__') and
                                 self.func.__name__ != '<lambda>')
        else:
            is_named_callable = (hasattr(self.func, 'func_name') and
                                 self.func.func_name != '<lambda>')
        if is_named_callable:
            # Don't do this for lambda functions or strange callable
            # objects, as it ends up being too fragile
            func_hash = self._hash_func()
            try:
                _FUNCTION_HASHES[self.func] = func_hash
            except TypeError:
                # Some callable are not hashable
                pass

    def _check_previous_func_code(self, stacklevel=2):
        """
            stacklevel is the depth a which this function is called, to
            issue useful warnings to the user.
        """
        # First check if our function is in the in-memory store.
        # Using the in-memory store not only makes things faster, but it
        # also renders us robust to variations of the files when the
        # in-memory version of the code does not vary
        try:
            if self.func in _FUNCTION_HASHES:
                # We use as an identifier the id of the function and its
                # hash. This is more likely to falsely change than have hash
                # collisions, thus we are on the safe side.
                func_hash = self._hash_func()
                if func_hash == _FUNCTION_HASHES[self.func]:
                    return True
        except TypeError:
            # Some callables are not hashable
            pass

        # Here, we go through some effort to be robust to dynamically
        # changing code and collision. We cannot inspect.getsource
        # because it is not reliable when using IPython's magic "%run".
        func_code, source_file, first_line = get_func_code(self.func)
        try:
            old_func_code, old_first_line = extract_first_line(
                self.store_backend.get_cached_func_code([self.func_id]))
        except (IOError, OSError):  # some backend can also raise OSError
            self._write_func_code(func_code, first_line)
            return False
        if old_func_code == func_code:
            return True

        # We have differing code, is this because we are referring to
        # different functions, or because the function we are referring to has
        # changed?

        _, func_name = get_func_name(self.func, resolv_alias=False,
                                     win_characters=False)
        if old_first_line == first_line == -1 or func_name == '<lambda>':
            if not first_line == -1:
                func_description = ("{0} ({1}:{2})"
                                    .format(func_name, source_file,
                                            first_line))
            else:
                func_description = func_name
            warnings.warn(JobLibCollisionWarning(
                "Cannot detect name collisions for function '{0}'"
                .format(func_description)), stacklevel=stacklevel)

        # Fetch the code at the old location and compare it. If it is the
        # same than the code store, we have a collision: the code in the
        # file has not changed, but the name we have is pointing to a new
        # code block.
        if not old_first_line == first_line and source_file is not None:
            if os.path.exists(source_file):
                _, func_name = get_func_name(self.func, resolv_alias=False)
                num_lines = len(func_code.split('\n'))
                with open_py_source(source_file) as f:
                    on_disk_func_code = f.readlines()[
                        old_first_line - 1:old_first_line - 1 + num_lines - 1]
                on_disk_func_code = ''.join(on_disk_func_code)
                possible_collision = (on_disk_func_code.rstrip() ==
                                      old_func_code.rstrip())
            else:
                possible_collision = source_file.startswith('<doctest ')
            if possible_collision:
                warnings.warn(JobLibCollisionWarning(
                    'Possible name collisions between functions '
                    "'%s' (%s:%i) and '%s' (%s:%i)" %
                    (func_name, source_file, old_first_line,
                     func_name, source_file, first_line)),
                    stacklevel=stacklevel)

        # The function has changed, wipe the cache directory.
        # XXX: Should be using warnings, and giving stacklevel
        if self._verbose > 10:
            _, func_name = get_func_name(self.func, resolv_alias=False)
            self.warn("Function {0} (identified by {1}) has changed"
                      ".".format(func_name, self.func_id))
        self.clear(warn=True)
        return False

    def clear(self, warn=True):
        """Empty the function's cache."""
        func_id = self.func_id
        if self._verbose > 0 and warn:
            self.warn("Clearing function cache identified by %s" % func_id)
        self.store_backend.clear_path([func_id, ])

        func_code, _, first_line = get_func_code(self.func)
        self._write_func_code(func_code, first_line)

    def call(self, path, args, kwargs):
        """ Force the execution of the function with the given arguments and
            persist the output values.
        """
        output = self.func(*args, **kwargs)
        self.store_backend.dump_item(path, output, verbose=self._verbose)
        return output

    def _persist_input(self, duration, path, args, kwargs,
                       this_duration_limit=0.5):
        """ Save a small summary of the call using json format in the
            output directory.

            output_dir: string
                directory where to write metadata.

            duration: float
                time taken by hashing input arguments, calling the wrapped
                function and persisting its output.

            args, kwargs: list and dict
                input arguments for wrapped function

            this_duration_limit: float
                Max execution time for this function before issuing a warning.
        """
        start_time = time.time()
        argument_dict = filter_args(self.func, self.ignore,
                                    args, kwargs)

        input_repr = dict((k, repr(v)) for k, v in argument_dict.items())
        # This can fail due to race-conditions with multiple
        # concurrent joblibs removing the file or the directory
        metadata = {"duration": duration, "input_args": input_repr}

        self.store_backend.store_metadata(path, metadata)

        this_duration = time.time() - start_time
        if this_duration > this_duration_limit:
            # This persistence should be fast. It will not be if repr() takes
            # time and its output is large, because json.dump will have to
            # write a large file. This should not be an issue with numpy arrays
            # for which repr() always output a short representation, but can
            # be with complex dictionaries. Fixing the problem should be a
            # matter of replacing repr() above by something smarter.
            warnings.warn("Persisting input arguments took %.2fs to run.\n"
                          "If this happens often in your code, it can cause "
                          "performance problems \n"
                          "(results will be correct in all cases). \n"
                          "The reason for this is probably some large input "
                          "arguments for a wrapped\n"
                          " function (e.g. large strings).\n"
                          "THIS IS A JOBLIB ISSUE. If you can, kindly provide "
                          "the joblib's team with an\n"
                          " example so that they can fix the problem."
                          % this_duration, stacklevel=5)
        return metadata

    def _load_item(self, path, metadata=None):
        return self.store_backend.load_item(path, metadata=metadata,
                                            timestamp=self.timestamp,
                                            verbose=self._verbose)

    def _print_duration(self, duration, context=''):
        _, name = get_func_name(self.func)
        msg = '%s %s- %s' % (name, context, format_time(duration))
        print(max(0, (80 - len(msg))) * '_' + msg)

    # XXX: Need a method to check if results are available.

    # ------------------------------------------------------------------------
    # Private `object` interface
    # ------------------------------------------------------------------------

    def __repr__(self):
        return '{class_name}(func={func}, location={location})'.format(
            class_name=self.__class__.__name__,
            func=self.func,
            location=self.store_backend.location,)


###############################################################################
# class `Memory`
###############################################################################
class Memory(Logger):
    """ A context object for caching a function's return value each time it
        is called with the same input arguments.

        All values are cached on the filesystem, in a deep directory
        structure.

        Read more in the :ref:`User Guide <memory>`.

        Parameters
        ----------
        location: str or None
            The path of the base directory to use as a data store
            or None. If None is given, no caching is done and
            the Memory object is completely transparent. This option
            replaces cachedir since version 0.12.

        backend: str, optional
            Type of store backend for reading/writing cache files.
            Default: 'local'.
            The 'local' backend is using regular filesystem operations to
            manipulate data (open, mv, etc) in the backend.

        cachedir: str or None, optional

            .. deprecated: 0.12
                'cachedir' has been deprecated in 0.12 and will be
                removed in 0.14. Use the 'location' parameter instead.

        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments.

        compress: boolean, or integer, optional
            Whether to zip the stored data on disk. If an integer is
            given, it should be between 1 and 9, and sets the amount
            of compression. Note that compressed arrays cannot be
            read by memmapping.

        verbose: int, optional
            Verbosity flag, controls the debug messages that are issued
            as functions are evaluated.

        bytes_limit: int, optional
            Limit in bytes of the size of the cache.

        backend_options: dict, optional
            Contains a dictionnary of named parameters used to configure
            the store backend.
    """
    # ------------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------------

    def __init__(self, location=None, backend='local', cachedir=None,
                 mmap_mode=None, compress=False, verbose=1, bytes_limit=None,
                 backend_options=None):
        # XXX: Bad explanation of the None value of cachedir
        Logger.__init__(self)
        self._verbose = verbose
        self.mmap_mode = mmap_mode
        self.timestamp = time.time()
        self.bytes_limit = bytes_limit
        self.backend = backend
        self.compress = compress
        if backend_options is None:
            backend_options = {}
        self.backend_options = backend_options

        if compress and mmap_mode is not None:
            warnings.warn('Compressed results cannot be memmapped',
                          stacklevel=2)
        if cachedir is not None:
            if location is not None:
                raise ValueError(
                    'You set both "location={0!r} and "cachedir={1!r}". '
                    "'cachedir' has been deprecated in version "
                    "0.12 and will be removed in version 0.14.\n"
                    'Please only set "location={0!r}"'.format(
                        location, cachedir))

            warnings.warn(
                "The 'cachedir' parameter has been deprecated in version "
                "0.12 and will be removed in version 0.14.\n"
                'You provided "cachedir={0!r}", '
                'use "location={0!r}" instead.'.format(cachedir),
                DeprecationWarning, stacklevel=2)
            location = cachedir

        self.location = location
        if isinstance(location, _basestring):
            location = os.path.join(location, 'joblib')

        self.store_backend = _store_backend_factory(
            backend, location, verbose=self._verbose,
            backend_options=dict(compress=compress, mmap_mode=mmap_mode,
                                 **backend_options))

    @property
    def cachedir(self):
        warnings.warn(
            "The 'cachedir' attribute has been deprecated in version 0.12 "
            "and will be removed in version 0.14.\n"
            "Use os.path.join(memory.location, 'joblib') attribute instead.",
            DeprecationWarning, stacklevel=2)
        if self.location is None:
            return None
        return os.path.join(self.location, 'joblib')

    def cache(self, func=None, ignore=None, verbose=None, mmap_mode=False):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

            Parameters
            ----------
            func: callable, optional
                The function to be decorated
            ignore: list of strings
                A list of arguments name to ignore in the hashing
            verbose: integer, optional
                The verbosity mode of the function. By default that
                of the memory object is used.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. By default that of the memory object is used.

            Returns
            -------
            decorated_func: MemorizedFunc object
                The returned object is a MemorizedFunc object, that is
                callable (behaves like a function), but offers extra
                methods for cache lookup and management. See the
                documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        if func is None:
            # Partial application, to be able to specify extra keyword
            # arguments in decorators
            return functools.partial(self.cache, ignore=ignore,
                                     verbose=verbose, mmap_mode=mmap_mode)
        if self.store_backend is None:
            return NotMemorizedFunc(func)
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, MemorizedFunc):
            func = func.func
        return MemorizedFunc(func, location=self.store_backend,
                             backend=self.backend,
                             ignore=ignore, mmap_mode=mmap_mode,
                             compress=self.compress,
                             verbose=verbose, timestamp=self.timestamp)

    def clear(self, warn=True):
        """ Erase the complete cache directory.
        """
        if warn:
            self.warn('Flushing completely the cache')
        if self.store_backend is not None:
            self.store_backend.clear()

    def reduce_size(self):
        """Remove cache elements to make cache size fit in ``bytes_limit``."""
        if self.bytes_limit is not None and self.store_backend is not None:
            self.store_backend.reduce_store_size(self.bytes_limit)

    def eval(self, func, *args, **kwargs):
        """ Eval function func with arguments `*args` and `**kwargs`,
            in the context of the memory.

            This method works similarly to the builtin `apply`, except
            that the function is called only if the cache is not
            up to date.

        """
        if self.store_backend is None:
            return func(*args, **kwargs)
        return self.cache(func)(*args, **kwargs)

    # ------------------------------------------------------------------------
    # Private `object` interface
    # ------------------------------------------------------------------------

    def __repr__(self):
        return '{class_name}(location={location})'.format(
            class_name=self.__class__.__name__,
            location=(None if self.store_backend is None
                      else self.store_backend.location))

    def __getstate__(self):
        """ We don't store the timestamp when pickling, to avoid the hash
            depending from it.
        """
        state = self.__dict__.copy()
        state['timestamp'] = None
        return state
