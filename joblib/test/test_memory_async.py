import asyncio
import shutil

from joblib.memory import Memory
from joblib.test.common import with_numpy, np
from .test_memory import (
    corrupt_single_cache_item,
    monkeypatch_cached_func_warn,
)


def check_identity_lazy_async(func, accumulator, location):
    """ Similar to check_identity_lazy_async for coroutine functions"""
    memory = Memory(location=location, verbose=0)
    func = memory.cache(func)

    @asyncio.coroutine
    def main():
        for i in range(3):
            for _ in range(2):
                value = yield from func(i)
                assert value == i
                assert len(accumulator) == i + 1

    asyncio.get_event_loop().run_until_complete(main())


def test_memory_integration_async(tmpdir):
    accumulator = list()

    @asyncio.coroutine
    def f(l):
        yield from asyncio.sleep(0.1)
        accumulator.append(1)
        return l

    @asyncio.coroutine
    def main():
        # Now test clearing
        for compress in (False, True):
            for mmap_mode in ('r', None):
                memory = Memory(location=tmpdir.strpath, verbose=10,
                                mmap_mode=mmap_mode, compress=compress)
                # First clear the cache directory, to check that our code can
                # handle that
                # NOTE: this line would raise an exception, as the database file is
                # still open; we ignore the error since we want to test what
                # happens if the directory disappears
                shutil.rmtree(tmpdir.strpath, ignore_errors=True)
                g = memory.cache(f)
                yield from g(1)
                g.clear(warn=False)
                current_accumulator = len(accumulator)
                out = yield from g(1)

            assert len(accumulator) == current_accumulator + 1
            # Also, check that Memory.eval works similarly
            evaled = yield from memory.eval(f, 1)
            assert evaled == out
            assert len(accumulator) == current_accumulator + 1

        # Now do a smoke test with a function defined in __main__, as the name
        # mangling rules are more complex
        f.__module__ = '__main__'
        memory = Memory(location=tmpdir.strpath, verbose=0)
        yield from memory.cache(f)(1)

    check_identity_lazy_async(f, accumulator, tmpdir.strpath)
    asyncio.get_event_loop().run_until_complete(main())


def test_no_memory_async():
    accumulator = list()

    @asyncio.coroutine
    def ff(x):
        yield from asyncio.sleep(0.1)
        accumulator.append(1)
        return x

    @asyncio.coroutine
    def main():
        memory = Memory(location=None, verbose=0)
        gg = memory.cache(ff)
        for _ in range(4):
            current_accumulator = len(accumulator)
            yield from gg(1)
            assert len(accumulator) == current_accumulator + 1

    asyncio.get_event_loop().run_until_complete(main())


@with_numpy
def test_memory_numpy_check_mmap_mode_async(tmpdir, monkeypatch):
    """Check that mmap_mode is respected even at the first call"""

    memory = Memory(location=tmpdir.strpath, mmap_mode='r', verbose=0)

    @memory.cache()
    @asyncio.coroutine
    def twice(a):
        return a * 2

    @asyncio.coroutine
    def main():
        a = np.ones(3)

        b = yield from twice(a)
        c = yield from twice(a)

        assert isinstance(c, np.memmap)
        assert c.mode == 'r'

        assert isinstance(b, np.memmap)
        assert b.mode == 'r'

        # Corrupts the file,  Deleting b and c mmaps
        # is necessary to be able edit the file
        del b
        del c
        corrupt_single_cache_item(memory)

        # Make sure that corrupting the file causes recomputation and that
        # a warning is issued.
        recorded_warnings = monkeypatch_cached_func_warn(twice, monkeypatch)
        d = yield from twice(a)
        assert len(recorded_warnings) == 1
        exception_msg = 'Exception while loading results'
        assert exception_msg in recorded_warnings[0]
        # Asserts that the recomputation returns a mmap
        assert isinstance(d, np.memmap)
        assert d.mode == 'r'

    asyncio.get_event_loop().run_until_complete(main())
