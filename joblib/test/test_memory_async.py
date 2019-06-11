import asyncio

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


def test_memory_async(tmpdir):
    accumulator = list()

    @asyncio.coroutine
    def coro(x):
        yield from asyncio.sleep(0.1)
        accumulator.append(1)
        return x

    check_identity_lazy_async(coro, accumulator, tmpdir.strpath)


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
