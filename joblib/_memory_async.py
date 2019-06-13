import asyncio
import time

from .memory import NotMemorizedResult, NotMemorizedFunc, MemorizedFunc


class AsyncNotMemorizedFunc(NotMemorizedFunc):

    async def call_and_shelve(self, *args, **kwargs):
        return NotMemorizedResult(await self.func(*args, **kwargs))


class AsyncMemorizedFunc(MemorizedFunc):

    async def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        if asyncio.iscoroutine(out):
            out = await out
        return out

    async def call_and_shelve(self, *args, **kwargs):
        mem_result = super().call_and_shelve(*args, **kwargs)
        if asyncio.iscoroutine(mem_result):
            mem_result = await mem_result
        return mem_result

    async def _call(self, path, args, kwargs, shelving):
        self._before_call(args, kwargs)
        start_time = time.time()
        out = await self.func(*args, **kwargs)
        return self._after_call(path, args, kwargs, shelving, out, start_time)
