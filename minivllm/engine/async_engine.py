"""AsyncLLMEngine：一个后台任务不停地 step()，每个请求一个 asyncio.Queue 收 (token, finished)。
step() 本身是同步的（GPU 在跑的时候 Python 等待），每步之间 await asyncio.sleep(0) 让 HTTP 层有机会收发。
没活的时候等 event，不空转。
"""
import asyncio

from ..sampling_params import SamplingParams
from .llm_engine import LLMEngine


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.queues: dict[str, asyncio.Queue] = {}
        self.wake = asyncio.Event()
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            if not self.engine.has_work():
                self.wake.clear()
                await self.wake.wait()
            for rid, tok, fin in self.engine.step():
                q = self.queues.get(rid)
                if q is not None:
                    q.put_nowait((tok, fin))
                    if fin:
                        self.queues.pop(rid, None)
            await asyncio.sleep(0)

    async def generate(self, prompt_ids: list[int], sampling: SamplingParams):
        """异步生成器：逐个 yield (token, finished)。"""
        q: asyncio.Queue = asyncio.Queue()
        rid = self.engine.add_request(prompt_ids, sampling)
        self.queues[rid] = q
        self.wake.set()
        while True:
            tok, fin = await q.get()
            yield tok, fin
            if fin:
                return
