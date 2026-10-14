"""OpenAI 兼容的 HTTP server，纯 asyncio 标准库，没有框架。

支持：POST /v1/completions、POST /v1/chat/completions（stream 和非 stream）、GET /v1/models、GET /health。
请求体里认：model, prompt | messages, max_tokens, temperature, top_p, top_k, stream, stream_options.include_usage,
ignore_eos（vLLM 扩展，跑分用）, logit_bias（只支持把 EOS 压到 -100 这一种用法：等价 ignore_eos）, stop（字符串列表）。
HTTP 只实现跑分和聊天客户端需要的子集：Content-Length 的请求体、chunked 的流式响应、keep-alive。
"""
import argparse, asyncio, json, time, uuid

from .engine.async_engine import AsyncLLMEngine
from .engine.llm_engine import LLMEngine
from .sampling_params import SamplingParams
from .tokenizer import ChatTokenizer, IncrementalDecoder


class Server:
    def __init__(self, engine: LLMEngine, tok: ChatTokenizer, model_name: str):
        self.aengine = AsyncLLMEngine(engine)
        self.tok, self.model_name = tok, model_name
        self.eos = engine.eos_ids

    # ---- HTTP 层 ----
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode().split("\r\n")
                method, path, _ = lines[0].split(" ", 2)
                headers = {k.lower(): v for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
                n = int(headers.get("content-length", "0").strip())
                body = await reader.readexactly(n) if n else b""
                await self.route(method, path, body, writer)
                if headers.get("connection", "").strip().lower() == "close":
                    break
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    async def route(self, method, path, body, writer):
        if method == "GET" and path in ("/health", "/v1/models"):
            payload = {"status": "ok"} if path == "/health" else {"object": "list", "data": [{"id": self.model_name, "object": "model"}]}
            return await self.send_json(writer, 200, payload)
        if method == "POST" and path in ("/v1/completions", "/v1/chat/completions"):
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                return await self.send_json(writer, 400, {"error": "bad json"})
            return await self.completion(req, writer, chat=path.endswith("chat/completions"))
        await self.send_json(writer, 404, {"error": "not found"})

    async def send_json(self, writer, status, payload):
        data = json.dumps(payload).encode()
        writer.write(f"HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
        await writer.drain()

    # ---- 推理 ----
    def parse(self, req, chat):
        if chat:
            ids = self.tok.apply_chat_template(req["messages"], enable_thinking=bool(req.get("enable_thinking", False)))
        else:
            p = req["prompt"]
            ids = self.tok.encode(p) if isinstance(p, str) else list(p)
        ignore_eos = bool(req.get("ignore_eos", False))
        lb = req.get("logit_bias") or {}
        if lb and all(float(v) <= -100 for v in lb.values()) and set(int(k) for k in lb) >= self.eos:
            ignore_eos = True
        stop = req.get("stop") or []
        stop = [stop] if isinstance(stop, str) else stop
        sp = SamplingParams(max_tokens=int(req.get("max_tokens", 256)), temperature=float(req.get("temperature", 1.0)),
                            top_p=float(req.get("top_p", 1.0)), top_k=int(req.get("top_k", 0)), ignore_eos=ignore_eos)
        return ids, sp, stop

    async def completion(self, req, writer, chat):
        ids, sp, stop = self.parse(req, chat)
        stream = bool(req.get("stream", False))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage", False))
        rid = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        dec = IncrementalDecoder(self.tok)
        n_out, text_all, finish = 0, "", "length"
        if stream:
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n")
            await writer.drain()
        async for tok, fin in self.aengine.generate(ids, sp):
            n_out += 1
            piece = dec.push(tok)
            if tok in self.eos and not sp.ignore_eos:
                piece, finish = "", "stop"
            text_all += piece
            if any(s in text_all for s in stop):
                finish = "stop"; fin = True
            if stream and piece:
                await self.send_chunk(writer, self.chunk(rid, created, piece, None, chat))
            if fin:
                break
        if stream:
            await self.send_chunk(writer, self.chunk(rid, created, "", finish, chat))
            if include_usage:
                await self.send_chunk(writer, {"id": rid, "object": "chat.completion.chunk" if chat else "text_completion", "created": created,
                                               "model": self.model_name, "choices": [], "usage": self.usage(len(ids), n_out)})
            done = b"data: [DONE]\n\n"
            writer.write(f"{len(done):x}\r\n".encode() + done + b"\r\n" + b"0\r\n\r\n")   # 最后一个 chunk + 结束
            await writer.drain()
        else:
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": text_all}, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text_all, "finish_reason": finish}
            await self.send_json(writer, 200, {"id": rid, "object": "chat.completion" if chat else "text_completion", "created": created,
                                               "model": self.model_name, "choices": [choice], "usage": self.usage(len(ids), n_out)})

    def chunk(self, rid, created, piece, finish, chat):
        if chat:
            choice = {"index": 0, "delta": ({"content": piece} if piece else {}), "finish_reason": finish}
        else:
            choice = {"index": 0, "text": piece, "finish_reason": finish}
        return {"id": rid, "object": "chat.completion.chunk" if chat else "text_completion", "created": created, "model": self.model_name, "choices": [choice]}

    @staticmethod
    def usage(n_in, n_out):
        return {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}

    async def send_chunk(self, writer, payload):
        data = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
        writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        await writer.drain()


async def main_async(a):
    engine = LLMEngine(a.model, max_seqs=a.max_seqs, max_len=a.max_len, max_batched_tokens=a.max_batched_tokens)
    tok = ChatTokenizer(a.model)
    srv = Server(engine, tok, a.served_model_name or a.model)
    srv.aengine.start()
    server = await asyncio.start_server(srv.handle, a.host, a.port)
    print(f"minivllm server on http://{a.host}:{a.port}  model={srv.model_name}  max_seqs={a.max_seqs} max_len={a.max_len} budget={a.max_batched_tokens}", flush=True)
    async with server:
        await server.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-seqs", type=int, default=32); ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-batched-tokens", type=int, default=512)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
