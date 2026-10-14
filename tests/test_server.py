"""第 5 步：server 端到端。起一个子进程跑 0.8B nvfp4，用 httpx 打 /v1/chat/completions 和流式 /v1/completions。"""
import json, os, socket, subprocess, sys, time
import httpx
import pytest

MODEL = "models/Qwen3.5-0.8B-nvfp4"


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "minivllm.server", "--model", MODEL, "--port", str(port), "--max-seqs", "4", "--max-len", "1024", "--max-batched-tokens", "256"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        proc.kill(); raise RuntimeError(proc.stdout.read())
    yield base
    proc.kill()


def test_chat_non_stream(server):
    r = httpx.post(server + "/v1/chat/completions", json={"model": "x", "messages": [{"role": "user", "content": "用一句话解释什么是 KV cache。"}], "max_tokens": 48, "temperature": 0}, timeout=120)
    assert r.status_code == 200
    d = r.json()
    text = d["choices"][0]["message"]["content"]
    assert "KV" in text and d["usage"]["prompt_tokens"] == 19 and 1 <= d["usage"]["completion_tokens"] <= 48
    assert d["choices"][0]["finish_reason"] in ("stop", "length")


def test_completions_stream_with_usage_and_ignore_eos(server):
    body = {"model": "x", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    chunks, usage, done = [], None, False
    with httpx.stream("POST", server + "/v1/completions", json=body, timeout=120) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True; break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            elif obj["choices"] and obj["choices"][0].get("text"):
                chunks.append(obj["choices"][0]["text"])
    assert done and usage["completion_tokens"] == 16 and len("".join(chunks)) > 0


def test_concurrent_requests(server):
    import concurrent.futures as cf
    def one(i):
        r = httpx.post(server + "/v1/completions", json={"model": "x", "prompt": f"Count: {i},", "max_tokens": 8, "temperature": 0, "ignore_eos": True}, timeout=120)
        return r.json()["usage"]["completion_tokens"]
    with cf.ThreadPoolExecutor(6) as ex:
        assert list(ex.map(one, range(6))) == [8] * 6
