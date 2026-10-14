"""分词 + chat 模板。用 HF 的 tokenizers（Rust 库，只读 tokenizer.json），chat 模板手写 ChatML，不引 jinja2。

Qwen3.5/3.8 的模板（enable_thinking=False 时）：
    <|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
enable_thinking=True 时 assistant 后面只有 "<think>\\n"，模型自己写完 </think> 再回答。
增量 detokenize：一个 token 可能只是一个 UTF-8 字符的一部分，decode 出来是 �，要攒到完整为止再吐。
"""
from pathlib import Path

from tokenizers import Tokenizer


class ChatTokenizer:
    def __init__(self, model_dir: str):
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def apply_chat_template(self, messages: list[dict], enable_thinking: bool = False) -> list[int]:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))
        return self.encode("".join(parts))


class IncrementalDecoder:
    """逐 token 吐文本：攒着 ids，decode 出来末尾不是 � 的部分就是可以安全输出的。"""

    def __init__(self, tok: ChatTokenizer):
        self.tok, self.ids, self.emitted = tok, [], ""

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        new = text[len(self.emitted):]
        self.emitted = text
        return new
