# SPDX-License-Identifier: MIT
# MTP seed/trim/replay structure adapted from mlx-vlm (c)2025 Prince Canuma.
# See LICENSE-MLX-VLM-MIT. Checkpoint is separately Apache-2.0 licensed.
"""Explicit opt-in singleton external MTP; no engine/target-verify routing.

New draft math reuses this engine's explicit norms/projections/attention.
Drafts are proposals only; target verification still needs independent strict
qualification. Target embedding/head owners are borrowed and never registered.
"""
from dataclasses import replace
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from ..cache import AttnCache
from ..context import ForwardContext
from ..model.attention import Attention
from ..model.layers import Linear, RMSNorm, apply_rope
from ..model.qwen3_5 import DecoderLayer
from .mtp_contract import checkpoint_metadata, require_bool, accept_plan, QUANTIZATION


class MTPKV(AttnCache):
    """offset counts stored draft K/V; position is the absolute RoPE frontier."""
    def __init__(self, position=0):
        super().__init__()
        self.position = position

    def trim(self, count):
        if type(count) is not int or not 0 <= count <= self.offset:
            raise ValueError('trim outside stored draft KV')
        self.offset -= count
        self.position -= count

    def update(self, keys, values):
        result = super().update(keys, values)
        self.position += keys.shape[2]
        return result


class MTPAttention(Attention):
    def _prefill(self, x, cache):
        if not isinstance(cache, MTPKV):
            raise TypeError('independent MTPKV required')
        count = x.shape[0]
        q, k, v, gate = self._project(x)
        q = apply_rope(q.transpose(1, 0, 2)[None], cache.position, self.rotary_dim, self.theta)
        k = apply_rope(k.transpose(1, 0, 2)[None], cache.position, self.rotary_dim, self.theta)
        keys, values = cache.update(k, v.transpose(1, 0, 2)[None])
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale,
                                                  mask='causal' if count > 1 else None)
        out = out[0].transpose(1, 0, 2).reshape(count, -1)
        return self.o_proj(out * mx.sigmoid(gate))


class ExternalMTPDraft(nn.Module):
    def __init__(self, target_cfg):
        super().__init__()
        cfg = replace(target_cfg, num_hidden_layers=1, layer_types=['full_attention'], quantization=dict(QUANTIZATION))
        self.fc = Linear(2 * cfg.hidden_size, cfg.hidden_size, cfg.quantization)
        self.pre_fc_norm_embedding = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, zero_centered=False)
        self.pre_fc_norm_hidden = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, zero_centered=False)
        layer = DecoderLayer(cfg, 'full_attention')
        layer.self_attn = MTPAttention(cfg)
        for norm in (layer.input_layernorm, layer.post_attention_layernorm,
                     layer.self_attn.q_norm, layer.self_attn.k_norm):
            norm.zero_centered = False
        self.layers = [layer]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, zero_centered=False)
        object.__setattr__(self, '_target', None)
        object.__setattr__(self, '_borrowed', None)
        object.__setattr__(self, '_parameter_identity', None)
        object.__setattr__(self, '_cache', [MTPKV()])
        object.__setattr__(self, '_seed_token', None)
        object.__setattr__(self, '_seed_hidden', None)
        object.__setattr__(self, '_last_drafts', None)
        self._round_appended = 0
        self._pending_round = False
        self.accept_lens, self.draft_lens = [], []

    def bind(self, target):
        cfg = target.cfg
        if cfg.rms_norm_eps != 1e-6 or cfg.rotary_dim != 64 or cfg.rope_theta != 1e7:
            raise ValueError('target RMS/RoPE contract differs')
        if (cfg.hidden_size, cfg.vocab_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.head_dim, cfg.intermediate_size) != (5120, 248320, 24, 4, 256, 17408):
            raise ValueError('unsupported target architecture')
        if cfg.quantization != QUANTIZATION or cfg.tie_word_embeddings:
            raise ValueError('fixed untied NVFP4 target required')
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_borrowed', (target.embed_tokens, target.lm_head))
        object.__setattr__(self, '_parameter_identity', tuple((name, id(a)) for name, a in tree_flatten(target.parameters())))
        return self

    def _bound(self):
        target = self._target
        if target is None or (target.embed_tokens is not self._borrowed[0] or target.lm_head is not self._borrowed[1]):
            raise RuntimeError('original borrowed embedding/head binding changed')
        if tuple((name, id(a)) for name, a in tree_flatten(target.parameters())) != self._parameter_identity:
            raise RuntimeError('target parameters changed after MTP binding')
        return target

    def reset(self, target=None):
        if target is not None:
            self.bind(target)
        self._bound()
        object.__setattr__(self, '_cache', [MTPKV()])
        object.__setattr__(self, '_seed_token', None)
        object.__setattr__(self, '_seed_hidden', None)
        object.__setattr__(self, '_last_drafts', None)
        self._round_appended, self._pending_round = 0, False
        self.accept_lens, self.draft_lens = [], []
        return self._cache

    def set_shared_kv(self, shared_kv_states, kv_offset, position=None, kv_valid_len=None, left_padding=None):
        # External MTP has its own KV; the target KV arrays are never borrowed.
        if left_padding is not None or type(kv_offset) is not int or kv_offset < 0:
            raise ValueError('single sequence with nonnegative integer frontier required')
        absolute = kv_offset if position is None else position
        if type(absolute) is not int or absolute < 0 or (kv_valid_len is not None and (type(kv_valid_len) is not int or kv_valid_len != kv_offset)):
            raise ValueError('unsupported MTP frontier metadata')
        cache = self._cache[0]
        if cache.offset == 0:
            cache.position = absolute
        elif cache.position != absolute:
            raise ValueError('drafter and committed target position differ')

    def forward_embedded(self, token_embed, hidden):
        if token_embed.ndim != 2 or token_embed.shape != hidden.shape or token_embed.shape[-1] != 5120:
            raise ValueError('paired [T,5120] embedding/normalized target hidden required')
        if token_embed.dtype != mx.bfloat16 or hidden.dtype != mx.bfloat16 or token_embed.shape[0] < 1:
            raise ValueError('nonempty BF16 draft inputs required')
        h = self.fc(mx.concatenate([self.pre_fc_norm_embedding(token_embed),
                                   self.pre_fc_norm_hidden(hidden)], axis=-1))
        ctx = ForwardContext('prefill', self._cache)
        return self.norm(self.layers[0](h, ctx, 0))

    def _forward_tokens(self, tokens, hidden):
        target = self._bound()
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype not in (mx.int32, mx.int64):
            raise ValueError('singleton integer [1,T] tokens required')
        if hidden.shape != (1, tokens.shape[1], 5120) or hidden.dtype != mx.bfloat16:
            raise ValueError('paired normalized hidden [1,T,5120] required')
        h = self.forward_embedded(target.embed_tokens(tokens[0]), hidden[0])
        return h[None]

    def _seed(self, hidden):
        target = self._bound()
        token = mx.argmax(target.lm_head_fn(hidden[0]), axis=-1).astype(mx.int32)[None]
        object.__setattr__(self, '_seed_token', token)
        object.__setattr__(self, '_seed_hidden', hidden)

    def prefill_from_target_hidden(self, input_ids, hidden, bonus_token):
        self._bound()
        if self._pending_round or self._cache[0].offset:
            raise RuntimeError('reset before MTP prompt seeding')
        if type(bonus_token) is not int or not 0 <= bonus_token < 248320:
            raise ValueError('target-sampled bonus token required')
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError('nonempty singleton prompt required')
        bonus = mx.array([[bonus_token]], dtype=input_ids.dtype)
        shifted = mx.concatenate([input_ids[:, 1:], bonus], axis=1)
        self._seed(self._forward_tokens(shifted, hidden))

    def draft_block(self, last_bonus, hidden, block_size=3):
        self._bound()
        if self._pending_round:
            raise RuntimeError('previous round must be accepted or reset before another draft')
        if type(last_bonus) is not int or not 0 <= last_bonus < 248320 or type(block_size) is not int or not 2 <= block_size <= 8:
            raise ValueError('bounded singleton greedy block2..8 required')
        tok = mx.array([[last_bonus]], dtype=mx.int32)
        h = hidden
        tokens = []
        self._round_appended = 0
        if self._seed_token is not None:
            tok, h = self._seed_token, self._seed_hidden
            tokens.append(tok)
            object.__setattr__(self, '_seed_token', None)
            object.__setattr__(self, '_seed_hidden', None)
        while len(tokens) < block_size - 1:
            h = self._forward_tokens(tok, h)
            self._round_appended += 1
            tok = mx.argmax(self._target.lm_head_fn(h[0]), axis=-1).astype(mx.int32)[None]
            tokens.append(tok)
        drafted = mx.concatenate(tokens, axis=1)
        object.__setattr__(self, '_last_drafts', drafted)
        self._pending_round = True
        return drafted

    def accept_verified_tokens(self, verify_hidden, draft_tokens, accepted, new_tokens):
        self._bound()
        if not self._pending_round or draft_tokens is not self._last_drafts:
            raise RuntimeError('must accept this exact outstanding draft object')
        plan = accept_plan(self._round_appended, draft_tokens.shape[1], accepted,
                           len(new_tokens), verify_hidden.shape[1])
        if verify_hidden.shape != (1, accepted + 1, 5120) or verify_hidden.dtype != mx.bfloat16:
            raise ValueError('only committed target final-normalized hidden prefix is allowed')
        if any(type(x) is not int or not 0 <= x < 248320 for x in new_tokens):
            raise ValueError('real target-verified delivery tokens required')
        # All checks precede state changes. Dropped speculative K/V is excluded
        # by valid offset; next update overwrites it, preserving vendor semantics.
        self._cache[0].trim(plan['trim'])
        chunks = [draft_tokens[:, i:i+1] for i in plan['replay_draft_indices']]
        hiddens = [verify_hidden[:, i:i+1] for i in plan['replay_draft_indices']]
        chunks.append(mx.array([[new_tokens[-1]]], dtype=mx.int32))
        hiddens.append(verify_hidden[:, accepted:accepted+1])
        self._seed(self._forward_tokens(mx.concatenate(chunks, axis=1), mx.concatenate(hiddens, axis=1)))
        self.accept_lens.append(accepted); self.draft_lens.append(draft_tokens.shape[1])
        self._round_appended, self._pending_round = 0, False
        object.__setattr__(self, '_last_drafts', None)

    def eval_arrays(self):
        cache = self._cache[0]
        return [a for a in (cache.keys, cache.values, self._seed_token, self._seed_hidden, self._last_drafts) if a is not None]

    def metadata(self):
        cache = self._cache[0]
        return {'draft_parameter_arrays': len(tree_flatten(self.parameters())),
                'borrowed_embedding_head_registered': False, 'draft_KV_valid_length': cache.offset,
                'draft_absolute_RoPE_frontier': cache.position, 'round_appended': self._round_appended,
                'pending_round': self._pending_round, 'accepted': list(self.accept_lens), 'drafted': list(self.draft_lens),
                'math': 'own explicit RMSNorm/QMM/attention, raw effective draft gamma; not vendor-byte-qualified',
                'engine_or_target_verify_integrated': False}


def load_draft(directory, target_cfg):
    config, index, _ = checkpoint_metadata(directory)
    if (target_cfg.hidden_size, target_cfg.num_attention_heads, target_cfg.num_key_value_heads,
            target_cfg.head_dim, target_cfg.intermediate_size, target_cfg.vocab_size,
            target_cfg.rms_norm_eps, target_cfg.rotary_dim, target_cfg.rope_theta) != (5120, 24, 4, 256, 17408, 248320, 1e-6, 64, 1e7):
        raise ValueError('fixed target architecture/RMS/RoPE required before draft allocation')
    if target_cfg.quantization != QUANTIZATION or target_cfg.tie_word_embeddings:
        raise ValueError('untied fixed NVFP4 target required before draft allocation')
    draft = ExternalMTPDraft(target_cfg)
    weights = {}
    for name in sorted(set(index.values())):
        weights.update(mx.load(str(Path(directory) / name)))
    # Checkpoint is converted MLX layout: its seven BF16 norm tensors already
    # contain effective gamma. Do not subtract1 or apply HF sanitize +1 again.
    draft.load_weights(list(weights.items()), strict=True)
    mx.eval(draft.parameters())
    return draft


class ExternalMTPAdapter:
    def __init__(self):
        self.enabled = False
        self.draft = None

    def prepare(self, target, directory, enabled=False):
        require_bool(enabled)
        if not enabled:
            self.enabled, self.draft = False, None
            return self
        # Build only after explicit opt-in. Never register it on target Module.
        draft = load_draft(directory, target.cfg)
        draft.reset(target)
        self.draft, self.enabled = draft, True
        return self

    def try_draft(self, last_bonus, hidden, *, block_size=3, batch_size=1, greedy=True, features=False, mode='decode'):
        if not self.enabled or self.draft is None:
            return None
        if type(batch_size) is not int or batch_size != 1 or greedy is not True or features is not False or mode != 'decode':
            return None  # Caller retains original normal/batched/non-greedy path.
        return self.draft.draft_block(last_bonus, hidden, block_size)
