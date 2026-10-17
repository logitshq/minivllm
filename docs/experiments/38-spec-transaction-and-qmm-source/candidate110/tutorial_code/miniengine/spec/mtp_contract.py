"""V110 CPU-only shape/frontier guards; no MLX or model imports."""
import json
from pathlib import Path
import struct

PROJECTIONS = {'fc': (10240, 5120), 'layers.0.self_attn.q_proj': (5120, 12288),
               'layers.0.self_attn.k_proj': (5120, 1024), 'layers.0.self_attn.v_proj': (5120, 1024),
               'layers.0.self_attn.o_proj': (6144, 5120), 'layers.0.mlp.gate_proj': (5120, 17408),
               'layers.0.mlp.up_proj': (5120, 17408), 'layers.0.mlp.down_proj': (17408, 5120)}
NORMS = {'pre_fc_norm_embedding.weight': 5120, 'pre_fc_norm_hidden.weight': 5120,
         'layers.0.input_layernorm.weight': 5120, 'layers.0.post_attention_layernorm.weight': 5120,
         'layers.0.self_attn.q_norm.weight': 256, 'layers.0.self_attn.k_norm.weight': 256,
         'norm.weight': 5120}
QUANTIZATION = {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}


def require_bool(value):
    if type(value) is not bool:
        raise TypeError('enabled must be strict bool')
    return value


def schema():
    out = {name: ('BF16', [dim]) for name, dim in NORMS.items()}
    for name, (inputs, outputs) in PROJECTIONS.items():
        out[name + '.weight'] = ('U32', [outputs, inputs // 8])
        out[name + '.scales'] = ('U8', [outputs, inputs // 16])
    return out


def checkpoint_metadata(directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    text = config.get('text_config') or {}
    expected = {'hidden_size': 5120, 'num_attention_heads': 24, 'num_key_value_heads': 4,
                'head_dim': 256, 'intermediate_size': 17408, 'vocab_size': 248320,
                'mtp_num_hidden_layers': 1, 'rms_norm_eps': 1e-6}
    if config.get('model_type') != 'qwen3_5_mtp' or config.get('quantization') != QUANTIZATION:
        raise ValueError('only fixed27B external NVFP4 MTP is supported')
    if any(text.get(k) != v for k, v in expected.items()) or config.get('block_size') != 3:
        raise ValueError('MTP architecture contract differs')
    rope = text.get('rope_parameters') or {}
    if text.get('tie_word_embeddings') is not False or rope.get('rope_theta') != 1e7 or rope.get('partial_rotary_factor') != 0.25:
        raise ValueError('MTP tied-head/RoPE contract differs')
    index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    if set(index) != set(schema()):
        raise ValueError('exact23 checkpoint keys required')
    headers = {}; tensors = {}
    for name in sorted(set(index.values())):
        path = directory / name
        with path.open('rb') as stream:
            n = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < n < 16 * 1024 * 1024:
                raise ValueError('invalid safetensors header length')
            headers[name] = json.loads(stream.read(n))
        tensors.update({k: v for k, v in headers[name].items() if k != '__metadata__'})
    if set(tensors) != set(schema()):
        raise ValueError('unexpected tensors in external draft shard')
    for key, (dtype, shape) in schema().items():
        actual = tensors[key]
        if actual['dtype'] != dtype or actual['shape'] != shape or key not in headers[index[key]]:
            raise ValueError('checkpoint tensor contract differs: ' + key)
    return config, index, tensors


def accept_plan(appended, draft_count, accepted, delivered_count, verify_rows):
    values = (appended, draft_count, accepted, delivered_count, verify_rows)
    if any(type(x) is not int for x in values):
        raise TypeError('integer MTP frontier counts required')
    if not (0 <= appended <= draft_count <= 7 and 0 <= accepted <= draft_count
            and delivered_count == accepted + 1 and verify_rows == accepted + 1):
        raise ValueError('only authoritative committed target hidden prefix may be replayed')
    keep = min(accepted, appended)
    return {'trim': appended - keep, 'keep_appended': keep,
            'replay_draft_indices': list(range(keep, accepted)), 'bonus_hidden_index': accepted}
