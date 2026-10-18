"""Freeze coding prompts and exact tokenizer/template inputs on CPU only.

Does not generate candidate answers, import MLX, start a listener or measure
performance. Input lengths are reported, never padded/truncated to win a run.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path('/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def content(task, source):
    return '\n\n'.join((source['shared_instruction'],
        task['title'], 'Required API:\n```python\n' + '\n'.join(task['api']) + '\n```',
        'Behavior:\n' + '\n'.join(f'{i}. {x}' for i, x in enumerate(task['requirements'], 1)),
        'Examples and edge cases:\n' + '\n'.join(f'- {x}' for x in task['examples']),
        source['shared_review_context']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--source', type=Path, default=HERE / 'fixtures/tasks-v1.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.model.resolve().name == DEFAULT_MODEL.name, 'new model needs a separate explicit preparation'
    source = json.loads(args.source.read_text())
    assert source['schema'] == 'E023-coding-tasks-v1' and len(source['tasks']) == 8
    assert len({x['id'] for x in source['tasks']}) == 8
    args.out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    raw = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    config = json.loads((args.model / 'generation_config.json').read_text())
    eos = config['eos_token_id']
    assert eos == [248046, 248044]
    rows = []
    for task in source['tasks']:
        messages = [{'role': 'user', 'content': content(task, source)}]
        kwargs = {'add_generation_prompt': True, 'enable_thinking': False}
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        assert raw.encode(prompt, add_special_tokens=False).ids == ids
        assert raw.decode(ids, skip_special_tokens=False) == prompt
        assert len(ids) + 512 <= 2048, 'raise model context explicitly in a new plan, never truncate fixtures'
        rows.append({'fixture_id': task['id'], 'title': task['title'], 'messages': messages,
                     'prompt_text': prompt, 'prompt_ids': ids, 'prompt_tokens': len(ids),
                     'max_output_tokens': 512, 'temperature': 0, 'normal_eos': True,
                     'eos_token_ids': eos, 'template_options': kwargs,
                     'native_path': '/v1/completions', 'expected_output': None})
    pins = {str(args.source.resolve()): sha(args.source), str(Path(__file__).resolve()): sha(__file__)}
    for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        pins[str(args.model / name)] = sha(args.model / name)
    payload = {'schema': 'E023-D-inputs-v1', 'status': 'CPU_INPUTS_ONLY_NOT_RUNTIME_QUALIFIED',
               'model': str(args.model.resolve()), 'revision': args.model.resolve().name,
               'source_sha256': sha(args.source), 'versions': {p: version(p) for p in ('transformers','tokenizers','jinja2')},
               'pins': pins, 'fixtures': rows, 'prompt_token_range': [min(r['prompt_tokens'] for r in rows), max(r['prompt_tokens'] for r in rows)],
               'no_padding_or_truncation': True, 'no_output_or_speed_based_selection': True,
               'normal_eos_policy': 'Actual EOS terminates normally; no minimum output length, no replacing short outputs.',
               'scope': 'One fixed eight-task coding mix; exact lengths reported, not an exact 1024-token synthetic shape.'}
    write(args.out / 'fixtures.json', payload)
    shutil.copyfile(args.source, args.out / 'tasks.json')
    write(args.out / 'receipt.json', {'status': payload['status'], 'fixture_sha256': sha(args.out / 'fixtures.json'),
        'prompt_tokens': {r['fixture_id']: r['prompt_tokens'] for r in rows}, 'max_output_tokens': 512,
        'no_model_loaded': True, 'versions': payload['versions']})
    print(json.dumps({'out': str(args.out), 'prompt_tokens': [r['prompt_tokens'] for r in rows],
                      'sha256': sha(args.out / 'fixtures.json')}))


if __name__ == '__main__':
    main()
