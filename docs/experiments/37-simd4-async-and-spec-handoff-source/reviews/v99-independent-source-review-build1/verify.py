"""V99 independent integration review; CPU stdlib, no imported worker/MLX/native."""
import ast
import hashlib
import json
from pathlib import Path
import sys

P = Path('/private/tmp/minivllm-engine-optimization-20261014')
D = Path(__file__).resolve().parent
Q = P / 'v99-simd4-cap64-fullqual-build1'
N = P / 'v99-native-simd4-cap64-build1'
C = P / 'candidate-v99-simd4-segment-build1-root/tutorial_code/miniengine'
A = P / 'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine'
refs = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    refs[str(path)] = sha(path)
    return path.read_text()


def exact_replace(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    required = {'fullqual.py': '3d42d8d2db315e0e35f57dfa169d818313920102339edf563651490811ba0c93',
                'source-pins.json': '3fcc81da7c5bcc2df278e7d23a70b915ff33eb72e0704667c81aa03365d39afa',
                'candidate-manifest.json': '735331f92c38c0d9c5dd5c07bdae1ff7816e60d4c4beac2c8e2af3354c27475e'}
    for name, expected in required.items():
        assert sha(Q / name) == expected
    pins = json.loads(read(Q / 'source-pins.json'))
    assert len(pins) == 153
    for path, expected in pins.items():
        assert sha(Path(path)) == expected, path
        refs[path] = expected
    assert sha(N / 'replay.mm') == '9e7c8b97038f864c7b66e13eb0dd926bf23f36d21b3b2b424058621e9e1ec384'
    assert sha(N / 'libv99replay.dylib') == '44790debc0f938797198664f8e4b0308c3d0563accaae8dc786e7a32a20c90a7'
    manifest = json.loads(read(Q / 'candidate-manifest.json'))['files']
    actual = {str(p.relative_to(C.parent.parent)): sha(p) for p in C.rglob('*.py')}
    assert len(actual) == 28 and actual == manifest
    parent_files = list(A.rglob('*.py'))
    assert len(parent_files) == 27
    for path in parent_files:
        rel = path.relative_to(A)
        if str(rel) != 'model/gdn.py':
            assert path.read_bytes() == (C / rel).read_bytes(), str(rel)
    helper = C / 'kernels/gdn_activated_simd4_segment_metal.py'
    assert helper.read_bytes() == (P / 'v95-gdn-simd4-segment-build1/gdn_activated_simd4_segment_metal.py').read_bytes()
    assert sha(helper) == '7f2d0136301079436839fbdee494a5c8849ffdd1f8861c9f78be05029819c5a6'
    gdn = read(C / 'model/gdn.py')
    gdn = exact_replace(gdn, '        self.set_simd4_segment(False)\n', '')
    gdn = exact_replace(gdn, '    def set_simd4_segment(self, enabled=True):\n        """Default-OFF V95 state transport; no new array parameters."""\n        if type(enabled) is not bool:\n            raise TypeError("enabled must be bool")\n        object.__setattr__(self, "_simd4_segment_enabled", enabled)\n\n', '')
    gdn = exact_replace(gdn, '        if object.__getattribute__(self, "_simd4_segment_enabled"):\n            from ..kernels.gdn_activated_simd4_segment_metal import gated_delta_step_activated_norm_reuse_metal\n', '')
    assert gdn == read(A / 'model/gdn.py')
    new_native = read(N / 'replay.mm')
    old_native = read(P / 'v89-native-capacity64-build1/replay.mm')
    restored = exact_replace(new_native, 'token<0||pos<0||pos>=(int)F().capacity||len!=pos+1||len>(int)F().capacity', 'token<0||pos<0||len!=pos+1||len>(int)F().capacity')
    restored = exact_replace(restored, ' const std::string expected_v95="custom_kernel_gdn_simd4_segment_cached_state_v95_build1_float_float_float_float_float_float_float_float_float";\n', '')
    restored = exact_replace(restored, 'const bool known=actual==expected||((actual==expected_v40||actual==expected_v95)&&ar.index==6&&br.index==9);', 'const bool known=actual==expected||(actual==expected_v40&&ar.index==6&&br.index==9);')
    old_report = next(line for line in old_native.splitlines(True) if 'for(auto&x:I().psos)' in line)
    new_report = next(line for line in restored.splitlines(True) if 'for(auto&x:I().psos)' in line)
    addition = 'NSMutableArray*args=[NSMutableArray array];for(auto&a:original.args)[args addObject:@{@"index":@(a.index),@"name":[NSString stringWithUTF8String:a.name.c_str()],@"type":@(a.type),@"access":@(a.access)}];'
    normalized_report = exact_replace(new_report, addition, '')
    normalized_report = exact_replace(normalized_report, '@"argument_reflection_observed":@(original.reflection),@"active_arguments":args,', '')
    assert normalized_report == old_report
    restored = exact_replace(restored, new_report, old_report)
    assert restored == old_native
    assert 'if(!a.active)continue;x.args.push_back({a.index,a.type,a.access,text(a.name)});' in new_native
    assert '(capacity!=64&&capacity!=256)' in new_native
    assert (N / 'ABI.h').read_bytes() == (P / 'v89-native-capacity64-build1/ABI.h').read_bytes()
    schema = json.loads(read(N / 'predicted-schema.json'))
    assert schema['status'] == 'PREDICTED_NOT_OBSERVED'
    assert schema['source95_helper_sha256'] == sha(helper)
    assert len(schema['active_arguments']) == 10
    assert schema['active_arguments'][6] == {'index': 6, 'name': 'S', 'type': 0, 'access': 0}
    assert schema['active_arguments'][9] == {'index': 9, 'name': 'S_new', 'type': 0, 'access': 1}
    worker = read(Q / 'fullqual.py')
    tree = ast.parse(worker)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    native_calls = [(n.lineno, n.args[0].value) for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'call' and n.args
                    and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)]
    replay_lines = [line for line, name in native_calls if name == 'v34_replay_step']
    assert len(replay_lines) == 1 and not any(name == 'v33_replay' for line, name in native_calls)
    schema_gate_line = next(i for i, line in enumerate(worker.splitlines(), 1) if 'assert actual_schema==predicted["active_arguments"]' in line)
    pairs_gate_line = next(i for i, line in enumerate(worker.splitlines(), 1) if 'assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)' in line)
    assert schema_gate_line < pairs_gate_line < replay_lines[0]
    baseline = ast.unparse(funcs['baseline_decode'])
    assert 'g.set_simd4_segment(False)' in baseline and 'g.set_simd4_segment(flag)' in baseline
    assert any(isinstance(n, ast.Try) and n.finalbody for n in funcs['baseline_decode'].body)
    assert 'helper_capture_calls==norm_capture_calls and not active_gdn' in worker
    assert 'len(norm_capture_calls)==48 and set(norm_capture_calls)==set(map(id,gdns))' in worker
    assert 'assert a_token==archived_cases[name]["A_ids"][step]' in worker
    assert 'canonical_digest!=archived["A_sha256"]' in worker and 'archived44==archived' in worker
    assert 'allclose' not in worker and 'vendor_rms' not in worker
    old_worker = ast.parse(read(P / 'v89-cap64-native-fullqual-build1/fullqual.py'))
    old_canonical = next(n for n in old_worker.body if isinstance(n, ast.FunctionDef) and n.name == 'canonical_output')
    assert dump(funcs['canonical_output']) == dump(old_canonical)
    # Source short-circuit ordering makes signed pos+1 bounded; exercise the declared scalar cases in Python.
    scalar_checks = 0
    def guard(token, pos, length, cap):
        return cap in (64, 256) and token >= 0 and pos >= 0 and pos < cap and length == pos + 1 and length <= cap
    for cap in (64, 256):
        for pos in (-2147483648, -1, 0, 1, 63, 64, 65, 255, 256, 2147483647):
            length = pos + 1 if 0 <= pos < cap else 0
            assert guard(0, pos, length, cap) == (0 <= pos < cap); scalar_checks += 1
            for bad in (-2147483648, -1, 0, 2147483647):
                assert not guard(0, pos, bad, cap); scalar_checks += 1
        for pos in range(cap):
            assert guard(0, pos, pos + 1, cap) and not guard(-1, pos, pos + 1, cap); scalar_checks += 2
    for cap in (0, 1, 32, 63, 65, 128, 257, 4294967295):
        assert not guard(0, 0, 1, cap); scalar_checks += 1
    assert scalar_checks == 748
    assert (N / 'guard-cpu.log').read_text().strip() == 'PASS_CPU_UBSAN_SCALAR_GUARD checks=748'
    assert not any(n == 'mlx' or n.startswith(('mlx.', 'numpy')) for n in sys.modules)
    out = {'status': 'PASS_CPU_V99_INTEGRATION_SOURCE_REVIEW_NO_GPU_QUALIFICATION',
           'reviewer_relationship': 'authored V95 dependency; independent V99 integration review, not independent rewrite/requalification of V95 mathematics',
           'source_pins_verified': len(pins), 'candidate_python_files': 28, 'other26_parentfiles_exact': True,
           'candidate_inverse_exact44': True, 'helper_exact95': True, 'native_inverse_exact89': True,
           'native_allowed_changes': ['exact95PSO S6 readonly/S_new9 nonreadonly alongside old exact rules', 'pos>=capacity rejects before pos+1', 'export existing original active-argument metadata'],
           'native_ABI_unchanged89': True, 'new_model_flag': 'constructor False, strict bool, only existing eligible44 route chooses95',
           'author_CPU_checks_rerun': 6, 'inert_rc': 0, 'scalar_cases_checked_in_Python': scalar_checks,
           'saved_UBSan_log_pinned_checks': 748, 'Metal_or_native_library_execution': False,
           'schema': 'predicted10 metadata only, actual exactPSO/full10+48bindingeligibility fail-stop before first native replay',
           'gate_line': schema_gate_line, 'pairs_count_gate_line': pairs_gate_line, 'first_native_replay_line': replay_lines[0],
           'fullmodel_target': 'actual trueCAP64, CPUcanonical256 only, old38+44 dual93x132 and96IDs each arm',
           'findings': [], 'remaining': ['root actual GPU reflection/full numerical qualification and saved lifecycle/resources', 'no performance or adoption claim'],
           'MLX_imported': False, 'GPU_used': False, 'frozen_files_modified': False}
    (D / 'receipt.json').write_text(json.dumps(out, indent=2) + '\n')
    (D / 'refsSHA.json').write_text(json.dumps(refs, indent=2) + '\n')
    print(json.dumps({'status': out['status'], 'pins': len(pins), 'schema_gate_before_replay': True, 'receipt_SHA256': sha(D / 'receipt.json')}))


if __name__ == '__main__':
    main()
