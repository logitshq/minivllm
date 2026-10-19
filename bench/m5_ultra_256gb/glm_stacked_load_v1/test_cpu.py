"""CPU admission/rebinding tests; no device, service, process or network actions."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def load(name):
    path=HERE/(name+'.py');sp=importlib.util.spec_from_file_location('_stacked_test_'+name,path)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class Tests(unittest.TestCase):
    def test_all_adapters_import_without_operational_effects(self):
        for name in ('checkpoint','load_probe','owner','controller','audit_restore','audit_closed','prepare_inputs','preregister','execute'):
            with self.subTest(module=name):self.assertIsNotNone(load(name))

    def test_default_operational_entries_are_inert(self):
        for name in ('load_probe','owner','prepare_inputs','preregister','execute'):
            with self.subTest(module=name),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(load(name).main([]),0)

    def test_actual_e036_header_stat_and_evidence(self):
        c=load('checkpoint');manifest=c.manifest();metadata=c.metadata()
        self.assertEqual(len(metadata['shard_stats']),58)
        self.assertEqual(manifest['summary']['output_tensors'],4960)
        self.assertEqual(sum(r['payload_bytes'] for r in manifest['shards'].values()),203976457080)
        self.assertFalse(metadata['weight_payload_SHA_rechecked'])

    def test_old_62_shard_receipts_are_rejected(self):
        p=load('load_probe')
        old=json.loads((ROOT/'.artifacts/m5-ultra-256gb/glm-flash-inputs-v1/probe-plan.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'real E036 derived receipts'):p.checkpoint_evidence(old)

    def test_derived_probe_rejects_original_snapshot_before_admission(self):
        p=load('load_probe')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';path.write_text(json.dumps(dict(schema='GLM-official-load-plan-v1',snapshot='original-checkpoint')))
            with self.assertRaisesRegex(ValueError,'fixed GLM plan'):
                p.prepare(SimpleNamespace(plan=path,plan_sha256=sha(path)))

    def test_owner_loads_the_real_local_probe(self):
        o=load('owner');p=o.helper('../glm_flash/load_probe.py')
        self.assertEqual(Path(p.__file__),HERE/'load_probe.py')
        self.assertEqual(p.SNAPSHOT,ROOT/'.artifacts/models/GLM-5.3-Flash-4bit-stacked-v1')
        self.assertEqual(o.FIXED['../glm_flash/load_probe.py'],sha(HERE/'load_probe.py'))

    def test_e035_actual_predecessor_is_closed_and_bound(self):
        c=load('controller');p,_=c.predecessor(c.predecessor_manifest()['binding'])
        self.assertEqual(p['state'],'RESTORED');self.assertEqual(p['replacement']['pid'],39988)
        self.assertEqual(p['trials'],['off','runner','full'])

    def test_old_e033_predecessor_is_rejected(self):
        c=load('controller');binding=c.predecessor_manifest()['binding'].copy()
        binding['path']=str(ROOT/'docs/experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2/restored-journal.json')
        with self.assertRaisesRegex(c.Refuse,'exact actual E035 archive pair'):c.predecessor(binding)

    def test_original_probe_runtime_algorithms_are_identical(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
        before=functions(HERE.parent/'glm_flash/load_probe.py');after=functions(HERE/'load_probe.py')
        excluded={'contract','prepare','model_metadata'}
        self.assertGreater(len(before),15)
        self.assertTrue(all(after[k]==v for k,v in before.items() if k not in excluded))

    def test_monitor_and_lifecycle_algorithms_use_local_dependencies(self):
        o=load('owner');c=load('controller')
        self.assertIs(o.run.__globals__,vars(o));self.assertIs(o._live.__globals__,vars(o))
        self.assertIs(c.recover.__globals__['authorize'],c.authorize)
        self.assertIs(c.recover.__globals__['reaped'],c.reaped)
        self.assertIs(c.recover.__globals__['monitored_load'],c.monitored_load)
        self.assertIs(c.stop.__globals__['authorize'],c.authorize)
        self.assertIs(c._original_phase_reaps.__globals__,vars(c))
        self.assertEqual(o.MONITOR_POLICY['physical_max_gap_seconds'],1.0)
        self.assertEqual(o.MONITOR_POLICY['metadata_max_gap_seconds'],5.0)

    def test_source_seals_and_current_predecessor_validate(self):
        load('preregister').require_seals()

    def test_owner_cpu_validation_does_not_accept_legacy_inventory(self):
        o=load('owner');p=load('load_probe');old=ROOT/'.artifacts/m5-ultra-256gb/glm-flash-inputs-v1/model-runtime-inventory.json'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'probe.json';path.write_text(json.dumps(p.FIXED_EVIDENCE))
            plan=dict(verification=o.ref(old),probe_plan=o.ref(path))
            with self.assertRaisesRegex(ValueError,'real derived inventory scope'):
                o.verify_inventory(plan,p,lambda:None)

if __name__=='__main__':unittest.main()
