"""CPU regressions, including the independently captured real v1 startup."""
import base64
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.c8_batched_http_v2 import trial as t
from bench.m5_ultra_256gb.c8_batched_http_v1.test_cpu import Fake,fake_spec

ACTUAL=t.HERE.parents[2]/'.artifacts/m5-ultra-256gb/c8-batched-http-v1-screen/trials/c8-batched-mb50-two-waves'
PINS={'spec.json':'8f7d0095c98d70bb4d72609a1a91e48581da32fc0695e6363f0fadbafa0c281c',
      'trace.jsonl':'1b72cbc2557312a25049e63e98d42c3c564675cf2120b815f59992a0f7b03e3f',
      'complete.json':'9714deef629cf6cb384b34de4869c4082d93a66f55a122097e16bc883b23e675'}


def actual():
    for name,digest in PINS.items():
        if t.sha(ACTUAL/name)!=digest:raise ValueError('closed real startup regression fixture changed')
    spec=t.read(ACTUAL/'spec.json');lines=(ACTUAL/'trace.jsonl').read_bytes().splitlines()
    assert len(lines)==1 and t.read(ACTUAL/'complete.json')['status']=='FAIL'
    startup=json.loads(lines[0]);assert startup['event']=='startup'
    return spec,startup['identity']


class Source(unittest.TestCase):
    def test_expected_identity_matches_independent_actual_server_startup(self):
        old,identity=actual();before=copy.deepcopy(old);new=t.derive(old)
        self.assertEqual(new['identity'],identity)
        self.assertEqual(old,before);self.assertTrue(old['identity']['diagnostic_only'])
        capture=t.old().module(t.old().CAPTURE)
        raw={'status':200,'body_base64':base64.b64encode(json.dumps({'data':[{'id':'native-target-only'}],'benchmark_identity':identity}).encode()).decode()}
        with self.assertRaisesRegex(ValueError,'startup identity'):capture.ready_identity(raw,old['identity'])
        self.assertTrue(capture.ready_identity(raw,new['identity']))
    def test_only_authorized_spec_fields_change_and_server_stays_frozen(self):
        old,_=actual();new=t.derive(old)
        self.assertEqual({k for k in old if old[k]!=new[k]}, {'schema','label','sources','scope','identity'})
        self.assertEqual(new['command'],old['command']);self.assertEqual(new['schema'],t.SCHEMA);self.assertEqual(new['label'],t.LABEL)
        self.assertEqual(new['sources'][str(t.PREVIOUS)],t.PREVIOUS_SHA)
        self.assertIn(str(t.HERE/'trial.py'),new['sources'])
        for key in ('numerical_qualified','full_state_qualified','performance_qualified'):self.assertIs(new[key],False)
    def test_missing_wrong_legacy_marker_or_changed_server_rejected(self):
        old,_=actual()
        for mutate in (lambda d:d['identity'].pop('diagnostic_only'),lambda d:d['identity'].update(diagnostic_only=False),
                       lambda d:d['command'].__setitem__(3,'other/server.py')):
            bad=copy.deepcopy(old);mutate(bad)
            with self.assertRaises(ValueError):t.derive(bad)
    def test_original_capture_code_private_validator_and_startup_schema(self):
        old=t.old();capture=old.module(old.CAPTURE);call=t.capture_function(capture,old.System)
        self.assertIs(call.__code__,capture.capture.__code__);self.assertIs(call.__globals__['validate'],t.validate)
        self.assertIsNot(capture.capture.__globals__['validate'],t.validate)
        self.assertEqual(call.__globals__['route_startup'].__globals__['ROUTE_SCHEMA'],t.server().ROUTE_SCHEMA)
    def test_prepare_keeps_original_full_prerequisite_call_and_default_off(self):
        old,_=actual();seen=[]
        with patch.object(t.base,'prepare',side_effect=lambda a,b:seen.append((a,b)) or copy.deepcopy(old)):
            result=t.prepare({'base':'synthetic'},{'predecessor':'synthetic'})
        self.assertEqual(seen,[({'base':'synthetic'},{'predecessor':'synthetic'})]);self.assertEqual(result['schema'],t.SCHEMA)
        with patch.object(t,'prepare',side_effect=AssertionError('default must be inert')),redirect_stdout(io.StringIO()):
            self.assertEqual(t.main([]),0)


class Lifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_original_two_wave_capture_and_cleanup_return_v2_receipt(self):
        spec=fake_spec();spec.update(schema=t.SCHEMA,label=t.LABEL)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'child';fake=Fake(spec)
            with patch.object(t,'validate',lambda value:None):result=await t.run(spec,out,checkpoint=lambda *a:None,backend=fake)
            self.assertEqual(result['schema'],'c8-batched-http-result-v2');self.assertEqual(result['status'],'PASS')
            self.assertEqual(fake.calls,[spec['jobs'][:8],spec['jobs'][8:]]);self.assertTrue(result['server_reaped'])
            self.assertEqual(t.read(out/'complete.json'),result);self.assertFalse(result['performance_qualified'])
    async def test_original_failure_still_preserved_and_reaped(self):
        spec=fake_spec();spec.update(schema=t.SCHEMA,label=t.LABEL)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'child';fake=Fake(spec,fail_first=True)
            with patch.object(t,'validate',lambda value:None):result=await t.run(spec,out,checkpoint=lambda *a:None,backend=fake)
            self.assertEqual(result['schema'],'c8-batched-http-result-v2');self.assertEqual(result['status'],'FAIL')
            self.assertEqual(len(fake.calls),1);self.assertTrue(result['server_reaped'])
            self.assertTrue((out/'client/warm/group.json').is_file());self.assertFalse((out/'client/screen').exists())


if __name__=='__main__':unittest.main()
