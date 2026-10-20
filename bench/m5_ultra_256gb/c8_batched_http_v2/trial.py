"""Default-OFF E052 v2: correct one inherited startup identity field.

Keep the frozen v1 server, candidate, two-wave transport and cleanup functions.
The failed v1 specification carried E047's diagnostic_only identity addition;
the actual combined server never emits that field. Retain all other identities.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'c8_batched_http_v1/trial.py'
PREVIOUS_SHA='49d07aad7045a2d05a4e8474476fb4d73e51834f309215e024ca178178957404'
SCHEMA='c8-batched-http-spec-v2'
LABEL='c8-batched-mb50-two-waves-v2'
SCOPE='Actual candidate warm8/screen8 HTTP capture v2 with corrected startup identity; numerical replay and independent audit outstanding.'


def donor():
    if hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()!=PREVIOUS_SHA:raise ValueError('frozen v1 trial changed')
    spec=importlib.util.spec_from_file_location('_e052_v2_frozen_trial',PREVIOUS)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

base=donor()
need,sha,read,write,ref,load=base.need,base.sha,base.read,base.write,base.ref,base.load
old,server=base.old,base.server


def source_pins():
    return {**base.source_pins(),str(PREVIOUS):PREVIOUS_SHA,str(Path(__file__).resolve()):sha(__file__)}


def derive(original):
    need(original['schema']==base.SCHEMA and original['identity'].get('diagnostic_only') is True,
         'exact inherited v1 diagnostic identity expected')
    result=copy.deepcopy(original)
    del result['identity']['diagnostic_only']
    need(result['command'][3]==str(PREVIOUS.parent/'server.py') and result['command'][4]=='--run'
         and result['identity']['server_source_path']==str(PREVIOUS.parent/'server.py'), 'same actual frozen v1 server')
    result.update(schema=SCHEMA,label=LABEL,sources=source_pins(),scope=SCOPE)
    return result


def prepare(base_spec_ref,predecessor_ref):return derive(base.prepare(base_spec_ref,predecessor_ref))
def validate(spec):
    need(spec['schema']==SCHEMA and old().same(spec,prepare(spec['base_spec'],spec['predecessor'])),
         'changed v2 actual candidate specification')


def clone(fn,**changes):
    function=FunctionType(fn.__code__,dict(fn.__globals__,**changes),fn.__name__,fn.__defaults__,fn.__closure__)
    function.__kwdefaults__=dict(fn.__kwdefaults__ or {});return function


def capture_function(capture,system_type):
    return clone(base.capture_function(capture,system_type),validate=validate)


async def run(spec,out,*,checkpoint,backend=None):
    result=await clone(base.run,capture_function=capture_function,SCOPE=SCOPE)(spec,out,checkpoint=checkpoint,backend=backend)
    result.update(schema='c8-batched-http-result-v2',scope=SCOPE)
    write(Path(out)/'complete.json',result);return result


def join_child(child,spec_sha256):
    output=clone(base.join_child,validate=validate,SCOPE=SCOPE)(child,spec_sha256)
    receipt=read(output/'receipt.json');receipt['schema']='c8-batched-http-joined-input-v2'
    write(output/'receipt.json',receipt);return output


def main(argv=None):
    return clone(base.main,prepare=prepare,join_child=join_child,SCHEMA=SCHEMA,SCOPE=SCOPE)(argv)


if __name__=='__main__':raise SystemExit(main())
