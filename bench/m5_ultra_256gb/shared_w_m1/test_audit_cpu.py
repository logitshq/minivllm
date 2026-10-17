"""Small synthetic raw evidence; no actual GPU/result fabrication or runtime imports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e027_raw_audit_test',HERE/'audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def record(out,name,raw,dtype,shape,owner):
    path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),dtype=dtype,shape=shape,
                strides=[shape[1],1],encoding=a.DTYPES[dtype][2],finite=True if dtype=='bfloat16' else None,owner_id=owner)


def layout_record(out,operand,label,name):
    raw=struct.pack('<4Q',*(operand['shape']+operand['strides']))
    path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    row=dict(label=label,input_owner_id=operand['owner_id'],shape=operand['shape'],actual_strides_elements=operand['strides'],
             ensure_row_contiguous=False,source_sha256=a.LAYOUT_SHA,
             descriptor=dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=32,dtype='uint64',shape=[4],encoding='uint64-le'))
    write(path.with_suffix('.json'),row);return row


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.h=a.helpers()

    def case_fixture(self,out):
        meta=a.expected_cases()[3];case=meta['case'];n,k,t=meta['N'],meta['K'],meta['T']
        words=[]
        for r in range(t):
            row=[0 if i%2==0 else 0x8000 for i in range(k)]
            for j,pos in enumerate((0,15,16,31,32,511,512,k-1)):row[(pos+r)%k]=(0x3f80,0xbf80,0x3f00,0xbf00)[(j+r)%4]
            words+=row
        x=struct.pack('<'+'H'*len(words),*words)
        w=struct.pack('<2I',0x76543210,0xfedcba98)*(n*k//16)
        sc=(a.SCALE_CODES*((n*k//16+9)//10))[:n*k//16]
        before={'x':record(out,'x-before',x,'bfloat16',[t,k],10),
                'weight':record(out,'w-before',w,'uint32',[n,k//8],11),
                'scales':record(out,'s-before',sc,'uint8',[n,k//16],12)}
        bankafter={name:record(out,name+'-bank-after',raw,dtype,shape,ident) for name,raw,dtype,shape,ident in
                   (('weight',w,'uint32',[n,k//8],11),('scales',sc,'uint8',[n,k//16],12))}
        after={'x':record(out,'x-after',x,'bfloat16',[t,k],10),**{name:a.bare(v) for name,v in bankafter.items()}}
        products={name:record(out,name,b'\0'*(t*n*2),'bfloat16',[t,n],20+i) for i,name in enumerate(('native','control','shared'))}
        row=dict(case=case,T=t,N=n,K=k,bank=meta['id'],status='PASS',inputs_unchanged=True,inputs_before=before,inputs_after=after,
                 outputs=products,route=dict(launch_count_before=0,launch_count_after=1,kernel_build_count=1,required_route='shared-W',fallback=False))
        for side in ('before','after'):row['layout_'+side]=layout_record(out,row['inputs_'+side]['x'],case+'/x-'+side,f'cases/{case}/x-layout-{side}.bin')
        checks=[dict(case=case,label=label,status='PASS',exact=True,reference=products['native'],actual=products[name]) for name,label in
                (('control','original-expression-control-vs-native-M1'),('shared','shared-W-vs-native-M1'))]
        bank=dict(inputs_before={name:before[name] for name in ('weight','scales')},inputs_after=bankafter)
        write(out/'cases'/case/'case.json',row)
        return row,meta,checks,bank

    def run_case(self,out,row,meta,checks,bank):
        write(out/'cases'/meta['case']/'case.json',row)
        return a.audit_case(out,row,meta,checks,self.h,{}, {},bank,{})

    def test_complete_small_raw_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);row,meta,checks,bank=self.case_fixture(out)
            self.assertEqual(self.run_case(out,row,meta,checks,bank)['comparisons'],2)

    def test_case_corruptions_are_rejected(self):
        for change in ('shared-bit','nonfinite','shape','fallback','build-count','offset-layout','readonly-owner','seed','missing-output','check-binding'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);row,meta,checks,bank=self.case_fixture(out)
                if change in ('shared-bit','nonfinite'):
                    data=bytearray((out/'shared').read_bytes());data[:2]=b'\x01\x00' if change=='shared-bit' else b'\x80\x7f'
                    row['outputs']['shared']=record(out,'shared',data,'bfloat16',[2,48],22);checks[1]['actual']=row['outputs']['shared']
                if change=='shape':row['outputs']['control']['shape']=[1,96]
                if change=='fallback':row['route']['fallback']=True
                if change=='build-count':row['route']['kernel_build_count']=2
                if change=='offset-layout':row['layout_before']['actual_strides_elements']=[5121,1]
                if change=='readonly-owner':row['inputs_after']['weight']['owner_id']=99
                if change=='seed':
                    raw=bytearray((out/'x-before').read_bytes());raw[0]^=1
                    for side in ('before','after'):row['inputs_'+side]['x']=record(out,'x-'+side,raw,'bfloat16',[2,5120],10)
                if change=='missing-output':row['outputs'].pop('shared')
                if change=='check-binding':checks[1]['case']='other'
                with self.assertRaises((ValueError,KeyError)):self.run_case(out,row,meta,checks,bank)

    def test_descriptor_requires_actual_raw_gpu_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);operand=record(out,'x',b'\0'*16,'bfloat16',[2,4],1)
            r=layout_record(out,operand,'case/x-before','descriptor.bin')
            a.layout(out,r,operand,'case/x-before','descriptor.bin')
            raw=struct.pack('<4Q',2,4,5,1);(out/'descriptor.bin').write_bytes(raw)
            r['descriptor']['sha256']=hashlib.sha256(raw).hexdigest();write(out/'descriptor.json',r)
            with self.assertRaisesRegex(ValueError,'raw GPU'):a.layout(out,r,operand,'case/x-before','descriptor.bin')

    def test_dtype_bytes_and_unsigned_finite_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);row=record(out,'signed-zero',b'\0\0\0\x80','bfloat16',[1,2],1)
            self.assertEqual(a.descriptor(out,row,'bfloat16',[1,2],self.h,{}), (out/'signed-zero').resolve())
            row['bytes']=True
            with self.assertRaises(ValueError):a.descriptor(out,row,'bfloat16',[1,2],self.h,{})

    def test_workload_counts_and_structured_prefix(self):
        self.assertEqual(len(a.expected_banks()),11);self.assertEqual(len(a.expected_cases()),33)
        self.assertEqual(sum(c['real_weights'] for c in a.expected_cases()),6)
        for bank in a.SYNTHETIC_BANKS[1:]:
            self.assertNotEqual(a.structured_input_sha(bank,2,5120),a.structured_input_sha(bank,4,5120))

    def test_physical_distinct_readings_keep_each_bound(self):
        raw={'unit':'byte','bytes per unit':1,'errors':[],'warnings':[],
             'processes':[{'pid':9,'footprint':65536,'auxiliary':{'phys_footprint':32768,'phys_footprint_peak':49152}}]}
        self.assertEqual(self.h.physical_values(raw,9)['peak_bytes'],65536)
        for field in ('footprint','phys_footprint','phys_footprint_peak'):
            bad=copy.deepcopy(raw)
            if field=='footprint':bad['processes'][0][field]=(2<<30)+1
            else:bad['processes'][0]['auxiliary'][field]=(2<<30)+1
            with self.subTest(field=field),self.assertRaises(ValueError):self.h.physical_values(bad,9)


if __name__=='__main__':unittest.main()
