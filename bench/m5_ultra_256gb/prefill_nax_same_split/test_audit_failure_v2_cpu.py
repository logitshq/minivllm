"""Independent audit parser challenges; stdlib only, no device or process actions."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('e025_failure_audit', Path(__file__).with_name('audit_failure_v2.py'))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def save_pair(path, left, right, shape=(2,)):
    payload = left + right
    header = {'reference': {'dtype':'BF16', 'shape':list(shape), 'data_offsets':[0,len(left)]},
              'candidate': {'dtype':'BF16', 'shape':list(shape), 'data_offsets':[len(left),len(payload)]}}
    text = json.dumps(header).encode(); path.write_bytes(struct.pack('<Q',len(text))+text+payload)


class ParserChallenges(unittest.TestCase):
    def test_signed_zero_difference_and_first_byte(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'pair.safetensors'
            save_pair(path, b'\0\0\0\x80', b'\0\0\0\0')
            row = a.compare_pair(path, [2])
            self.assertFalse(row['bytes_equal'])
            self.assertEqual((row['differing_bytes'],row['differing_elements'],row['first_differing_byte_index']), (1,1,3))
            self.assertEqual(row['max_absolute_difference'],0)

    def test_equal_nan_bits_do_not_pass_finite_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'pair.safetensors'
            save_pair(path,b'\xc0\x7f'*2,b'\xc0\x7f'*2)
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                a.compare_pair(path,[2])

    def test_shape_or_trailing_payload_cannot_pass(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'pair.safetensors'; save_pair(path,b'\0'*4,b'\0'*4)
            with self.assertRaisesRegex(ValueError,'shape/dtype'):
                a.compare_pair(path,[1,2])
            path.write_bytes(path.read_bytes()+b'\0')
            with self.assertRaisesRegex(ValueError,'trailing'):
                a.tensors(path)

    def test_physical_identity_units_and_bound_fail_closed(self):
        raw={'unit':'byte','bytes per unit':1,'errors':[],'warnings':[],
             'processes':[{'pid':123,'footprint':200,'auxiliary':{'phys_footprint':100,'phys_footprint_peak':150}}]}
        self.assertEqual(a.physical(raw,123)['peak_bytes'],200)
        for mutation in ('pid','unit','budget','peak'):
            bad=copy.deepcopy(raw)
            if mutation=='pid': bad['processes'][0]['pid']=True
            elif mutation=='unit': bad['bytes per unit']=True
            elif mutation=='budget': bad['processes'][0]['footprint']=(2<<30)+1
            else: bad['processes'][0]['auxiliary']['phys_footprint_peak']=99
            with self.assertRaises(ValueError): a.physical(bad,123)


if __name__=='__main__': unittest.main()
