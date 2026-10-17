"""Root CPU binding utility. Never imports MLX or runs a server."""
import argparse
import json
from pathlib import Path
from adapter import qualification

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('receipt','release'):ap.add_argument('--'+k,type=Path,required=True);ap.add_argument('--'+k+'-sha256',required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert a.out.is_absolute() and a.receipt.is_absolute() and a.release.is_absolute()
    row={'status':'BOUND_V117_ACTUAL_PASS','receipt':str(a.receipt),'receipt_sha256':a.receipt_sha256,'release':str(a.release),'release_sha256':a.release_sha256}
    with a.out.open('x') as f:json.dump(row,f,indent=2);f.write('\n')
    qualification(a.out)
    print(json.dumps({'status':'V117_ACTUAL_BINDING_VALIDATED','binding':str(a.out),'inference_imported':False}))

if __name__=='__main__':main()
