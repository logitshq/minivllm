"""CPU-only regression for nested model EOS and tokenizer special-token stopping metadata."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from tutorial_code.miniengine.eos import read_eos_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/'config.json').write_text(json.dumps({'text_config':{'eos_token_id':11}}))
        assert read_eos_ids(tmp)=={11}
        (root/'generation_config.json').write_text(json.dumps({'eos_token_id':[12,13]}))
        assert read_eos_ids(tmp)=={11,12,13}
        (root/'tokenizer_config.json').write_text(json.dumps({'eos_token':'<end>'}))
        (root/'tokenizer.json').write_text(json.dumps({'added_tokens':[{'id':14,'content':'<end>'}]}))
        assert read_eos_ids(tmp)=={11,12,13,14}
    report={'nested_config_eos':True,'generation_config_eos_list':True,'tokenizer_special_token_id_lookup':True,'scope':'metadata only; tokenizer encoding is outside the core'}
    print(json.dumps(report,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
