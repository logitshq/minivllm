"""Inventory baseline roles and local compatibility. Availability is not a benchmark result."""
import argparse
import json
import platform
import shutil
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core-report',type=Path);ap.add_argument('--omlx-report',type=Path)
    ap.add_argument('--lithos-report',type=Path,help='An actual local measurement artifact, never a README projection')
    ap.add_argument('--lithos-preparation-report',type=Path,help='CPU preparation only; never counted as a GPU measurement')
    ap.add_argument('--out',type=Path);args=ap.parse_args()
    macos=platform.mac_ver()[0]
    major=int(macos.split('.')[0]) if macos else None
    workspace=Path(__file__).resolve().parents[2]
    isolated_binary=workspace/'.artifacts/lithos-baseline/.venv/bin/lithos-metal'
    lithos=str(isolated_binary) if isolated_binary.exists() else shutil.which('lithos-metal')
    rows=[{'name':'HF PyTorch/MPS','role':'correctness and framework reference',
           'artifact':'tutorial_code/examples/chapter01-trace.json','performance_comparison':'not used for optimized-engine speedup'},
          {'name':'tutorial miniengine','role':'our independently implemented token-ID core engine','timing_scope':'engine scheduling + evaluated GPU work'},
          {'name':'oMLX','role':'optimized Apple Silicon engine reference','timing_scope':'HTTP SSE client; includes server and transport overhead'},
          {'name':'Lithos Metal','role':'candidate Apple Silicon engine reference',
           'local_binary':lithos,'required_macos_major':26,'local_macos':macos or None,
           'macos_requirement_met':major>=26 if major is not None else False,
           'hardware_note':'M4 Max 32/40-core paths were unmeasured in the cited upstream hardware table; verify local kernel route before interpreting speed.',
           'draft_note':'DSpark is a separate draft setup; do not label it DFlash2.',
           'upstream_source':'https://github.com/lithos-ai/lithos-metal#models-and-hardware',
           'upstream_projection':'168 token/s is a published M5 Max target-compute projection under assumed acceptance; it excludes draft/accept/rewind and is not a local end-to-end measurement.',
           'upstream_projection_source':'https://github.com/lithos-ai/lithos-metal/blob/46b2bc4cda57826c072d94afe90a58633aaeb6d9/docs/research/m5max-27b-backend-verification.md'}]
    for row,artifact in zip(rows[1:],[args.core_report,args.omlx_report,args.lithos_report]):
        row['measurement_status']='local artifact supplied' if artifact else 'unmeasured in this report'
        row['measurement_artifact']=str(artifact.resolve()) if artifact else None
        if artifact:
            data=json.loads(artifact.read_text())
            row['artifact_keys']=sorted(data.keys())
            row['interpretation']='Inspect the actual timing scope, prompt IDs, output counts and weight format in the linked artifact.'
    if args.lithos_preparation_report:
        preparation=json.loads(args.lithos_preparation_report.read_text())
        assert preparation['status']=='CPU prepared, GPU unmeasured'
        assert preparation['server_started'] is False and preparation['native_device_created'] is False
        rows[-1]['role']='Apple Silicon engine reference; installed and CPU prepared'
        rows[-1]['preparation_status']='prepared_not_measured'
        rows[-1]['preparation_artifact']=str(args.lithos_preparation_report.resolve())
        installation=workspace/'tutorial_code/baselines/lithos-installation.json'
        rows[-1]['installation']=json.loads(installation.read_text())
        if args.lithos_report is None:
            rows[-1]['measurement_status']='prepared_not_measured; no GPU server run'
    report={'host':platform.platform(),'baselines':rows,
            'rule':'Keep correctness tests, API measurements, core measurements and published projections as separate evidence.'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
