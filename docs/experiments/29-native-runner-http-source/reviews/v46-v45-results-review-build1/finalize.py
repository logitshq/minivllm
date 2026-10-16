from pathlib import Path
import hashlib,json,re
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).parent;O=P/'v23-root-control-build1/native-http-v45-attempt1';C=O/'child-output';r=json.loads((D/'receipt.json').read_text());rows=[]
for p in sorted(C.glob('owner-resources-??-raw.json'))+sorted(O.glob('external-resource-??-raw.json')):
 v=json.loads(p.read_text());assert int(v['pressure_level']['text'])==v['pressure_level_value']==1
 assert int(v['processinfo_thermal_metadata']['raw']['stdout'])==v['processinfo_thermal_metadata']['code']==0
 assert int(re.search(r'free percentage: (\d+)%',v['memory_pressure']['text'])[1])==v['free_percent']
 assert v['power_metadata']['source'] in ('AC Power','Battery Power') and v['process_inventory_ok'] and not v['known_inference_processes']
 rows.append({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'label':v['label'],'power':v['power_metadata']['source'],'free_percent':v['free_percent']})
assert len(rows)==3 and [x['power'] for x in rows]==['AC Power','Battery Power','AC Power']
r['V45']['owner_and_external_raw_resource_samples']=rows;r['V45']['all_raw_resource_samples']=r['V45']['per_arm_raw_samples']+len(rows)
(D/'receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
(D/'REVIEW.md').write_text('''# V46 actual and V45 STOP independent CPU audit
1. PASS audit of saved records and frozen sources only; no independent GPU recomputation.
2. V46 actual ab6987a4…67252:58pins,12408 strict hashes match corresponding actual44 A rows,124native/4reset/ICB[64,60].
3. Three queued requests validate132 outputs at all93 native steps; fallback temp.7/topk1 and max1 prefill-only work; reentry checks32IDs+final132 only.
4. Capture131seed and retained MLX owners readonly; terminal batch/staging/ahead/request cleanup passes; original41 adapter bf7bea08…fea0 unchanged.
5. V46 resources18 raw samples AC/thermal0/pressure1,minfree47%,swap/SO0; owner69496/child69505 reaped0,externally strictly absent.
6. V45 controller212757fa…a4321,pins56564d41…6b7e:14controller/61runtime pins match; source binds actual46/candidate44/native40build2.
7. request/group AST equals frozen official attempt2; NOL/OLN/LNO plan remains9fresh,3warm3measure; SIGTERM raises KeyboardInterrupt into cleanup.
8. Actual STOP:4/9 completed; no suite-complete and no owner-cleanup JSON. Final main AC guard also raises on Battery; external release supplies owner cleanup proof.
9. First Battery is C1-B2-lithos after-C1-measure-1/resources-12-raw.json,bf2cb2ee…8145; all warm samples before it were AC.
10. Keep that arm’s2 raw measure groups but exclude entire incomplete arm; no AC/Battery pooling,continuation,formal PASS or victory claim.
11. Recomputed four completed arm X/Y/TTFT/E2E match suite-failure;29 savedHTTP groups/928tokens include incomplete arm; native192rawIDs all canonical32.
12. Native completed HTTP arm proves186steps/6reset/ICB[96,90],48S pingpong; startup ordinary warm32 remains explicit and outside client54-request plan.
13. All85 saved raw resource samples replayed:thermal0/pressure1,per-arm swap/SO0,minfree44%; firstbad remains Battery even though external post-release power is AC.
14. Owner70057/driver70059 reapedexit1; five servers70078/70257/70441/70625/70811 waited-15 and externally strictly absent; five ports ECONNREFUSED61.
15. Four completed arm numbers are descriptive only; no nine-load C1 campaign/Pareto/adoption conclusion; no C2/C4 evidence.
''')
print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'REVIEW.md',D/'receipt.json',D/'audit.py',D/'finalize.py')},indent=2))
