#!/usr/bin/env python3
"""Fresh recording: baseline, fixed KF, frozen-confidence KF, every-frame matcher."""
import argparse,json,subprocess,time
from pathlib import Path
from adaptivefusion.state import file_sha256
def main():
 p=argparse.ArgumentParser()
 for k in ['selection','datasets','paths','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();repo=Path(__file__).resolve().parents[1];selection=json.loads(a.selection.read_text());paths=json.loads(a.paths.read_text());parent_path=Path(paths['confidence_protocol']);parent=json.loads(parent_path.read_text())
 if a.output.exists():raise ValueError('refuse overwrite')
 if file_sha256(parent_path)!=selection['parent_confidence_protocol_sha256'] or parent['config']!=selection['confidence']:raise ValueError('frozen confidence configuration changed')
 for path,sha in selection['frozen_source_sha256'].items():
  if file_sha256(repo/path)!=sha:raise ValueError('source changed: '+path)
 a.output.mkdir();commands=[];status=[];begin=time.perf_counter()
 def run(stage,command):
  commands.append(dict(stage=stage,command=list(map(str,command))));(a.output/'commands.json').write_text(json.dumps(commands,indent=2));start=time.perf_counter()
  with (a.output/(stage+'.log')).open('x') as f:r=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,timeout=4200)
  value=dict(stage=stage,returncode=r.returncode,seconds=time.perf_counter()-start,status='complete' if r.returncode==0 else 'failed');status.append(value);(a.output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(value),flush=True);return value
 py=__import__('sys').executable
 command=[py,'-B',repo/'python/run_independent_keyframe_validation.py','--selection',a.selection,'--datasets',a.datasets,'--parent-protocol',paths['fixed_protocol'],'--baseline-binary',paths['baseline_binary'],'--replay-binary',paths['replay_binary'],'--graph-binary',paths['graph_binary'],'--native-binary',paths['native_binary'],'--lightglue-source',paths['lightglue_source'],'--weights',paths['weights'],'--native-library-path',paths['native_library_path'],'--output',a.output/'study']
 run('matched-study',command)
 manifest=a.output/'study/keyframe-protocol.json';sources=[]
 if manifest.exists():
  m=json.loads(manifest.read_text())
  if m['matcher']!=parent['frozen_matching'] or m['graph']!=parent['frozen_base_graph']:raise ValueError('matching/graph parameters changed')
  for scene in selection['selected_sequences']:
   if scene in m['scenes'] and (a.output/'study/auxiliary'/scene/'corrected.tum').exists():sources.append(dict(scene=scene,split='new_independent_recording',manifest=str(manifest.resolve()),evidence=str((a.output/'study/auxiliary'/scene).resolve())))
 (a.output/'confidence-sources.json').write_text(json.dumps(sources,indent=2)+'\n')
 execution=parent.copy();execution.update(stage='V34 frozen confidence fresh recording',variants=['confidence_quadratic'],fit_budget=len(sources),evaluation_scope='new independent recording, all parameters fromV33 unchanged; controls already fitted in matched-study')
 (a.output/'confidence-execution-protocol.json').write_text(json.dumps(execution,indent=2)+'\n')
 if sources:
  run('confidence',[py,'-B',repo/'python/run_confidence_fusion.py','--protocol',a.output/'confidence-execution-protocol.json','--sources',a.output/'confidence-sources.json','--binary',paths['confidence_binary'],'--native-library-path',paths['native_library_path'],'--output',a.output/'confidence'])
 else:
  status.append(dict(stage='confidence',status='missing_baseline_or_fixed_constraints'));(a.output/'status.json').write_text(json.dumps(status,indent=2))
 for path,sha in selection['frozen_source_sha256'].items():
  if file_sha256(repo/path)!=sha:raise ValueError('source changed during study')
 (a.output/'wall.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,GT_read=False),indent=2))
 seal={str(p.relative_to(a.output)):file_sha256(p) for p in a.output.rglob('*') if p.is_file()}
 (a.output/'grand-fit-seal.json').write_text(json.dumps(seal,indent=2)+'\n');print('ALL_FITS_SEALED_WITHOUT_GT',flush=True)
if __name__=='__main__':main()
