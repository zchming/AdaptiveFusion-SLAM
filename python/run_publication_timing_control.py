#!/usr/bin/env python3
"""Frozen ABBA full-sequence publication-timing control, one candidate profile."""
import argparse,json,time,traceback
from pathlib import Path
from adaptivefusion.state import file_sha256
from run_live_cpp_auxiliary import run_arm
from run_async_auxiliary_output import save
def main():
 p=argparse.ArgumentParser()
 for name in ["protocol","output"]:p.add_argument("--"+name,type=Path,required=True)
 a=p.parse_args();plan=json.loads(a.protocol.read_text())
 if a.output.exists():raise ValueError("refuse overwrite")
 for path,sha in plan["input_sha256"].items():
  if file_sha256(path)!=sha:raise ValueError("frozen input changed: "+path)
 sources={s["scene"]:s for s in json.loads(Path(plan["sources"]).read_text())}
 a.output.mkdir();records=[];begin=time.perf_counter()
 for arm in plan["arms"]:
  effective=dict(plan);effective["producer_binary"]=plan["profile_binaries"][arm["profile"]]
  try:
   value=run_arm(sources[arm["scene"]],arm,effective,a.protocol,a.output/(arm["scene"]+"-"+arm["mode"]))
   value["profile"]=arm["profile"]
  except Exception as error:value=dict(scene=arm["scene"],mode=arm["mode"],profile=arm["profile"],status="failed",reason=str(error),traceback=traceback.format_exc())
  records.append(value);(a.output/"status.json").write_text(json.dumps(records,indent=2)+"\n");print(json.dumps(value),flush=True)
 for path,sha in plan["input_sha256"].items():
  if file_sha256(path)!=sha:raise ValueError("input changed during timing control: "+path)
 save(a.output/"wall.json",dict(seconds=time.perf_counter()-begin,GT_read=False))
 save(a.output/"fit-seal.json",{str(p.relative_to(a.output)):file_sha256(p) for p in a.output.rglob("*") if p.is_file()})
 print("PUBLICATION_TIMING_SEALED_WITHOUT_GT",flush=True)
if __name__=="__main__":main()
