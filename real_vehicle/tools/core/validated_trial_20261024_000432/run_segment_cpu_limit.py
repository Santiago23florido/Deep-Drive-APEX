"""Temporarily reduce Pi CPU peak consumption; keep all trial gates intact."""
import argparse,json,subprocess as sp,time
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-khz',type=int,choices=[1500000,1800000,2000000],default=2000000)
parser.add_argument('--runner',choices=['run_segment.py','run_oval.py'],default='run_segment.py')
args=parser.parse_args()
root=Path(__file__).resolve().parents[3]
diag=root/'data/diagnostics/20261022_open_segment'/('cpu_ceiling_'+time.strftime('%Y%m%d_%H%M%S'))
diag.mkdir()
state=[]
for policy in Path('/sys/devices/system/cpu/cpufreq').glob('policy*'):
 state.append({'path':str(policy),'max_khz':int((policy/'scaling_max_freq').read_text()),'governor':(policy/'scaling_governor').read_text().strip()})
(diag/'before.json').write_text(json.dumps(state,indent=2)+'\n')
try:
 for row in state:
  (Path(row['path'])/'scaling_max_freq').write_text(str(min(row['max_khz'],args.max_khz)))
 (diag/'requested_max_khz.txt').write_text(str(args.max_khz)+'\n')
 print(f'Temporary CPU ceiling <={args.max_khz/1e6:g} GHz; governor unchanged',flush=True)
 result=sp.run(['sudo','-H','-u','ensta','python3',str(root/'data/diagnostics/20261022_open_segment'/args.runner),'--execute'],timeout=650 if args.runner=='run_oval.py' else 360)
 (diag/'trial_exit.txt').write_text(str(result.returncode)+'\n')
finally:
 for row in state:
  (Path(row['path'])/'scaling_max_freq').write_text(str(row['max_khz']))
 after=[{'path':row['path'],'max_khz':int((Path(row['path'])/'scaling_max_freq').read_text()),'governor':(Path(row['path'])/'scaling_governor').read_text().strip()} for row in state]
 (diag/'after.json').write_text(json.dumps(after,indent=2)+'\n')
 print('Restored original CPU ceiling; evidence '+str(diag),flush=True)
