"""Command-free timing of the current controller-only IK cost after warmup."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from igris_ik import UpperBodyIK
from igris_ik.geometry import UpperBodyTargets
from umi_replay import config


def run():
    out=Path(__file__).resolve().parent/'results'/f'{time.strftime("%Y%m%d_%H%M%S")}_ik_timing'
    out.mkdir()
    session=Path(Path('/tmp/igris_manual_vr_output').read_text())
    q=np.array(json.loads((session/'motions/HOME.json').read_text())['motions'][0]['position'])
    ik=UpperBodyIK(config());ik.warmup(q);ik.lock_to(q);zero=ik.forward(q)
    assert ik.check_configuration(q,zero).success
    summary=dict(gpu=torch.cuda.get_device_name(),config='controller-only VR: local 40 iterations, elbow-out 0.2, scale independent',
                 scope='Standalone solver including FK/constraint validation, warmup excluded; no robot IO or physical tracking',groups={})
    traces=[]
    for name,count in [('local_hold',200),('local_small_reachable_motion',300),('local_unreachable',100),('global_hold',60)]:
        assert ik.check_configuration(q,zero,update_warm_start=True).success
        for i in range(count):
            if name=='local_small_reachable_motion':
                moved=q.copy();phase=np.sin(2*np.pi*i/150)
                moved[[15,22]]+=.025*phase;moved[[18,25]]-=.035*phase;moved[0]+=.025*phase
                target=ik.forward(moved)
            elif name=='local_unreachable':
                target=UpperBodyTargets(zero.left.translated([.3,0,0]),zero.right,torso=zero.torso)
            else:target=zero
            if name=='global_hold':ik.reset()
            warm=ik.warm_start();began=time.perf_counter();result=ik.solve(q,target)
            wall_ms=(time.perf_counter()-began)*1000
            if not result.success:ik.restore_warm_start(warm)
            traces.append(dict(group=name,index=i,wall_ms=wall_ms,solve_ms=result.diagnostics.solve_time_s*1000,
                               used_global=result.diagnostics.used_global,success=result.success,reasons=list(result.reasons),
                               position_error_mm=result.diagnostics.max_hand_position_error_m*1000))
        rows=traces[-count:];ms=[r['wall_ms'] for r in rows]
        summary['groups'][name]=dict(count=count,success_count=sum(r['success'] for r in rows),
            global_calls=sum(r['used_global'] for r in rows),mean_ms=float(np.mean(ms)),
            p50_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),max_ms=float(np.max(ms)),
            max_position_error_mm=max(r['position_error_mm'] for r in rows))
        assert summary['groups'][name]['global_calls']==(count if name=='global_hold' else 0)
        print(name,summary['groups'][name],flush=True)
    (out/'timing.json').write_text(json.dumps(summary,indent=2))
    (out/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in traces))
    print('OUTPUT',out,flush=True)


if __name__=='__main__':run()
