"""Live Quest probe: observe raw solver constraints without changing commands or gates."""
import argparse,json,threading,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from igris_ik import UpperBodyIK,IKConfig
from igris_ik.gui.robot_monitor import RobotMonitor
from igris_ik.realtime.robot_control import ArmController
from igris_ik.realtime.vr_teleop import run_vr_teleop
from curobo.inverse_kinematics import InverseKinematics
import evaluate


def install_probe(path):
    log=path.open('a',buffering=1);local=threading.local();lock=threading.Lock();sequence=0
    def write(event):
        with lock:log.write(json.dumps(dict(t_wall=time.time(),**event))+'\n')
    original_result=InverseKinematics._get_result
    original_solve=UpperBodyIK.solve
    original_monitor=RobotMonitor._solve
    original_start=RobotMonitor.start
    original_stop=ArmController.stop
    def raw_result(self,opt_result,metrics_result,num_seeds,return_seeds):
        result=original_result(self,opt_result,metrics_result,num_seeds,return_seeds)
        record=getattr(local,'record',None)
        if record is not None:
            index=int(result.debug_info['seed_idx'].flatten()[0].item())
            def selected(collection):
                return {n:v[index].detach().cpu().tolist() for n,v in zip(collection.names,collection.values)}
            costs=metrics_result.costs_and_constraints
            record['raw']=dict(success=bool(result.success.flatten()[0]),feasible=bool(result.feasible.flatten()[0]),
                seed_index=index,num_seeds=num_seeds,q_active=result.solution.flatten().detach().cpu().tolist(),
                constraints=selected(costs.constraints),hybrid_constraints=selected(costs.hybrid_costs_constraints))
            if not record['raw']['success']:
                record['raw']['robot_spheres']=metrics_result.state.cuda_robot_model_state.robot_spheres[index].detach().cpu().tolist()
        return result
    def solve(self,q,targets,**kwargs):
        nonlocal sequence
        sequence+=1;start=time.perf_counter()
        seed=np.asarray(q.q_sdk if hasattr(q,'q_sdk') else q)
        lo,hi=self.model.joint_limits()
        record=dict(event='ik',solve_id=sequence,started_wall=time.time(),q_seed=seed.tolist(),
                    joint_names=list(self.model.active_joint_names),lower=lo.tolist(),upper=hi.tolist(),
                    targets={s:dict(position=getattr(targets,s).position.tolist(),wxyz=getattr(targets,s).wxyz.tolist()) for s in ('left','right')})
        local.record=record
        try:
            result=original_solve(self,q,targets,**kwargs)
            record.update(success=result.success,reported_collision_free=result.collision_free,reasons=result.reasons,
                          q_returned=result.q_sdk.tolist(),diagnostics=asdict(result.diagnostics))
            raw=np.array(record['raw']['q_active'])
            record['raw_limit_excess_rad']=np.maximum(np.maximum(lo-raw,raw-hi),0).tolist()
            record['observed_total_s']=time.perf_counter()-start
            write(record)
            return result
        finally:local.record=None
    def monitor(self,**kwargs):
        result=original_monitor(self,**kwargs)
        if result is not None:
            write(dict(event='monitor',last_solve_id=sequence,success=result.success,reasons=result.reasons,
                       q=result.q_sdk.tolist(),mode=self.controller.mode,branch_rejections=self.branch_guard_rejections))
        return result
    def start(self,*args,**kwargs):
        result=original_start(self,*args,**kwargs);self.set_recording(True);return result
    def stop(self,reason='manual'):
        write(dict(event='stop',reason=reason,mode=self.mode))
        return original_stop(self,reason)
    InverseKinematics._get_result=raw_result;UpperBodyIK.solve=solve
    RobotMonitor._solve=monitor;RobotMonitor.start=start;ArmController.stop=stop
    return log


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    out=evaluate.HERE/'results'/f'{time.strftime("%Y%m%d_%H%M%S")}_probe_off_0.5kg';out.mkdir()
    log=install_probe(out/'probe.jsonl')
    if args.check:
        ik=UpperBodyIK(IKConfig(waist_mode='ik',neck_mode='fixed'))
        q=ik.default_configuration().q_sdk.copy();q[[15,22]]=-.5;q[[18,25]]=-1.7
        ik.lock_to(q);ik.solve(q,ik.forward(q))
        row=json.loads((out/'probe.jsonl').read_text().splitlines()[-1])
        assert len(row['raw']['q_active'])==len(row['joint_names'])
        assert 'self_collision' in row['raw']['constraints']
        assert len(row['raw_limit_excess_rad'])==len(row['joint_names'])
        print('PROBE CHECK PASS',out,flush=True);log.close();return
    evaluate.scene(.5,out);evaluate.setup(out)
    evaluate.run([str(evaluate.ROOT/'sim_local/start_dds_bridge.sh')],stdout=open(out/'bridge-default.log','w'))
    (out/'record.trigger').write_text(str(time.time()))
    Path('/tmp/igris_manual_vr_output').write_text(str(out))
    (out/'manual_session.json').write_text(json.dumps(dict(input='live Quest',arm_tracking='off',mass_kg=.5,
        scale=1.,wrist_kp=30,wrist_kd=3,free_base=True,url='https://127.0.0.1:8012/?ws=wss://127.0.0.1:8012',start_wall=time.time()),indent=2))
    print('PROBE READY',out,flush=True)
    try:
        run_vr_teleop('127.0.0.1',5557,state_endpoint='tcp://172.17.0.2:5560',cmd_endpoint='tcp://172.17.0.2:5561',
                     viser_port=8093,simulation=True,delta_scale=1.,orientation_weight=1.,arm_tracking='off',log_dir=str(out))
    finally:evaluate.ctl('sim','pause');log.close()

if __name__=='__main__':main()
