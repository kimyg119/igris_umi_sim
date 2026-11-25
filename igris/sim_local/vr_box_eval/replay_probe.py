"""Replay captured Quest poses/grips at original cadence through the vendor simulation."""
import argparse
import copy
import json
from pathlib import Path
import threading
import time

import zmq

import evaluate
from igris_ik.gui.robot_monitor import RobotMonitor
from igris_ik.realtime.vr_teleop import run_vr_teleop


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.capture.open()]
    enabled=lambda row:all(row['payload']['buttons'][s]['hand_trigger']>.5 for s in ('left','right'))
    end=max(i for i,row in enumerate(rows) if enabled(row))
    rows=rows[:end+1]
    origin=rows[0]['received_wall']; duration=rows[-1]['received_wall']-origin
    out=evaluate.HERE/'results'/f'{time.strftime("%Y%m%d_%H%M%S")}_cartesian_probe_replay'
    out.mkdir()
    (out/'metadata.json').write_text(json.dumps(dict(capture=str(args.capture.resolve()),arm_tracking='off',scale=1.,
        free_base=True,mass_kg=.5,wrist_kp=30.,wrist_kd=3.,duration_s=duration,synthetic_release_prefix_s=1.),indent=2))
    evaluate.scene(.5,out);evaluate.setup(out)
    evaluate.run([str(evaluate.ROOT/'sim_local/start_dds_bridge.sh')],stdout=open(out/'bridge-default.log','w'))
    # Same recording and control path as a live run; no modifications to IK results.
    old_start=RobotMonitor.start
    def start(self,*args,**kwargs):
        result=old_start(self,*args,**kwargs);self.set_recording(True);ready.set();return result
    RobotMonitor.start=start
    ready=threading.Event();done=threading.Event();errors=[]
    def publish():
        context=zmq.Context();socket=context.socket(zmq.PUB);socket.bind('tcp://127.0.0.1:5577')
        try:
            if not ready.wait(30):raise RuntimeError('Teleop did not become ready')
            began=time.monotonic();(out/'record.trigger').write_text(str(time.time()));index=0;seq=0
            with (out/'replay_input.jsonl').open('w',buffering=1) as log:
                while not done.is_set():
                    elapsed=time.monotonic()-began-1.
                    while index+1<len(rows) and rows[index+1]['received_wall']-origin<=elapsed:index+=1
                    packet=copy.deepcopy(rows[index]['payload']);packet.pop('arms',None)
                    packet['seq']=seq;packet['t_capture']=time.time()
                    if elapsed<0 or elapsed>duration:
                        for side in ('left','right'):packet['buttons'][side]['hand_trigger']=0.
                    socket.send_json(packet);log.write(json.dumps(dict(t_wall=time.time(),payload=packet))+'\n');seq+=1
                    if elapsed>duration+2:break
                    done.wait(1/60)
        except BaseException as exc:errors.append(exc)
        finally:done.set();socket.close(0);context.term()
    thread=threading.Thread(target=publish);thread.start()
    try:
        result=run_vr_teleop('127.0.0.1',5577,state_endpoint='tcp://172.17.0.2:5560',cmd_endpoint='tcp://172.17.0.2:5561',
            viser_port=8093,simulation=True,arm_tracking='off',orientation_weight=1.,log_dir=str(out),stop_event=done)
        (out/'teleop_summary.json').write_text(json.dumps(result,indent=2))
        if errors:raise errors[0]
    finally:done.set();thread.join();RobotMonitor.start=old_start;evaluate.ctl('sim','pause')
    print('REPLAY COMPLETE',out,flush=True)


if __name__=='__main__':main()
