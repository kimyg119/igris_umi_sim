"""Controlled gain ablation: identical timed approach + 30 mm/hand preload hold."""
import sys,json
import evaluate
from igris_ik.gui.robot_monitor import RobotMonitor
kp=float(sys.argv[1]);outputs=[]
if kp not in (30.,60.):raise ValueError('Use wrist Kp 30 or 60')
original_start=RobotMonitor.start
original_virtual=evaluate.virtual
original_stage=evaluate.stage

def start(self,*a,**kw):
    r=original_start(self,*a,**kw);self.set_recording(True);outputs.append(self.log_dir);return r

def virtual(t,*a,**kw):return original_virtual(min(t,23.),*a,**kw)

def stage(t,motion='box'):return 'squeeze' if t>=19 else original_stage(t,motion)

RobotMonitor.start=start
evaluate.virtual=virtual;evaluate.stage=stage
evaluate.SQUEEZE_M=.030
evaluate.DURATION=42.
try:
    evaluate.trial('off',.5,contact_driven=False,wrist_kp=kp)
finally:
    if outputs:
        p=outputs[-1]/'metadata.json';m=json.loads(p.read_text());m.update(experiment='wrist gain ablation: timed approach then hold 30 mm/hand preload; no lift requested',squeeze_hold=True);p.write_text(json.dumps(m,indent=2))
    evaluate.ctl('sim','pause')
