"""Keep the neutral upper-body reference active during private WALK_LOW bringup.

The replay publisher explicitly waits for this publisher to acknowledge its
shutdown before taking ownership. Legs retain zero outgoing gains and torque.
"""
import argparse
import json
import os
from pathlib import Path
import time
import tomllib

from run import ROOT,JOINTS


def run(out):
    if not Path('/.dockerenv').exists():raise RuntimeError('Private simulation container required')
    runtime=json.loads((out/'runtime.json').read_text())
    if runtime['network']!='none' or runtime['ipc']!='private':raise RuntimeError('Private simulation required')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from igris_c_sdk.msg import LowCmd,RobotState
    model=json.loads((out/'model.json').read_text())
    owned=[0,1,2,*range(15,29)]
    q=[model['hinge_q'].get(name,0.) for name in JOINTS]
    if any(abs(q[i])>1e-6 for i in owned):raise RuntimeError('Expected the audited neutral upper-body state')
    profile=tomllib.loads((ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/replay_gains.toml').read_text())['motion']['gains']
    kp=[0.]*31;kd=[0.]*31
    for i in owned:
        name=JOINTS[i].replace('l_','left_',1) if JOINTS[i].startswith('l_') else JOINTS[i].replace('r_','right_',1) if JOINTS[i].startswith('r_') else JOINTS[i]
        kp[i]=profile[name]['kp'];kd[i]=profile[name]['kd']
    rclpy.init();node=rclpy.create_node('isaac_room_upper_hold')
    publisher=node.create_publisher(LowCmd,'igris_c_001/rt/lowcmd',qos_profile_sensor_data)
    environment=None;received=0.
    def state(message):
        nonlocal environment,received
        environment=message.environment;received=time.monotonic()
    node.create_subscription(RobotState,'igris_c_001/rt/robotstate',state,qos_profile_sensor_data)
    start=time.monotonic();count=0;error=None
    try:
        while time.monotonic()-start<180:
            if (out/'upper_hold.stop').exists() or (out/'stop.request').exists():break
            rclpy.spin_once(node,timeout_sec=0)
            if environment is None:
                if time.monotonic()-start>10:raise RuntimeError('No simulation RobotState')
                time.sleep(.01);continue
            if environment!=RobotState.ROBOT_ENV_SIM:raise RuntimeError('Simulation RobotState required')
            if time.monotonic()-received>.5:raise RuntimeError('Stale simulation RobotState')
            message=LowCmd();message.header.seq=count;message.kinematic_modes=[1]*5
            for i,motor in enumerate(message.motors):
                motor.id=i;motor.q=q[i] if i in owned else 0.;motor.dq=0.;motor.tau=0.
                motor.kp=kp[i];motor.kd=kd[i]
            publisher.publish(message);count+=1
            if count==100:(out/'upper_hold.ready').write_text(str(os.getpid()))
            time.sleep(.01)
        else:raise RuntimeError('Bringup upper hold exceeded 180 seconds')
    except Exception as exc:
        error=str(exc);(out/'stop.request').write_text('upper_hold: '+error)
    finally:
        node.destroy_node();rclpy.shutdown()
        (out/'upper_hold_result.json').write_text(json.dumps(dict(error=error,samples=count,
            q=q,kp=kp,kd=kd,hardware_executed=False)))
        (out/'upper_hold.stopped').write_text(str(time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out.resolve())
