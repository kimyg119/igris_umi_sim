"""Measure actual native payload orientation, including unsuccessful replays."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def measure(out,rotation_limit_deg=3.):
    out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text())
    model=json.loads((out/'model.json').read_text())
    command=json.loads(sorted(out.glob('command_*.json'))[-1].read_text())
    basic=json.loads((out/'replay_result.json').read_text())
    report={'pass':False,'native_physics':True,'rotation_limit_deg':rotation_limit_deg,
            'replay_pass':basic['pass'],'hardware_executed':False}
    samples=[s for s in command['samples'] if s['archive_index']>=0]
    if samples:
        motion=np.load(command['motion'])
        progress=motion['original_command_t_s']
        width=2+7*len(model['bodies'])
        raw=np.fromfile(out/'poses.bin');rows=raw[:len(raw)//width*width].reshape(-1,width)
        rows=rows[(rows[:,0]>=samples[0]['wall'])&(rows[:,0]<=samples[-1]['wall'])]
        p=np.interp(rows[:,0],[s['wall'] for s in samples],[progress[s['archive_index']] for s in samples])
        bi=next(i for i,b in enumerate(model['bodies']) if b['name']==manifest['payload']['body'])
        box=rows[:,2+7*bi:2+7*(bi+1)]
        rotation=Rotation.from_quat(box[:,[4,5,6,3]])
        # Reference the settled box immediately before squeezing, not a tilted
        # box after a failed grasp. Include pickup, carry, landing and release.
        start=int(np.argmin(abs(p-29.705294881166235)))
        allowed_yaw=np.zeros(len(rows))
        mode=str(motion['target_rotation']) if 'target_rotation' in motion else 'world_fixed'
        if mode=='follow_waist_yaw':
            allowed_yaw=np.interp(rows[:,0],[s['wall'] for s in samples],
                [motion['expected_box_yaw_rad'][s['archive_index']] for s in samples])
        expected=Rotation.from_euler('z',(allowed_yaw-allowed_yaw[start])[:,None])*rotation[start]
        if command.get('walking_transfer') or command.get('inplace_transfer'):
            expected_matrices=expected.as_matrix()
            indices=np.maximum(0,np.searchsorted([s['wall'] for s in samples],rows[:,0],side='right')-1)
            for i,slot in enumerate(indices):
                walking=samples[slot].get('walking')
                if walking:expected_matrices[i]=walking['expected_payload_rotation_world']
            expected=Rotation.from_matrix(expected_matrices)
            spec=manifest.get('inplace_transfer',manifest.get('walking_transfer',{}))
            mode=('waist-yaw pickup and explicit in-place waist arc; ' if command.get('inplace_transfer') else 'waist-yaw pickup; measured torso-relative carry; ')+('world-position/torso-orientation lowering/open'
                if spec.get('world_position_lowering',False) else 'torso-relative lowering/open'
                if manifest.get('walking_transfer',{}).get('torso_relative_lowering',False) else 'fixed-world lowering')
            # Unwrap the commanded world heading, retaining the payload's
            # authored non-Z vertical axis in the absolute tilt calculation.
            relative=expected.as_matrix()@rotation[start].as_matrix().T
            allowed_yaw=np.unwrap(np.arctan2(relative[:,1,0],relative[:,0,0]))
        drift=np.rad2deg((expected.inv()*rotation).magnitude())
        # The payload's local vertical axis need not be Z (this USD uses Y).
        initial_matrix=rotation[start].as_matrix()
        vertical_axis=int(np.argmax(abs(initial_matrix[2])))
        vertical=rotation.as_matrix()[:,:,vertical_axis]*np.sign(initial_matrix[2,vertical_axis])
        tilt=np.rad2deg(np.arccos(np.clip(vertical[:,2],-1,1)))
        report['target_rotation']=mode
        active=p>=29.705294881166235
        if active.any():
            report.update(max_rotation_deg=float(drift[active].max()),
                          p95_rotation_deg=float(np.percentile(drift[active],95)),
                          final_rotation_deg=float(drift[-1]),
                          max_tilt_deg=float(tilt[active].max()),
                          expected_yaw_change_deg=float(np.rad2deg(allowed_yaw[-1]-allowed_yaw[start])),
                          reached_original_time_s=float(p[-1]))
            report['pass']=bool(basic['pass'] and report['max_rotation_deg']<=rotation_limit_deg
                                and report['max_tilt_deg']<=rotation_limit_deg)
            if manifest.get('evaluation_profile')=='relaxed_transport':
                relaxed_limit=float(manifest['relaxed_rotation_limit_deg'])
                report.update(evaluation_profile='relaxed_transport',relaxed_rotation_limit_deg=relaxed_limit,
                              relaxed_pass=bool(basic.get('relaxed_pass') and report['max_rotation_deg']<=relaxed_limit and report['max_tilt_deg']<=relaxed_limit))
        np.savez_compressed(out/'transport_trace.npz',wall=rows[:,0],sim=rows[:,1],
                            original_time=p,box_pose=box,rotation_drift_deg=drift,
                            allowed_yaw_rad=allowed_yaw,tilt_deg=tilt)
    (out/'transport_result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(measure(a.out),indent=2))
