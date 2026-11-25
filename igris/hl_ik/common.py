"""Paths and explicit IGRIS pose conventions. Quaternion order is wxyz."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
IGRIS = HERE.parent
ROOT = IGRIS.parent
OUTPUT = ROOT / 'outputs/hl_ik_20260915'
GMR = IGRIS / 'retargeting/GMR'
DESCRIPTION = IGRIS / 'igris_c_description'
VENDOR = IGRIS / 'igris_c_description_public'
SDK_NAMES = tuple(['waist_yaw', 'waist_roll', 'waist_pitch'] +
    [f'{s}_{j}' for s in ('l', 'r') for j in ('hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_pitch','ankle_roll')] +
    [f'{s}_{j}' for s in ('l', 'r') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','wrist_yaw','wrist_roll','wrist_pitch')] +
    ['neck_yaw','neck_pitch'])
ARM_NAMES = SDK_NAMES[15:29]
EE_NAMES = ('end_effector_left', 'end_effector_right')
ELBOW_NAMES = ('l_elbow_bracket', 'r_elbow_bracket')
# Vendor torso/pelvis variants place these link frames at opposite joint ends.
# Translation of an old pelvis-model point into the new torso-model link frame.
PELVIS_TO_TORSO_LINK_OFFSET = {'waist_frame': np.array([.05,0,-.026]),
                              'waist_follower': np.array([0,0,-.04])}
HAND_Q = {f'{s}_hand_dof_joint_{j}': np.deg2rad(v) for s in ('l','r')
          for j,v in [('mcp',10),('dip',0),('cmc',-60),('mcp_thumb',-60)]}

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def origin(el):
    o = el.find('origin'); t = np.eye(4)
    if o is not None:
        t[:3,3] = np.fromstring(o.get('xyz','0 0 0'),sep=' ')
        t[:3,:3] = Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t

def pose7(t):
    return np.r_[t[:3,3], Rotation.from_matrix(t[:3,:3]).as_quat(scalar_first=True)]

def matrix7(p):
    p = np.asarray(p,dtype=float)
    if p.shape != (7,) or not np.isfinite(p).all() or np.linalg.norm(p[3:])<1e-8:
        raise ValueError('pose must be finite xyz + nonzero wxyz')
    t=np.eye(4);t[:3,3]=p[:3];t[:3,:3]=Rotation.from_quat(p[3:],scalar_first=True).as_matrix()
    return t
