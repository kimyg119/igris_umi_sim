"""Highlight the exact pelvis visual in the existing pelvis-root model.
Run from Project_Humanoid with MUJOCO_GL=egl PYTHONPATH=. and gmr Python.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from igris.hl_ik.common import ROOT, OUTPUT, SDK_NAMES, sha256, save_json

OUT = ROOT / 'outputs/report_figures/pelvis'
OUT.mkdir(parents=True, exist_ok=True)
urdf = ROOT/'igris/igris_c_description/urdf/igris_c_rcv_hand_pelvis.urdf'
xml = OUTPUT/'model/reference_pelvis.xml'
provenance = json.loads((OUTPUT/'model/provenance.json').read_text())
assert provenance['sources'][str(urdf)] == sha256(urdf)
robot = ET.parse(urdf).getroot()
pelvis_visual = robot.find("link[@name='pelvis']/visual/geometry/mesh")
assert pelvis_visual is not None
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
data.qpos[:] = model.qpos0
data.qpos[2] = 1.0
values = dict(l_shoulder_roll=.30, r_shoulder_roll=-.30, l_elbow_pitch=-.30, r_elbow_pitch=-.30)
for name, value in values.items():
    data.qpos[model.joint(name).qposadr[0]] = value
mujoco.mj_forward(model, data)
pid = model.body('pelvis').id
selected = np.flatnonzero((model.geom_bodyid == pid) & (model.geom_group == 1))
assert len(selected) == 1
gid = int(selected[0])
assert model.mesh(int(model.geom_dataid[gid])).name == 'pelvis'
for i in range(model.ngeom):
    model.geom_rgba[i] = [1.0, .40, .06, 1.] if i == gid else [.50, .55, .61, 1.]
model.vis.headlight.ambient[:] = .60
model.vis.headlight.diffuse[:] = .75
model.vis.headlight.active = 1
model.vis.global_.offwidth = 1000
model.vis.global_.offheight = 1100
model.vis.quality.offsamples = 4
option = mujoco.MjvOption()
option.geomgroup[:] = 0
option.geomgroup[1] = 1
mid = int(model.geom_dataid[gid])
start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
vertices = model.mesh_vert[start:start+count]@data.geom_xmat[gid].reshape(3, 3).T+data.geom_xpos[gid]
center = (vertices.min(0)+vertices.max(0))/2
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(size):
    return ImageFont.truetype(font_path, size)

renderer = mujoco.Renderer(model, height=1100, width=1000)
def render(close=False):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = center if close else [0, 0, .83]
    camera.distance = .62 if close else 2.30
    camera.azimuth = 210
    camera.elevation = -13 if close else -8
    renderer.update_scene(data, camera, option)
    picture = Image.fromarray(renderer.render())
    arr = np.array(picture)
    orange = (arr[:, :, 0] > 140) & (arr[:, :, 0].astype(int) > arr[:, :, 1].astype(int)*1.30) & (arr[:, :, 2] < 140)
    assert orange.sum() > 300
    if not close:
        yy, xx = np.nonzero(orange)
        x, y = (xx.min()+xx.max())/2, (yy.min()+yy.max())/2
        draw = ImageDraw.Draw(picture)
        draw.line((x+20, y, 760, y-55), fill=(255, 179, 77), width=4)
        draw.text((763, y-90), 'pelvis', font=font(40), fill=(255, 179, 77))
    return picture

try:
    whole, zoom = render(), render(True)
finally:
    renderer.close()
whole.save(OUT/'whole-body.png')
zoom.save(OUT/'pelvis-closeup.png')
sheet = Image.new('RGB', (2040, 1400), 'white')
draw = ImageDraw.Draw(sheet)
draw.text((45, 28), 'IGRIS-C  |  pelvis 링크', font=font(50), fill='#182c3e')
draw.text((45, 103), '주황색: pelvis의 실제 visual 메시  ·  회색: 주변 링크', font=font(29), fill='#536678')
sheet.paste(whole.resize((880, 968)), (40, 195))
sheet.paste(zoom.resize((1000, 1100)), (1000, 195))
draw.text((42, 156), '전신에서의 위치', font=font(27), fill='#182c3e')
draw.text((1002, 156), '골반 링크 확대', font=font(27), fill='#182c3e')
draw.text((50, 1200), '허리 아래 · 좌우 고관절 위', font=font(33), fill='#182c3e')
draw.text((50, 1257), '인접한 허리·고관절 링크와 구분하여 강조', font=font(25), fill='#536678')
draw.text((50, 1345), '모델: igris_c_rcv_hand_pelvis.urdf  |  link = pelvis  |  mesh = pelvis.stl  |  모델 렌더링', font=font(26), fill='#536678')
sheet.save(OUT/'igris-c-pelvis-highlight.png', dpi=(220, 220))
sheet.resize((1224, 840)).save(OUT/'preview.png')
q = np.zeros(31)
for i, name in enumerate(SDK_NAMES[:29]):
    q[i] = data.qpos[model.joint(name).qposadr[0]]
np.savetxt(OUT/'q-position.csv', np.c_[np.arange(31), q], delimiter=',', header='sdk_index,q_rad', comments='')
save_json(OUT/'provenance.json', dict(urdf=str(urdf), urdf_sha256=sha256(urdf), mjcf=str(xml), mjcf_sha256=sha256(xml),
    highlighted_link='pelvis', highlighted_visual_geom_ids=selected.tolist(), visual_mesh=pelvis_visual.attrib,
    neighboring_links=[j.find('child').get('link') for j in robot.findall('joint') if j.find('parent').get('link')=='pelvis'],
    pelvis_frame_world_position=data.xpos[pid].tolist(), pelvis_visual_bbox_world_m=[vertices.min(0).tolist(), vertices.max(0).tolist()],
    pose='visualization only: waist/legs/neck zero; shoulder rolls +/-0.30rad; elbows -0.30rad; fixed hands from compiled model',
    physics_steps=0, hardware_commands_sent=0))
print('Verified exact pelvis visual geometry; rendered whole-body and close-up views.')
