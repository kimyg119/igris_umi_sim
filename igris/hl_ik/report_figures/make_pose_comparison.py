"""Re-render an existing matched prior ablation; no physics steps or robot I/O.

Run from Project_Humanoid using the existing gmr Python environment:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  PYTHONPATH=. python -m igris.hl_ik.report_figures.make_pose_comparison
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import csv
import json
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from igris.hl_ik.common import ROOT, HERE, OUTPUT, SDK_NAMES, ARM_NAMES, EE_NAMES, ELBOW_NAMES, Rotation, matrix7, sha256, save_json
from igris.hl_ik.solver import HLIK

OUT = ROOT / 'outputs/report_figures/pose_comparison'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'outputs/hl_ik_pipeline_20260915/final'
LABELS = ('qp_priority_none', 'qp_priority_full')
manifest = json.loads((SOURCE/'manifest.json').read_text())
run_meta = [json.loads((SOURCE/'runs'/s/'metrics.json').read_text()) for s in LABELS]
for meta in run_meta:
    assert meta['manifest_sha256'] == sha256(SOURCE/'manifest.json')
    assert all(sha256(HERE/name) == value for name, value in meta['source_sha256'].items())
configs = [manifest['configs'][s].copy() for s in LABELS]
assert configs[0].pop('prior') is None
checkpoint = configs[1].pop('prior')
assert configs[0] == configs[1]
assert manifest['prior_sha256'][checkpoint] == sha256(checkpoint)

# Select for visibility of posture change, not for the direction of improvement.
# Only completed recorded fixtures, with <1 mm and <0.5 deg EE residual in BOTH runs.
candidates = []
for fixture in manifest['fixtures']:
    name = fixture['name']
    if fixture['kind'] != 'recorded':
        continue
    if not all(next(r for r in m['fixtures'] if r['name'] == name)['completed'] for m in run_meta):
        continue
    runs = [np.load(SOURCE/'runs'/s/f'{name}.npz') for s in LABELS]
    assert np.array_equal(runs[0]['goals'], runs[1]['goals'])
    assert np.array_equal(runs[0]['time'], runs[1]['time'])
    assert np.array_equal(runs[0]['q'][:, :15], runs[1]['q'][:, :15])
    good = np.ones(len(runs[0]['q']), dtype=bool)
    for z in runs:
        good &= (z['metrics'][:, 0] < .001) & (z['metrics'][:, 1] < np.deg2rad(.5)) & (z['metrics'][:, 5] == 1)
    delta = np.linalg.norm(runs[1]['q'][:, 15:29]-runs[0]['q'][:, 15:29], axis=1)
    for i in np.flatnonzero(good):
        candidates.append(dict(fixture=name, frame=int(i), arm_delta_l2_rad=float(delta[i])))
    for z in runs:
        z.close()
selected = max(candidates, key=lambda r: r['arm_delta_l2_rad'])
name, frame = selected['fixture'], selected['frame']
fixture_meta = next(r for r in manifest['fixtures'] if r['name'] == name)
fixture_path = SOURCE/'fixtures'/f'{name}.npz'
assert sha256(fixture_path) == manifest['fixture_sha256'][name]
fixture = np.load(fixture_path)
runs = [np.load(SOURCE/'runs'/s/f'{name}.npz') for s in LABELS]
target = runs[0]['goals'][frame].astype(float)
qs = [z['q'][frame].copy() for z in runs]
solver = HLIK()
reference_elbows = fixture['elbows'][frame].copy()
reference_elbows[:, :3] += solver.shoulder
records = []
states = []
for q, z, tag in zip(qs, runs, ('without_cost', 'with_cost')):
    solver.set_q(q)
    state = solver.states()
    actual = state[:, :7].copy()
    actual[:, :3] += solver.shoulder
    elbows = state[:, 7:].copy()
    elbows[:, :3] += solver.shoulder
    position = np.linalg.norm(actual[:, :3]-target[:, :3], axis=1)
    angle = (Rotation.from_quat(actual[:, 3:], scalar_first=True).inv()*Rotation.from_quat(target[:, 3:], scalar_first=True)).magnitude()
    elbow_pos = np.linalg.norm(elbows[:, :3]-reference_elbows[:, :3], axis=1)
    elbow_rot = (Rotation.from_quat(elbows[:, 3:], scalar_first=True).inv()*Rotation.from_quat(reference_elbows[:, 3:], scalar_first=True)).magnitude()
    assert np.all(position < .001) and np.all(angle < np.deg2rad(.5))
    assert np.isclose(elbow_pos.mean(), z['metrics'][frame, 8], atol=1e-7)
    assert solver.geometry.collision(solver.configuration.data) is None
    arm = q[15:29]
    assert np.all(arm >= solver.limits[:, 0]) and np.all(arm <= solver.limits[:, 1])
    assert np.all(solver.A@arm <= solver.b+1e-8)
    end = solver.configuration.q.copy()
    solver.set_q(z['q'][frame-1])
    begin = solver.configuration.q.copy()
    assert solver.geometry.corridor(solver.configuration.data, begin, end, max_step=.01) is None
    velocity = (q[15:29]-z['q'][frame-1, 15:29])*30
    previous_velocity = (z['q'][frame-1, 15:29]-z['q'][frame-2, 15:29])*30
    assert np.abs(velocity).max() <= 3.00001
    assert np.abs((velocity-previous_velocity)*30).max() <= 30.0001
    states.append(elbows)
    records.append(dict(condition=tag, ee_position_mm=(position*1000).tolist(), ee_orientation_deg=np.rad2deg(angle).tolist(),
                        elbow_reference_position_mm=(elbow_pos*1000).tolist(), elbow_reference_orientation_deg=np.rad2deg(elbow_rot).tolist(),
                        elbow_reference_position_mean_mm=float(elbow_pos.mean()*1000), elbow_reference_orientation_mean_deg=float(np.rad2deg(elbow_rot).mean()),
                        collision=None, joint_limits_pass=True))

np.savez_compressed(OUT/'figure-3-pose-data.npz', q_without_cost=qs[0], q_with_cost=qs[1], target=target,
                    reference_elbows=reference_elbows, actual_elbows=states, warmup=fixture['warmup'], joint_names=SDK_NAMES)
with (OUT/'figure-3-joint-positions.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['joint', 'without_cost_rad', 'with_cost_rad', 'difference_deg'])
    writer.writerows((n, a, b, float(np.rad2deg(b-a))) for n, a, b in zip(SDK_NAMES, qs[0], qs[1]))
with (OUT/'figure-3-ee-targets.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['side', 'x_m', 'y_m', 'z_m', 'qw', 'qx', 'qy', 'qz'])
    writer.writerows((side, *p) for side, p in zip(('left', 'right'), target))

# Real mesh rendering from measured saved q, with identical cameras and targets.
model = solver.model
model.vis.global_.offwidth = 1200
model.vis.global_.offheight = 780
model.vis.headlight.ambient[:] = .65
model.vis.headlight.diffuse[:] = .75
model.vis.headlight.active = 1
model.vis.quality.offsamples = 4
option = mujoco.MjvOption()
option.geomgroup[:] = 0
option.geomgroup[1] = 1
rgb = [(37, 104, 147), (120, 80, 163)]
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(size):
    return ImageFont.truetype(font_path, size)

renderer = mujoco.Renderer(model, height=780, width=1200)
def segment(a, b, color, width=.005, kind=mujoco.mjtGeom.mjGEOM_ARROW):
    geom = renderer.scene.geoms[renderer.scene.ngeom]
    renderer.scene.ngeom += 1
    mujoco.mjv_initGeom(geom, kind, np.ones(3), np.zeros(3), np.eye(3).ravel(), np.array(color, np.float32))
    mujoco.mjv_connector(geom, kind, width, a, b)

def render(q, color, view):
    solver.set_q(q)
    for k in range(model.ngeom):
        if model.geom_group[k] != 1:
            continue
        body = model.body(int(model.geom_bodyid[k])).name
        is_arm = body.startswith(('l_', 'r_')) and any(s in body for s in ('shoulder', 'upper_arm', 'elbow', 'forearm', 'wrist'))
        model.geom_rgba[k] = [*(np.array(color)/255 if is_arm else [.60, .63, .66]), 1.]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.08, 0, .24]
    camera.distance = 1.65
    camera.azimuth = 170
    camera.elevation = -12
    if view == 'oblique':
        camera.lookat[:] = [.02, -.37, .25]
        camera.distance = .60
        camera.azimuth = 145
        camera.elevation = -20
    renderer.update_scene(solver.configuration.data, camera, option)
    # RGB arrows show the same six-DoF target; no target is moved to match a result.
    for pose in target:
        t = matrix7(pose)
        for axis, c in enumerate(([.9, .18, .18, 1], [.16, .64, .30, 1], [.20, .42, .94, 1])):
            segment(t[:3, 3], t[:3, 3]+.095*t[:3, axis], c, .003)
    # Amber arrows show retarget reference elbow hinge +Y. White arrows show robot +Y.
    for body_name, pose in zip(ELBOW_NAMES, reference_elbows):
        body = model.body(body_name).id
        pos = solver.configuration.data.xpos[body]
        rot = solver.configuration.data.xmat[body].reshape(3, 3)
        segment(pos, pos+.15*rot[:, 1], [1, 1, 1, 1], .006)
        t = matrix7(pose)
        segment(t[:3, 3], t[:3, 3]+.15*t[:3, 1], [1, .68, .12, 1], .004)
    return Image.fromarray(renderer.render())

panels = []
try:
    for idx, (q, color) in enumerate(zip(qs, rgb)):
        front = render(q, color, 'front')
        oblique = render(q, color, 'oblique')
        front.save(OUT/f'figure-3-{idx}-front.png')
        oblique.save(OUT/f'figure-3-{idx}-oblique.png')
        panels.append((front, oblique))
finally:
    renderer.close()

W, H = 2560, 2060
sheet = Image.new('RGB', (W, H), 'white')
draw = ImageDraw.Draw(sheet)
draw.text((60, 35), '동일한 양손 EE 목표에서 인간 자세 cost 적용 전후 비교', font=font(48), fill='#182c3e')
draw.text((60, 112), '학습된 팔꿈치 자세 prior → IK 자세 cost  |  같은 목표 시퀀스·초기 이력·solver·제약', font=font(29), fill='#536678')
for i, ((front, oblique), color, rec) in enumerate(zip(panels, rgb, records)):
    x = 40+i*1260
    draw.rounded_rectangle((x, 185, x+1220, 258), radius=8, fill=color)
    title = '인간 자세 cost 미적용' if i == 0 else '인간 자세 cost 적용'
    draw.text((x+30, 199), title, font=font(34), fill='white')
    sheet.paste(front, (x+10, 280))
    sheet.paste(oblique.resize((720, 468)), (x+10, 1090))
    draw.text((x+30, 1103), '오른팔 팔꿈치 확대', font=font(26), fill='white')
    draw.text((x+750, 1110), '리타게팅 기준 대비', font=font(27), fill='#536678')
    draw.text((x+750, 1160), '팔꿈치 위치 오차', font=font(28), fill='#182c3e')
    draw.text((x+750, 1202), f"{rec['elbow_reference_position_mean_mm']:.2f} mm", font=font(47), fill=color)
    draw.text((x+750, 1280), '팔꿈치 방향 오차', font=font(28), fill='#182c3e')
    draw.text((x+750, 1322), f"{rec['elbow_reference_orientation_mean_deg']:.2f}°", font=font(47), fill=color)
    draw.text((x+750, 1400), '위치·방향: 양팔 평균', font=font(22), fill='#536678')
    draw.text((x+750, 1440), 'EE 위치 오차 < 0.001 mm', font=font(22), fill='#536678')
    draw.text((x+750, 1480), 'EE 방향 오차 < 0.001°', font=font(22), fill='#536678')
draw.text((60, 1592), 'RGB 축: 양손의 공통 EE 목표  ·  흰 화살표: 로봇 팔꿈치 힌지축  ·  금색 화살표: 리타게팅 기준 힌지축', font=font(28), fill='#182c3e')
draw.line((60, 1660, 2460, 1660), fill='#d7e1e8', width=2)
lines = [
    f'사례: AMASS / {name}, 비교 시퀀스 frame {frame} ({runs[0]["time"][frame]:.3f} s). 최대 팔 관절 차이 사례를 시각화용으로 선택.',
    f'선정 조건: 양쪽 모두 EE < 1 mm / 0.5°. 완료된 기록 기반 비교에서 이 조건을 만족한 {len(candidates):,}개 프레임 중 선택.',
    '초기 관절 이력은 동일하며 이후 이력은 각 조건의 생성 결과로 갱신됨. 허리·하체·목 관절 및 카메라는 양쪽 동일.',
    '팔꿈치 기준은 사람 동작을 로봇 체형에 맞춘 리타게팅 결과이며, 실측 인간 자세 오차가 아님.',
    '단일 사례의 운동학 렌더링임. 전체 성능 개선·물리 접촉·실물 실행을 뜻하지 않음. physics steps = 0, hardware commands = 0.',
]
for j, line in enumerate(lines):
    draw.text((60, 1692+j*55), line, font=font(25), fill='#536678')
sheet.save(OUT/'figure-3-human-pose-cost-comparison.png', dpi=(220, 220))
sheet.resize((1536, 1236)).save(OUT/'figure-3-preview.png')
files = [SOURCE/'manifest.json', fixture_path, Path(checkpoint), OUTPUT/'model/igris_c_rcv_hand_torso.xml']
files += [SOURCE/'runs'/s/f'{name}.npz' for s in LABELS]
report = dict(selected=selected, time_s=float(runs[0]['time'][frame]), source_motion=fixture_meta['source'],
              selection='Largest L2 arm joint change among completed recorded fixtures where BOTH outputs have EE error below 1 mm / 0.5 deg. Selection does not use elbow improvement.',
              eligible_frames=len(candidates), candidates=candidates, comparisons=records,
              history='same five initial observations; each condition then updates its own generated FK history at 30 Hz',
              coordinate_frame='torso; xyz metres + quaternion wxyz; identical caller-owned waist/lower-body/neck for both rendered poses',
              elbow_reference='robot-morphology retarget of AMASS human motion; not measured anatomical ground truth',
              collision_model=solver.geometry.provenance(), source_sha256={str(p):sha256(p) for p in files},
              reused_solver_source_sha256=run_meta[0]['source_sha256'], physics_steps=0, hardware_commands_sent=0, execution_ready=False)
save_json(OUT/'figure-3-validation.json', report)
print(json.dumps(dict(selected=selected, eligible_frames=len(candidates), comparisons=records), indent=2))
