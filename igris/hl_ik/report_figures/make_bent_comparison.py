"""Saved box-fixture ablation, bent arms. FK/render only; no physical commands.

From Project_Humanoid:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
PYTHONPATH=. python -m igris.hl_ik.report_figures.make_bent_comparison
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import csv
import json
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from igris.hl_ik.solver import HLIK
from igris.hl_ik.common import ROOT, HERE, OUTPUT, SDK_NAMES, ELBOW_NAMES, Rotation, sha256, save_json

OUT = ROOT / 'outputs/report_figures/bent_elbows'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT/'outputs/hl_ik_pipeline_20260915/final'
FRAME = 65
LABELS = ['qp_priority_none', 'qp_priority_full']
manifest = json.loads((SOURCE/'manifest.json').read_text())
meta = [json.loads((SOURCE/'runs'/s/'metrics.json').read_text()) for s in LABELS]
config = [manifest['configs'][s].copy() for s in LABELS]
assert config[0].pop('prior') is None
checkpoint = config[1].pop('prior')
assert config[0] == config[1]
assert sha256(checkpoint) == manifest['prior_sha256'][checkpoint]
for m in meta:
    assert m['manifest_sha256'] == sha256(SOURCE/'manifest.json')
    assert all(sha256(HERE/n) == h for n, h in m['source_sha256'].items())
collisions = [next(r for r in m['fixtures'] if r['name'] == 'box')['collision'] for m in meta]
assert collisions[0] == collisions[1]
assert sha256(SOURCE/'fixtures/box.npz') == manifest['fixture_sha256']['box']
runs = [np.load(SOURCE/'runs'/s/'box.npz') for s in LABELS]
assert np.array_equal(runs[0]['goals'], runs[1]['goals'])
assert np.array_equal(runs[0]['time'], runs[1]['time'])
q = np.stack([z['q'][FRAME] for z in runs])
assert np.array_equal(q[0, :15], q[1, :15]) and np.array_equal(q[0, 29:], q[1, 29:])
goal = runs[0]['goals'][FRAME]
solver = HLIK(hand_envelope=collisions[0]['hand_envelope'])
assert solver.geometry.provenance() == collisions[0]
elbows, metrics = [], []
for cur, z in zip(q, runs):
    solver.set_q(cur)
    state = solver.states()
    actual = state[:, :7].copy()
    actual[:, :3] += solver.shoulder
    elbow = state[:, 7:10]+solver.shoulder
    elbows.append(elbow)
    ep = np.linalg.norm(actual[:, :3]-goal[:, :3], axis=1)*1000
    er = np.rad2deg((Rotation.from_quat(actual[:, 3:], scalar_first=True).inv()*Rotation.from_quat(goal[:, 3:], scalar_first=True)).magnitude())
    assert ep.max() < .001 and er.max() < .001
    assert np.abs(cur[[18, 25]]).min() > np.deg2rad(45)
    assert solver.geometry.collision(solver.configuration.data) is None
    assert np.all(cur[15:29] >= solver.limits[:, 0]) and np.all(cur[15:29] <= solver.limits[:, 1])
    assert np.all(solver.A@cur[15:29] <= solver.b+1e-8)
    end = solver.configuration.q.copy()
    solver.set_q(z['q'][FRAME-1])
    assert solver.geometry.corridor(solver.configuration.data, solver.configuration.q.copy(), end, max_step=.01) is None
    velocity = (cur[15:29]-z['q'][FRAME-1, 15:29])*30
    acceleration = (cur[15:29]-2*z['q'][FRAME-1, 15:29]+z['q'][FRAME-2, 15:29])*900
    assert abs(velocity).max() <= 3.00001 and abs(acceleration).max() <= 30.0001
    metrics.append(dict(ee_position_mm=ep.tolist(), ee_orientation_deg=er.tolist(),
                        elbow_pitch_abs_deg=np.rad2deg(abs(cur[[18, 25]])).tolist(),
                        speed_max_rad_s=float(abs(velocity).max()), acceleration_max_rad_s2=float(abs(acceleration).max())))
elbows = np.array(elbows)
shift_cm = np.linalg.norm(elbows[1]-elbows[0], axis=1)*100

with (OUT/'joint-positions.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['joint', 'without_cost_rad', 'with_cost_rad', 'difference_deg'])
    w.writerows((n, a, b, np.rad2deg(b-a)) for n, a, b in zip(SDK_NAMES, q[0], q[1]))
with (OUT/'ee-targets.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['side', 'x_m', 'y_m', 'z_m', 'qw', 'qx', 'qy', 'qz'])
    w.writerows((s, *p) for s, p in zip(['left', 'right'], goal))
np.savez_compressed(OUT/'pose-data.npz', q=q, goal=goal, elbows=elbows, joint_names=SDK_NAMES)

W, H = 1100, 900
model = solver.model
model.vis.global_.offwidth = W
model.vis.global_.offheight = H
model.vis.headlight.ambient[:] = .65
model.vis.headlight.diffuse[:] = .75
model.vis.quality.offsamples = 4
option = mujoco.MjvOption()
option.geomgroup[:] = 0
option.geomgroup[1] = 1
colors = [(37, 104, 147), (120, 80, 163)]
gold = (255, 198, 67)
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(size):
    return ImageFont.truetype(font_path, size)

renderer = mujoco.Renderer(model, H, W)
def project(point):
    cams = renderer.scene.camera
    pos = (cams[0].pos+cams[1].pos)/2
    forward, up = cams[0].forward, cams[0].up
    rel = np.array(point)-pos
    depth = rel@forward
    assert depth > 0
    near = cams[0].frustum_near
    bottom, top = cams[0].frustum_bottom, cams[0].frustum_top
    center = (cams[0].frustum_center+cams[1].frustum_center)/2
    x = W/2 + (near*(rel@np.cross(forward, up))/depth-center)*H/(top-bottom)
    y = H - (near*(rel@up)/depth-bottom)*H/(top-bottom)
    assert 0 <= x < W and 0 <= y < H
    return np.array([x, y])

panels = []
try:
    for i, cur in enumerate(q):
        solver.set_q(cur)
        for k in range(model.ngeom):
            if model.geom_group[k] != 1:
                continue
            body = model.body(int(model.geom_bodyid[k])).name
            arm = body.startswith(('l_', 'r_')) and any(s in body for s in ['shoulder', 'upper_arm', 'elbow', 'forearm', 'wrist'])
            model.geom_rgba[k] = [*(np.array(colors[i])/255 if arm else [.60, .63, .67]), 1.]
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.15, 0, .24]
        camera.distance = 1.35
        camera.azimuth = 210
        camera.elevation = -15
        renderer.update_scene(solver.configuration.data, camera, option)
        picture = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(picture)
        for p in goal[:, :3]:
            x, y = project(p)
            draw.line((x-12, y, x+12, y), fill='white', width=3)
            draw.line((x, y-12, x, y+12), fill='white', width=3)
            draw.text((x+17, y-20), 'EE', font=font(23), fill='white', stroke_width=2, stroke_fill='black')
        for side, p in enumerate(elbows[i]):
            x, y = project(p)
            draw.ellipse((x-25, y-25, x+25, y+25), outline=gold, width=4)
            if i == 1:
                old = project(elbows[0, side])
                draw.ellipse((old[0]-17, old[1]-17, old[0]+17, old[1]+17), outline=(210, 215, 220), width=3)
                draw.line((*old, x, y), fill=gold, width=4)
                direction = np.array([x, y])-old
                direction /= np.linalg.norm(direction)
                normal = np.array([-direction[1], direction[0]])
                tip = np.array([x, y])
                draw.polygon([tuple(tip), tuple(tip-14*direction+7*normal), tuple(tip-14*direction-7*normal)], fill=gold)
                label = f'{shift_cm[side]:.1f} cm'
                lx = x+34 if side == 0 else x-150
                draw.text((lx, y+24), label, font=font(32), fill=gold, stroke_width=2, stroke_fill='black')
        picture.save(OUT/f'pose-{i}.png')
        panels.append(picture)
finally:
    renderer.close()

sheet = Image.new('RGB', (2280, 1470), 'white')
draw = ImageDraw.Draw(sheet)
draw.text((40, 28), '팔꿈치를 굽힌 자세에서: 손 목표는 유지하고 팔 배치를 조절', font=font(47), fill='#182c3e')
draw.text((40, 100), '동일한 양손 EE 위치·방향  |  양팔 elbow_pitch 약 105°  |  학습된 자세 cost 적용 전후', font=font(29), fill='#536678')
for i, picture in enumerate(panels):
    x = 30+i*1130
    draw.rounded_rectangle((x, 172, x+1100, 240), radius=8, fill=colors[i])
    draw.text((x+24, 181), '인간 자세 cost '+('미적용' if i == 0 else '적용'), font=font(35), fill='white')
    sheet.paste(picture, (x, 260))
draw.text((42, 1191), '오른팔 팔꿈치 6.26 cm · 왼팔 3.58 cm 이동', font=font(38), fill='#182c3e')
draw.text((42, 1250), '금색 원: 현재 팔꿈치  ·  회색 원: cost 미적용 위치  ·  흰 십자: 양쪽에 동일한 EE 목표', font=font(28), fill='#536678')
draw.text((42, 1308), '수치 FK 검증: 두 조건 모두 EE 위치 오차 < 0.001 mm / 방향 오차 < 0.001°. 동일 카메라·허리·하체 자세.', font=font(26), fill='#536678')
draw.text((42, 1360), '기존 box 목표 시퀀스 frame 65 (2.167 s)의 운동학 사례. 동일 초기 이력 후 각 조건의 생성 이력을 사용.', font=font(25), fill='#536678')
draw.text((42, 1408), '구조를 설명하는 단일 사례이며, 파지 성공이나 전체 성능 향상의 지표가 아님. 물리 step 0 / 실물 명령 0.', font=font(25), fill='#536678')
sheet.save(OUT/'figure-3b-bent-elbows.png', dpi=(220, 220))
sheet.resize((1368, 882)).save(OUT/'preview.png')
gif_frames = []
for i, picture in enumerate(panels):
    im = Image.new('RGB', (W, H+90), '#182c3e')
    im.paste(picture, (0, 90))
    ImageDraw.Draw(im).text((30, 20), '인간 자세 cost '+('미적용' if i == 0 else '적용'), font=font(36), fill='white')
    gif_frames.append(im.resize((660, 594)))
gif_frames[0].save(OUT/'cost-toggle.gif', save_all=True, append_images=gif_frames[1:], duration=[1300, 1300], loop=0)
paths = [SOURCE/'manifest.json', SOURCE/'fixtures/box.npz', Path(checkpoint), Path(collisions[0]['hand_envelope']), OUTPUT/'model/igris_c_rcv_hand_torso.xml']
paths += [SOURCE/'runs'/s/'box.npz' for s in LABELS]
report = dict(fixture='box', frame=FRAME, time_s=float(runs[0]['time'][FRAME]), metrics=metrics,
              elbow_shift_cm=shift_cm.tolist(), elbow_displacement_m=(elbows[1]-elbows[0]).tolist(),
              selection='Illustrative bent-arm frame selected after screening completed paired runs. Both elbows exceed45deg and both EE residuals below1mm/0.5deg. Choose large elbow spatial displacement; do not interpret as a population metric.',
              history='identical initial history; then each run uses its own generated history',
              source_sha256={str(p):sha256(p) for p in paths}, solver_source_sha256=meta[0]['source_sha256'],
              collision_model=collisions[0], physics_steps=0, hardware_commands_sent=0, execution_ready=False)
save_json(OUT/'validation.json', report)
print(json.dumps(dict(metrics=metrics, elbow_shift_cm=shift_cm.tolist()), indent=2))
