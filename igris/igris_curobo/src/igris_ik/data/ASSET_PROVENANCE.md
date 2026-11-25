# IGRIS-C pelvis-base URDF + RCV hand (2026-08-26)

## 2026-09-11 end-effector update

- Refreshed from the later source revision, SHA-256
  `f06f037e61a756b528bae73daf57f65da4462b1e525c33d511f4995b30bede6d`.
  The packaged URDF now matches the supplied file byte-for-byte (no duplicate links).
  EE origins remain `0 0 -0.0082`; local RPY is `(0, pi/2, +pi/2)` for left,
  `(0, pi/2, -pi/2)` for right. This changes each old EE orientation by 120 degrees.
  Added the source's `l_wrist_cam` fixed frame. All 31 motor joints and existing
  mesh contents are unchanged; collision spheres therefore need no refit.
- Historical EE pose quaternions must be converted with
  `T_target_new = T_target_old @ inverse(T_wrist_oldEE) @ T_wrist_newEE`.
  Joint recordings remain in the same motor convention. VR calibration must be
  recaptured after restarting the solver with the updated model.
- Source: `igris/igris_c_description/urdf/igris_c_rcv_hand_pelvis.urdf` supplied locally.
- Imported `end_effector_left/right`, three camera links and four camera meshes.
- Removed one identical duplicate `l_hand_본체1_5` link in this packaged copy;
  the supplied source is retained unchanged. Existing joints and mesh files are unchanged.
- IK targets, calibration, Cartesian interpolation and FK errors use the new EE frames.
  Shoulder reach and human arm landmarks still refer to wrist connector origins.
- Native vendor MJCF retains its free base, linkage dynamics and 0.6 kg fixed hands;
  fixed EE/camera frames are copied by `sim_local/vr_box_eval/bake_fixed_hands.py`.
  Camera meshes and EE markers are visual only, matching the new links' lack of collisions.

## Project import provenance

- Imported: 2026-08-27
- Source archive: `igris_c_rcv_hand_pelvis_urdf_2026-08-26 (1).zip`
- Source SHA-256: `e5cbb8e24da2dde575d5c69e492aef3ee2afc95f6b811b267aa562211a43be89`
- Phase 1 use: canonical URDF/mesh source for PyRoki, Viser, and the
  fixed-hand MuJoCo derivative
- Not imported: the companion USD archive; it remains reserved for later
  Isaac/Isaac Lab integration

Runtime code resolves this checked-in package tree and does not access the
original file under `~/Downloads`.

## 내용
- `igris_c_description/urdf/igris_c_rcv_hand_pelvis.urdf` — 이 파일 하나가 로봇 모델
- `igris_c_description/meshes/igris_c_v2/` (35 STL, 몸체) · `meshes/igris_c_rcv_hand/{left,right}/` (34 STL, RCV hand)
- `igris_c_description/package.xml` — URDF 의 메시 경로가 `package://igris_c_description/...` 이라 ROS 패키지명 해석용

## 구조
- 루트 `base_link` = **골반**(pelvis). 체인: base_link → pelvis(고정) → waist_pitch → waist_roll → waist_yaw → torso → 팔·목 / pelvis → 양쪽 다리
- 몸체 가동 관절 31 (다리 12, 허리 3, 팔 14, 목 2) — 전부 직렬 revolute(joint 공간, 링키지 없음)
- RCV hand 8 관절(`l/r_hand_dof_joint_{cmc,dip,mcp,mcp_thumb}`) 가동
- 관절 limit 은 igris_c_description v2.1.0 값
- 손 장착: `l_hand`/`r_hand` 고정 조인트, `wrist_connector` 기준 xyz (0, ±0.0109, −0.024), rpy (π, 0, ∓π/2)
  → 영자세(모든 관절 0, 팔 내림)에서 **손바닥이 몸통을 정면으로 향하고 손가락은 아래**. 1DOF 그리퍼(xacro `end_effector:=1dof`)의 장착 각도와 정합.

## 주의
- RCV hand 링크 질량은 0.001 kg(placeholder). 물리 시뮬에 쓰려면 실측 관성 필요 (손 전체 약 0.6 kg 추정).
- ROS 없이 쓸 땐 `package://igris_c_description/` 을 압축 푼 폴더 경로로 치환.
- 관절 이름은 description 규약(`l_hip_pitch`, `waist_yaw`, …). 리타게팅 산출물의 `Joint_Hip_Pitch_Left` 식 이름과의 대응: l_/r_ 접두 → _Left/_Right 접미, 나머지 Title_Case.
