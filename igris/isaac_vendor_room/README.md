# Isaac Sim room + IGRIS-C vendor backend

사용자가 선택한 **벤더 동역학 우선 구성**이다. 실제 물리는 기존 MuJoCo **3.2.7**이 계산하고,
Isaac Sim **6.0.0-rc.59**는 그 결과를 렌더링한다. Newton solver는 실행하지 않는다.
Newton/MuJoCo-Warp로 교체하거나 PhysX로 로봇을 다시 시뮬레이션하는 구성이 아니다.

```text
상체 q [rad]
  → 기존 SDK ROS LowCmd (PJS, 100 Hz)
  → 기존 bridge_sim → 기존 controller sim (WALK_LOW)
  → 기존 MuJoCo 3.2.7: 로봇 + 방 충돌체 + 동적 박스
  → 읽기 전용 body pose 관측 (최대 30 Hz)
  → Isaac Sim USD world pose → RGB / depth
```

`igris/sim_local/robot.yaml`과 `igris_c_v2_rcv_hand.xml`을 재사용한다. 원본 파일은 수정하지 않는다.
하체는 벤더 WALK_LOW 정책이 맡고, 예제 클라이언트는 허리 3축·양팔 14축·목 2축만 명령한다.
로봇 베이스는 floating이다. 기본 고정 손 구성과 아래의 MCP·DIP 컴플라이언스 구성을 구분한다. 원본 RCV 손 메시 76개의
충돌 비트만 `contype=2, conaffinity=0`으로 설정하고 방에는 `conaffinity=3`을 적용한다.
이는 사용자가 승인한 손–방 접촉 추가다. 원래 손은 visual-only였다.
로봇끼리·손끼리·원래 바닥과의 충돌 필터는 원본 그대로다. 기본 고정 손 구성은 손 질량·관성·형상·구동기를 바꾸지 않는다. MCP·DIP 구성은 사용자가 요청한 기존 손 모델로만 교체한다.

방은 `assets/room_real2sim/room_corner_igris.usd`에서 읽는다. 기존 정적 롤테이너를 유지하고,
팔레트 1개와 박스 38개를 동적 강체로 옮긴다. Cube/Cylinder 충돌체 214개를 정확한 위치·크기로
변환한다. 중복 바닥 충돌을 피하려고 USD 바닥 collider 하나는 제외하고 벤더 ground plane을 유지한다.
새 방 물체의 MuJoCo 관성은 collision volume과 USD 질량으로 계산한다. 마찰은 USD dynamic friction을
sliding friction으로 사용하고 torsional/rolling friction은 MuJoCo 기본값을 쓴다.
따라서 방의 PhysX 접촉 결과와 수치적으로 같다는 뜻은 아니다.

## 실행

필요한 기존 환경: 호스트 `python3`의 `pxr`, `numpy`, `yaml`; `g++`, Docker,
`igris-sim:jazzy` 이미지와 저장소의 `igris/ws/install`, `/home/rcv/isaacsim/python.sh`.
새 라이브러리를 설치하지 않는다. 이미지가 없으면 기존 `igris/sim_local/Dockerfile`로 빌드한다.

저장소 루트에서 새 출력 디렉터리를 선택한다. 이미 존재하는 실험 디렉터리는 덮어쓰지 않는다.

```bash
ROOM_RUN="$PWD/outputs/isaac_vendor_room_$(date +%Y%m%d_%H%M%S)"
python3 igris/isaac_vendor_room/prepare.py --out "$ROOM_RUN"
python3 igris/isaac_vendor_room/run.py start --out "$ROOM_RUN"
python3 igris/isaac_vendor_room/run.py bringup --out "$ROOM_RUN"
```

`start`는 이름이 겹치지 않는 컨테이너를 만들고, `--network none`, private IPC, loopback DDS로
격리한다. 호스트 및 실기 DDS와 연결하지 않는다. `bringup`은 해당 컨테이너 안에서만
reset → resume → torque on → HOME → reset → WALK_LOW를 실행한다.
사용자가 허용한 **sparse Jacobian이 기본값**이다. 나머지 물리 옵션은 제조사 원본과 모두
일치해야 시작할 수 있다. 원본 dense 비교는 `prepare.py --dense`로 별도 출력에 생성한다.
`bringup`은 관측기가 확인한 실제 물리 스레드를 가장 높은 최대 주파수의 CPU에 고정하고
FIFO 50을 적용한다. 특정 CPU는 `--physics-cpu 8`처럼 지정할 수 있다.
Isaac과 중복되는 숨겨진 MuJoCo 뷰어의 그리기는 기본적으로 끈다. 물리 함수는 그대로 실행한다.
그리기 비용 비교에만 `start --native-draw`를 사용한다.

별도 터미널에서 Isaac 창을 연다. 위의 `ROOM_RUN` 값을 같은 경로로 설정한다.

```bash
/home/rcv/isaacsim/python.sh igris/isaac_vendor_room/render.py \
  --out "$ROOM_RUN" --gui --live --record --seconds 20
```

기본값은 **Overview(기존 구도), RobotFacing(로봇 전방), RolltainerDown(롤테이너 위에서 내려다봄)**
세 구도의 동시 기록이다. 한 물리 샘플을 모든 링크에 적용한 다음 한 번의 렌더 평가로
세 카메라를 읽는다. 각 카메라의 `recording.mp4`, PNG, depth와 공통 timestamp 배열을 저장한다.
라이브 영상은 물리 샘플의 실제 wall 간격을 반영하는 가변 프레임률이며, 인코딩 후 전체 디코딩을 검사한다.
정면 구도에서 팔레트·롤테이너에 의한 자연스러운 가림은 유지한다. 롤테이너 위 구도만 필요하면
`--camera /World/Cameras/RolltainerDown`을 지정한다. 창을 계속 열어두려면 `--seconds 0`으로 실행하고,
유한 길이 녹화는 위처럼 양수 시간을 지정한다. 렌더러는 물리 CPU와 그 SMT 형제를 피해서 실행한다.

화질 옵션(2026-11-24 추가, 주지 않으면 이전과 같은 동작): 렌더러는 자기 프레임을 여러 장 누적해 한 장을 만든다.
저장 프레임마다 강체를 다음 포즈로 옮긴 뒤 기본 2장만 누적하므로 움직이는 부분이 이전 위치와 섞여 뭉개진다.
`--subframes 16 --antialiasing 4`(원본 해상도 DLAA)면 깨끗해지고 프레임당 시간은 약 2.4배다(PRO6000, 카메라 1대
0.13 → 0.31초). 기록 재생에서 `--interpolate`를 주면 다음 기록 행 대신 프레임 시각의 포즈를 보간해 쓴다
(`pose_frames.py`; 약 30 Hz 기록을 그대로 고르면 프레임 간격이 고르지 않다). 측정은
`igris/umi_walklow_replay/README.md`의 "렌더 화질" 절에 있다.

Isaac의 Play 버튼으로 물리를 시작할 필요가 없다. 원본 USD를 참조한 별도 렌더 장면을 만들고,
그 장면에서는 물리·관절 스키마를 제거한다. 143개 로봇 링크와 39개 방 강체의 world pose를
MuJoCo에서 받아 적용한다. 고정 손·카메라 링크는 관절 계층의 고정 변환을 유지한다.
RGB와 depth는 동일한 외부 자세로 평가하며 Isaac timeline의 시간을 진행시키지 않는다.
렌더링 샘플마다 원본 MuJoCo wall/sim timestamp를 기록한다.

상체 목표 예제:

```bash
python3 igris/isaac_vendor_room/run.py command --out "$ROOM_RUN" \
  --q '{"waist_yaw":0.05,"l_elbow_pitch":-0.3,"r_elbow_pitch":-0.3}' \
  --seconds 30
```

관절 이름은 `run.py`의 `JOINTS` 순서를 사용한다. 목표는 절대 PJS 관절각(rad)이다.
지정하지 않은 상체 축은 명령 시작 시 측정각을 유지한다. 하체 q/dq/tau/kp/kd는 0으로 보내며
WALK_LOW가 하체를 제어한다. URDF limit·유한값·SIM 상태·WALK_LOW·상태 freshness를 확인하고,
최대 0.3 rad/s의 quintic ramp 후 남은 시간 동안 유지한다. 허리에는 벤더 hold gain,
팔에는 벤더 kp와 최대 5의 kd, 목에는 30/3을 사용한다.
`command_*.json`에 목표, 실제 q, 오차, mode, pelvis, wall/sim timestamp가 남는다.

기존 SDK 또는 상위 bridge를 쓸 때는 동일한 **격리 컨테이너 안에서** namespace
`igris_c_001`, domain 0의 `rt/lowcmd`를 사용한다. 컨테이너 이름은 `runtime.json`에 기록된다.
단일 command publisher만 사용한다. 호스트 DDS를 열어 실기와 섞는 설정은 이 실행기에 없다.

종료:

```bash
python3 igris/isaac_vendor_room/run.py stop --out "$ROOM_RUN"
```

해당 실험 컨테이너만 정지한다. 소스·로그·기존 컨테이너는 삭제하지 않는다.

## MCP·DIP 손과 이송 높이 +10 cm (2026-09-18)

`box_replay.py prepare --compliant-hands`는 기존
`igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/scene.xml`의 손 subtree와
`compliant_hands.py`, `hand_motor.cpp`, HAND admittance/profile을 그대로 사용한다.
MCP 기준각은 30°, 물리 관절 범위는 0–60°, 기존 실행 검증 범위는 20–40°다.
DIP는 구동기가 없는 torsion spring이며 강성 0.049015873 Nm/rad, damping 0.0001,
preload 0°, 실행 검증 범위 −10–45°를 유지한다. 이 강성은 기존 코일 추정값이며 실측 보정값은 아니다.
손 하나의 총질량은 0.6 kg이다. 기존 손–방 접촉 비트를 적용하고 손 이외의 제조사 모델은 보존한다.
특히 과거 손 실험의 `noslip_iterations=10`은 가져오지 않으며 제조사 값 0을 유지한다.

사용 URDF는 `outputs/hl_ik_20260915/model/igris_c_rcv_hand_torso.urdf`다.
`base_link → torso`가 fixed이고 EE 두 개가 각각 MCP 회전축 위에 있다.
EE 원점은 MCP joint 원점에서 축 방향으로 14.7 mm 떨어져 있다.
URDF의 torso root는 IK 기준이며 native 로봇의 floating pelvis를 고정하지 않는다.
`test_hands.py`는 손 자세 12개와 native snapshot의 torso→EE FK를 비교한다.
제조사 모델의 출력각은 motor q에 passive backlash joint q를 더한 값이며,
이 합을 사용하면 native 3.2.7과 URDF가 수치 정밀도 범위에서 일치한다.

`lift_motion.py`는 원본 q 파일을 보존하고 별도 archive를 만든다. 양손의 이송 구간에
torso 기준 Z +0.10 m를 부드럽게 추가하고 집기·내려놓기 높이와 허리 q는 유지한다.
원래 손 방향을 유지한 Z 단독 변경은 오른쪽 wrist yaw limit에 걸려, 이송 중 X +0.08 m도 추가한다.
이는 박스의 실제 world Z 상승량을 보장하는 값이 아니라 IK 목표 변경량이다.
검증된 기본 파생본은 8,102 samples / 81.01 s이며 최대 FK 위치 오차 0.094 mm,
방향 오차 0.256°와 기존 손 충돌 envelope 및 coupled joint limits를 확인했다.

```bash
export PYTHONPATH="$PWD/igris/igris_curobo/src:$PWD/igris-core/igris_core:$PWD/igris/real_client:$PWD"
LIFT_RUN="$PWD/outputs/isaac_lift_$(date +%Y%m%d_%H%M%S)"
/home/rcv/workspace/curobo/.venv/bin/python igris/isaac_vendor_room/lift_motion.py --out "$LIFT_RUN/motion"
python3 igris/isaac_vendor_room/box_replay.py prepare --out "$LIFT_RUN/room" \
  --compliant-hands --motion "$LIFT_RUN/motion/motion_100hz.npz"
python3 igris/isaac_vendor_room/run.py start --out "$LIFT_RUN/room/run"
python3 igris/isaac_vendor_room/run.py bringup --out "$LIFT_RUN/room/run" --physics-cpu 8
python3 igris/isaac_vendor_room/run.py command --out "$LIFT_RUN/room/run" \
  --compensate-backlash --motion "$LIFT_RUN/motion/motion_100hz.npz" \
  --gains "$PWD/igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/replay_gains.toml"
python3 igris/isaac_vendor_room/run.py stop --out "$LIFT_RUN/room/run"
python3 igris/isaac_vendor_room/box_replay.py analyze --out "$LIFT_RUN/room/run"
```

오류로 `command`가 끝나도 `stop`을 실행한다. 같은 컨테이너만 정지하고 로그는 보존한다.
`bringup`은 원래 실험과 같이 reset/WALK_LOW 호출을 80 ms 간격으로 시작한다.
상체 준비 자세를 마친 뒤 방 물체의 초기 위치를 현재 pelvis XY/yaw에 맞춰 한 번 배치한다.
박스를 큰 박스 네 개 위에 쌓고 팔레트는 발에서 멀리 옮긴다. 2초간 안정화한 뒤 replay한다.
`room_setup.cpp`는 이 초기화 이후 물체를 이동시키지 않으며 로봇 상태와 물리 계수를 수정하지 않는다.

`--compensate-backlash`는 기존 bridge의 native backlash reader로 출력각을 추종하도록
LowCmd를 보정한다. 제조사 backlash 물성은 유지하며 보정된 motor target을 별도 기록한다.
보정은 기존 individual/coupled limits와 0.5 rad/s를 지킨다. 파생 archive의 가속도·jerk 검증을
온라인 보정 명령까지 통과했다는 뜻은 아니다. 이 옵션에서는 원본 LowCmd와 완전히 같은 q를 보내지 않는다.
필터·LowState·MCP 상태가 오래되거나 손의 기존 검증 범위를 벗어나면 중단한다.

사용자의 2026-09-18 크기 확정에 따라 `--compliant-hands`의 기본 payload는
**15×15×15 cm / 1 kg**이다. Body 시각 메시와 Body_Collider 파지면을 함께 맞춘다.
기존 손 실험의 cube/contact 속성을 사용하며, `--no-vendor-box`로 이전 18×18×10 cm 비교를 재현할 수 있다.
원본 asset은 수정하지 않는다. `replay_result.json`은 q 송신·추종, 실제 상승·무지지 운반·착지·해제와
양손 palm/tip 접촉을 구분하며 기존 1 kg force 기준(각 palm 6.131 N, 각 tip 0.2 N)을 검사한다.
carry의 95% 이상에서 네 접촉점과 무지지 상승 조건을 만족해야 전체 성공으로 판정한다.
과거 원본 실험도 palm/tip 동시 접촉률은 19.6%, 최종 `task_success=false`였다.
손 모델 복원이나 q 추종만으로 안정적인 전체 손 파지가 검증된 것은 아니다.

Isaac 렌더러는 새 MCP/DIP 네 body의 native 움직임까지 받아 동일한 세 구도에 반영한다.
녹화는 앞 절의 `render.py`를 사용한다. 저장된 physics replay 렌더는 실행 당시 wall timestamp를
유지하므로 영상 속도를 높여 성능 문제를 가리지 않는다.

최근 `outputs/isaac_compliant_carry_20260918/gt_bringup`에서는 기존 크기 박스가 15.06 cm 상승해
23.93초간 지지대 없이 양손 접촉으로 이동했다. 실측 RTF는 0.9815, LowCmd는 100 Hz였고
내려놓는 중 MCP 범위 이탈로 7,295/8,102 samples에서 중단했다. 전체 손 접촉 기준과 착지는 미통과다.
해당 실행의 세 영상은 1280×960, 8 fps, 595개 공통 native samples이며 전체 디코딩 검사를 통과했다.
전체 실험·비교·검증 기록은 `outputs/isaac_compliant_carry_20260918/REPORT.md`에 있다.

15 cm 크기 확정 후 새 실행 `outputs/isaac_cube15_20260918/retest_01`에서는 시각 메시와 충돌면
치수 검사를 통과했다. 14.28 cm 상승, 11.52초 무지지 양손 접촉, 실행된 carry의 palm+tip 동시
접촉률 92.35%를 기록했지만, 박스 기울기 24.04°와 오른손 MCP 19.908°에서 중단했다.
크기 적용은 완료됐으며 안정 파지·착지 성공은 아니다. 결과와 재현 절차는
`outputs/isaac_cube15_20260918/REPORT.md`에 있다.

## 기존 고정 손 1 kg 작업 q 재생

`box_replay.py`는 `outputs/synthetic_box_motion_20260914/motion_100hz.npz`의
80.6초·8,061개 SDK 상체 q를 그대로 사용한다. IK를 다시 풀거나 본편 q를 보정하지 않는다.
준비 구간에서 팔을 옆으로 벌리고 팔꿈치를 접은 뒤 기록의 첫 자세에 접근한다.
이 준비 구간은 `archive_index=-1`, 본편은 원본 샘플 번호로 구분된다.
본편은 매 물리 step을 관측하는 기존 `hl_ik/native_clock`에 맞추며, 100 Hz 벽시계 송신에서
같은 샘플을 반복할 수 있으나 샘플을 건너뛰지는 않는다. 1배속 벽시계 실행을 뜻하지 않는다.

새 배치는 원본 USD를 덮어쓰지 않는다. room asset의 작은 박스(18×18×10 cm)를 1 kg으로
설정하고 높이가 18 cm가 되도록 세워 큰 박스 위에 놓는다. 형상 크기는 변경하지 않는다.
집기 받침 높이는 약 72 cm, 받는 쪽은 약 68 cm다. payload 마찰계수 1.2는 기존 1 kg 시험에서 가져온다.
큰 받침 박스를 엇갈려 쌓아 발·팔의 여유를 확보하며, 롤테이너는 +Y 방향으로 1 m,
팔레트는 대각선 바깥쪽으로 10 cm 옮긴다. 나머지 박스도 모두 동적·충돌 활성 상태로 보존한다.
각 좌표·질량·입력 hash는 `layout.json`과 `run/manifest.json`에 기록된다.

원래 q 파일을 만든 시험은 15 cm 정육면체, MCP 30도와 passive DIP spring을 가진 별도 손,
`noslip_iterations=10`을 사용했다. 이전 고정 손 실험은 제조사 고정 손과 `noslip_iterations=0`을
유지한다. 따라서 같은 q 재생이 같은 파지 결과를 보장하지는 않는다. 본편 q와 저장된
`replay_gains.toml`은 보정하지 않으며, SDK 메시지의 float32 변환만 적용된다.

```bash
BOX_RUN="$PWD/outputs/isaac_box_replay_$(date +%Y%m%d_%H%M%S)"
python3 igris/isaac_vendor_room/box_replay.py prepare --out "$BOX_RUN"
python3 igris/isaac_vendor_room/run.py start --out "$BOX_RUN/run"
python3 igris/isaac_vendor_room/run.py bringup --out "$BOX_RUN/run"
python3 igris/isaac_vendor_room/run.py command --out "$BOX_RUN/run" \
  --motion "$PWD/outputs/synthetic_box_motion_20260914/motion_100hz.npz" \
  --gains "$PWD/igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/replay_gains.toml"
python3 igris/isaac_vendor_room/run.py stop --out "$BOX_RUN/run"
python3 igris/isaac_vendor_room/box_replay.py analyze --out "$BOX_RUN/run"
```

`command` 합격은 q 추종 판정이다. 운반·착지는 `replay_result.json`의 별도 결과로 확인한다.
운반 판정에는 원본 q 전부 송신, q 추종, 25 mm 이상 상승, 받침 접촉 없이 양손 지지 1초 이상,
최종 받는 박스와의 접촉, 정착 및 손 접촉 해제를 함께 요구한다. 손바닥·손끝 압력 품질은 별도다.
`payload_contacts.jsonl`은 실제 payload–손/받침 접촉력을 기록한다. 저장된 물리 상태는 위의
렌더 명령으로 재생할 수 있다. 라이브 녹화를 먼저 끝내려면 해당 capture 디렉터리에
`stop.request` 파일을 만들면 정상적으로 인코딩하고 종료한다.

2026-09-17 기본 배치와 같은 `outputs/isaac_box_replay_20260917/layout_m`에서 8,061개 q
전량 재생, 4.8 cm 상승, 연속 8.22초의 받침 없는 양손 지지, 받는 박스와의 최종 접촉을 확인했다.
그러나 왼손 접촉 1.24–2.47 N이 남아 최종 `replay_result.json`은 **pass=false**다.
받침을 63 cm로 낮춘 `layout_q`에서는 박스가 바닥으로 떨어졌다. 현재 기본값은 완성된
pick-and-place 성공 프리셋이 아니라 가장 진행이 좋았던 재현용 배치다.

## 검증과 성능 경계

```bash
python3 igris/isaac_vendor_room/test_room.py
```

`start`는 실제 벤더 `libmujoco.so.3.2.7`로 원본과 합성 장면을 비교한다.
로봇 mass/inertia, body/joint frame, damping/friction/armature, actuator gain/bias/gear/limits,
equality, robot collision geometry/contact parameters, keyframe, solver option을 검사한다.
결과는 `model.json`, 입력 SHA-256은 `manifest.json`에 남는다.
38개 박스의 충돌 필터와 원본 로봇/바닥의 모든 충돌 비트 쌍도 검사한다.
`contacts.jsonl`은 실제 실행에서 발생한 로봇–방 접촉의 물체 이름·거리·접촉력을 기록하고,
`physics_stats.json`은 실제 물리 thread ID와 계산 비용을 기록한다.

2026-09-17 사용자가 sparse 계산 표현을 허용했다. 변경하는 것은 Jacobian 표현이며,
solver 종류·반복수·허용오차·로봇 질량·관성·구동기·자기충돌은 원본을 유지한다. 물리 주기는 원본
`0.00047619048 s`(약 2,100 Hz), native 상태 decimation은 7이다. 그러나 방 전체의 접촉 계산 때문에
벽시계 기준 상태 전달률이 300 Hz에 못 미친다. WALK_LOW policy의 벽시계 타이머도 원래 코드 그대로이므로
실시간보다 느린 실행을 원본과 같은 시간 응답으로 간주하면 안 된다.
실시간 성능을 얻기 위해 원본 로봇 관성·접촉체·제약·dt를 임의로 낮추지 않았다.

**2026-09-17 초기 밀집 배치에서는 1배속 요구가 미달성이었다.** 카메라가 없어도 약 0.56–0.62배속이었다.
동일 실행에서 차례로 측정한 1카메라 구간은 0.512배속, 3카메라 구간은 0.652배속이었다.
접촉 상태가 다른 구간이므로 3대가 더 빠르다는 인과 비교는 아니다. 카메라 수를 줄이는 것만으로
물리 병목이 해결되지 않는다는 근거다. 네이티브 뷰어 그리기를 꺼 CPU 한 코어 정도의 부하는
줄었지만 1배속은 얻지 못했다. 기록 영상을 시간 압축해 1배속으로 표시하지 않는다.

손–방 접촉과 sparse가 켜진 구성에서 75초간 7,500개 상체 명령(허리 yaw·양팔 elbow)을
전송하며 WALK_LOW를 유지했고 최대 목표 오차는 0.0166 rad였다.
38개 박스 × 팔·하퇴·양손의 152개 독립 접촉 검사를 통과했다. 검증 출력은
`outputs/isaac_room_refine_20260917`에 보존한다.

별도의 19축·65초 시험은 39.37초까지 최대 목표 오차 0.0176 rad로 추종했으나,
이후 물리 pose 갱신이 2.286초 끊겨 freshness 검사에서 중단됐다. 벤더 controller도
SHM stale을 감지해 JOINT_HOLD로 전환했다. 이 시험은 실패이며 중단 원인은 아직 확정하지 못했다.
12초 동안의 3구도 녹화가 정상이라는 결과와 전체 제어 시험의 실패를 구분한다.

Isaac과 native 프로세스를 종료한 뒤, 같은 완전한 MuJoCo 상태·입력을 복원하며 200회씩
한 스텝을 비교했다. dense 9.046 ms, sparse 0.873 ms였으며 sparse 2/4 worker도
0.851/0.855 ms였다. 1배속 예산 0.476 ms를 모두 초과한다. 최대 dense/sparse 차이는
qpos 4.37e-15, qvel 9.17e-12, 접촉 wrench 4.89e-12였다. 이는 한 상태에서의 수치 검사이며
장시간 궤적 동일성이나 비트 단위 동일성을 보증하지 않는다. 병렬 worker는 실행 기본값에 추가하지 않았다.

- [최신 검증 결과와 미해결 항목](../../outputs/isaac_room_refine_20260917/latest_validation.json)
- [75초 통과 실행 중 동기화된 3구도 영상](../../outputs/isaac_room_refine_20260917/three_views_walk_low.mp4)
- [롤테이너 위 단독 영상](../../outputs/isaac_room_refine_20260917/no_native_draw/three_cameras/RolltainerDown/recording.mp4)

검증된 범위는 해당 목표에서의 상체 q 명령 추종, WALK_LOW 기립, 동일 물리 상태의 Isaac RGB/depth 표시다.
보행 속도 명령, 박스 파지·운반, force/IMU 센서의 Isaac 변환은 검증하지 않았다.
IMU·관절·접촉은 벤더 출력을 계속 사용한다. 실물 로봇은 동작시키지 않았다.

이전 `outputs/newton_room_20260917/verified`는 손 충돌이 비활성인 구성이다.
후속 `recorded2`의 dense 실험은 물리 지연으로 WALK_LOW를 잃고 넘어졌다.
이전 영상이나 렌더러의 `pass`를 현재 제어 안정성의 근거로 사용하지 않는다.

## 기록 재렌더링

```bash
/home/rcv/isaacsim/python.sh igris/isaac_vendor_room/render.py \
  --out "$ROOM_RUN" --seconds 6 --fps 15
```

기본은 마지막 6초의 **벽시계 구간**이다. `--start-wall`로 command JSON의 첫 wall timestamp를
지정할 수 있다. 재생은 저장된 실제 MuJoCo 자세이며 IK 애니메이션이 아니다.
라이브는 카메라별 `first.png`, `last.png`, `depth_first.npy`와 공통 `render_report.json`을 저장한다.
매 프레임 PNG 저장은 `--record`를 추가할 때만 수행한다. 캡처를 반복하면 새로운 `--capture` 경로를 지정한다.

독립 충돌 검사·물리 프로파일(외부 명령 전송 없음):

```bash
g++ -O2 -std=c++17 -Iigris/ws/install/igris_c_mujoco/include \
  igris/isaac_vendor_room/check_physics.cpp \
  -Ligris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -Wl,-rpath,"$PWD/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco" \
  -l:libmujoco.so.3.2.7 -o "$ROOM_RUN/check_physics"
taskset -c 8 "$ROOM_RUN/check_physics" "$ROOM_RUN/scene.xml" "$ROOM_RUN/physics_check.json" 1000 0
```

이 검사는 각 박스를 별도 `mjData`에서 팔·하퇴·양손 위치로 옮겨 실제 접촉과 양의 접촉력을
검사한다. 모든 박스를 원래 적층 위치에서 로봇이 실제로 건드린다는 뜻은 아니다.
프로파일은 원본 solver 설정을 수정하지 않으며, 모델 reset 후 무제어 자유 동역학을 측정한다.
라이브 WALK_LOW 성능은 별도로 `poses.bin`의 wall/sim timestamp와 native 로그에서 평가한다.

`observer.cpp`는 최근 완전한 `mjSTATE_INTEGRATION`을 `state.bin`에 저장한다.
동일 상태의 dense/sparse 비교는 위 컴파일 명령에서 `check_physics.cpp`를 `compare_step.cpp`로
바꾸어 빌드하고 `compare_step scene.xml state.bin comparison.json`으로 실행한다.
벤치마크 때 native/Isaac을 종료하고, worker 비교에는 여러 P-core의 affinity를 허용해야 한다.

설계 선택의 근거: [NVIDIA Newton Physics Backend](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/newton_physics.html).
설치된 Newton 쪽 MuJoCo는 3.8.1이며 벤더 3.2.7과 동일한 런타임이 아니다.

## 2026-09-19: 접촉 계산을 명시한 별도 조작 실험

제조사 기본 `noslip_iterations=0`은 기본값으로 유지한다. HL-IK 이송 연구에서
로봇이 없는 받침대만 분리해 native MuJoCo3.2.7로90초 비교했을 때,0은
서서히 미끄러져 무너졌고3은 초기2초 이후 payload XY 이동 약3.31mm였다.
이 분리 검사는 로봇 파지 성공 증거가 아니다.

`transport_trial.py --contact-noslip-iterations 3`은 새 출력 폴더에만 전역
마찰 후처리를 설정하며 `contact_solver_audit.json`과 manifest에 기록한다.
질량·마찰계수·접촉 형상·관절·모터·시간 간격은 보존하되, 발을 포함한 접촉
계산은 제조사 기본과 다르므로 전신 안정성을 별도로 검증해야 한다.
Native audit는 명시된 반복수만 허용하고 그 밖의 solver 옵션 변경은 거부한다.
설명: https://mujoco.readthedocs.io/en/3.2.7/modeling.html#preventing-slip

전역 noslip3 전신 시험은 준비 중 passive DIP 검증 범위 이탈로 종료됐다.
후속 `--pair-friction-time .001` 실험은 전역 noslip0과 발 접촉을 유지하고,
손–payload76쌍 및 사용 중인 받침/적재 박스–바닥91쌍의 `solreffriction`만
설정한다. 분리된90초 받침대 시험에서 초기2초 이후 payload XY 이동은3.17mm였다.
마찰계수 및 법선 접촉 설정은 유지되며, 이 시간 상수는 실측 실리콘 물성값이
아닌 접선 접촉 계산 설정이다. 실제 native contact의 값도 기동 전 검사한다.

이후 긴 접근 시험에서는 오른발이 90초 동안 옆으로 약19cm 미끄러져 받침대와
충돌했다. `--refine-foot-friction`은 `--pair-friction-time`과 함께 명시한 새
실험에서만 두 발–바닥 접촉에도 같은 접선 시간 상수를 적용한다. 발의 우선순위로
결정되는 기존 법선 solref/solimp, 마찰계수1.0, condim3은 보존하며 native audit가
이 동일성을 검사한다. 이 옵션은 기존 발 접촉 계산과 다르고, 전신 실험 결과로
별도 평가해야 한다. 발 고정이나 외력 보조를 추가하지 않는다.

제자리 이송 경로는 허리 yaw를 따르는 박스 회전을 허용한다. 손끝보다 손바닥이
먼저 닿도록 좌/우 EE에 local-X7/−6.2도 기울기를 주고, 이후 허리 pitch offset을
−0.15rad로 지정한 경로가 학습 FiSTA 기반 HL-IK 전 구간 및 보간 검사를 통과했다.
이는 오프라인 경로 검사이며 native 파지·이송 성공은 아직 확인되지 않았다.

후속 native 하중 검사에서는 실제 손 자세가 목표보다 바깥으로 약1.5–1.7도
벌어져 손끝 접촉이 사라졌다. `--orientation-feedback`은 측정된 SDK 출력각의
FK와 native torso 자세로 EE local-X 오차를 구하고, 손바닥 힘2N 이상에서
초당0.3도·총4도 이내로 목표를 보정한다. `--grasp-travel`은 별도로 bounded EE
압착 보정 이동량을 지정하며, 목표 힘14N과 MCP/DIP·전류·토크 제한을 바꾸지 않는다.
이 보정은 HL-IK 목표에 적용되므로 기존 학습 prior·관절·연속구간 충돌 검사를 거친다.

손끝 접촉에는 받침대와 같은0.001초 설정이 적절하지 않았다. Native3.2.7에서
손바닥 위치를 규정하고 박스를 고정한 별도 손가락 진단은0.001초에서17–50N
충격과 DIP 진동을 보였다.0.005초의1.5mm/s 진입에서는 최대0.492N이었다.
이 분리 진단은 실제 파지 성공 근거가 아니다. `--hand-friction-time .005`는
손 접촉 시간 상수만 별도로 지정하며, native audit는 손과 받침/발 값을 구분해
검사한다. 이전 출력 장면과 제조사 원본은 수정하지 않는다.

현재 `transport_trial.py`는 새 실험마다 `contact_material.py`로 USD 정적/동적
마찰계수와 native 손–payload 76쌍을 모두 0.6으로 설정한다. 이는 실리콘–종이의
문헌 기반 근사이며 이 손에서 측정한 값은 아니다. Native 3.2.7의 실제 contact
76개를 생성해 계수를 검사한 뒤에만 실행한다. 이 설정에서 1 kg 파지 판정은
각 손바닥 12.2625 N 이상과 각 손끝 0.2 N 이상을 요구한다.

`transport_plan.py`와 `live_ik.py`는 FiSTA checkpoint를 읽은 `PipelineIK`를 사용한다.
오프라인 전체 경로와 보간 검사를 통과한 archive만 실행하며, 실시간 world EE 목표는
측정 torso 자세와 접촉 힘에 따라 보정하고 같은 HL-IK로 다시 푼다. 실행 결과의
`prior_sha256`, `prior_status`, 관절·충돌·속도·가속도 판정을 함께 확인한다.

`--narrow-source-support`는 기존 큰 박스 두 개와 작은 박스 두 개를 회전·재배치해
받침 높이 0.7408 m를 유지하면서 맨 위 받침 폭을 0.10 m로 만든다. 받침 asset의
형상과 질량은 그대로이며, 실제 사용한 body 목록을 `source_support_bodies`에 기록해
접촉 설정과 변위 분석에 사용한다. 대상 박스의 초기 자세·크기·질량은 유지한다.

`--pregrasp-waist-pitch-offset`은 접촉 전에 허리 pitch를 조정하고 들기 구간에서
기존 운반 pitch로 연결한다. 이때 양손의 world 목표는 보존한다. 오프라인
`--joint-margin`은 원래 관절 한계 안쪽의 여유를 더 확보하는 옵션이며, 보간 후에도
원래 individual/coupled 한계와 충돌 검사를 통과해야 한다. 이 옵션들의 구현이나
오프라인 통과를 native 파지·이송 성공으로 해석하지 않는다.

`--contact-impratio 100`은 새 실험에서 마찰/법선 constraint impedance 비율을
명시적으로 변경한다. 마찰계수, 질량, 형상, 관절·모터, 시간 간격을 바꾸지 않는다.
MuJoCo의 부드러운 접촉 계산에서 생기는 느린 미끄러짐을 줄이기 위한 수치 설정이며,
실측 재료 물성값이 아니다. 발에도 영향을 주므로 전신 안정성을 별도로 확인한다.
`friction_impedance_audit.json`, manifest 및 native `model.json`에 원래/적용 값을
기록한다. native audit는 선언한 값만 허용하고 다른 solver 변경은 계속 거부한다.

`payload_contacts.jsonl`에는 현재 native solver에서 읽은 world 힘, payload 질량중심
기준 토크, 접촉점과 상대 속도도 저장한다. `state.bin`만 복원해 `mj_forward`한 힘은
실행 중 actuator parameter/callback을 복원하지 않아 원 실행의 힘과 다를 수 있다.
실제 하중 분석에는 실행 중 기록을 사용한다. `test_contact_wrench.cpp`는 경사진
1 kg 하중에서 힘 방향·크기와 양쪽 geom 순서, 관측 시 상태 보존을 검사한다.

`--synchronized-grasp`는 첫 접촉한 손바닥이 반대 손보다 먼저 큰 힘을 가하지 않도록
3 N의 가벼운 접촉으로 기다린다. 양쪽 손바닥의 필터 힘이 각각 2 N 이상으로 0.2초
유지되면 공통 목표를 2 N/s로 14 N까지 올린다. 파지 확인은 기존 각 손바닥
12.2625 N/손끝0.2 N/0.3초 기준이며, 확인 후 접촉이 줄어도 남은 손의 목표를
3 N으로 낮추지 않는다. EE 속도·이동량·손가락·모터 제한은 그대로다.

SDK 명령 루프는 한 실험 동안 cyclic GC를 미뤄 누적 로그 수집으로 인한 긴 명령
공백을 방지하고 종료 시 원래 GC 상태를 복원한다. SDK/IK/손 입력 신선도 기준은
유지한다. 관련 회귀 검사는 `test_control_gc.py`, `test_grasp_feedback.py`다.

`--box-orientation-feedback`는 파지가 확인된 뒤 payload의 측정 방향을 사용해
양손 EE 목표를 계획된 박스 중심 주위로 함께 회전시킨다. 허리를 따르는 계획 yaw를
제외한 오차를 적분하며, 보정은0.3 deg/s 및10 deg 이내다. 양쪽 손바닥 접촉이
2 N 미만이면 적분을 멈추고, 놓기 단계에는 부드럽게 보정을 줄인다. 손 사이의
상대 pose와 간격은 보존하며, 물체 pose/속도/접촉 힘은 직접 수정하지 않는다.
기존 손목 local-X 보정도 이 공통 보정을 포함한 목표를 추종한다.
`test_box_orientation_feedback.py`는 기울기 오차 억제, 의도한 yaw 보존,
회전 속도/크기 제한과 양손 상대 pose 보존을 확인한다. 동역학 성공 검사는 별도다.

선택적 `--collision-braking`의 먼 접촉 탐색은 현재 whole-tick 관절 속도 상한과
감속 한계로 body pair마다 줄어든다. 각 접촉 Jacobian의 절댓값 상한을 B_j,
속도 상한을 s_j라고 할 때 `margin + Σ B_j(s_j²/a + 3 dt s_j)`보다 먼 행은
현재 QP의 명령 범위에서 제약으로 작용할 수 없다. 원래0.15 m 탐색 범위 이내에서만
이 상한을 적용하며 기본0.02 m 범위는 유지한다. 선택된 최근접 구의 순서와 거리,
미분값 및 마지막 전체 비선형 충돌/corridor 검사는 유지한다.

`--native-collision-contacts`는 동일한 구 접촉 후보 선택을 C++로 계산한다.
형상·거리 임계값·pair별 최근접 개수는 동일하며, 거리가 정확히 같은 후보는
기존 NumPy 정렬로 되돌아가 순서를 보존한다. 생성 라이브러리는 소스 해시별로
`.local/cache/hlik-native-contacts/`에 저장하며 g++가 필요하다. 활성화 여부와
소스 해시는 `live_ik_ready.json`, tie fallback 횟수는 종료 결과에 기록한다.
실제 기록 25,168개 접촉 비교에서 구 쌍·거리·순서·미분값이 일치했다.
이는 계산 동등성 검사이며 파지·이송 성공의 증거는 아니다.

`--simulation-rate-limit`은 초기 준비 후 `pacing.request`부터 native step 사이에
벽시계 대기만 추가하는 진단 옵션이다. 같은 actuator 입력의 native 물리 상태는
보존하지만, 벤더 정책 추론은 벽시계 50 Hz인 반면 하위 제어는 시뮬레이션 시간에
동기화되어 있다. 따라서 이 옵션은 정책의 시뮬레이션 시간당 추론 횟수를 바꾼다.
기존 환경과 동일 조건의 이송 성공 검증에는 사용하지 않는다. Trial46은 이 차이를
확인하고 진단용으로 종료했으며 Trial47부터 이 옵션을 사용하지 않는다.

`transport_plan.py --lift-waist-pitch`는 들기 완료 시 허리 pitch와 회전 후 pitch를
나눈다. 들기에는 이 값을 사용하고, 회전하면서 `--waist-pitch-offset`으로 연결한 뒤
놓기에는 기존 `--place-waist-pitch`로 연결한다. 양손의 world 목표와 의도한 박스 yaw는
유지한다. 미지정하면 기존 경로와 같으며, 모든 새 경로는 learned IK와 보간 후
관절 한계·충돌·FK 검사를 다시 통과해야 한다.
