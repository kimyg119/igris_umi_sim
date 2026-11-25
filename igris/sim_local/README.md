# IGRIS-C 로컬 시뮬레이션 + GMR 텔레옵

NUC에서 돌던 `igris_c_controller` 시뮬 스택과 `igris_c_gmt_public` 정책을 이 PC에서 실행한다.
검증 완료: **텔레옵(ROS2) 모드로 로봇이 정책 제어 하에 z≈0.89 m 기립 유지.**

## 왜 Docker인가

| 문제 | 내용 |
|---|---|
| glibc | NUC 바이너리는 `GLIBC_2.38` / `GLIBCXX_3.4.32` 요구. 이 PC는 glibc 2.35 (Jammy). 애플리케이션 레벨 우회 불가 |
| ROS | 바이너리는 apt ROS Jazzy(`/opt/ros/jazzy`) 기준. 이 PC엔 RoboStack(conda) ROS만 있고 ABI가 다름 |
| 부수 효과 | 컨테이너는 `/dev/shm`이 격리되어 실기 스택과 SHM 이름 충돌이 없다 |

이미지에서 따로 챙긴 것: **RBDL 3.3.1**(apt에 없음, `--recursive` + Boost 소스 빌드 →
SONAME `librbdl.so.3.3.1` 일치 확인), **ONNX Runtime 1.20.1**(바이너리가 `VERS_1.20.1`
심볼 요구), mesa/NVIDIA GL, redis-server.

## 준비 (1회)

```bash
cd sim_local
docker build -t igris-sim:jazzy .
./up.sh

docker exec igris-sim bash -lc '
  source /opt/ros/jazzy/setup.bash && source /igris/ws/install/setup.bash
  cd /igris/sim_local/tools_ws && colcon build --packages-select igris_sim_ctl
  cd /igris/gmt_ws && colcon build --packages-select igris_c_gmt_public'
```

## 실행 (터미널 4개)

```bash
./start_sim.sh          # 1: 시뮬 스택
./start_policy.sh       # 2: GMT 정책 (기본 = 텔레옵 params_teleop.yaml)
./start_frame_pub.sh    # 3: 텔레옵 프레임 소스 (헤드셋 없는 검증용)
./bringup.sh            # 4: 기동 시퀀스
```

정상이면 터미널 2:

```
motion_mode=live  robot_state=ROBOT_STATE_LOW  robot_env=ROBOT_ENV_SIM
low_command_mode=true  action_hz=50  onnx_inference_ms=0.10
```

터미널 1: `[LOW_LEVEL_MODE] Control: 299.99 Hz`, `LowCmd: 50 Hz, MotorStatus: 300 Hz`.
베이스 높이 약 0.89 m.

녹화 리플레이로 되돌리려면 `./start_policy.sh /igris/sim_local/params_sim.yaml`
(이 경우 터미널 3 불필요, `motion_mode=onnx_replay`).

## 텔레옵 경로

```
PICO ─ xrobotoolkit_sdk ─→ GMR 리타게팅 ─→ Redis ─→ redis_to_ros2_bridge.py
                                                          │
                                          /gmr/teleop/retarget_frame (std_msgs/String)
                                                          ↓
                                       igris_c_gmt_public (motion_source.type: ros2)
```

`igris_c_gmt_public/GMR/`의 파일을 `retargeting/GMR/`에 배치해 두었다:

| 원본 | 배치 위치 |
|---|---|
| `xrobot_teleop_to_robot_w_hand.py`, `redis_to_ros2_bridge.py`, `retarget_teleop.sh`, `run_ros2_redis_bridge.sh` | `retargeting/GMR/scripts/` |
| `fps_monitor.py`, `motion_utils.py`, `params.py`, `twist2_redis.py` | `retargeting/GMR/general_motion_retargeting/utils/` |

### 소스 두 가지

**A. 진짜 GMR (PICO 필요)** — `./start_gmr_bridge.sh`
브릿지 자체는 검증했다(Redis 시딩·발행 동작 확인). 다만 **생산자 쪽은 PICO 헤드셋과
`xrobotoolkit_sdk`가 있어야 한다**(`general_motion_retargeting/xrobot_utils.py`가 임포트).
헤드셋 쪽 GMR은 호스트에서 python3.10으로 돌리고(`bash scripts/retarget_teleop.sh`,
`--redis_ip`를 이 PC로), 컨테이너 Redis는 `127.0.0.1:6379`로 공개돼 있다.

> **주의**: GMR 생산자가 없으면 브릿지는 기본 자세만 Redis에 시딩하는데, 그 프레임은
> `keybody_pos_local`과 `link_body_list`가 **비어 있다**. `obs.yaml`이 `motion_body_pos`
> (`kMotionBodyNames` 14개 바디)를 요구하므로 관측 구성이 실패하고 **`action_hz=0`**,
> 로봇이 무너진다. 실측 확인함. 정책을 돌리려면 실제 리타게팅 데이터가 있어야 한다.

**B. 헤드셋 없는 검증용** — `./start_frame_pub.sh` (`tools/gmr_frame_pub.py`)
녹화 모션 ONNX를 GMR 페이로드로 변환해 같은 토픽에 발행한다.
매핑은 `src/motion/frame_source.cpp`의 onnx_replay 경로를 그대로 따랐다:

```
joint_pos/joint_vel       -> dof_pos / dof_vel
body_pos_b                -> keybody_pos_local
body_names(metadata)      -> link_body_list
body_pos_w[base_link]     -> root_pos
body_quat_w[base_link]    -> root_rot (wxyz)
body_lin_vel_b[base_link] -> root_vel     (바디->월드 회전 후 발행)
body_ang_vel_b[base_link] -> root_angvel  (동일)
```

수신측이 `root_vel`/`root_angvel`을 월드 프레임으로 보고 바디 프레임 사본을 역산하므로
발행 전에 회전시킨다. 다른 모션도 인자로 줄 수 있다:
`./start_frame_pub.sh /igris/igris_c_gmt_public/data/onnx/recorded_walk_4.onnx`

## 수동 제어

```bash
./ctl.sh sim    resume|pause|reset|reload
./ctl.sh torque on|off
./ctl.sh mode   low|lowwalk|hold|stop|walk|home|preset:WALK_STANDBY
./down.sh
```

## 기동 순서가 왜 그런가

`bringup.sh`가 하는 일과 이유:

1. `sim resume` — MuJoCo는 **일시정지로 시작**한다. 안 하면 `MotorStatus: 0 Hz`,
   컨트롤러는 "Motor SHM stale"만 반복한다. 크래시가 아니다.
2. `torque on` — 토크가 꺼져 있으면 LOW 요청이 거부된다:
   `power: ON, emergency: OFF, torque: OFF. Switching to TORQUE_OFF_MODE for safety.`
3. `mode home` — 컨트롤러가 HIGH_LEVEL_MODE에서 HOME 관절 자세를 **붙잡게** 만든다.
4. `sim reset` **반복** — 여기가 핵심이다. `mujoco.fixed: "false"`(자유 부유)라
   아무도 안 잡아주면 로봇은 넘어져 있다. **넘어진 상태에서 HOME을 줘도 관절 보간이라
   못 일어난다.** reset은 스폰 자세로 되돌리는데, **컨트롤러가 이미 자세를 붙잡고 있을 때만**
   그대로 서 있는다. 그래서 3번 다음에 4번이고, 한 번에 안 되면 반복한다.
5. `mode low` — 정책 인계. `LOW_MODE=lowwalk ./bringup.sh` 로 WALK_LOW 인계(아래 `mode lowwalk` 참고).

`WALK_STANDBY` 프리셋도 있지만, 서 있는 상태에서 전환하면 무너졌다(z 0.887 → 0.046).
보행 정책용으로 보이며 현재 standstill 레퍼런스에는 필요 없다.

`mode lowwalk`(`LOW_LEVEL_WALKMODE_ON`, robotstate `ROBOT_STATE_WALK_LOW`, 0903 추가)는 컨트롤러 **내부 기립 정책**이
다리를 잡은 채 외부 `rt/lowcmd` 를 통과시킨다 — 서 있는 로봇에 주면 외부 명령 없이도 z 0.869 유지, 팔 lowcmd 는 그대로
따라간다(igris_ik 로봇 모니터의 정답 모드). `mode low` 는 순수 PD 라 정적 홀드만으로는 2 s 안에 주저앉는다.

## 밟았던 함정

1. **MuJoCo 일시정지 시작** (위 1번).
2. **토크 OFF면 LOW 거부** (위 2번).
3. **넘어진 채로는 HOME으로 못 일어남 → 자세를 잡은 채 reset** (위 3~4번).
4. **ONNX Runtime 스레드 폭주.** `onnx_runner.cpp`가 스레드 수를 지정하지 않아 코어 수만큼
   스레드를 띄우고 스핀 대기한다. 32코어에서 **21코어**를 먹고 MuJoCo/컨트롤러를 굶겨
   제어 루프가 **300 Hz → 1.8 Hz**로 붕괴한다. `start_policy.sh`가 `taskset`으로 코어 4개에
   묶어 회피한다. 근본 해결은 `onnx_runner.cpp`에
   `SetIntraOpNumThreads(1)` + `AddConfigEntry("session.intra_op.allow_spinning","0")`. 정확한 기전(굶는 건 컨트롤러가 아니라
   MuJoCo 물리 스레드, ORT 스레드는 코어 고정이라 taskset 으로 못 가둠)은 "300 Hz 가 떨어지는 진짜 이유" 절.
4b. **컨테이너가 만든 파일에 자물쇠 아이콘(root 소유).** `docker exec igris-sim ...` 는 기본이 root(uid 0)라,
   마운트된 `/igris`(=이 리포)에 파일을 만들면 호스트에서 root:root 가 되어 rcv(uid 1000)가 수정·삭제 못 한다(GNOME 자물쇠).
   컨테이너의 `ubuntu` 유저가 uid 1000=rcv 이므로 **파일을 만드는 컨테이너 명령은 `-u 1000:1000` 로 실행**하면 rcv 소유로 떨어진다:
   `docker exec -u 1000:1000 igris-sim bash -lc '... render_replay.py / state_recorder.py / colcon build ...'`
   (RT 승격 `chrt -f`·`taskset` 등 파일 안 만드는 명령은 root 유지 무방.) 이미 잠긴 파일은
   `docker exec igris-sim chown -R 1000:1000 <경로>` 로 되돌린다. redis 는 리포 최상위에 root 소유 `dump.rdb` 를 떨구므로
   `start_stack.sh` 가 `--save '' --dir /tmp`(RDB 저장 비활성)로 시작한다.
5. **NUC `cyclonedds.xml`이 NIC `eno1` 고정.** 이 PC엔 없다. `cyclonedds_local.xml`
   (autodetermine)과 `robot.yaml` 오버레이로 대체. NUC 원본 `configs/`는 건드리지 않는다.
6. **`sim reset` 직후 MuJoCo가 명령을 거부할 때가 있다**:
   `command time delay is too high! sim time: 0, command time: 1.77`.
   시뮬 시계가 0으로 돌아가며 생기는 과도 현상. `bringup.sh`가 reset을 반복해 넘어간다.

## 파일

| 파일 | 역할 |
|---|---|
| `Dockerfile` | Ubuntu 24.04 + ROS Jazzy + RBDL 3.3.1 + ONNX Runtime 1.20.1 + GL + redis |
| `up.sh` / `down.sh` / `sh.sh` | 컨테이너 기동 / 정리 / 셸 |
| `start_sim.sh` / `start_policy.sh` / `bringup.sh` / `ctl.sh` | 실행 |
| `start_gmr_bridge.sh` / `start_frame_pub.sh` | 텔레옵 소스 (진짜 GMR / 검증용) |
| `start_dds_bridge.sh` | DDS↔ZMQ 브리지 (컨테이너). 호스트의 `igris-robot-monitor` 가 붙을 JSON 엔드포인트 5560/5561 을 연다 — `igris/igris_ik/README.md` 의 "로봇 모니터" 절 참조 |
| `robot.yaml` | `configs/robot.yaml` 오버레이 (uri_file만 변경) |
| `cyclonedds_local.xml` | NIC 자동탐지 |
| `params_teleop.yaml` / `params_sim.yaml` | gmt_public params (ros2 텔레옵 / onnx 리플레이) |
| `tools/gmr_frame_pub.py` | 모션 ONNX → GMR 페이로드 발행 |
| `tools_ws/` | `igris_ctl` — sim/torque/mode DDS 명령 도구 |

## 주의: 리포 변경 1건

`config/params.yaml`이 가리키는 `models/wbc_teleop/student_130.onnx`가 커밋
`20ee814`("remove models")로 삭제돼 있었다. `action.yaml` 게인이 130 기준이라 복원했다:

```bash
git -C ../igris_c_gmt_public checkout 20ee814^ -- models/wbc_teleop/student_130.onnx
```

## RCV Hand URDF (2026-08-24)

현재 `sim_local/robot.yaml`은 **`igris_c_rcv_hand_ctrl.urdf`** 를 컨트롤러 모델로 쓴다.
전체 파이프라인(기립 → LOW → 정책 텔레옵) 6분+ 연속 안정 동작 검증 완료.

### URDF 변형 4종

| 파일 | 용도 | 상태 |
|---|---|---|
| `igris_c_rcv_hand.urdf` | CAD 원본 (143링크/39DOF) | RBDL 로드 불가 (관성 프레임 회전 8개) |
| `igris_c_rcv_hand_rbdl.urdf` | 관성 회전 흡수판 | RBDL 로드됨. 시각화/외부 도구용 |
| `igris_c_rcv_hand_fixed.urdf` | 손 관절 fixed + 0.6kg | RBDL 31 DOF. **컨트롤러 사용 불가** (아래) |
| `igris_c_rcv_hand_ctrl.urdf` | **컨트롤러용 (현재 사용)** | 33링크, 손 0.6 kg을 wrist_connector에 합성 |

### 실측으로 확인한 컨트롤러 제약 2가지

1. **링크 수/구조 민감성.** 컨트롤러가 검증된 구조는 xacro 생성본
   (`xacro igris_c_v2.urdf.xacro base_type:=pelvis parallel:=false end_effector:=none`,
   33링크 — `sim_local/ref_none.urdf`) 뿐이다. 링크를 **2개만 추가해도**(정적
   `igris_c_v2.urdf`의 hand_dummy 포함 35링크) HIGH_LEVEL HOME 유지가 무너져 기립
   불가. 143링크 모델은 로봇이 뒤로 1 m 튕겨나감. → 컨트롤러 URDF는 33링크 구조를
   유지하고 **관성값만** 수정해야 한다. (TOCABI 계열 고정 링크 배열 추정, 바이너리라 미확인)
2. **손 질량 임계.** wrist_connector 합성 질량에 따라 LOW 모드(LowCmd 수신 중)
   컨트롤러 스레드 성능이 갈린다:
   - 손 0.15 kg / 0.3 kg: control 300 Hz 정상 (실험용 `test_hand015/030.urdf` 는 2026-08-26 삭제 — 옛 장착 기준이라 유지하지 않음)
   - 손 0.6 kg: control 5~7 Hz로 저하 (사이클당 ~200 ms). **로봇 기립·정책 50 Hz는
     정상 유지** — LowCmd 전달은 update 스레드가 담당. 원인은 바이너리 내부
     (중력보상 OFF 시 정상 → 중력보상 경로의 질량 의존 로직). 임계는 0.3~0.6 사이.

### 실측 질량 반영 시

`igris_c_rcv_hand_ctrl.urdf`의 `l/r_wrist_connector` inertial만 수정하면 된다
(구조 변경 금지). 합성 공식: wrist 원본(0.032 kg) ⊕ 손 럼프(질량 m, CoM은
`l_hand` 조인트 변환으로 wrist 프레임 이동). inertial `rpy`는 반드시 `0 0 0`.
측정 질량이 0.3 kg을 넘으면 LOW 모드 control 스레드 저하를 감수하거나 로브로스에
문의할 것. 시뮬 검증 용도로는 저하 상태로도 문제없음이 확인됨.

### MJCF에도 RCV 핸드 반영 (2026-08-24 오후)

컨트롤러 RBDL만이 아니라 **MuJoCo 물리에도 손이 실제로 달리도록** 정적 MJCF를 생성했다.
이제 물리(1.2 kg 손 관성 포함)·컨트롤러(중력보상)·시각(핸드 메시) 세 층이 모두 정합한다.

| 파일 | 내용 |
|---|---|
| `sim_local/igris_c_v2_rcv_hand.xml` | **현재 사용 MJCF.** xacro 산출물(base_type=pelvis, parallel=true, end_effector=none)에 후처리 |
| `sim_local/mjcf_base_gen.xml` | 후처리 전 xacro 산출물 (참고용) |
| `sim_local/videos/home_gripper_aligned_closeup.png` | 렌더 증빙 (EGL 오프스크린; 2026-08-26 그리퍼 각도 장착 기준, 옛 `rcv_hand_mjcf*.png` 는 삭제) |

후처리 내용 (`sim_local/robot.yaml`의 `mujoco.robot_model_path`가 이 파일을 가리킴):

1. `meshdir` 절대경로화 (`/robros/configs/description/meshes/igris_c_v2/`) — 정적 파일이라 launch의 자동 절대화를 못 받으므로.
2. `l/r_wrist_connector` 아래에 **welded body `l/r_hand_rcv`** 추가 — URDF `l/r_hand` 조인트 변환(xyz `0 ±0.0109 -0.024`, rpy `π 0 0`) 그대로, inertial은 `igris_c_rcv_hand_fixed.urdf`의 hand_root 값(0.6 kg, bbox CoM/관성). DOF 추가 없음(qpos 117 / act 93 불변), 액추에이터 무변경.
3. 시각: `igris_c_rcv_hand_rbdl.urdf`의 fixed 체인을 FK로 펼쳐 **손당 visual geom 38개** (`class="visual"` — contype/conaffinity 0, 밀도 0 시각 전용. 유니크 메시 asset 34개, 절대경로). 충돌 geom은 넣지 않음(자기충돌 방지, 질량은 inertial로만).
4. `<visual><global offwidth/offheight>` — 오프스크린 렌더용 (무해).

검증 결과 (mujoco 3.2.7 파이썬으로 컴파일 확인 후 전체 파이프라인 재실행):

- 컴파일: nbody 67(+2), ngeom 155(+76), 총질량 59.82 kg(+1.2), 손 바디 parent = wrist_connector ✔
- 기립: home→reset #1에 z=0.885 — **물리에 1.2 kg 손이 달려도 기립 성공**
- LOW + 텔레옵 정책: z=0.889로 100초+ 유지, action_hz=50, motion_mode=live
- 컨트롤러 control 스레드: **300 Hz 유지** — 이전에 관찰된 0.6 kg 저하(5~7 Hz)가
  이 실행에서는 재현되지 않음. 저하는 비결정적일 수 있음 — 실기 전 재관찰 필요.
- 손목 처짐(kp=kd=0 limp): 손 CoM이 손목 바로 아래에 매달리는 평형이라 처짐이 작다
  — wrist motor output 기준 최대 ±0.18 rad, wrist_yaw ±0.05 rad. 현 상태로도 기립·
  트래킹에 지장 없음. 팔을 수평으로 드는 모션에서는 커질 수 있으니 필요 시
  action.yaml kp/kd를 31개 벡터로 바꿔 손목 홀드 게인 부여.

주의: python mujoco로 검증할 땐 **반드시 3.2.7** — 최신 버전은 이 MJCF의
`<position kv=... dampratio=...>` 조합을 거부한다(시뮬 바이너리는 3.2.7이라 무관).

## 손목 홀드 게인 실험 (2026-08-24)

AMASS 클립 재생에서 손(0.6 kg×2) 장착 시 낙상 → 손 유무 대조 → 손목 홀드로 해결까지:

| 구성 | 무술 클립 (ACCAD D9) | GRAB 사진찍기 |
|---|---|---|
| 손 있음, 손목 limp (kp=kd=0) | t=20.4s 낙상 | t=12.3s 낙상 |
| 손 없음 (원본 모델) | 22s 완주 | 22s 완주 |
| **손 있음 + 손목 홀드 (kp=30, kd=3)** | **22s 완주** | **22s 완주** |

결론: 낙상 원인은 무동력 손목에 매달린 0.6 kg 진자. 손목 홀드 게인만으로 억제됨
(손목 흔들림 ±0.18 → ~0.13 rad, 베이스 z 안정).

- 설정: `action_wrist_hold.yaml` — kp/kd를 31개(시스템 모터 순서)로 확장,
  손목 6모터에 kp=30/kd=3, 목은 0 유지(변인 통제). 23개 정책 관절 값은 불변.
  `params_teleop_wristhold.yaml` 이 이 파일을 참조.
- 홀드 방식: ActionBuilder가 비정책 관절에 "현재 측정각"을 목표로 보내므로,
  손목은 고정 자세 홀드가 아니라 tick마다 앵커가 따라오는 점성 홀드로 동작.
- 클립 재생: `tools/gmr_pkl_pub.py --pkl <retargeted pkl>` (13,409개 사용 가능).
  루프 랩 순간의 레퍼런스 점프가 낙상 트리거가 될 수 있음 — 평가 시 주의.
- 주의: 조건당 1회 실측. 통계적 확정에는 반복 필요.

## MuJoCo 뷰어 창 관리 (2026-08-24)

`igris_c_mujoco`는 GLFW 필수(헤드리스 모드 없음)라 DISPLAY가 있으면 항상 뷰어 창을
연다. 창을 닫으면 launch `respawn=True`가 2초 뒤 되살리므로 "꺼도 다시 뜨는" 현상이
생긴다. 해법: `up.sh` 기본값이 **가상 디스플레이(Xvfb :99)** — 창은 가상 화면에 뜨고
실제 화면엔 안 보인다. 라이브 뷰어를 보고 싶으면:

```bash
IGRIS_SIM_GUI=1 ./up.sh     # 실제 DISPLAY로 — 창이 뜨고, 닫아도 respawn됨(정상)
```

## 텔레옵 녹화 pkl 재생 → 영상 (2026-08-24 밤)

`box_pickup.pkl`(PICO 텔레옵 녹화, `xrobot_teleop_to_robot_w_hand.py --save_pkl_*` 청크)을
GMT 정책 + 컨트롤러로 재생해 `videos/box_pickup.mp4` 를 만들었다. **AMASS 리타겟 pkl과
스키마가 다르다** — `tools/gmr_pkl_pub.py` 가 `record_meta.source` 로 자동 감지한다.

| 항목 | AMASS 리타겟 pkl | 텔레옵 녹화 청크 |
|---|---|---|
| `root_rot` | xyzw | **wxyz** (qpos[3:7]) |
| `dof_pos` | 29 (목 없음 → 0 패딩) | **31** |
| 로컬 바디 | `local_body_pos` (34, base_link 포함) | `keybody_pos_local` (33, base_link 없음 — 수신기가 [0,0,0] 삽입) |
| fps | `fps` 키 그대로 | **`fps`=30 은 `--save_pkl_fps` 라벨일 뿐.** 프레임은 매 스텝 저장 → 실제 92.5 Hz. `record_start/end_unix` 로 복원 |
| 저장된 속도 | — | dt=1/30 기준이라 3배 작음 → 버리고 실제 dt 로 재계산 |

퍼블리셔 추가 옵션: `--start_file F` (F 가 생길 때까지 첫 프레임을 속도 0으로 홀드 →
기동 시퀀스 뒤 원하는 순간에 시작), `--no-loop` (끝에서 마지막 프레임 유지, 랩 점프 방지).

```bash
# 컨테이너 안(-d 백그라운드, 로그 /tmp/*.log)에서 순서대로:
ros2 launch igris_c_controller simulation.launch.py config_file:=/robros/configs/robot.yaml
taskset -c <끝 4코어> ros2 run igris_c_gmt_public igris_c_gmt_public_node \
    --config /igris/sim_local/params_teleop_wristhold.yaml --namespace igris_c_001
python3 tools/gmr_pkl_pub.py --pkl box_pickup.pkl --start_file /tmp/play.trigger --no-loop
./bringup.sh                       # 호스트
python3 tools/state_recorder.py --out videos/box_pickup.npz --duration 22 --rate 30 &
sleep 3; touch /tmp/play.trigger   # 재생 시작
MUJOCO_GL=egl python3 tools/render_replay.py --npz videos/box_pickup.npz --out videos/box_pickup.mp4
```

결과 (손 장착 MJCF + 손목 홀드 게인): 기립 z=0.883 → 재생 중 **0.727(t≈8.4s, 상자
집으러 웅크림)** → 0.86 → 0.79(내려놓기) → 0.876, 낙상 없음. `action_hz=50`,
`motion_mode=live`, 컨트롤러 300 Hz 유지. 레퍼런스 대비 정책 23관절 RMS 0.133 rad
(다리 0.14~0.16, 허리 0.09; 무릎 ref 0→0.84 rad 를 0.12→0.90 으로 추종). 팔(손목 포함)
RMS 0.25~0.27 — 손목은 정책 밖(점성 홀드)이라 큼.

### 레퍼런스 vs 시뮬 비교 영상 — `videos/box_pickup_compare.mp4`

좌: pkl 직접 FK(레퍼런스), 우: MuJoCo 시뮬(GMT 정책+컨트롤러). 같은 viz 모델·카메라로 렌더해
프레임 1:1 hstack. 레퍼런스는 초기 yaw/xy 만 시뮬 스폰에 정렬(z 는 그대로 — 레퍼런스 골반
0.82~1.03 m 로 시뮬 0.73~0.88 보다 높다. 텔레옵 조작자가 무릎을 편 채 서 있어서).

```bash
# t0 = 퍼블리셔 '트리거 감지' 로그 시각 − 녹화 시작 로그 시각 (이번 실행 2.578 s)
python3 tools/pkl_to_ref_npz.py --pkl box_pickup.pkl --sim_npz videos/box_pickup.npz --t0 2.578 --out videos/box_pickup_ref.npz
MUJOCO_GL=egl python3 tools/render_replay.py --npz videos/box_pickup_ref.npz --out videos/box_pickup_ref.mp4   # 컨테이너
ffmpeg -i videos/box_pickup_ref.mp4 -i videos/box_pickup.mp4 -filter_complex "[0:v]drawtext=...[l];[1:v]drawtext=...[r];[l][r]hstack" videos/box_pickup_compare.mp4
```

육안 소견: 6개 시점(1/6/8.4/12/16/20 s) 모두 자세 일치. 최저점(8.4 s)에서 시뮬이 레퍼런스보다
약간 덜 내려감(손 높이 차). 손목은 정책 밖(점성 홀드)이라 자세 차이 있음.

## 정제 A/B — 레퍼런스만 고치면 스텝이 나오나? (2026-08-25)

`retargeting/tools/clean_motion.py`(프레임별 지면 정렬 + 지지발 잠금 루트 보정, 설계
`docs/superpowers/specs/2026-08-25-teleop-motion-to-training-npz-design.md`)로 `box_pickup.pkl`을 정제한 뒤
현 정책(`student_130`, 손목 홀드)으로 재생. 지표는 `tools/replay_metrics.py`(골반 기반 viz 모델 FK).

| 입력 | ref yaw 변화 | sim yaw 변화 | 발 들림(>2 cm) L/R | 최대 들림 L/R | base z 최소 |
|---|---|---|---|---|---|
| 원본 (08-24) | −89° | +2° | 0/0 | 1.4/1.9 cm | 0.726 |
| 기준선: 지면 정렬만 (`--no-skate-fix`) | −89° | 0° | 0/1 | 1.6/2.0 cm | 0.726 |
| **정제본: 지면 정렬 + 지지발 잠금** | **−41°** | −1° | 0/0 | 1.0/1.6 cm | 0.730 |

- 결론: **레퍼런스를 물리적으로 정합시켜도 현 정책은 스텝을 내지 않는다** (경로 C 부정). 상체·웅크림은 3건 모두 정상 추종, 낙상 없음.
- 정제본의 yaw 변화가 −89° → −41° 로 준 것은 버그가 아니다: 원본 회전의 절반 이상이 "양발이 바닥에서 미끄러지며 도는" 성분이라
  지지발을 잠그면 사라지고, 0.3 s 잰걸음 두 번(L 8.6–8.9 s, R 9.4–9.8 s)으로 실제 이동한 41° 만 남는다.
- 따라서 경로 A ③(혼합 재학습)의 데이터로는 정제본(`retargeting/npz/box_pickup_clean.npz`)을 쓰되, "회전 스텝"을 가르치려면
  스텝이 실제로 있는 클립(AMASS 회전/걷기)과의 혼합이 필수다.
- 재현: `retargeting/README.md` 의 파이프라인 → 위 "텔레옵 녹화 pkl 재생" 절차에서 `--pkl` 만 정제본으로. t0 는 퍼블리셔 '트리거 감지'
  로그 시각 − 녹화 시작 로그 시각.

## S0 플래너 브릿지 (2026-08-26)

얼린 G1 MotionBricks 플래너(`planner_sonic.onnx`)를 G1 공간에서 돌리고, 50 Hz 서빙 프레임만
Φ 계수표로 IGRIS-C 로 디코드해 Redis → `redis_to_ros2_bridge.py` → GMT 정책으로 흘린다.
브릿지 본체는 `tools/planner_bridge.py`, 호스트 실행 래퍼는 `start_planner.sh`.

### 실행 순서

```bash
# 컨테이너 (터미널 1~3)
./start_sim.sh            # 1: 시뮬 스택
./start_policy.sh         # 2: GMT 정책 (params_teleop.yaml, student_130)
./start_gmr_bridge.sh     # 3: Redis -> /gmr/teleop/retarget_frame

# 호스트 (터미널 4) — bringup 전에 먼저 띄운다. 플래너는 idle 로 즉시 발행 시작.
./start_planner.sh --duration 1200

# 컨테이너 (터미널 5)
./bringup.sh              # sim resume -> torque on -> mode home -> sim reset -> mode low
```

`mode low` 시점에 `keybody_pos_local` 이 있는 유효 프레임이 흐르고 있어야 한다(없으면
`action_hz=0`, 로봇 붕괴). 그래서 플래너가 `bringup.sh` 보다 **먼저** 떠 있어야 한다.

스크립트 시퀀스로 바꿀 때는 idle 플래너를 **먼저 새 플래너로 덮고 나서** 죽인다 — Redis 는
같은 키를 덮어쓰므로 잠깐 두 프로세스가 겹쳐도 무해하고, 반대로 100 ms 이상 비면
(`frame_timeout_ms`) 정책이 프레임을 잃는다. ONNX 로드에 ~40 s 걸린다.

```bash
./start_planner.sh --script tools/scripts/s0_walk.json --duration 100   # 게이트 4
./start_planner.sh --keyboard                                            # 게이트 5
docker exec igris-sim bash -lc 'source /opt/ros/jazzy/setup.bash && \
  python3 /igris/sim_local/tools/state_recorder.py --out /igris/sim_local/logs/s0_walk.npz --duration 108 --rate 50'
python3 tools/s0_metrics.py --npz logs/s0_walk.npz --t0 <오프셋> \
  --seg 0:10:idle 10:40:slowWalk@0.3 40:70:walk@0.5 70:85:turn 85:100:idle2
```

`--t0` = (플래너 종료 unix 시각 − `--duration`) − 녹화 시작 unix 시각. 플래너 루프는 벽시계에
고정돼 있어 종료 시각에서 역산하는 편이 로그 문자열보다 정확하다.

### 플래너 경로·지연

`GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` (774 MB, 11 입력 V2).
`start_planner.sh` 가 `env_isaaclab_2.3.2` 의 CUDA-12 pip 라이브러리 + `gmr` 의 cudnn 9 를
`LD_LIBRARY_PATH` 로 빌려준다. 로드되면 `[planner] providers=['CUDAExecutionProvider', ...]` 가
찍히고 **11~12 ms/plan**, 실패하면 조용히 CPU 로 떨어져 **57~77 ms/plan**. 재계획은 10 Hz
(모드별 0.1~1.0 s 간격), 서빙은 50 Hz.

### 계수표 요약 (`retargeting/phi/phi_igris.json`, format `igris-phi/2`)

| 항목 | 값 |
|---|---|
| 루트 xy 스케일 `s_xy` | 0.8709 |
| 루트 z | `z_igris = 0.8313·z_g1 + 0.2629` |
| 의도 속도 스케일 `v_scale` | 1.1482 |
| 허리·어깨 | closed-form + post-affine (어깨 RMS 0.093 / 0.110 rad) |
| 팔꿈치 | L `a=0.988 b=−1.550` (r²=0.996), R `a=0.970 b=−1.535` (r²=0.993) |
| 손목·목 | 고정 (keyframe default) |

### 게이트 결과

**게이트 0 — 보행 전제 (정책이 보행 레퍼런스를 따라갈 수 있나): FAIL.**
`ACCAD/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii` (237 프레임 / 7.9 s, 레퍼런스
루트 xy 4.95 m 이동)을 `gmr_pkl_pub.py` 로 재생. 낙상은 없지만 **걷지 않는다**:

| 모델 규약 | 시뮬 xy 이동 | 레퍼런스 대비 | yaw (sim/ref) | 발 들림(>2 cm) L/R, 최대 | base z 최소 |
|---|---|---|---|---|---|
| torso (`retargeting/retargeted/`) | 0.36 m | 7 % | −5° / +35° (−15 %) | 1/5, 2.1/2.7 cm | 0.853 |
| pelvis (`retargeting/phi/igris/`) | 0.06 m | 1 % | +1° / +53° (2 %) | 6/0, 2.9/0.9 cm | 0.884 |

발 들림은 2~3 cm 잰걸음 수준으로 실제 스텝이 아니다. 이 리포의 08-25 정제 A/B 결론
("레퍼런스를 물리적으로 정합시켜도 현 정책은 스텝을 내지 않는다")과 같은 결과다.

**게이트 3 — 플래너 idle 폐루프 기립: PASS.**
`bringup.sh` 인계 후 z=0.892, 스크립트 idle 구간(0~10 s) 동안 z 0.885~0.890 유지, 실현 속도
0.001 m/s. 정책 `motion_mode=live`, `action_hz=50`, `onnx_inference_ms` 0.2~0.4,
컨트롤러 `Control: 299.99 Hz` / `MotorStatus: 300 Hz`.

**게이트 4 — 스크립트 폐루프 100 s: FAIL (낙상).**
`tools/scripts/s0_walk.json` (idle 10 s → slowWalk 0.3 → walk 0.5 → 제자리 회전 → idle):

| 구간 (스크립트 t) | 명령 | 실현 | 판정 |
|---|---|---|---|
| 0~10 s idle | — | 0.001 m/s, z 0.885~0.890 | OK |
| 10~40 s slowWalk | 0.3 m/s | 낙상 전 10~28 s 구간 0.027 m/s (9 %) | **t=30.7 s 낙상** |
| 40~100 s | — | z 0.046 (넘어진 채) | — |

z 곡선: 0.885 유지 → t=25.5 s 부터 0.86 아래로 서서히 침하 → t=30.6 s 0.80 → t=32 s 0.23 (전방
붕괴, x 가 0.48 → 1.26 m 로 튐). 즉 **모드 전환 순간이 아니라 slowWalk 를 20 s 추종하다가 무너진다.**

**게이트 4 변형 — slowWalk 만 60 s (`tools/scripts/s0_slowwalk.json`): 동일하게 FAIL.**
t=24.3 s (slowWalk 시작 14.3 s 뒤) 낙상, 낙상 전 실현 0.03 m/s 수준. 두 번 모두 최저속(0.3 m/s)
구간에서 무너지므로 **속도 상한이나 플래너 전환이 아니라 정책의 보행 추종 자체가 원인**이다
(게이트 0 과 일치). 브릿지 오프라인 `--audit` 은 G-K 전 항목 통과(한계 위반 0, 발 최저 z −2.2 mm,
지지발 부유 +3.6 mm, 이음새 0.11 rad, 실현 0.243@0.3 / 0.442@0.5 m/s)이므로 계수표·플래너가
아니라 **트래커(student_130) 쪽 문제**로 분리된다.

**게이트 5 — 키보드: 스모크만 PASS, 실주행은 사용자 몫 (Ruling R11).**
`script -q -c "./start_planner.sh --keyboard --duration 70" /dev/null` 로 의사 터미널에서 기동 →
HELP 출력 → 키 입력 반영 확인(`w` move=[1,0], `a` face 15° 회전, `r` 정지) → `o` 로 종료,
exit 0, termios 복구 정상.

### 키보드 운전 (사용자용)

터미널 1~3 + `bringup.sh` 까지 끝난 상태(플래너 idle 이 이미 돌고 있어야 `mode low` 가 산다)에서
idle 플래너를 새 키보드 플래너로 덮은 뒤 죽인다:

```bash
cd sim_local
./start_planner.sh --keyboard          # 진짜 TTY 가 필요하다 (raw 모드)
```

| 키 | 동작 |
|---|---|
| `1` `2` `3` `4` | 현재 모션 세트의 1~4 번 모드. **기본 세트는 `1`=idle, `2`=slowWalk, `3`=walk, `4`=run** |
| `N` / `P` | 모션 세트 이동 (0: idle/보행, 1: 점프류, 2: 기타, 3: 17~26) |
| `W` / `S` | 헤딩 방향 전진 / 후진 |
| `,` / `.` | 좌 / 우 스트레이프 |
| `A` / `D` | 헤딩 ±15° (이동 방향도 같이 회전) |
| `Q` / `E` | 제자리 회전 ±30° (헤딩만) |
| `9` / `0` | 속도 ∓0.1 m/s (0.1~3.0) |
| `-` / `=` | 골반 높이 ∓0.05 m (0.2~1.0) |
| `R` | 정지 (move=0, 속도 기본값) |
| `O` | 종료 |

녹화하려면 컨테이너에서 `state_recorder.py --out ... --duration 300` 을 같이 돌린다.
**주의**: 현 정책으로는 `2` 이상(보행 모드)에서 20 s 안팎에 넘어진다(게이트 4). `1`(idle) +
`-`/`=` 높이 조절 + `Q`/`E` 헤딩 정도가 현재 안전하게 확인 가능한 범위다.

### 밟았던 함정 (S0)

7. **`record_meta.unix_ms = 0`.** `igris_codec.PayloadBuilder.frame()` 이 페이로드에 시각을 0 으로
   박아 두는데, `redis_to_ros2_bridge.py` 는 `|t_action − record_meta.unix_ms| > stale_warn_ms(150)`
   이면 프레임을 **버리고 `keybody_pos_local` 이 빈 기본 프레임으로 대체**한다 → 정책
   `action_hz=0`, 로봇 붕괴. 브릿지 로그의 `local_kb_pos: []` 가 신호다. `RedisSink.push` 에서
   발행 시각을 도장 찍어 해결(`planner_bridge.py`).
8. **`pkill -f planner_bridge.py`가 자기 셸을 죽인다** — 명령줄에 그 문자열이 들어 있으면
   pkill 이 자신을 매칭한다. PID 로 죽일 것.

### 알려진 한계

- **보행 불가.** 위 게이트 0/4. 재학습(경로 A ③) 전까지 폐루프 이동은 나오지 않는다.
- **제자리 회전 미끄러짐.** 게이트 4 의 회전 구간은 낙상 뒤라 미측정. 오프라인 감사에서도
  회전은 지지발 고정 없이 도는 성분이 남는다.
- **고속 모드 밖.** 계수표는 slowWalk/walk 대역(0.3~0.5 m/s)에서만 검증했다. `run`(모드 3)과
  세트 1~3 (점프 등)은 미검증.
- **손목·목 고정.** Φ 는 손목·목을 keyframe default 로 둔다. 손목은 정책 밖(점성 홀드).
- **레퍼런스 골반 높이가 10 cm 높다.** 플래너 idle 의 디코드 골반 z = 0.990 m 인데 시뮬 기립은
  0.885~0.892 m. 기립만으로는 견디지만 `motion_root_state` 관측에 상시 바이어스로 들어간다 —
  침하·낙상의 보조 원인 후보. 계수표 재교정 시 확인할 것.
- **키보드 실주행 미완.** 5 분 운전은 사용자 인계 (Ruling R11).

### 2026-08-26 재검증 — 게이트 0/4 실패 원인은 트래커가 아니었다

위 게이트 0/4 결과(S0 첫 폐루프)는 두 가지 환경 결함 아래서 측정된 것이라 **무효**다.

1. **정책 노드 ONNX Runtime 스레드 폭주**(함정 4). `taskset` 만으로는 부족했고(4코어에서도 LowCmd 30~40 Hz), 학습·렌더가 같이 돌면
   MuJoCo/컨트롤러가 굶어 `MotorStatus 10 Hz / Control 5~10 Hz` 로 물리가 5배 느려졌다(`ros_sync` "Command time delay is high" 경고 연발).
   50 Hz 벽시계로 흘러가는 레퍼런스를 5배 느린 로봇이 쫓다 넘어진 것. **근본 수정**: `igris_c_gmt_public/src/modules/onnx_runner.cpp`
   에 `SetIntraOpNumThreads(1)`, `SetInterOpNumThreads(1)`, `session.intra_op.allow_spinning=0` 추가 후 재빌드 → 정책 CPU 1300 % → 9 %,
   `action_hz=50`, `onnx_inference_ms≈1`, 물리 300 Hz 유지. `start_stack.sh` 는 시뮬을 `SIM_CORES`(기본 16-27)에 고정한다.
2. **손목 홀드 미적용**. 스택이 `params_teleop.yaml`(→ `action.yaml`, 손목 kp 0)을 써서 8/24 실험의 "0.6 kg 손 진자" 낙상이 재현됐다.
   `start_stack.sh` 기본을 `params_teleop_wristhold.yaml`(`action_wrist_hold.yaml`, 손목 kp 30/kd 3)로 바꿈.

재실행(`s0_walk.json`, 학습 프로세스 동시 가동 중, `logs/s0_walk_v3.npz`, `videos/s0_walk_v3.mp4`):
LOW 인계 후 기립 → slowWalk/walk 구간에서 **20 s 에 7 m 전진(≈0.35 m/s)** → 회전 세그먼트에서 +y 로 5 m 이동 → idle 에서 70 s 이상 기립 유지.
낙상 0, 물리 300 Hz·정책 50 Hz 유지. (`s0_metrics.py` 의 "t=6.7 s 생존 FAIL" 은 녹화를 bringup 전에 시작해 reset 낙하가 잡힌 것.)
따라서 **S0 = 얼린 G1 플래너 + 계수표 + 기존 student_130 으로 IGRIS-C 가 걷는다.** 남은 관찰: 회전 세그먼트가 제자리 회전이 아니라 이동을 동반(플래너 `move=0`
+ `face` 변경 시 브릿지의 재계획 규칙 확인 필요), 실현 속도가 명령보다 낮음(0.35 vs 0.5).

### 2026-08-26 14:30 재검증 — 우리 학생(igris_c_wbt) 붕괴도 스택 결함이었다

위 두 수정(ORT 1스레드 + 물리 스레드 FIFO 50) 뒤 `run_replay.sh` 로 `igris_c_wbt` 학생들을 같은 ONNX·설정으로 재실행(06:08 런과의 차이는 그 둘뿐):
**DC 학생(`params_teleop_wbt_dc.yaml`)이 클립 15 s 전 구간을 서서 추종**(2회, z 최소 0.67/0.78, yaw 추종 93 %, 실제 스텝 발생) — 06:08 엔 인계 즉시 z 0.10.
학습게인/배포게인 학생은 여전히 붕괴, student_130 대조는 README 정제 A/B 표와 동일 재현. 06:08 런 로그도 Control 300 Hz 였으므로 평균 Hz 가 아니라
ORT 스핀 스레드가 정책 노드 타이밍에 준 영향으로 추정(미확정). 남은 낙상은 클립 종료 후 퍼블리셔 홀드 구간(Isaac 에서도 재현) — `igris_c_wbt/README.md` 가 정본.
주의: `replay_metrics.py` 의 "base z 최소" 는 2.8~17.8 s 창만 보므로 클립 종료 후 낙상은 `videos/<name>.npz` 의 `base[:,2]` 로 확인할 것.
관찰(미해결): FIFO 승격 후에도 dc_v3 렌더 단계·delay_v2 인계 직후에 Control 2~6 Hz 구간이 각 1회 있었다(다른 4런은 없음). 녹화 창 안이면 결과 무효 처리.

### 2026-08-26 300 Hz 가 떨어지는 진짜 이유 — 물리 스레드 락스텝 (실측)

"Control: 300 Hz" 는 자유 타이머가 아니다. MuJoCo 가 2100 Hz 물리 스텝 7개마다(=300 Hz) SHM SPMC 링(futex)으로 모터 상태를
발행하고, 컨트롤러 명령을 받아 10 스텝 지연 큐로 적용한다(`ws/install/igris_c_mujoco/include/igris_c_mujoco/ros_controller.h`:
`MUJOCO_DATA_DECIMATION 7`, `MUJOCO_COMMAND_DELAY 10`). 제어 루프는 이 상태 스트림에 **락스텝**으로 종속되어
`Control Hz = 물리 실시간 비율 × 300` 이다. 컨트롤러의 세 스레드(Update/Inference/Control)는 **SCHED_FIFO 68~70** 이라
CPU 경쟁에 면역이고, 스택에서 굶을 수 있는 유일한 스레드는 **MuJoCo 물리 스레드(SCHED_OTHER, E-코어에서 듀티 ~50 %)** 다.

실측(idle LOW, 정책 가동. 스핀 부하 = `taskset -c N yes`):

| 조건 | 물리 스레드 CPU | Control / MotorStatus |
|---|---|---|
| 기준 | 50 % | 300 / 300 Hz |
| 물리 스레드 코어에 스핀 4개 — 호스트(user.slice)에서 | 52 % | 300 Hz (cgroup 공정분배가 컨테이너 몫 50 % 보장 → 호스트 부하로는 재현 불가) |
| 같은 스핀 4개를 컨테이너 안에서 | 21 % | 105 / 110 Hz (비례) |
| 시뮬 코어 16-27 에 3개/코어 (컨테이너) | 25 % | 140~145 Hz |
| 32코어 전부 1개/코어 (컨테이너) | 44~50 % | 205~265 Hz |
| 패치 전 ORT 폭주(68 스레드, 2071 %) + idle | 54 % | 295 Hz |
| 위 폭주 + 렌더 2개 + 보행 | 51 % | 210~305 Hz |
| 정책 코어 28-31 만 3개/코어 | – | 300 Hz (LowCmd 40~50) |
| **물리 스레드 SCHED_FIFO 50** + 시뮬 코어 3개/코어 | 55 % | **300 Hz** |
| **물리 스레드 SCHED_FIFO 50** + 32코어 스핀 | 55 % | **300 Hz** |

기각한 가설: (1) 한 번의 긴 지연이 락스텝 위상을 깨뜨린다 — 컨트롤러/MuJoCo `SIGSTOP` 50~300 ms 후 즉시 300 Hz 복귀;
(2) 바닥 접촉으로 스텝 비용 폭증 — 넘어져 누운 상태(z 0.14)에서도 300 Hz, 스텝 0.12 ms(P-코어, ncon 8);
(3) GPU 포화 — 40 s 100 % 동안 300 Hz; (4) 모드 고유 속도 — JOINT_HOLD 도 300 Hz.

ORT 폭주의 정체(함정 4 보강): 패치 전 ORT 는 물리 코어 수(24)만큼 스레드를 만들고 **각 스레드를 코어에 명시적으로 고정**
(`sched_setaffinity`: 2,3 / 4,5 / … / 16 / … / 31)한다 → `taskset -c 28-31` 로 가둘 수 없고, `allow_spinning=1` 이라 100 % 스핀.
조용한 머신에선 물리 스레드가 공정분배 50 % 를 지켜 295 Hz 지만, 학습·렌더·(중복 기동된) 정책이 겹쳐 지분이 더 깎이면 위 표대로
비례 감속한다. 패치 후에도 45 스레드 중 코어별 고정 idle 스레드 23개가 남는데 CPU 0 % (출처 미확인, 무해).

**물리 갭 ≥ 0.5~1 s 는 별개 문제**: 컨트롤러 freshness 감시가 `Motor SHM stale → Runtime SHM degraded → forceStop(JOINT_HOLD)`
로 안전정지시키고 정책 명령을 더 받지 않는다(`low_command_mode=false`, `robot_state=STOP`). MuJoCo 를 1 s 얼리면 재현(0.5 s 는 안 걸림).
복구는 `./bringup.sh` 뿐(`mode low` 거부). 트립 직후 수 초간 `MotorStatus 0 Hz / Control 9.99 Hz`(100 ms 타임아웃 페이싱)가 보인다 —
패치 전 "9 Hz" 관찰은 이 감시 트립·낙상 국면과 겹쳤을 가능성이 크다. bringup 의 TORQUE_OFF/HOME 단계에서 보이는
`MotorStatus 10 Hz / Control 5 Hz / delay 0.0038` 도 명령이 없을 때 MuJoCo 가 100 ms 타임아웃으로 스텝하는 정상 동작(바이너리에
100 ms·200 ms 상수)이며 런타임 문제가 아니다.

처방(적용됨): `rt_physics.sh` — 1 s 동안 가장 바쁜 `igris_c_mujoco` 스레드(=물리)를 `chrt -f 50` 으로 승격. `bringup.sh` 가
`sim resume` 직후 자동 호출한다. FIFO 50 < 컨트롤러 68~70 이라 우선순위 역전 없음, 듀티 50 % 라 코어 독점 없음(RT throttling
95 % 가 안전망). ORT 패치·`SIM_CORES` 고정·손목 홀드는 그대로 유지. 참고: `SIM_CORES` 기본 16-27 은 E-코어(0-15 가 P-코어 HT 쌍)라
물리 스텝이 0.24 ms(P-코어 0.12 ms)지만, FIFO 승격 후엔 코어 종류가 Hz 를 좌우하지 않는다.

## S2 SONIC zero-shot 기립 게이트 (2026-08-27)

얼린 SONIC 트래커(G1 학습)를 재학습 없이 코덱으로 감싸 IGRIS-C 기립을 버티는지 측정하는 S2 첫 게이트(G1).
student_130 대신 `tools/sonic/sonic_tracker.py`(50 Hz, Φ_enc+SONIC enc/dec+코덱 Φ_dec → SDK lowcmd)를
유일 lowcmd 소스로 세운다. 정지(idle) G1 레퍼런스를 redis `sonic_g1_reference` 로 흘려준다.

### 실행 순서 (핸드오프)
```
# 컨테이너: sim + 브릿지 (student_130 은 곧 내려감)
./start_stack.sh
# 호스트: 정지 G1 레퍼런스를 redis sonic_g1_reference 로 발행
./start_planner.sh --reference --duration 300 &      # (idle=mode0, --script 불필요)
```
그다음 **핸드오프는 sonic-먼저-then-mode-low** 로 한다(브릿지 런북의 bringup→start_sonic 순서와 다름, 이유 아래):
```
# 1) HIGH 스탠바이 기립까지만 (mode low 는 아직): torque on; mode home; sim reset 반복(z>0.7)
# 2) ./start_sonic.sh          # student_130 내리고 sonic_tracker 1개 기동 (LowCmd 50 Hz 확인)
# 3) igris_ctl mode low        # sonic 이 이미 50 Hz 로 흐르는 상태에서 LOW 인계 → stale 갭 0
# 4) state_recorder.py --out logs/s2_standing.npz --duration 40 --rate 50
# 5) python3 tools/sonic/s2_metrics.py --rec logs/s2_standing.npz --sim-log logs/sim.log \
#        --sonic-log logs/sonic.log --out logs/g1_standing.json
```
왜 bringup(mode low) 을 그대로 못 쓰나: `--reference` 모드 플래너는 `sonic_g1_reference` 만 발행하고
student_130 이 읽는 `action_retarget_frame` 은 안 낸다. 그래서 bringup 의 `mode low` 시점에 student_130 은
모션 소스가 없어 즉시 붕괴(실측 z 0.88→0.05)한다. sonic 을 먼저 스트리밍시킨 뒤 mode low 하면 stale 갭 없이
바로 sonic 에 인계된다. 함정: sonic_tracker 를 중복 기동하면 LowCmd 100 Hz(두 퍼블리셔 충돌) 로 즉붕 —
`pkill -9 -f sonic_tracker.py` 로 1개만 유지. 실측 지표는 sim.log 의 `LowCmd: Hz`(권위값, sonic stdout 은 버퍼링됨).

### G2 측정 — 보행 zero-shot (2026-08-27 04:10, Φ_enc 닫힌형 역 `640c9ca` 적용 후)

플래너를 `--reference --script --start-file` 로 띄우면(`--start-file`: 파일이 생기는 순간이 스크립트 t=0, 그 전엔 idle
발행) 인계 순간에 맞춰 보행 레퍼런스를 시작할 수 있다. `SKIP_LOW=1 ./bringup.sh` 는 4단계(HIGH 스탠바이 기립)까지만
돌고 끝난다. 두 타이밍으로 측정(`tools/scripts/s2_walk_fwd{,_b}.json`, 녹화 `logs/s2_walk_fwd{,_b}.npz`,
영상 `videos/s2_zeroshot_walk_fwd.mp4`):

| 시도 | 레퍼런스 타이밍 | 결과 |
|---|---|---|
| A | idle 로 인계 → LOW +1.5 s 뒤 slowWalk 0.3 트리거 | **인계 +2.8 s 붕괴**(보행 명령 도달 직전, 기립 게이트 재현. z 0.884→0.12, 뒤-우측) |
| B | 인계 직전에 트리거(slowWalk 이 인계 시점부터 흐름) | **인계 +0.94 s 붕괴**, 첫 스텝 없이 뒤로 0.4 m 밀리며 낙상 |

LowCmd 50 Hz·물리 300 Hz 는 두 시도 모두 낙상 구간까지 유지(sim.log). 보행 레퍼런스는 기립을 돕지 않고 오히려
앞당긴다 — zero-shot 보행 갭은 기립 갭보다 크다. S3(디코더 LoRA) 착수 근거는 그대로.

### 결과: **G1 FAIL** — 자세는 유지, 균형은 못 버팀 (`logs/g1_standing.json`)
| 지표 | 값 | 게이트 |
|---|---|---|
| 생존시간 | **4.4 s** (30 s 필요) | ✗ |
| base z | 0.884(시작) → 0.054(min) → 0.06(끝) | ✗ 정지 자세 ~2.5 s 유지 → ~3 s 틸트 |
| action_hz | 50.0 (sim LowCmd) | ✓ |
| physics_hz | 300.0 | ✓ |
| tracking RMS(23관절) | 0.11 rad (기립 3.6 s 창) | — |

**낙상 서명**: 0–2.5 s 정지 유지(관절 이동 <0.01 rad, z=0.884, lean 2.9° → sonic 이 정확한 정지 자세를 붙잡음)
→ ~3 s 틸트 시작(IGRIS 동역학에서 기립 평형이 불안정) → 3–4.4 s 균형 캐치 실패(sonic 이 무릎 0→1.6 rad·발목
배굴을 크게 명령하지만 IGRIS 1.6× 질량엔 약·늦음) → ~4.5 s 붕괴. tracking RMS 0.11 은 이 실패한 캐치 국면
(무릎·발목·엉덩이)이 지배하며, 안정 구간(첫 2.5 s)의 자세오차는 <0.01 rad.

**ALIGN_ROT 정렬(spec §6) 시도·결론**: 정직-상단 기립에서 트래커가 sonic 에 주는 중력이 `[+0.080,0,-0.997]`
(4.6° 전방 피치 잔차, odom 은 base 거의 수직 2°) → 물리적으로 근거 있는 knob. 피치 스윕
`{-0.0805,-0.04,0,+0.04,+0.0805} rad` 를 단일-퍼블리셔로 측정: **전부 2–4 s 붕괴**, 게다가 같은 ALIGN_ROT 에서도
낙상 방향이 매번 다름(identity 가 한 번은 뒤로, 한 번은 앞으로) = **고정 프레임 편향이 아니라 한계 불안정**.
정렬 상수로 기립을 못 살림 → 근거 없는 비항등값을 넣지 않고 identity(spec 기본) 유지.

**진단**(교정됨, Ruling R-t8): 명령 경로는 정확하다 — 정지 자세를 <0.01 rad 로 붙잡으니 배선·매핑·코덱 타깃은
옳다. 그러나 갭은 순수 동역학 서사가 **아니다**. 갭 = **raw 동역학/PD 불일치 + 미배제 Φ_enc 피드백 오염
co-cause**이며 현 증거로 둘을 분리할 수 없다. Φ_enc 역아핀은 균형 임계 관절에서 R² 가 낮고(ankle-roll 0.36/0.52,
waist-roll/pitch 0.30/0.38) 이 관절들은 관측되며 동시에 구동되므로, 균형 이탈 구간에 로봇이 Φ_enc 적합이
가장 나쁜 영역으로 들어가 얼린 디코더가 오염된 proprio 를 먹는다 — 질량 갭과 같은 낙상 서명. G0(G1-native)는
Φ_enc 를 우회하므로 배제 근거가 아니다. **S3(LoRA) 확정 전** 판별 실험(이탈 구간 ankle-roll/waist Φ_enc 잔차
bound 또는 채널 ablate 후 재게이트) 필요. 상세·권고: `../docs/superpowers/specs/2026-08-26-s2-gap-report.md`.

## S2 재측정 — 4요소 이식 + 대조실험 (2026-08-31)

Isaac zero-shot(걷기 55% 완주, ref_v25)을 배포 심으로 가져오는 Track A. 결론 먼저:
**Isaac 성능은 배포 성능이 아니다.** 이상적 serial 플랜트의 55%가 이 플랜트(병렬 링키지+백래시)
에서는 기립 ~5 s로 붕괴하고, 같은 플랜트에서 student_130은 무한 직립한다(플랜트 무죄).
잔여 갭 = G1-학습 정책의 플랜트 모델 불일치(서서히 기움) — 학습 없이는 닫히지 않는다.

### 측정표 (인계 후 기립 생존, 게이트별)
| 게인 | 코덱 | 튜닝 | 생존 |
|---|---|---|---|
| 벤더(robros_lab) | matrix | 클램프+LPF | 2.74 s |
| 구형(wrist-hold) | matrix | 클램프+LPF | 3.44 s |
| 구형 | matrix/affine | off | 4.3~5.1 s (S2 4.4 패리티) |
| 구형 | matrix | **클램프만** | **5.20 s (최적)** |
| student_130 | — | — | 60 s+ 무한 |

### 새 러너 게이트 (start_sonic.sh / sonic_tracker.py)
- `SONIC_PARAMS_YAML=/igris/sim_local/params_sonic_vendor.yaml` — 벤더 게인(**이 플랜트에선 비권장**,
  위 표). 미설정 = 구형 action_wrist_hold(= student 실발행 게인과 동일, lowcmd 캡처로 확인).
- `SONIC_CODEC_MATRIX=1`(기본) — Isaac zero-shot과 동일한 29×29 행렬 코덱(`newcodec_linear.npz`).
  구코덱의 hip_yaw ±0.32 rad 데이터-적합 오프셋 제거. 왕복 항등 1e-16.
- `SONIC_DEPLOY_TUNE=1`(기본) + `SONIC_TUNE_LPF=0`(**권장**) — sonic-x2 dev-클램프(leg0.7/waist0.45/
  arm2.0)만. 다리 16 Hz LPF는 균형 보정을 지연시켜 −1 s.

### ⚠️ lowcmd 발행 경로 변경 (필수 지식)
sim의 네이티브 LowCmd 리더는 `TypeConsistency.DisallowTypeCoercion` — XTypes 타입객체를 보내는
현행 cyclonedds-python(0.10.5 소스빌드/11.x 휠 모두)의 자작 IdlStruct는 **매칭 자체가 거부**된다
(디스커버리는 되므로 조용히 0 Hz). S2 시절은 타입객체 없던 구바인딩이라 이름 매칭으로 통과했던 것.
현행 경로: `LowCmdPublisherROS`(lowcmd_pub.py) — student_130과 동일하게 `igris_c_sdk/msg/LowCmd`를
ROS 토픽 `igris_c_001/rt/lowcmd`로 발행, `igris_c_bridge`가 네이티브로 포워딩. 주의: ROS msg의
`frame_id`는 str만(`_FRAME_ID` char 배열 넣으면 publish에서 "bad argument type").
`LowCmd: N Hz` 카운터는 sim.log의 `[LOW_LEVEL_MODE]` 줄에서만 유효(HIGH에선 항상 0).

### 재현 도구
- 컨테이너 재생성 시 cyclonedds/scipy는 start_sonic.sh가 멱등 설치(이미지에 없음).
- 측정 1사이클 자동화: `/tmp/measure_once.sh <tag>` (reset→bringup→sonic→60 s 녹화→판정;
  tools/ 에 영구 보관..
- 플랜트: 직렬 관절 17개에 robros_lab 물리(armature/점성/마찰) 이식됨. 발목·waist p/r 링키지와
  백래시는 벤더 배포 모델 원값(student 직립으로 유효성 검증).

### 다음(P4 수정판): 배포-정렬 학습
학습은 배포에서 보낼 게인(구형)·지연(DelayedPD 0-2스텝)·obs 노이즈로 정렬한 Isaac 환경 +
KL-anchored PPO(표준 PPO는 ref_v26까지 전 조합 반증), **ckpt 선정은 이 심의 measure_once 생존시간**
으로 한다(sim2sim 게이트). 성공 기준: 기립 5.2 s → 30 s+.
