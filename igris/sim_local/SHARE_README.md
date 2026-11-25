# IGRIS-C sim_local — 공유용 번들

IGRIS-C SDK 바이너리 스택(컨트롤러 + MuJoCo 시뮬)과 GMT 정책 노드를 **Docker 컨테이너 안에서**
통째로 돌리는 환경이다. 호스트에는 ROS도 MuJoCo도 필요 없다. Docker만 있으면 된다.

검증된 상태: 텔레옵(ROS2) 모드로 로봇이 정책 제어 하에 **z ≈ 0.89 m 기립 유지**, control 300 Hz.

자세한 설계·함정·실험 기록은 **`sim_local/README.md`** 가 정본이다(이 파일은 다른 서버에서
처음 띄우는 데 필요한 것만).

## 전제

| 항목 | 요구 |
|---|---|
| OS | x86_64 Linux, Docker (컨테이너가 Ubuntu 24.04 / ROS Jazzy) |
| 디스크 | 압축 해제 ~450 MB + 이미지 ~4 GB |
| 네트워크 | `docker build` 1회에 필요 (RBDL 소스 빌드, ONNX Runtime 다운로드, pip) |
| GPU | **선택**. nvidia-container-toolkit 이 있으면 자동으로 `--gpus all`, 없으면 mesa 소프트웨어 렌더링 |
| CPU | 코어 16개 이상 권장 (시뮬 12코어 + 정책 4코어로 분리해 고정한다) |

## 5분 기동

```bash
unzip igris-sim-local.zip && cd igris-sim-local/sim_local

docker build -t igris-sim:jazzy .    # 10~20분 (RBDL 소스 빌드 포함), 1회만
./up.sh                              # 컨테이너 기동. 창을 보려면: IGRIS_SIM_GUI=1 ./up.sh
```

터미널 4개로 나눠 띄운다:

```bash
./start_sim.sh          # 1: 시뮬 스택 (MuJoCo + 컨트롤러)
./start_policy.sh /igris/sim_local/params_teleop_wristhold.yaml   # 2: GMT 정책
./start_frame_pub.sh    # 3: 모션 프레임 소스 (헤드셋 없는 검증용)
./bringup.sh            # 4: 기동 시퀀스 (resume → torque on → home → reset 반복 → LOW)
```

또는 1~3을 한 방에:

```bash
./start_stack.sh        # sim + 정책 + Redis 브릿지를 백그라운드로 (logs/*.log)
./start_frame_pub.sh    # 다른 터미널
./bringup.sh
```

**정상 판정** — 터미널 2:

```
motion_mode=live  robot_state=ROBOT_STATE_LOW  robot_env=ROBOT_ENV_SIM
low_command_mode=true  action_hz=50
```

터미널 1에 `[LOW_LEVEL_MODE] Control: 299.99 Hz`, `bringup.sh` 마지막 줄 `z=0.89` 근처.

`action_hz=0` 이면 모션 프레임이 없거나 무효다(터미널 3 확인).

## 수동 제어 / 정리

```bash
./ctl.sh sim    resume|pause|reset|reload
./ctl.sh torque on|off
./ctl.sh mode   low|hold|stop|home|preset:WALK_STANDBY
./stop_stack.sh     # 프로세스만 내림 (컨테이너 유지)
./down.sh           # 컨테이너 제거
```

## 들어있는 것

| 경로 | 내용 |
|---|---|
| `sim_local/` | Dockerfile, 실행 스크립트, params, MJCF, README(정본) |
| `sim_local/tools_ws/` | `igris_ctl` — sim/torque/mode DDS 명령 도구 (**빌드된 install 포함**) |
| `ws/install/` | IGRIS-C SDK 바이너리 워크스페이스 (controller, mujoco, bridge, can, shm …) |
| `gmt_ws/install/` | GMT 정책 노드 `igris_c_gmt_public_node` (**빌드 완료 — 바로 실행됨**) |
| `igris_c_gmt_public/` | 정책 소스 + `models/wbc_teleop/student_130.onnx` + 녹화 모션 ONNX |
| `configs/` | `/robros/configs` 로 마운트되는 로봇 설정, 게인, safe-set, 메시 |
| `igris_c_description_public/` | URDF + 메시 (컨트롤러 모델 `igris_c_rcv_hand_ctrl.urdf`) |
| `retargeting/GMR/` | Redis→ROS2 브릿지와 그 기본 자세 파라미터만 (GMR 본체 1.9 GB assets 제외) |

`ws/build`, `gmt_ws/build`, 로그·영상은 런타임에 안 쓰므로 뺐다. 정책을 고칠 거면
컨테이너 안에서 다시 빌드한다:

```bash
docker exec -u 1000:1000 igris-sim bash -lc '
  source /opt/ros/jazzy/setup.bash && source /igris/ws/install/setup.bash
  cd /igris/gmt_ws && colcon build --packages-select igris_c_gmt_public'
```

> `-u 1000:1000` 을 빼면 컨테이너 root 가 만든 파일이 호스트에서 잠긴다. 이미 잠겼으면
> `docker exec igris-sim chown -R 1000:1000 <경로>`.

## 안 들어있는 것

- **SONIC 트래커 경로** (`start_sonic.sh`, `tools/sonic/`): 스크립트는 들어있지만 아티팩트
  (`GR00T-WholeBodyControl/.../release/model_{encoder,decoder}.onnx` 87 MB, `retargeting/tools`
  코덱, `retargeting/phi/*.json`)가 빠져 그대로는 안 돈다.
- **S0 플래너 브릿지** (`start_planner.sh`): 호스트 conda env(`gmr`) + `planner_sonic.onnx` 전제.
- **진짜 GMR 텔레옵** (`start_gmr_bridge.sh`): PICO 헤드셋 + `xrobotoolkit_sdk` + GMR 본체 필요.
  헤드셋 없이 정책을 검증하려면 `start_frame_pub.sh` 를 쓴다.

## 처음 띄울 때 자주 걸리는 것

1. **`MotorStatus: 0 Hz`** — MuJoCo는 일시정지로 시작한다. `./ctl.sh sim resume`. (`bringup.sh` 1단계)
2. **`Switching to TORQUE_OFF_MODE`** — 토크가 꺼져 있으면 LOW 요청이 거부된다. (2단계)
3. **로봇이 넘어져 있음** — 모델이 자유 부유(`fixed: false`)라, HOME으로 붙잡은 **상태에서**
   `sim reset` 해야 선다. 넘어진 뒤엔 HOME을 줘도 못 일어난다. (3~4단계, `bringup.sh` 가 반복)
4. **Control Hz 가 300 → 10 Hz 로 떨어짐** — ONNX Runtime 스레드 폭주 또는 다른 부하가 MuJoCo
   물리 스레드를 굶긴 것. `start_*.sh` 가 `taskset` 으로 코어를 나누고, `bringup.sh` 가
   `rt_physics.sh` 로 물리 스레드를 SCHED_FIFO 50 으로 올린다. 같은 머신에서 학습을 돌리면
   학습 쪽을 `taskset -cp 0-15 <pid>` 로 묶어라.
5. **코어가 16개 미만** — `start_stack.sh` 의 `SIM_CORES=16-27` 기본값이 유효하지 않다.
   `SIM_CORES=0-7 ./start_stack.sh` 처럼 머신에 맞춰 준다.

나머지(URDF 변형, 손 질량 임계, DDS 설정, 300 Hz 락스텝 분석)는 `sim_local/README.md`.

## 다시 압축하기

```bash
./pack_share.sh [출력경로]     # 기본: <번들루트>/igris-sim-local.zip
```
