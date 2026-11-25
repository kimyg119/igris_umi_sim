# UR5e 책상 pick-and-place 셀 (Isaac Sim USD)

실제 UR5e + Robotiq 2F-85 + RealSense D455 셀을 옮긴 Isaac Sim 씬입니다. 이 폴더만 있으면 씬을 열고, 배치를 바꾸고, 에셋을 다시 만들 수 있습니다.

- 검증 환경: Isaac Sim 6.0.0-rc.59 (Kit 110). `OmniPBR.mdl`, `OmniGlass.mdl`은 Isaac Sim 내장 MDL을 씁니다.
- 단위와 좌표: 미터, Z-up. 원점은 책상 상판 중심 바로 아래 바닥이고, 상판 윗면이 z = 0.73, 벽은 +Y 쪽입니다.
- 모든 참조는 상대 경로입니다. 폴더 구조를 유지한 채 통째로 옮기면 됩니다.

## 여는 법

Isaac Sim에서 `ur5e_desk_scene.usda`를 엽니다. 로봇은 에피소드 0의 홈 자세로 놓여 있고, Play를 누르면 그 자세를 유지합니다.

## 씬 구조

| Prim | 에셋 파일 | 비고 |
|---|---|---|
| `/World/Room`, `/World/Desk`, `/World/RobotStand` | 씬 파일에 직접 작성 | 벽·바닥, 책상(1400 × 600 mm), 로봇 스탠드 |
| `/World/UR5e` | `assets/robots/UniversalRobots/ur5e/ur5e.usd` | `Gripper` variant = `Robotiq_2f_85` |
| `/World/UR5e/wrist_3_link/d455_mount` | `assets/d455_mount/d455_mount.usd` | 출력 브라켓 + D455 (`assets/sensors/RealSense/D455/rsd455.usd`) |
| `/World/Pallet` | `assets/pallet/pallet.usd` | 210 mm 격자 팔레트 |
| `/World/Blocks/block_XY_L` | `assets/blocks/block.usd` | 25 mm 큐브 36개. variant `color`(5색), `grain`(나뭇결 g0~g7) |
| `/World/Basket` | `assets/basket/basket.usd` | 바구니 |
| `/World/Lights`, `/World/Looks`, `/World/PhysicsScene` | 씬 파일에 직접 작성 | 조명, 재질, 물리(240 Hz, TGS) |

손목 카메라(640 × 480, 실측 intrinsic과 왜곡 적용) prim:
`/World/UR5e/wrist_3_link/d455_mount/D455/RSD455/Camera_OmniVision_OV9782_Color`

## 배치를 바꾸는 법

- **Isaac Sim에서 직접**: 팔레트, 바구니, 블록, 로봇은 각각 독립된 reference입니다. prim의 `translate`와 `rotateZ`를 바꾸면 됩니다.
- **스크립트로 재생성**: `scripts/build_scene.py` 위쪽의 상수(`PALLET_IN_BASE`, `BASKET_IN_BASE`, `PALLET_X`, `ROBOT_YAW`, `ROBOT_Q`, `STACKS` 등)를 고치고 다시 실행합니다. 이때 `ur5e_desk_scene.usda`를 덮어쓰므로, Isaac Sim에서 손으로 고친 내용은 사라집니다.

```bash
python scripts/build_scene.py
```

## 스크립트

| 스크립트 | 하는 일 | 필요한 것 |
|---|---|---|
| `build_scene.py` | 씬 USD 생성 | usd-core, numpy, Pillow |
| `build_props.py` | 팔레트, 브라켓+D455, 블록 에셋 생성 (`inputs/`의 STL·JSON·YCB 스캔 사용) | 위 + trimesh, opencv-python |
| `render_check.py [--physics]` | 렌더와 3초 물리 안정성 점검 | Isaac Sim의 `python.sh` |
| `check_gripper.py` | 그리퍼를 물체 없이 완전히 닫았다 여는 동작 점검 | Isaac Sim의 `python.sh` |
| `check_camera.py` | 손목 카메라 투영이 캘리브레이션과 맞는지 점검 | Isaac Sim의 `python.sh` |
| `replay_episode.py [N]` | 실제 에피소드를 재생해 파지와 카메라 프레임 비교 | Isaac Sim의 `python.sh`, ffmpeg |
| `fetch_episode.py [N]` | 데이터셋에서 다른 에피소드 받기 | pandas, pyarrow, ffmpeg |
| `build_textures.py`, `build_basket.py` | 책상·벽 텍스처와 바구니 재생성 | 원본 사진·영상(이 패키지에 없음) |

`assets/reference/episode_000/`은 `sonnyjin/ur5e_cube_auto_200_success_5826` 에피소드 0의 관절 기록과 손목 카메라 프레임입니다.

## 알아둘 가정

- 로봇 스탠드의 형상과 치수는 가정입니다. 로봇 베이스 위치(책상 앞 가장자리에서 0.30 m, 상판보다 60 mm 위)는 에피소드 0에서 복원했습니다.
- 그리퍼 파지 토크는 6 N·m(패드 기준 약 55 N)으로 낮춰 두었습니다. 실제 그리퍼의 힘 설정은 데이터에 없습니다.
- 카메라 extrinsic은 CAD 공칭값이며 실측 캘리브레이션이 아닙니다.
- 스택은 모두 4층이고, 아래층 블록 색은 임의로 정했습니다.
- 그리퍼 손가락은 손목 카메라 영상에 맞춰 검정으로 했습니다. D455 본체는 에셋 기본 재질 그대로입니다.

## 출처와 라이선스

- `assets/robots`, `assets/sensors`, `assets/Materials`: NVIDIA Isaac Sim 6.0 에셋 팩에서 가져온 파일입니다. 각 폴더의 `LICENSE`를 따릅니다.
- `inputs/ycb`: ai-habitat/ycb의 colored wood blocks 스캔입니다. `inputs/ycb/LICENSE.txt`를 따릅니다.
