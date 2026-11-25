# 방 사진 기반 Isaac Sim USD

## 현재 장면: 모서리 접촉 + IGRIS C RCV hand

**`room_corner_igris.usd`**가 최신 요청을 적용한 장면이다. `room_corner_igris.usdz`는 텍스처를 포함한 단일 파일이다.
기존 `room_interactive.usd`와 아래 원본 에셋들은 비교 및 재생성을 위해 보존했다.

```bash
"$ISAAC_SIM_DIR/python.sh" run_isaac.py --gui --scene room_corner_igris.usd
```

- 기존 박스 38개와 각 지지체 사이의 상대 위치·회전을 행렬 그대로 보존했다.
- 평면에서 롤테이너와 팔레트의 앞면은 90° V 형태이며, 외곽 사각형의 모서리 하나만 접한다. 시각 모델의 둥근 모서리는 미세한 틈으로 보일 수 있다.
- 로봇 **발밑 원점은 (0, 0, 0)**, 전방은 +X, 왼쪽은 +Y, 위는 +Z다. 초기 원점에서 각 지지체의 로봇 쪽 앞면까지 바닥 평면상의 수선 거리는 **각각 0.20m**다. 로봇 표면에서의 여유 거리가 아니다.
- 접점은 **(0.2828427125, 0, 0)m**이다. 수선의 발은 양쪽 실제 앞면 선분 안에 있다.
- **이 장면의 롤테이너 차체·캐스터·바퀴는 모두 정적 충돌체로 고정했다.** 위의 박스들은 독립 동적 강체다. 원본 이동형 `silver_rolltainer.usd`는 변경하지 않았다.
- `igris_c_rcv_hand.usd`는 재사용 가능한 로봇 단독 USD다. 기존 로컬 `igris/igris_c_isaacsim_public/usd/igris_c_rcv_hand.usd`의 골반 기준 모델을 복사·통합했으며 원본은 수정하지 않았다. RCV hand를 포함한 143개 링크와 142개 관절(가동 관절 31개)을 유지했다.
- 로봇 베이스는 월드에 고정하지 않았다. 골반은 발바닥 기준 약 0.89m 높이다. 중립 자세 유지용 관절 드라이브만 적용했으며 보행·균형 정책은 연결하지 않았다. 물리 실행 후 자유 베이스가 움직일 수 있으므로 20cm는 저장된 초기 배치의 조건이다.

| 지지체 | 바닥 중심 위치 XYZ (m) | Z축 회전 |
|---|---|---:|
| 팔레트 | (0.2828427125, -0.7778174593, 0) | -135° |
| 롤테이너 | (0.2050609665, 0.5727564928, 0) | -45° |

`corner_validation.json`은 상대 배치 보존·20cm 수선·단일 모서리 접촉·로봇 자유 베이스·참조 누락 검증이다.
로봇 초기 전체 메시와 두 앞면 평면 사이에는 최소 약 4.76cm의 여유가 있다.
`previews/corner/simulation_validation.json`과 같은 폴더의 네 PNG는 Isaac Sim 6.0 / PhysX 240Hz로 5초 실행한 검증 결과다.
정확한 물리 변위는 해당 JSON에 기록했다. 이 검증은 5초 서기와 적층 유지에 한정하며 장시간 균형·조작 성공을 의미하지 않는다.

```bash
python compose_corner_layout.py --robot-usd /path/to/original/igris_c_rcv_hand.usd
python check_scene.py
"$ISAAC_SIM_DIR/python.sh" run_isaac.py --arrangement-only \
  --scene room_corner_igris.usd --out previews/corner
```

## 원래 사진 복원 장면

입력: IMG_2415.jpg, IMG_2416.jpg, IMG_2417.jpg, IMG_2418.jpg와 사용자가 제공한 네 가지 치수.
사진에 보이는 팔레트 / 롤테이너의 적재물을 복원했다. 벽에 따로 쌓인 박스, 삼각대, 텐트, 케이블과 장비 케이스는 제외했다.

## 열기

`room_interactive.usd`가 전체 장면이다. `textures/`와 다른 USD 파일을 같은 폴더에 유지한다.
ZIP을 옮길 때는 폴더 전체를 압축 해제한다. `room_interactive.usdz`는 전체 장면과 텍스처를 담은 단일 파일이다.

Isaac Sim의 **File → Open**으로 열고 카메라 **/World/Cameras/Overview**를 선택한다.
검증에 사용한 엔진은 **Isaac Sim 6.0 / PhysX**이다. 아래 실행 스크립트는 엔진과 화면 설정까지 적용한다.

```bash
# 이 폴더에서 실행. ISAAC_SIM_DIR는 자신의 설치 위치로 지정한다.
"$ISAAC_SIM_DIR/python.sh" run_isaac.py --gui
```

**Play → Shift + 마우스 왼쪽 드래그**로 박스를 잡아 움직인다.
Physics Settings의 Mouse Interaction Enabled / Mouse Grab은 켜져 있고,
Mouse Grab Ignore Invisible은 꺼져 있다. 충돌 전용 형상이 보이지 않도록 분리돼 있기 때문이다.
Stop은 초기 장면으로 되돌리는 용도이며, 초기 배열은 파일에 그대로 보존돼 있다.

## 요청한 네 종류의 개별 에셋

| 파일 | 내용 | 치수 cm / 원점 |
|---|---|---|
| `large_box.usd` | 큰 박스 1개, 자유 강체 | 34.5 × 23 × 17 / 박스 중심 |
| `small_box.usd` | 작은 박스 1개, 자유 강체 | 18 × 18 × 10 / 박스 중심 |
| `pallet_loaded.usd` | 초록 팔레트 + 큰 박스 13개 | 지지체 110 × 110 × 15 / 지지체 바닥 중심 |
| `rolltainer_loaded.usd` | 은색 롤테이너 + 큰 박스 18개 + 작은 박스 7개 | 지지체 92 × 70 × 170 / 지지체 바닥 중심 |

빈 지지체도 `green_pallet.usd`, `silver_rolltainer.usd`로 포함돼 있다.
지지체 치수는 적재물 포함 치수가 아니다. 모든 USD는 metresPerUnit=1, Z-up이다.
박스의 충돌 치수는 입력값과 같고, 시각 형상에는 0.3mm 미만 두께의 포장 테이프가 추가돼 있다.

## 배열과 근거

X는 사진 오른쪽, Y는 벽 방향, Z는 위쪽이다. 파일의 `layout.json`에 박스별 이름·좌표·회전·근거를 기록했다.

- **팔레트:** 정면에서 왼쪽부터 4 / 4 / 3 / 1단. 그 위에 큰 박스 1개를 90° 돌려 두 열에 걸쳐 배치했다. 총 13개. 뒤쪽 격자는 비워 뒀다.
- **롤테이너 큰 박스:** 정면 3열 × 깊이 2열 × 높이 3단, 총 18개. 정면의 9개와 사진에 보이는 상단 면을 기준으로 구성했다. **뒤쪽 9개의 완전한 배열은 가려져 있어 추정이며, 사용자 확인 전이다.**
- **롤테이너 작은 박스:** 가운데 앞 3단, 오른쪽 뒤 2단, 오른쪽 앞 1개, 왼쪽 뒤 1개. 총 7개. 사진에서 읽은 위치와 작은 회전 차이를 적용했다.
- 팔레트와 롤테이너 사이 간격, 벽까지 거리 및 바닥 범위는 사진 기반 추정이다. 방 전체의 실측 치수로 해석하면 안 된다.

사진 네 장을 참고해 치수·형상·재질을 재구성한 장면이며, 포토그래메트리 스캔은 아니다.
종이 섬유, 갈색 테이프, 아연도금, 초록 플라스틱, 검정 칩 타일은 절차적으로 만든 재질이다.
사진마다 존재하는 눌림·오염·테이프 주름을 전부 실측 복제한 것은 아니다.

## 상호작용과 물성

- 38개 박스 모두 독립적인 동적 강체이며, 박스끼리 연결하거나 고정하지 않았다.
- 팔레트도 동적 강체다. 지게차 삽입 공간과 격자 구멍을 형상으로 만들었다.
- 롤테이너는 차체 1개 + 캐스터 4개 + 바퀴 4개, 회전 관절 8개로 구성했다. 월드에 고정하지 않았다.
- 롤테이너 측면 / 뒤 프레임은 차체와 함께 움직인다. 문 개폐 관절이나 천 스트랩 변형은 구현 범위에 포함하지 않았다.
- 바퀴 내부 구조는 사진에 잘 보이지 않아 4개 회전 캐스터로 가정했다. 실제 제품의 고정 / 회전 바퀴 조합을 확정한 모델은 아니다.
- 박스는 강체 모델이다. 찌그러짐·찢어짐·뚜껑 열기는 구현하지 않았다.

| 물성 | 설정 | 근거 |
|---|---:|---|
| 큰 박스 질량 | 0.55 kg | 추정, 내용물 미확인 |
| 작은 박스 질량 | 0.16 kg | 추정, 내용물 미확인 |
| 팔레트 질량 | 12 kg | 추정 |
| 롤테이너 총 질량 | 37.2 kg | 차체 32 + 캐스터 4×0.6 + 바퀴 4×0.7, 추정 |
| 박스 정지 / 동마찰 | 0.58 / 0.44 | 추정 |
| 팔레트 정지 / 동마찰 | 0.56 / 0.42 | 추정 |
| 아연도금 정지 / 동마찰 | 0.42 / 0.336 | 추정 |
| 바퀴 정지 / 동마찰 | 0.8 / 0.64 | 추정 |
| 중력 / 물리 주기 | 9.81 m/s² / 240 Hz | 시뮬레이션 설정 |

질량은 `layout.json`에서 수정 후 재생성하거나 각 강체의 Physics Mass 속성에서 수정한다.
충돌 전용 박스 형상은 외관의 작은 접힘·테이프 돌출을 제외한 단순화다.
프레임·팔레트는 여러 기본 충돌 형상을 조합했으며, 공간 전체를 하나의 볼록 껍질로 막지 않았다.

## 검증과 재실행

`offline_validation.json`: 네 종류 치수 검사, 강체 / 관절 수, 박스 간 관통 검사, 외부 참조 누락 검사.
`previews/simulation_validation.json`: 실제 PhysX 적층 안정성, 박스 운동, 롤테이너 힘 응답, 이미지 통계.
`previews/interaction.mp4`: 초기 속도를 준 큰 / 작은 박스가 떨어지고, 18N 힘을 받은 롤테이너가 이동하는 기록.
로봇 파지 성공이나 실제 물체의 무게 / 마찰 검증 결과는 아니다.

```bash
# 재생성: pxr, numpy, Pillow가 있는 Python
python build_scene.py
python check_scene.py

# 설치된 Isaac Sim에서 검증 / 렌더 / 원본 프레임 저장
"$ISAAC_SIM_DIR/python.sh" run_isaac.py

# 30 fps 검증 영상 생성
ffmpeg -y -framerate 30 -i previews/frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 previews/interaction.mp4
```

물리 구성 기준: [NVIDIA Physics Simulation Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html),
[NVIDIA Colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html).
GUI 드래그는 설치된 `omni.physx.ui`의 Mouse Interaction 문서와 설정을 기준으로 했다.
