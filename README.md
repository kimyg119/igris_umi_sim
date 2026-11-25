# igris_umi_sim

UMI 박스 적재 데이터를 IGRIS-C가 시뮬레이션에서 수행하게 하는 코드 묶음이다. 세 가지가 들어 있다.

- **시뮬레이션**: 제조사 WALK_LOW + SDK LowCmd 위에서 UMI 손 궤적을 재생한다(물리는 제조사 MuJoCo).
- **IK**: 손 포즈 목표를 허리 3축 + 양팔 14축 관절 명령으로 바꾸는 mink QP(관절·속도·결합 한계, 자기충돌).
- **ACT**: 로봇 카메라 3대와 관절 각을 보고 상대 EE 행동을 내는 정책의 학습과 폐루프 실행.

Isaac Sim은 설치되어 있다고 가정한다. 실기는 한 번도 움직이지 않았다: 모든 결과는 시뮬레이션이다.

```text
UMI 에피소드 ─ labels ─→ 로봇이 따라갈 수 있는 포즈 라벨(15 Hz)
  ─ 시뮬 재생 ─→ 관절 각·손 목표·몸체 포즈 기록
  ─ ee_dataset ─→ 프레임: 관절 → 링크 → EE(FK), 로봇 카메라 3대
  ─ act_ee ─→ 상대 EE 행동 묶음으로 ACT 학습
  ─ 폐루프: policy_server(카메라 + 관절 각 → 행동) ⇄ worker(IK → LowCmd) ⇄ 제조사 제어기 + 물리
```

자세한 설계, 측정값, 한계는 [igris/umi_walklow_replay/README.md](igris/umi_walklow_replay/README.md)에 있다
(작업 기록이라 `/DATA/ykkim/...` 같은 경로는 작업한 기기의 것이다).

## 폴더

| 폴더 | 내용 |
|---|---|
| `igris/umi_walklow_replay/` | 본체: 재생(`run.py`, `worker.py`), IK(`mink_ik.py`, `ik.py`), 전처리 라벨(`labels.py`), 카메라(`cameras.py`), 데이터셋(`ee_dataset.py`, `lerobot_export.py`), ACT(`act_ee.py`, `ee_features.py`), 폐루프(`policy_server.py`, `policy_link.py`), 서버용 실행기(`container_run.py`) |
| `igris/isaac_vendor_room/` | 본체가 호출하는 기동(bringup)·관측기·장면 배치·Isaac 렌더러 |
| `igris/hl_ik/` | 시뮬 시계 훅(`native_clock`)과 HL-IK 망 정의(팔꿈치 prior 선택 기능용) |
| `igris/sim_local/` | 제조사 로봇 모델(MJCF), 시뮬 이미지 `Dockerfile`, 제어 도구(`tools_ws`) |
| `igris/ws/install/` | **제조사 빌드 결과물**(제어기, MuJoCo 노드, SDK). 컨테이너에 `/igris/ws/install`로 물린다 |
| `igris/configs/`, `igris/igris_curobo/…/data/`, `igris/igris_c_description*/` | 제조사 설정·메시, 관절 한계, 충돌 구, 손목 카메라 링크가 있는 URDF |
| `assets/room_real2sim/`, `real2sim/ur5e_desk/` | 장면 USD 자산 |
| `third_party/lerobot/` | ACT에 쓴 LeRobot 0.6.2 소스(Apache-2.0; PyPI에는 0.6.1과 0.7.0만 있다) |
| `server/` | 호스트에서 docker와 rsync만 쓸 수 있는 서버용 스크립트 |
| `tools/` | 패키지 설치 스크립트, 결과 집계·분석 스크립트 |

들어 있지 않은 것: Isaac Sim, 데이터셋, 학습된 체크포인트, 실행 출력, 서버 주소.

## 준비

1. **Isaac Sim의 파이썬**(3.12, `pxr` 포함)으로 실행한다. 아래에서 `$PY`가 그것이다.
2. **시뮬 이미지** `igris-sim:jazzy`: `docker build -t igris-sim:jazzy igris/sim_local`.
   결과는 이미 있던 이미지로 얻었고, 이 `Dockerfile`로 다시 빌드해 보지는 않았다.
3. **파이썬 패키지**: `tools/install_packages.sh $PY`가 `vendor_py/`에 받는다(환경에는 설치하지 않는다).
4. **데이터셋**: Hugging Face `rcvsungmincho/Box_palletizing_UMI_macos_intern_merged`를 로컬 폴더로 받는다.
5. 장면 텍스처에 한글 글꼴이 필요하다(`/usr/share/fonts/truetype/nanum/` 또는 Noto Sans CJK).

```bash
export PY=/path/to/isaac-sim/python
export PYTHONPATH=$PWD:$PWD/third_party/lerobot/src:$PWD/vendor_py/train_host:$PWD/vendor_py/mink_host
export DATASET=/path/to/Box_palletizing_UMI_macos_intern_merged
```

## 실행 (Docker와 GPU가 있는 한 기기)

출력 폴더는 매번 새 이름이어야 한다. 시뮬 컨테이너는 네트워크 없이 뜬다.

```bash
# 1. 시연 1회: 전처리 라벨을 한 샘플씩 재생 (약 4분). 세 박스가 쌓였는지는 run/analysis.json
$PY -m igris.umi_walklow_replay.run all --out $OUT/demo_ep007 --dataset $DATASET --episode 7 \
    --stream --slack calibrated --no-render

# 2. 그 실행을 학습 프레임으로: 관절 → EE(FK), 카메라 3대 (dataset/ 폴더)
MUJOCO_GL=egl $PY -m igris.umi_walklow_replay.ee_dataset render $OUT/demo_ep007

# 3. ACT 학습: $OUT 아래의 성공한 실행을 모두 쓴다 (에피소드 7·13·21·29는 평가용으로 빠진다)
$PY -m igris.umi_walklow_replay.act_ee train --runs $OUT --out $CKPT

# 4. 폐루프: 정책이 카메라와 관절 각만 보고 로봇을 구동한다
$PY -m igris.umi_walklow_replay.run all --out $OUT/policy_ep007 --dataset $DATASET --episode 7 \
    --policy $CKPT/last --policy-seconds 105 --no-render

# 5. 여러 실행을 LeRobot v3.0 데이터셋 하나로
MUJOCO_GL=egl $PY -m igris.umi_walklow_replay.lerobot_export --runs $OUT --out $OUT/lerobot_dataset
```

- `--no-render`를 빼면 `all`이 Isaac Sim으로 3구도 영상까지 만든다(약 17분).
- 폐루프의 허리 처리: `--waist {pinned,reference,yaw,free}`(기본 `pinned`).
- 검사: `$PY -m pytest igris/umi_walklow_replay -q`. 렌더 검사까지 하려면 `MUJOCO_GL=egl`을 준다.
- 결과 집계: `$PY tools/analysis/summarize.py <실행 폴더들이 든 폴더>`.

## 서버에서 (호스트에서는 docker와 rsync만 쓸 수 있을 때)

`server/env.example.sh`를 `server/env.local.sh`로 복사해 주소·계정·GPU 번호를 채운다(이 파일은 git에 들어가지 않는다).
서버에 `igris-sim:jazzy` 이미지와, 파이썬 3.12 + PyTorch가 든 GPU 이미지가 있어야 한다. 파이썬 패키지는 서버의
`<REMOTE>/DATA/vendor_py/{mink_cp312,prep_cp312,train_cp312}`에 컨테이너 안에서 `pip --target`으로 받는다.

```bash
cd server
./sync.sh && ./gpu.sh up                                    # 코드 전송, GPU 컨테이너 기동
./batch.sh gen 8 - jobs/gen.txt                             # 시연 생성 (34개 에피소드 × 4회, 8개 동시)
./render.sh gen                                             # 실행별 dataset/
./gpu.sh run "python -m igris.umi_walklow_replay.act_ee train --runs /data/outputs/umi_act/gen --out /data/outputs/umi_act/train/main"
./batch.sh eval 8 /data/outputs/umi_act/train/main/last jobs/held.txt      # 폐루프 평가
./gpu.sh run "python /data/scripts/summarize.py /data/outputs/umi_act/eval"
```

이 스크립트들은 결과를 낸 스크립트에서 주소·계정·경로만 변수로 바꾼 것이다. 바꾼 뒤에는 문법 검사만 했고 서버에서 다시
돌려 보지 않았다.

## 얻은 결과 (시뮬레이션, 2026-11-25)

과제는 케이스 위 박스 3개를 롤테이너에 쌓는 것이다. 학습에 쓰지 않은 장면(에피소드 7·13·21·29)에서:

| 조건 | 제자리에 놓인 박스 | 3개 모두 |
|---|---|---|
| 시연: 라벨 재생 + 방 고정 보정(시뮬의 골반 실측) | 41/48 | 10/16 |
| 시각 없이 라벨만 재생 | 12/33 | 2/11 |
| 상대 EE ACT (카메라 + 관절 각만) | 42/69 | 5/23 |
| 절대 EE ACT | 24/36 | 3/12 |

- 정책 실행에서 로봇이 넘어진 적은 없다.
- 세 번째 박스가 약하다(5/23). 이 시뮬에서는 쥔 박스가 손 안에서 초당 약 1 cm씩 미끄러진다.
- 절대 EE는 성공률이 비슷하지만 관절 가속도가 4배다.
- 학습 데이터: 시연 137회 중 세 박스가 모두 쌓인 94회(122,261프레임).

## 이 폴더로 다시 확인한 것 (PRO6000, 2026-11-25)

- 검사 208개 통과, 6개 건너뜀(팔꿈치 prior 체크포인트 4, 텔레옵 데이터셋 1, 렌더 1은 `MUJOCO_GL=egl`에서 통과).
- 위 1~3번: 에피소드 7 시연이 세 박스를 쌓았고(실시간의 0.99배, 명령 100 Hz), 데이터셋 1,131프레임을 만들었고,
  그 한 실행으로 ACT 학습이 60스텝 돌았다.
- 4번: 서버에서 학습한 체크포인트(30,000스텝)로 에피소드 7 폐루프 1회. 박스 2개를 제자리에 놓았고 세 번째는 놓지 못했다
  (실시간의 0.99배, 정책의 답은 관측 뒤 24 ms = 0.3샘플, 건너뛴 샘플 0).
- 확인하지 않은 것: `Dockerfile`로 이미지 다시 빌드, 서버 스크립트, 5번, Isaac 렌더, 팔꿈치 prior 선택 기능.
