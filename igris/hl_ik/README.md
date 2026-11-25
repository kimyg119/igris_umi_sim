# IGRIS-C HL-IK

HL-IK 논문 v3의 FiSTA elbow prior와 SE(3) Levenberg–Marquardt IK를 재구현한
로컬 실험 모듈이다. 입력·학습·충돌 검사·MuJoCo 물리 기록을 연결한다.
실험 출력 루트는 `outputs/hl_ik_20260915`이다.

VR 실행은 [HL-IK VR 안내](../docs/hl_ik_vr_teleop.md)를 참조한다.
기존 `igris-vr-teleop` CLI의 기본 backend가 `hlik`이다. `vr_adapter.py`가
`PipelineIK`/학습 checkpoint를 기존 clutch·명령 controller·feedback에 연결한다.

**완료 결과:** 두 조건 × 4 seed 학습, 전체 파일 품질 집계, 공통 test,
생성 이력 rollout, native 1 kg 박스 시험을 완료했다.
[결과 보고서](/home/rcv/workspace/Project_Humanoid/igris/docs/hl_ik_results_2026-09-15.md)에 수치와 측정 영상이 있다.
전체 데이터 모델이 더 정확했으나 안정적 박스 운반·착지는 통과하지 못했다.

## 논문과 구현의 대응

| 논문 요소 | 구현 |
|---|---|
| 5개 과거 EE/elbow pose + 다음 EE goal → 다음 elbow pose | `network.py` |
| GRU temporal branch, 두 pose token의 spatial attention | `FiSTA.temporal`, `attention` |
| goal 조건부 sigmoid scale/shift FiLM | `FiSTA.goal`, `forward` |
| MSE 학습, quaternion 자세 | `train.py`, window별 hemisphere 정렬 |
| SE(3) log residual, LM, 이전 시각 q에 대한 smoothing | `solver.py` |
| EE (50,50,50,40,40,40), elbow (20,20,20,5,5,5), q .35 | 잔차에 각 weight의 제곱근 적용 |
| 사람 동작의 robot retarget | 사용자 요청에 따라 GMR로 교체 |

원문: [프로젝트](https://hl-ik.github.io/),
[arXiv v3](https://arxiv.org/abs/2509.20263v3).
전체 논문·그림·미공개 설정·버전 차이는
`igris/docs/hl_ik_paper_implementation_notes_2026-09-15.md`에 기록했다.
공개 실행 코드/학습 checkpoint는 제공되지 않아 저자 구현의 복제본으로 주장하지 않는다.

본문은 단일 팔 history 14D / goal 7D이고 그림은 양팔 28D / 14D라 서로 다르다.
본문에 맞춰 좌우 독립 FiSTA를 학습한다. hidden 128, attention 4 heads,
dropout .1, AdamW 1e-3, batch 2048, 60 epochs, cosine schedule은 미공개 값을
대신하는 명시적 IGRIS 설정이다. 두 데이터 조건 모두 seed 42/123/789/1024를 사용한다.

## 로봇·좌표계

`build_models.py`는 vendor xacro의 `base_type=torso`, `parallel=false`,
`end_effector=none`에서 시작하여 최신 pelvis RCV hand URDF의 손 전체 subtree와
카메라·`end_effector_left/right`를 이식한다. 이전 hand-root를 EE로 대신 쓰지 않는다.
vendor의 torso/pelvis 변환에서 원점이 바뀌는 `waist_frame`, `waist_follower`는
충돌구에도 동일한 변환을 적용한다. 원본 파일은 변경하지 않는다.

- API pose: **torso 기준**, `[x,y,z,qw,qx,qy,qz]`, m/rad.
- 네트워크 pose: shoulder 원점 기준, **축은 torso에 고정**. shoulder의 관절 회전을
  포함하는 가동 frame을 다음 시각의 입력에 넣지 않는다.
- SDK q: `common.SDK_NAMES`의 31축 순서. 팔 14축만 IK로 변경하며 waist와 다리는 caller 소유.
- 현재 학습용 MJCF는 neck 0, MCP 10°, DIP 0°, CMC/thumb MCP −60°로 고정한다.
  공개 API는 neck이 0이 아니면 거부한다. raw URDF에는 손 관절을 유지한다.
- 양손 EE 축과 전체 FK를 64개 자세로 교차 검증한다.
- 학습용 MJCF는 FK/GMR용이다. 실제 관성·접촉 실험은 별도의 기존 native
  free-base 모델과 0.6 kg/hand 손 모델을 사용한다.

## 데이터와 품질

`human.py`는 SMPL-X의 필요한 22 skeletal joint를 공식 모델과 같은 수식으로
계산한다. 원 데이터의 실제 frame rate로부터 30 Hz로 SLERP/선형 보간한다.
`retarget.py`는 설치된 Mink API에 맞게 `limits=`를 명시하여 GMR의 두 단계 IK를 수행한다.
어깨 위치와 팔/다리 segment 길이는 IGRIS URDF에 맞춰 구성한다.

모든 파일을 manifest로 집계한다. stage-i, 손상, 짧은 기록 및 실패는 누락하지 않고
결과에 남긴다. 모든 프레임에 유한값, 개별/연동 관절 한계, 3,666개 충돌구의
340개 설정 쌍, target 위치, wrist 방향, 팔 segment 방향, 속도, 가속도 검사를 한다.
프레임 사이도 최대 .02 rad 간격으로 자기충돌 검사한다. 이는 연속 swept-volume 보증이 아니다.
기존 collision ignore 목록은 유지하고 실패 쌍을 자동으로 제외하지 않는다.

학습 창의 **여섯 프레임 전부**가 통과해야 한다. pack에서 양쪽 elbow 방향 오차
40° 조건도 추가한다. 안전한 frame만 따로 이어 붙여 가짜 연속 이력을 만들지 않는다.
다리 동작은 충돌 문맥이며, 보행 균형·발 접촉 성공을 검증한 데이터라는 뜻은 아니다.

`splits.py`는 원본 manifest와 독립된 source-ID 분할을 만든다. EyesJapan의 wrapper
폴더를 제거하고 DFaust/MoSh의 동일 500xx ID는 보수적으로 같은 분할에 둔다.
CMU에는 한 사람이 여러 subject 번호를 가질 수 있어 완전한 person-disjoint라는
주장은 하지 않는다. 3개 미만 group corpus는 train-only이며 BMLhandball은 로컬에 없다.

두 학습 조건:

1. `paper`: 로컬 ACCAD + CMU + SFU. 로컬 CMU 버전의 motion 수는 원 논문과 다르다.
2. `full`: 전체 corpus의 적격 train 분할.

최종 비교는 **같은 full test window 전부**를 사용한다. 각 모델의 normalization이
다르므로 모델 간 우열은 물리 단위의 m/deg로 비교한다. checkpoint는 validation으로
고르며 test 결과로 seed를 고르지 않는다. box 및 rollout의 대표 seed는 사전에 42로 고정했다.

## 실행

저장소 root에서 실행한다. 이 작업에서 검증한 환경은
`/home/rcv/miniconda3/envs/gmr/bin/python`이다. BLAS worker 과다 생성을 피하도록
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`을 설정한다.

```bash
python -m igris.hl_ik.build_models
python -m igris.hl_ik.retarget --jobs 12
python -m igris.hl_ik.retarget --retry-failed --jobs 12
python -m igris.hl_ik.splits
python -m igris.hl_ik.train pack --scope paper
python -m igris.hl_ik.train pack --scope full
python -m igris.hl_ik.train train --scope paper
python -m igris.hl_ik.train train --scope full
python -m igris.hl_ik.data_report
python -m igris.hl_ik.evaluate common
python -m igris.hl_ik.evaluate fixtures
python -m igris.hl_ik.evaluate rollouts --scope numerical
python -m igris.hl_ik.evaluate rollouts --scope paper
python -m igris.hl_ik.evaluate rollouts --scope full
python -m igris.hl_ik.test_core
```

기존 checkpoint가 있으면 학습은 덮어쓰지 않고 중단한다. source clip은 기존 처리
기록을 재사용하며 `--retry-failed`는 실패 파일만 다시 처리한다.

## IK 호출 계약

```python
from igris.hl_ik.solver import HLIK, ElbowPrior

solver = HLIK(ElbowPrior(checkpoint_path))
solver.reset(q_sdk31)  # 초기 이력은 같은 자세 5개로 채움
result = solver.solve(two_torso_ee_poses, q_sdk31, dt=1/30)
if result.success:
    q_sdk31 = result.q
```

`success=False`의 q는 진단용이다. 기본 pose 허용치는 5 mm/2°이며 collision과
관절 제한을 별도로 검사한다. `solve`가 성공하면 이력을 한 번 추가하므로 같은
tick에서 `observe`를 다시 호출하지 않는다. 실제 관측 이력을 직접 관리하려면
`record_history=False`로 풀고 관측값에 대해 `observe`를 한 번 호출한다.
네트워크 이력은 30 Hz이다. 100 Hz controller를 사용할 때는 command를 별도로
보간해야 하며, 100 Hz마다 이력을 추가하면 학습 시간 간격과 달라진다.

## 박스 실험

`box_plan.py`는 기존 1 kg 실험의 보정된 torso EE FK 경로를 목표로 사용한다.
초기 arm q 이후에는 reference elbow를 제공하지 않고 생성된 FK로 이력을 갱신한다.
strict residual 실패도 기록하여 누적 오차를 측정한다. 충돌 corridor에서는 중단하고
그 이후 command를 내보내지 않는다. 이 진단 모드는 실패 q를 실행 허가한 API가 아니다.

`native_sim.py`는 `igris-hlik-sim-20260915`라는 전용 container를 사용한다.
network `none`, private IPC이며 DDS 명령은 그 container 안에서만 발행한다.
기존 WBC container/학습 및 실물 endpoint를 사용하지 않는다. native MuJoCo 3.2.7의
실제 `mj_step` 결과를 기록하며 qpos/박스 위치를 재생해서 성공을 만드는 방식은 없다.
renderer의 qpos 재생은 저장된 **측정 결과를 시각화할 때만** 사용한다.

최종 비교는 native simulation time으로 경로 진행을 맞추고 wall-clock 100 Hz로
command를 발행한다. 계산 부하와 실제 RTF는 결과에 별도로 기록한다. 물리 관측이
학습 모듈의 이력으로 실시간 되먹임되는 online IK 시험은 별도 단계이다.

native helper는 Python 3.11 이상인 cuRobo 환경으로 실행한다. 새 container에는
Python 3.12용 `cyclonedds==11.0.1` wheel이 필요하며, 다운로드는 host에서 하고
`--no-index`로 전용 container에만 설치한다. 로봇 네트워크 연결은 필요하지 않다.

박스 재현 예시(기존 출력과 다른 새 이름 사용):

```bash
# gmr Python: offline IK
python -m igris.hl_ik.box_plan reproduce_full --scope full
# cuRobo Python: native simulator, then score even if a trial stops on a guard
python -m igris.hl_ik.native_sim --prepare reproduce_full --motion outputs/hl_ik_20260915/box_plans/reproduce_full/motion.npz
python -m igris.hl_ik.native_sim --trial outputs/hl_ik_20260915/native/reproduce_full
MUJOCO_GL=egl PYTHONPATH=igris/igris_curobo/src:igris/sim_local/vr_box_eval python igris/sim_local/vr_box_eval/render_results.py outputs/hl_ik_20260915/native/reproduce_full --metrics-only
python -m igris.hl_ik.native_tracking reproduce_full reproduce_full
MUJOCO_GL=egl python -m igris.hl_ik.visualize video --trial reproduce_full --plan reproduce_full --label 'HL-IK / All corpora'
python -m igris.hl_ik.visualize plots
```

`visualize`의 matplotlib/Pillow는 cuRobo 환경에 설치되어 있다. native trial은
한 container에서 순차 실행한다. 저장된 결과를 다시 읽을 때는 simulator를 재실행하지 않는다.

결과는 경로 완료, EE 추종, 박스 들림/운반/착지, 양손 손바닥+손끝 하중,
자기충돌, 지지대 접촉, MCP/DIP 범위 및 중단 사유로 구분한다. 이 체크포인트와
시뮬레이션 결과는 실물 구동 승인이나 손 관성·마찰·전류 모델의 실측 보정을 의미하지 않는다.

## 고도화 retargeting 재학습

`refined_training.py`는 실행 중인 고도화 배치의 완료 파일을 해시로 고정하고,
같은 후보 source-ID 집합에서 기존 GMR / 보정·원래 시간 / 보정·감속 데이터를
각각 학습한다. 출력 루트를 별도로 지정해 기존 checkpoint를 보존한다.
`freeze --require-complete`는 대상 배치 전체가 완료되지 않으면 중단한다.
감속 데이터도 실제 연속 30 Hz 이력을 유지하며 원본 시간 구간별 창 수를 제한한다.

293개 후보의 부분 데이터로 학습한 12개 모델과 전체 모델 비교는
[고도화 학습 결과](/home/rcv/workspace/Project_Humanoid/igris/docs/hl_ik_refined_training_results_2026-09-15.md)에 있다.
이 결과는 고도화 데이터 전체 학습이 아니며, 현재 부분 모델은 기존 전체 모델보다
IK 및 박스 경로 성능이 낮다. 모델을 자동으로 교체하지 않았다.

기존 native 재생의 100 Hz wall publisher는 30 Hz 관측 파일의 시각을 읽어
명령이 반복된 뒤 점프하는 문제가 확인됐다. 현재는 `native_clock.cpp`가 매 물리
step의 시각을 공유하고 `native_clock.py`가 일관성과 freshness를 확인한다.
이전 영상으로 데이터 정제의 jitter 개선 효과를 판단하면 안 된다.
[속도·jitter 진단](/home/rcv/workspace/Project_Humanoid/igris/docs/hl_ik_speed_jitter_diagnosis_2026-09-15.md)을 참조한다.

## EE 우선 파이프라인

`PipelineIK`는 기존 `HLIK`와 checkpoint를 재사용하는 선택 가능한 수치 solver다.
먼저 EE 잔차를 줄이고, EE Jacobian의 null space에서 elbow prior와 자세 연속성을
반영한다. 비선형 EE 잔차가 나빠지는 secondary 이동은 줄인다. 이는 논문 원형에
추가한 IGRIS 구현이며, 7 DoF 팔의 모든 자세나 접촉 동역학을 보장하지 않는다.
ProxQP 재시도까지 수렴하지 않으면 설치된 DAQP로 같은 문제를 최대 100회 반복
한도에서 한 번 더 푼다. 양쪽 모두 동일한 제약 검사를 통과해야 하며 재시도 수는
`result.qp_fallbacks`로 확인한다. gmr 환경의 `proxsuite`와 `daqp`를 사용한다.

```python
from igris.hl_ik.solver import ElbowPrior
from igris.hl_ik.pipeline import PipelineIK

solver = PipelineIK(ElbowPrior(checkpoint_path), acceleration_limit=30.)
solver.reset(q_sdk31)
result = solver.solve(two_torso_ee_poses, q_sdk31)
if not result.accepted:
    raise RuntimeError(result.reason)  # caller stops; do not export this path
# accepted: validated tracking tick; success: additionally within 5 mm / 2 deg
q_sdk31 = result.q
```

이 API는 30 Hz이며 팔 14축에 전체 tick 기준 속도 3 rad/s, 가속도 30 rad/s²,
개별/연동 관절 한계와 기존 전체 충돌 모델을 적용한다. 충돌 경계에 0.5 mm,
관절 경계에 0.0001 rad 추가 여유를 둔다. 독립적인 jerk 제한은 없다.
이력은 초기 5개 실측으로 준비할 수 있고, 이후 `record_history=False`와 `observe`
한 번으로 관리한다. caller가 waist/다리를 바꾸면 그 변화를 포함한 corridor도
검사해야 한다. 거절 이후 급정지의 가속도를 보장하는 실행 controller는 아니다.

`pipeline_experiment.py`는 같은 목표·초기 이력·충돌 모델에서 solver/prior/가속도
11개 조건을 비교한다. 첫 거절에서 중단하고 남은 전체 목표를 실패로 집계한다.
`export`는 완료 경로만 C2 보간하고 속도/가속도, 관절 및 전체 충돌 corridor를
다시 검사한 뒤 100 Hz 시뮬레이션 명령을 저장한다. 검사 실패 시 motion 파일을
생성하지 않는다. 보간 결과의 추종 오차는 별도로 보고한다.

```bash
python -m igris.hl_ik.pipeline_experiment prepare --out outputs/hl_ik_pipeline_new
python -m igris.hl_ik.pipeline_experiment run --out outputs/hl_ik_pipeline_new --label qp_priority_full
python -m igris.hl_ik.pipeline_experiment export --out outputs/hl_ik_pipeline_new --label qp_priority_full --name pipeline_new
python -m pytest igris/hl_ik/test_core.py igris/hl_ik/test_pipeline.py -q
```

[파이프라인 고도화 결과](/home/rcv/workspace/Project_Humanoid/igris/docs/hl_ik_pipeline_results_2026-09-15.md)에
수치 solver의 효과, 기존/부분 고도화 checkpoint 비교와 native 물리 결과를 구분했다.
