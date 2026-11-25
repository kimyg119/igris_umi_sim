"""Per-frame stages of the IGRIS Web Record pick/place sequence."""

PHASE_KEY = "complementary_info.phase"
PHASE_LABELS = {
    "0": "미표기 (기존 데이터)",
    "1": "에피소드 시작 ~ Pre Pick 후 TOW 인계 완료",
    "2": "수동 집기 ~ Post Grasp/적재 버튼 클릭 직전",
    "3": "Post Grasp/적재 클릭 ~ 팔 동작 완료",
    "4": "적재 위치로 허리 회전 ~ TOW 인계 완료",
    "5": "수동 내려놓기·놓기 ~ Home 클릭 직전",
    "6": "Home 클릭 ~ 에피소드 종료",
}
PHASE_FEATURE = {
    "dtype": "int64",
    "shape": (1,),
    "names": ["phase"],
    "info": {"phase_labels": PHASE_LABELS},
}


def unknown_phase_stats(count):
    return {
        **{key: [0.0] for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")},
        "count": [int(count)],
    }
