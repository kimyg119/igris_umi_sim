"""Three named recording motions; no transport, sensors or hardware side effects."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .act_pregrasp import ARMS, UPPER_BODY, PreGraspReset
from .igris_c import joint_names

STEPS = (
    ("pre_pick", "Pre Pick", "Home 팔 유지 · 허리 +0.844 rad"),
    ("post_grasp_place", "Post Grasp → 적재 위치 회전", "Post Grasp 궤적 완료 → 허리 −1.266 rad"),
    ("return_home", "Home → 정면", "팔 Home 도착 확인 → 허리 0 rad"),
)


class RecordPoses:
    def __init__(self, path, preview_root="outputs/record_pose_previews"):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        names = [joint_names()[i].replace("left_", "l_").replace("right_", "r_") for i in ARMS]
        if (
            self.data.get("schema") != "igris-record-poses-v1"
            or self.data.get("unit") != "rad"
            or self.data.get("joint_names") != names
        ):
            raise ValueError("record poses require ordered 14-arm joint coordinates in radians")
        self.home = self._arms("home_positions")
        self.post = self._arms("post_grasp_positions")
        for key in ("pre_pick_yaw", "place_yaw", "shoulder_lift_deg"):
            if type(self.data.get(key)) not in (float, int) or not np.isfinite(self.data[key]):
                raise ValueError(f"{key}: finite number required")
        self.preview_dir = Path(preview_root) / self.sha256

    def _arms(self, key):
        value = np.asarray(self.data.get(key), dtype=float)
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError(f"{key}: 14 finite values required")
        return value

    def descriptors(self):
        return [
            {
                "id": key,
                "name": name,
                "description": description,
                "preview": f"/pose-preview/{self.sha256}/{key}.mp4"
                if (self.preview_dir / f"{key}.mp4").is_file()
                else None,
            }
            for key, name, description in STEPS
        ]

    def goals(self, action, initial, *, tolerance=0.2):
        q = np.asarray(initial, dtype=float).copy()
        if q.shape != (31,) or not np.isfinite(q).all():
            raise ValueError("31 finite starting joint coordinates required")
        request = self.motion_request(action)
        if action == "pre_pick" and np.max(abs(q[ARMS] - self.home)) > tolerance:
            raise ValueError("먼저 Home → 정면으로 복귀한 뒤 Pre Pick을 실행하세요")
        goals = []
        for segment in request["segments"]:
            for axis, value in segment["targets"].items():
                q[int(axis)] = value
            goals.append(q.copy())
        return goals

    def motion_request(self, action):
        """Only targets cross IPC; Viser captures the current measured start."""
        if action not in {step[0] for step in STEPS}:
            raise ValueError("unknown pose action")
        request = dict(schema="igris-record-motion-v1", action=action, profile_sha256=self.sha256)
        if action == "pre_pick":
            request.update(
                required_start=dict(zip(map(str, ARMS), self.home.tolist())),
                segments=[dict(targets={"0": self.data["pre_pick_yaw"]}, arrival_axes=UPPER_BODY)],
            )
        else:
            contact = action == "post_grasp_place"
            arms = self.post if contact else self.home
            request["segments"] = [
                dict(
                    targets=dict(zip(map(str, ARMS), arms.tolist())),
                    arrival_axes=[] if contact else UPPER_BODY,
                ),
                dict(
                    targets={"0": self.data["place_yaw"] if contact else 0.0},
                    arrival_axes=[0, 1, 2] if contact else UPPER_BODY,
                ),
            ]
        return request

    def prepare(self, action, target_factory, initial, kp, kd, output, *, execute):
        """Validate every segment before a writer can open, using the shared pose planner."""
        q = np.asarray(initial).copy()
        resets = []
        for index, goal in enumerate(self.goals(action, q)):
            path = Path(output) / f"step-{index}.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "igris-init-pose-v1",
                        "coordinate_space": "joint",
                        "unit": "rad",
                        "joint_names": [
                            joint_names()[i].replace("left_", "l_").replace("right_", "r_")
                            for i in UPPER_BODY
                        ],
                        "positions": goal[UPPER_BODY].tolist(),
                        "name": action,
                    }
                )
                + "\n"
            )
            arrival_axes = None
            if action == "post_grasp_place":
                # Box contact prevents arm target arrival; only the subsequent waist turn needs it.
                arrival_axes = () if index == 0 else (0, 1, 2)
            reset = PreGraspReset(target_factory(q), path, kp, kd, execute=execute, arrival_axes=arrival_axes)
            reset.plan(0.0)
            resets.append(reset)
            q = goal
        return resets
