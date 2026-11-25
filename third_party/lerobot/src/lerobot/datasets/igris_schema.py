"""The cumulative IGRIS warehouse contract: 17 named positions/torques and image-only tactile."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .feature_utils import get_hf_features_from_features
from .io_utils import load_stats, write_stats, write_table_one_row_group_per_episode
from .record_phases import PHASE_FEATURE, PHASE_KEY, unknown_phase_stats

POSITION_NAMES = (
    "waist_yaw.pos",
    "waist_roll.pos",
    "waist_pitch.pos",
    "left_shoulder_pitch.pos",
    "left_shoulder_roll.pos",
    "left_shoulder_yaw.pos",
    "left_elbow_pitch.pos",
    "left_wrist_yaw.pos",
    "left_wrist_roll.pos",
    "left_wrist_pitch.pos",
    "right_shoulder_pitch.pos",
    "right_shoulder_roll.pos",
    "right_shoulder_yaw.pos",
    "right_elbow_pitch.pos",
    "right_wrist_yaw.pos",
    "right_wrist_roll.pos",
    "right_wrist_pitch.pos",
)
POSITION_KEYS = ("observation.state", "action")
TORQUE_KEYS = ("observation.joint_tau_est_Nm", "observation.motor_tau_est_Nm")
PROJECTED_KEYS = POSITION_KEYS + TORQUE_KEYS
SDK_POSITION_NAMES = (
    *POSITION_NAMES[:3],
    *(
        f"{side}_{joint}.pos"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
    ),
    *POSITION_NAMES[3:],
    "neck_yaw.pos",
    "neck_pitch.pos",
)
SDK_TORQUE_NAMES = tuple(name.removesuffix(".pos") for name in SDK_POSITION_NAMES)
# SDK MOTOR_NAMES uses coupled actuator slots at the waist, ankles and wrists.
SDK_MOTOR_NAMES = (
    "waist_yaw",
    "waist_left_motor",
    "waist_right_motor",
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee_pitch",
    "left_ankle_outer_motor",
    "left_ankle_inner_motor",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee_pitch",
    "right_ankle_outer_motor",
    "right_ankle_inner_motor",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_wrist_yaw",
    "left_wrist_front_motor",
    "left_wrist_back_motor",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_wrist_yaw",
    "right_wrist_front_motor",
    "right_wrist_back_motor",
    "neck_yaw",
    "neck_pitch",
)
SDK_UPPER_BODY_INDICES = (0, 1, 2, *range(15, 29))
TORQUE_NAMES = tuple(SDK_TORQUE_NAMES[index] for index in SDK_UPPER_BODY_INDICES)
MOTOR_NAMES = tuple(SDK_MOTOR_NAMES[index] for index in SDK_UPPER_BODY_INDICES)
_NAMED_FEATURES = {
    "observation.state": (POSITION_NAMES, SDK_POSITION_NAMES),
    "action": (POSITION_NAMES, SDK_POSITION_NAMES),
    "observation.joint_tau_est_Nm": (TORQUE_NAMES, SDK_TORQUE_NAMES),
    "observation.motor_tau_est_Nm": (MOTOR_NAMES, SDK_MOTOR_NAMES),
}


def _is_raw_tactile(key: str) -> bool:
    return "tactile" in key.lower() and key.lower().endswith("_raw")


def _joint_indices(features: dict, key: str) -> list[int]:
    feature = features.get(key, {})
    target_names, source_names = _NAMED_FEATURES[key]
    names = feature.get("names")
    shape = list(feature.get("shape", []))
    if key in TORQUE_KEYS and names is None and shape == [31] and feature.get("dtype") == "float32":
        # Original IGRIS tau arrays use the fixed 31-axis SDK order without names.
        names = source_names
    if (
        feature.get("dtype") != "float32"
        or shape not in ([17], [31])
        or not isinstance(names, list | tuple)
        or len(names) != feature["shape"][0]
        or not all(isinstance(name, str) for name in names)
        or len(set(names)) != len(names)
        or set(names) != set(source_names if shape == [31] else target_names)
    ):
        raise ValueError(f"{key} must contain uniquely named 17D or 31D IGRIS float32 joints")
    return [names.index(name) for name in target_names]


def validate_features(features: dict) -> None:
    """Require 17 named positions, joint torques and corresponding physical motor torques."""
    for key in PROJECTED_KEYS:
        _joint_indices(features, key)
        if list(features[key]["shape"]) != [17] or tuple(features[key]["names"]) != _NAMED_FEATURES[key][0]:
            raise ValueError(f"{key} must use the canonical 17D warehouse joint order")
    if any(_is_raw_tactile(key) for key in features):
        raise ValueError("Warehouse datasets store tactile images only, without raw tactile fields")


def project_features(features: dict) -> dict:
    """Select matching position/torque joints and drop raw tactile; preserve other features."""
    result = {key: copy.deepcopy(value) for key, value in features.items() if not _is_raw_tactile(key)}
    for key in PROJECTED_KEYS:
        _joint_indices(features, key)
        result[key]["shape"] = (17,) if isinstance(features[key]["shape"], tuple) else [17]
        result[key]["names"] = list(_NAMED_FEATURES[key][0])
    validate_features(result)
    return result


def _slice_stat(value, indices: list[int], dimension: int, name: str):
    if name == "count":
        return value
    values = np.asarray(value)
    if values.shape != (dimension,):
        raise ValueError(f"Invalid joint statistic {name}: expected {dimension} values")
    return values[indices]


def project_dataset(source: Path, target: Path, *, add_unknown_phase: bool = False) -> None:
    """Create a separate canonical dataset, retaining images and all unrelated columns verbatim."""
    source, target = Path(source), Path(target)
    if target.exists() or target.resolve().is_relative_to(source.resolve()):
        raise ValueError("Projection target must be a new directory outside the source dataset")
    info = json.loads((source / "meta/info.json").read_text())
    features = project_features(info["features"])
    fill_phase = add_unknown_phase and PHASE_KEY not in features
    if fill_phase:
        features[PHASE_KEY] = PHASE_FEATURE
    indices = {key: _joint_indices(info["features"], key) for key in PROJECTED_KEYS}
    dimensions = {key: info["features"][key]["shape"][0] for key in PROJECTED_KEYS}
    hf_features = get_hf_features_from_features(
        {key: {**feature, "shape": tuple(feature["shape"])} for key, feature in features.items()}
    )
    stats = load_stats(source)
    if stats is None:
        raise ValueError("Finalized source dataset requires statistics")
    stats = {key: value for key, value in stats.items() if not _is_raw_tactile(key)}
    if fill_phase:
        stats[PHASE_KEY] = unknown_phase_stats(info["total_frames"])
    for key in PROJECTED_KEYS:
        stats[key] = {
            name: _slice_stat(value, indices[key], dimensions[key], name)
            for name, value in stats[key].items()
        }

    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".cache", ".git"))
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        table = table.drop([key for key in table.column_names if _is_raw_tactile(key)])
        if fill_phase:
            table = table.append_column(PHASE_KEY, pa.array(np.zeros(len(table), dtype=np.int64)))
        for key in PROJECTED_KEYS:
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (len(table), dimensions[key]) or not np.isfinite(values).all():
                raise ValueError(f"Invalid joint values for {key}: {path}")
            column = pa.array(values[:, indices[key]].tolist(), type=pa.list_(pa.float32(), 17))
            table = table.set_column(table.schema.get_field_index(key), key, column)
        metadata = dict(table.schema.metadata or {})
        metadata.pop(b"pandas", None)
        metadata[b"huggingface"] = hf_features.arrow_schema.metadata[b"huggingface"]
        table = table.replace_schema_metadata(metadata)
        write_table_one_row_group_per_episode(table, target / path.relative_to(source))

    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        if fill_phase:
            phase_stats = [unknown_phase_stats(length) for length in table["length"].to_pylist()]
            for name in unknown_phase_stats(0):
                table = table.append_column(
                    f"stats/{PHASE_KEY}/{name}", pa.array([stats[name] for stats in phase_stats])
                )
        table = table.drop(
            [
                key
                for key in table.column_names
                if key.startswith("stats/") and _is_raw_tactile(key.split("/")[1])
            ]
        )
        for key in PROJECTED_KEYS:
            for column in table.column_names:
                if not column.startswith(f"stats/{key}/"):
                    continue
                name = column.removeprefix(f"stats/{key}/")
                if name == "count":
                    continue
                values = [
                    _slice_stat(value, indices[key], dimensions[key], name).tolist()
                    for value in table[column].to_pylist()
                ]
                table = table.set_column(table.schema.get_field_index(column), column, pa.array(values))
        pq.write_table(
            table.replace_schema_metadata(None), target / path.relative_to(source), compression="snappy"
        )
    info["features"] = features
    (target / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    write_stats(stats, target)
