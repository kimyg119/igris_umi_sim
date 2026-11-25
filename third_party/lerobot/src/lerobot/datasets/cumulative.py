"""Commit finalized sessions to one cumulative LeRobot dataset without replacing history."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import WeakFileLock

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.igris_schema import project_dataset, project_features, validate_features
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.record_phases import PHASE_FEATURE, PHASE_KEY
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH
from lerobot.utils.constants import HF_LEROBOT_HOME

DEFAULT_REPO_ID = "kimyg119/Box_Palletizing_Scenario-A_Bucheon_Warehouse"
LEDGER = "meta/ingest.json"
PROJECTION = "igris-waist-arms-17d-named-torque-tactile-images-v2"


def dataset_files(root: Path) -> list[Path]:
    files = [root / "meta" / name for name in ("info.json", "stats.json", "tasks.parquet")]
    for folder, pattern in (("meta/episodes", "*.parquet"), ("data", "*.parquet"), ("videos", "*.mp4")):
        files.extend(sorted((root / folder).rglob(pattern)))
    if not all(path.is_file() for path in files):
        raise ValueError(f"Incomplete finalized dataset: {root}")
    return sorted(files)


def file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in dataset_files(root):
        with path.open("rb") as stream:
            hashes[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def validate_dataset(root: Path) -> dict:
    """Reject incomplete sessions, invalid indices and non-finite numeric observations before publication."""
    dataset_files(root)
    info = json.loads((root / "meta/info.json").read_text())
    if PHASE_KEY in info["features"] and info["features"][PHASE_KEY] != {**PHASE_FEATURE, "shape": [1]}:
        raise ValueError(f"Invalid recording phase schema: {root}")
    if info["total_episodes"] <= 0 or info["total_frames"] <= 0:
        raise ValueError(f"Empty dataset: {root}")
    shards = sorted((root / "data").rglob("*.parquet"))
    episodes = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not shards or not episodes:
        raise ValueError(f"Missing parquet files: {root}")
    indices = []
    for path in shards:
        frame = pd.read_parquet(path)
        for key, feature in info["features"].items():
            if feature["dtype"] in ("video", "image", "string"):
                continue
            if key not in frame:
                raise ValueError(f"Missing feature {key}: {path}")
            values = np.asarray(frame[key].tolist())
            if values.size != len(frame) * int(np.prod(feature["shape"])):
                raise ValueError(f"Wrong feature shape for {key}: {path}")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {key}: {path}")
            if key == PHASE_KEY and (values.dtype.kind not in "iu" or not np.isin(values, range(7)).all()):
                raise ValueError(f"Invalid recording phase values: {path}")
        indices.append(frame[["index", "episode_index", "frame_index", "task_index"]])
    frames = pd.concat(indices, ignore_index=True)
    metadata = []
    for path in episodes:
        shard = pd.read_parquet(path)
        references = {
            DEFAULT_EPISODES_PATH.format(chunk_index=int(chunk), file_index=int(file))
            for chunk, file in zip(
                shard["meta/episodes/chunk_index"], shard["meta/episodes/file_index"], strict=True
            )
        }
        if references != {path.relative_to(root).as_posix()}:
            raise ValueError(f"Episode metadata points to the wrong shard: {path}")
        metadata.append(shard)
    meta = pd.concat(metadata, ignore_index=True)
    if len(frames) != info["total_frames"] or len(meta) != info["total_episodes"]:
        raise ValueError(f"Finalized counts do not match parquet rows: {root}")
    if not np.array_equal(frames["index"], np.arange(len(frames))):
        raise ValueError(f"Non-contiguous frame indices: {root}")
    if not np.array_equal(meta["episode_index"], np.arange(len(meta))):
        raise ValueError(f"Non-contiguous episode metadata: {root}")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if not set(frames["task_index"]).issubset(set(tasks["task_index"])):
        raise ValueError(f"Unknown task index: {root}")
    cursor = 0
    for row in meta.to_dict("records"):
        start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        part = frames.iloc[start:end]
        if (
            start != cursor
            or end <= start
            or end > len(frames)
            or len(part) != row["length"]
            or not len(part)
            or not (part["episode_index"] == row["episode_index"]).all()
            or not np.array_equal(part["frame_index"], np.arange(len(part)))
        ):
            raise ValueError(f"Invalid episode range {row['episode_index']}: {root}")
        cursor = end
        data_path = info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        )
        if not (root / data_path).is_file():
            raise ValueError(f"Missing data shard {data_path}")
        for key, feature in info["features"].items():
            if feature["dtype"] == "video":
                path = info["video_path"].format(
                    video_key=key,
                    chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]),
                )
                if not (root / path).is_file():
                    raise ValueError(f"Missing video {path}")
    if cursor != len(frames):
        raise ValueError(f"Episode metadata does not cover every frame: {root}")
    return info


def _refresh_index_stats(root: Path) -> None:
    stats = load_stats(root)
    frames = pd.concat(
        [
            pd.read_parquet(path, columns=["index", "episode_index", "task_index"])
            for path in sorted((root / "data").rglob("*.parquet"))
        ],
        ignore_index=True,
    )
    for key in frames:
        values = frames[key].to_numpy(dtype=np.float64)
        stats[key] = {
            "min": np.array([values.min()]),
            "max": np.array([values.max()]),
            "mean": np.array([values.mean()]),
            "std": np.array([values.std()]),
            "count": np.array([len(values)]),
            **{f"q{q:02d}": np.array([np.quantile(values, q / 100)]) for q in (1, 10, 50, 90, 99)},
        }
    write_stats(stats, root)


def _verify_remote(api: HfApi, repo_id: str, revision: str, root: Path) -> None:
    remote = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    files = {item.rfilename: item for item in remote.siblings}
    for path in [*dataset_files(root), root / LEDGER, root / "README.md"]:
        name = path.relative_to(root).as_posix()
        item = files.get(name)
        if item is None or item.size != path.stat().st_size:
            raise ValueError(f"Remote file missing or wrong size: {name}@{revision}")
        if item.lfs:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            expected = item.lfs.sha256
        else:
            actual = hashlib.sha1(f"blob {item.size}\0".encode() + path.read_bytes()).hexdigest()
            expected = item.blob_id
        if actual != expected:
            raise ValueError(f"Remote hash mismatch: {name}@{revision}")


def append_datasets(
    source_roots: list[Path], repo_id: str, *, private: bool = True, cache_dir: Path | None = None
) -> dict:
    """Append whole sessions once; optimistic parent commits prevent lost concurrent uploads.

    Sources stay untouched. A failed/uncertain upload can be retried using the same roots.
    The Hub ledger identifies sessions by all dataset file hashes, independent of their directory names.
    """
    if not source_roots:
        raise ValueError("At least one finalized source is required")
    cache_dir = Path(cache_dir or HF_LEROBOT_HOME / ".cumulative")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(cache_dir / "append.lock"):
        sources = []
        for root in map(Path, source_roots):
            if (root / LEDGER).exists():
                raise ValueError("Source must be an independent recording, not a cumulative dataset")
            info = validate_dataset(root)
            project_features(info["features"])
            hashes = file_hashes(root)
            fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            sources.append((root, info, hashes, fingerprint))
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        remote = api.dataset_info(repo_id, revision="main")
        if remote.private != private:
            raise ValueError(
                "Target privacy differs from recording configuration; repository was not changed"
            )
        parent = remote.sha
        ledger = {"version": 1, "schema": PROJECTION, "sources": []}
        roots = []
        names = {item.rfilename for item in remote.siblings}
        if "meta/info.json" in names:
            if LEDGER not in names:
                raise ValueError(
                    "Existing target has no ingest ledger; refusing to replace an unmanaged dataset"
                )
            previous = Path(snapshot_download(repo_id, repo_type="dataset", revision=parent))
            ledger = json.loads((previous / LEDGER).read_text())
            if ledger.get("version") != 1:
                raise ValueError("Unsupported ingest ledger version")
            previous_info = validate_dataset(previous)
            validate_features(previous_info["features"])
            roots.append(previous)
        else:
            if names - {".gitattributes"}:
                raise ValueError("Target is not an empty or managed dataset")
            previous_info = {"total_episodes": 0, "total_frames": 0}
        known = {entry["fingerprint"] for entry in ledger["sources"]}
        known_names = {entry["name"]: entry["fingerprint"] for entry in ledger["sources"]}
        pending = []
        for source in sources:
            if source[0].name in known_names and known_names[source[0].name] != source[3]:
                raise ValueError(
                    f"Previously committed session changed; refusing duplicate episodes: {source[0]}"
                )
            if source[3] not in known:
                pending.append(source)
                known.add(source[3])
                known_names[source[0].name] = source[3]
        if not pending:
            _verify_remote(api, repo_id, parent, roots[0])
            result = {
                "repo_id": repo_id,
                "revision": parent,
                "already_present": True,
                "total_episodes": previous_info["total_episodes"],
                "total_frames": previous_info["total_frames"],
            }
            (cache_dir / "last_commit.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        with tempfile.TemporaryDirectory(prefix="append-", dir=cache_dir) as work:
            annotated = any(PHASE_KEY in source[1]["features"] for source in pending) or (
                PHASE_KEY in previous_info.get("features", {})
            )
            if roots and annotated and PHASE_KEY not in previous_info["features"]:
                previous_projected = Path(work) / "previous"
                project_dataset(roots[0], previous_projected, add_unknown_phase=True)
                roots[0] = previous_projected
            for index, source in enumerate(pending):
                projected = Path(work) / f"source-{index}"
                project_dataset(source[0], projected, add_unknown_phase=annotated)
                validate_features(validate_dataset(projected)["features"])
                roots.append(projected)
            target = Path(work) / "dataset"
            # shortcut: rebuild metadata locally per commit; use incremental aggregation if dataset scale warrants it.
            aggregate_datasets(
                repo_ids=[repo_id] * len(roots),
                roots=roots,
                aggr_repo_id=repo_id,
                aggr_root=target,
                concatenate_videos=False,
                concatenate_data=False,
            )
            _refresh_index_stats(target)
            result_info = validate_dataset(target)
            validate_features(result_info["features"])
            offset = previous_info["total_episodes"]
            for root, info, hashes, fingerprint in pending:
                if file_hashes(root) != hashes:
                    raise ValueError(f"Source changed during append; no commit created: {root}")
                ledger["sources"].append(
                    {
                        "name": root.name,
                        "fingerprint": fingerprint,
                        "sha256": hashes,
                        "episode_from": offset,
                        "episode_to": offset + info["total_episodes"],
                        "total_frames": info["total_frames"],
                        "projection": PROJECTION,
                    }
                )
                offset += info["total_episodes"]
            if result_info["total_episodes"] != offset or result_info["total_frames"] != (
                previous_info["total_frames"] + sum(item[1]["total_frames"] for item in pending)
            ):
                raise ValueError("Aggregate counts differ from source totals")
            (target / LEDGER).write_text(json.dumps(ledger, indent=2) + "\n")
            (target / "README.md").write_text(
                "---\ntags:\n- LeRobot\n- robotics\n---\n\n# " + repo_id.split("/")[-1] + "\n\n"
                "Cumulative IGRIS recordings: waist and both arms 17D state/action and named 17D "
                "joint/motor torque; tactile images only.\n\n"
                "Raw tactile arrays and leg/neck position/torque axes are excluded. Timing fields, "
                "task labels and original video bytes are preserved. The joint/motor tau_est fields "
                "are estimates in Nm, not electrical current in amperes. Joint torque follows joint "
                "coordinates; motor torque follows actuator coordinates, with distinct names for "
                "coupled waist/wrist motors. Original local sessions remain unchanged.\n\n"
                f"{result_info['total_episodes']} episodes; {result_info['total_frames']} frames; "
                f"nominal {result_info['fps']} FPS. See `meta/ingest.json` for source hashes and episode ranges.\n\n"
                "Pin an exact commit SHA and a separate session-based split for training. "
                "Pass revision explicitly to LeRobotDataset; no moving v3.0 alias is maintained. "
                "The `main` branch grows; it is not a frozen benchmark. Source task labels may differ.\n"
            )
            commit = api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                revision="main",
                folder_path=target,
                parent_commit=parent,
                commit_message=f"Append {len(pending)} sessions ({offset} episodes total)",
            )
            _verify_remote(api, repo_id, commit.oid, target)
            result = {
                "repo_id": repo_id,
                "revision": commit.oid,
                "already_present": False,
                "total_episodes": offset,
                "total_frames": result_info["total_frames"],
            }
            (cache_dir / "last_commit.json").write_text(json.dumps(result, indent=2) + "\n")
            return result


def append_dataset(
    source_root: Path, repo_id: str, *, private: bool = True, cache_dir: Path | None = None
) -> dict:
    return append_datasets([source_root], repo_id, private=private, cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            append_datasets(
                args.source_root, args.repo_id, private=not args.public, cache_dir=args.cache_dir
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
