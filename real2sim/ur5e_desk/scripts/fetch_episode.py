"""Pull one episode of the real dataset into assets/reference/episode_XXX/ for replay_episode.py.

    python scripts/fetch_episode.py [episode_index]      # needs pandas + pyarrow, ffmpeg

Writes episode.json (30 Hz joint states/actions in rad, gripper 0..1, step labels) and frames/NNN.jpg (wrist camera).
Only episodes stored in the dataset's first data/video file are supported (episode 0 is).
"""
import json, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

REPO = "https://huggingface.co/datasets/sonnyjin/ur5e_cube_auto_200_success_5826/resolve/main/"
ROOT = Path(__file__).resolve().parents[1]
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cache = ROOT / "work/hf"; out = ROOT / f"assets/reference/episode_{ep:03d}"; (out / "frames").mkdir(parents=True, exist_ok=True)


def get(rel):
    dst = cache / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        urllib.request.urlretrieve(REPO + rel, dst)
    return dst


meta = pd.read_parquet(get("meta/episodes/chunk-000/file-000.parquet")); m = meta[meta.episode_index == ep].iloc[0]
assert m["data/file_index"] == 0 and m["videos/observation.images.front/file_index"] == 0, "episode is not in file-000"
df = pd.read_parquet(get("data/chunk-000/file-000.parquet")); d = df[df.episode_index == ep]
label = next(json.loads(l) for l in open(get("labels.jsonl")) if json.loads(l)["episode_index"] == ep)
json.dump(dict(dataset="sonnyjin/ur5e_cube_auto_200_success_5826", episode_index=ep, fps=30,
               joint_names=["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"],
               state=np.stack(d["observation.state"].values).round(6).tolist(), action=np.stack(d["action"].values).round(6).tolist(),
               steps=label["steps"], label=label), open(out / "episode.json", "w"))
t0, t1 = m["videos/observation.images.front/from_timestamp"], m["videos/observation.images.front/to_timestamp"]
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t0:.4f}", "-i",
                str(get("videos/observation.images.front/chunk-000/file-000.mp4")), "-frames:v", str(len(d)), "-vsync", "0",
                "-start_number", "0", "-q:v", "2", str(out / "frames/%03d.jpg")], check=True)
print(f"episode {ep}: {len(d)} frames, steps {label['steps']} -> {out}")
