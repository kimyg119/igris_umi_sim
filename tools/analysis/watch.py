"""A video of one run for people: the view from outside next to the three robot cameras, at the stream rate.
usage: watch.py <run output folder> <out.mp4> [seconds from the replay start] [seconds to the end]"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from igris.umi_walklow_replay import cameras

out = Path(sys.argv[1]); run = out / 'run'
log = np.load(run / 'replay_log.npz'); moving = np.isin(log['phase'], ('replay', 'hold'))
start = float(log['sim'][log['phase'] == 'replay'][0]); end = float(log['sim'][moving][-1])
first = start + (float(sys.argv[3]) if len(sys.argv) > 3 else 0.); last = min(end, start + float(sys.argv[4])) if len(sys.argv) > 4 else end
wall, sim, poses = cameras.read_poses(run)
stamps = np.arange(first, last, 1. / 15.)
rows, _ = cameras.rows_at(stamps, sim, poses)
view = cameras.Cameras(run)
writer = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x960', '-r', '15', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p', sys.argv[2]], stdin=subprocess.PIPE)
for row in rows:
    images = view.render(row)
    top = np.concatenate([view.overview(row), images['head_cam']], axis=1)
    bottom = np.concatenate([images['left_cam'], images['right_cam']], axis=1)
    writer.stdin.write(np.concatenate([top, bottom], axis=0).tobytes())
writer.stdin.close(); writer.wait(); view.close()
print(json.dumps(dict(video=sys.argv[2], frames=len(rows), seconds=round(float(last - first), 1))))
