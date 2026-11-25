"""Draw the three robot cameras of a finished run at a few moments (inside the GPU container)."""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from igris.umi_walklow_replay import cameras

run = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
moments = [float(value) for value in sys.argv[3:]] or [.1, .3, .5, .7, .9]
wall, sim, poses = cameras.read_poses(run)
view = cameras.Cameras(run)
start = time.perf_counter(); count = 0
for share in moments:
    row = poses[int(share * (len(poses) - 1))]
    images = view.render(row); count += 1
    strip = np.concatenate([images[name] for name in cameras.NAMES], axis=1)
    Image.fromarray(strip).save(out / f'probe_{int(share * 100):03d}.jpg', quality=90)
for _ in range(60):
    view.render(poses[len(poses) // 2]); count += 1
view.close()
print("frames", count, 'seconds per frame (3 views)', (time.perf_counter() - start) / count, 'records', len(poses),
      'record rate', (len(sim) - 1) / (sim[-1] - sim[0]))
