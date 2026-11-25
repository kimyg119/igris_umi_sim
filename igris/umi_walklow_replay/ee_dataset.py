"""Training data from simulated replays: what the robot's cameras saw, where its links were, what it was told.

One finished run of the pose-stream replay (``run.py ... --stream``, ``container_run.py``) becomes one episode
at the rate of the stream (15 Hz), written next to the run in ``<output>/dataset``:

  frames.npz       per frame: joint angles, the hands through the links (pelvis frame and room), the chest, the
                   pelvis, the hand targets the controller was following and the waist angles it commanded
  head_cam.mp4, left_cam.mp4, right_cam.mp4      the three cameras of cameras.py at full size
  head_cam.jpgs, ... + images.json               the same frames small, one JPEG after the other, for training

Frame ``k`` is the moment stream sample ``k`` was handed to the controller.  Its observation is what was known
then: the newest camera record, the newest joint state.  Its action is that sample as the controller followed
it - the hand targets one sample period later, when the stream has arrived at it, and the waist angles the IK
commanded with them.  The targets are kept in the pelvis frame as commanded and in the room (through the pelvis
pose of the simulator): ``ee_features.relative_actions`` turns either into the relative actions a policy learns.

  python -m igris.umi_walklow_replay.ee_dataset frames <output>            numbers only
  python -m igris.umi_walklow_replay.ee_dataset render <output> [...]      numbers and pictures (needs MUJOCO_GL)
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from . import cameras, ee_features
from .kinematics import UpperBody

SMALL = (320, 240)                # training pictures: width, height
JPEG_QUALITY = 92


def pose_matrices(rows):
    """4x4 transforms of N x 7 poses (position, w,x,y,z)."""
    rows = np.asarray(rows, dtype=float)
    out = np.zeros((len(rows), 4, 4))
    out[:, :3, :3] = cameras._matrices(rows[:, 3:]); out[:, :3, 3] = rows[:, :3]; out[:, 3, 3] = 1.
    return out


def frames(out, model=None):
    """The numbers of one run as a dict of arrays (see the module text); raises if the run is not a finished
    pose-stream replay."""
    out = Path(out); run = out / 'run'
    result = json.loads((run / 'replay_result.json').read_text())
    if not result.get('pass') or not result.get('stream_hz') or 'replay_start_sim' not in result:
        raise ValueError(f'{out} is not a finished pose-stream replay')
    log = np.load(run / 'replay_log.npz')
    if 'target_left_pose' not in log.files:
        raise ValueError('The replay log has no hand target poses (older worker)')
    rate = float(result['stream_hz']); start = float(result['replay_start_sim'])
    # The lead-in to the first pose and the replay itself; the hold after it logs no followed targets.
    moving = np.isin(log['phase'], ('approach', 'replay'))
    sim = log['sim'][moving]
    count = int(np.floor((sim[-1] - start) * rate + 1e-6))                                       # samples k that were reached
    stamps = start + np.arange(count) / rate
    seen = np.clip(np.searchsorted(sim, stamps, side='right') - 1, 0, len(sim) - 1)              # newest tick at each frame
    reached = np.clip(np.searchsorted(sim, stamps + 1. / rate, side='left') - 1, 0, len(sim) - 1)    # last tick before the next sample
    model = model or UpperBody.from_mjcf()
    measured = log['q_measured'][moving][seen]
    pelvis = pose_matrices(log['pelvis'][moving])
    hands = np.empty((count, 2, 4, 4)); chest = np.empty((count, 4, 4)); state = np.empty((count, ee_features.STATE_SIZE))
    for index, q in enumerate(measured):
        hands[index], chest[index], waist = ee_features.body(model, q)
        state[index] = ee_features.state(hands[index], chest[index], waist)
    target_pelvis = np.stack([log['target_left_pose'][moving][reached], log['target_right_pose'][moving][reached]], axis=1)
    analysis = json.loads((run / 'analysis.json').read_text()) if (run / 'analysis.json').exists() else {}
    manifest = json.loads((run / 'manifest.json').read_text())['umi_replay']
    return dict(
        t=np.arange(count) / rate, sim=stamps, rate_hz=rate, episode=int(manifest['episode']),
        q_measured=measured, q_command=log['q_command'][moving][seen], state=state,
        ee_pelvis=hands, chest_pelvis=chest, pelvis_room=pelvis[seen], ee_room=pelvis[seen][:, None] @ hands,
        target_pelvis=target_pelvis, target_room=pelvis[reached][:, None] @ target_pelvis,
        waist_target=log['q_command'][moving][reached][:, ee_features.WAIST],
        parcels_room=log['parcels'][moving][seen], phase=log['phase'][moving][seen],
        boxes_stacked=int(analysis.get('boxes_stacked', -1)), stack_success=bool(analysis.get('stack_success', False)))


class JpegStore:
    """Small pictures of one camera, one JPEG after the other in a single file."""

    def __init__(self, path):
        self.path = Path(path); self.stream = self.path.open('wb'); self.offsets = [0]

    def add(self, image):
        import cv2
        small = cv2.resize(image, SMALL, interpolation=cv2.INTER_AREA)
        done, data = cv2.imencode('.jpg', small[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not done:
            raise RuntimeError('JPEG encoding failed')
        self.stream.write(data.tobytes()); self.offsets.append(self.offsets[-1] + len(data))

    def close(self):
        self.stream.close()
        return self.offsets


def read_jpeg(blob, offsets, index):
    """Picture ``index`` (H x W x 3 uint8, RGB) of a JPEG store read as bytes or a memory map."""
    import cv2
    data = np.frombuffer(blob[offsets[index]:offsets[index + 1]], dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1]


def video_writer(path, width, height, rate):
    return subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                             '-r', f'{rate:g}', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-g', '15',
                             '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)


def render(out, width=640, height=480, video=True, overwrite=False):
    """Write ``<out>/dataset`` for one run: the numbers and the three cameras.  Returns a summary."""
    out = Path(out).resolve(); run = out / 'run'; target = out / 'dataset'
    if (target / 'images.json').exists() and not overwrite:
        return json.loads((target / 'images.json').read_text())
    numbers = frames(out)
    target.mkdir(exist_ok=True)
    wall, sim, poses = cameras.read_poses(run)
    rows, used = cameras.rows_at(numbers['sim'], sim, poses)
    view = cameras.Cameras(run, width, height)
    stores = {name: JpegStore(target / f'{name}.jpgs') for name in cameras.NAMES}
    videos = {name: video_writer(target / f'{name}.mp4', width, height, numbers['rate_hz']) for name in cameras.NAMES} if video else {}
    try:
        for row in rows:
            images = view.render(row)
            for name in cameras.NAMES:
                stores[name].add(images[name])
                if video:
                    videos[name].stdin.write(images[name].tobytes())
    finally:
        view.close()
        offsets = {name: store.close() for name, store in stores.items()}
        for writer in videos.values():
            writer.stdin.close(); writer.wait()
    np.savez_compressed(target / 'frames.npz', camera_record=used, camera_age_s=numbers['sim'] - sim[used], **numbers)
    summary = dict(frames=len(rows), rate_hz=numbers['rate_hz'], episode=numbers['episode'], small=list(SMALL), full=[width, height],
                   cameras=list(cameras.NAMES), offsets=offsets, boxes_stacked=numbers['boxes_stacked'],
                   stack_success=numbers['stack_success'], camera_age_p99_s=float(np.percentile(numbers['sim'] - sim[used], 99)))
    (target / 'images.json').write_text(json.dumps(summary) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['frames', 'render'])
    parser.add_argument('out', type=Path, nargs='+', help='Run output folders')
    parser.add_argument('--no-video', action='store_true', help='render: only the small training pictures')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for out in args.out:
        try:
            if args.action == 'frames':
                numbers = frames(out)
                print(json.dumps(dict(out=str(out), frames=len(numbers['t']), stack_success=numbers['stack_success'])), flush=True)
            else:
                summary = render(out, video=not args.no_video, overwrite=args.overwrite)
                print(json.dumps(dict(out=str(out), **{key: summary[key] for key in ('frames', 'stack_success', 'boxes_stacked', 'camera_age_p99_s')})), flush=True)
        except (ValueError, FileNotFoundError) as error:
            print(json.dumps(dict(out=str(out), error=str(error))), flush=True)


if __name__ == '__main__':
    main()
