"""The policy's side of a closed-loop run: look, answer, write.

Runs next to the simulator container (on the GPU), sharing only the run folder with it.  Every ``replan``
samples of the controller's clock it takes an observation - the controller's newest joint state
(``policy/state.bin``) and the three robot cameras drawn from the simulator's newest body poses
(``latest.bin``) - asks the network for a chunk of relative actions, turns them into hand targets in the pelvis
frame and waist joint targets with the hands and the waist of that same observation, and writes them
(``policy/chunk.bin``).  Nothing else of the simulator is read: no pelvis pose, no object pose.

  python -m igris.umi_walklow_replay.policy_server --run <output>/run --checkpoint <act_ee checkpoint folder>
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from . import cameras, ee_features, policy_link
from .kinematics import UpperBody


def latest_rows(run, count):
    """The newest body poses the simulator's observer wrote (count x 7), or None while there is no whole record."""
    try:
        row = np.frombuffer((run / 'latest.bin').read_bytes(), dtype=np.float64)
    except FileNotFoundError:
        return None
    if len(row) != 2 + 7 * count or not np.isfinite(row).all():
        return None
    return row[2:].reshape(count, 7)


def serve(run, checkpoint, replan=4, wait_seconds=900., video=True, device='cuda', absolute=None):
    """Answer the controller of ``run`` until it says it is done.  Returns a summary of the timing."""
    from .act_ee import Policy
    run = Path(run).resolve(); folder = run / 'policy'
    deadline = time.monotonic() + wait_seconds

    def wait_for(test, what):
        while not test():
            if time.monotonic() > deadline:
                raise TimeoutError(f'No {what} from the simulator')
            time.sleep(.05)

    # Everything slow happens before the controller starts its clock: it waits for ``policy_ready`` (worker.py).
    policy = Policy(checkpoint, device)
    absolute = policy.meta.get('absolute', False) if absolute is None else absolute
    paired = policy.meta.get('paired', False)                        # checkpoints from before the pair had a relative action per hand
    model = UpperBody.from_mjcf()
    wait_for(lambda: (run / 'model.json').exists() and (run / 'scene.xml').exists(), 'prepared run')
    count = len(json.loads((run / 'model.json').read_text())['bodies'])
    view = cameras.Cameras(run)
    wait_for(lambda: latest_rows(run, count) is not None, 'body poses')
    for _ in range(3):                                              # the first calls of a network are slow (CUDA start-up)
        hands, chest, waist = ee_features.body(model, np.zeros(31))
        policy.predict(view.render(latest_rows(run, count)), ee_features.state(hands, chest, waist))
    (run / 'policy_ready').write_text(str(checkpoint) + '\n')
    wait_for(lambda: (folder / 'state.bin').exists(), 'controller state')
    writer = None
    if video:
        width = 3 * 320
        writer = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x240',
                                   '-r', '3.75', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p',
                                   str(folder / 'observations.mp4')], stdin=subprocess.PIPE)
    log = dict(observed=[], elapsed=[], answered_elapsed=[], wall_seconds=[], render_seconds=[], network_seconds=[], actions=[])
    wanted = 0; sequence = 0; last_seen = time.monotonic(); seen = None      # seen: the controller's last sequence number
    try:
        while True:
            state = policy_link.read_state(folder)
            if state is None:
                time.sleep(.002); continue
            if state['done']:
                break
            if state['sequence'] != seen:
                seen = state['sequence']; last_seen = time.monotonic()
            elif time.monotonic() - last_seen > 30.:
                break                                               # the controller stopped without saying so
            sample = int(np.floor(state['elapsed'] * state['rate_hz'] + 1e-9))
            if state['elapsed'] < 0. or sample < wanted:
                time.sleep(.001); continue
            rows = latest_rows(run, count)
            if rows is None:
                time.sleep(.001); continue
            clock = time.perf_counter()
            images = view.render(rows)
            rendered = time.perf_counter()
            hands, chest, waist = ee_features.body(model, state['q_measured'])
            actions = policy.predict(images, ee_features.state(hands, chest, waist))
            if absolute:
                targets = ee_features.pair_targets(actions) if paired else \
                    np.stack([ee_features.from9(actions[:, :9]), ee_features.from9(actions[:, 9:18])], axis=1)
                waists = actions[:, 18:21]
            else:
                targets, waists = ee_features.absolute_targets(hands, waist, actions, paired)
            sequence += 1
            policy_link.write_chunk(folder, sequence, sample, targets, waists)
            done = time.perf_counter()
            after = policy_link.read_state(folder)
            log['observed'].append(sample); log['elapsed'].append(state['elapsed'])
            log['answered_elapsed'].append(after['elapsed'] if after is not None else np.nan)
            log['wall_seconds'].append(done - clock); log['render_seconds'].append(rendered - clock)
            log['network_seconds'].append(done - rendered); log['actions'].append(actions.astype(np.float32))
            if writer is not None:
                import cv2
                strip = np.concatenate([cv2.resize(images[name], (320, 240), interpolation=cv2.INTER_AREA) for name in cameras.NAMES], axis=1)
                writer.stdin.write(np.ascontiguousarray(strip).tobytes())
            wanted = (sample // replan + 1) * replan
    finally:
        view.close()
        if writer is not None:
            writer.stdin.close(); writer.wait()
    arrays = {key: np.array(value) for key, value in log.items()}
    np.savez_compressed(folder / 'server_log.npz', **arrays)
    rate = float(state['rate_hz']) if state is not None else 15.
    late = (arrays['answered_elapsed'] - arrays['elapsed']) * rate if len(arrays['observed']) else np.zeros(1)
    summary = dict(checkpoint=str(checkpoint), replan=replan, chunks=int(len(arrays['observed'])), absolute=bool(absolute),
                   answer_samples_mean=float(np.nanmean(late)), answer_samples_p95=float(np.nanpercentile(late, 95)),
                   wall_ms_mean=float(1e3 * arrays['wall_seconds'].mean()) if len(arrays['observed']) else None,
                   render_ms_mean=float(1e3 * arrays['render_seconds'].mean()) if len(arrays['observed']) else None,
                   network_ms_mean=float(1e3 * arrays['network_seconds'].mean()) if len(arrays['observed']) else None)
    (folder / 'server.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='The run folder the simulator container writes (<output>/run)')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--replan', type=int, default=4, help='Samples between observations')
    parser.add_argument('--wait-seconds', type=float, default=900.)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    print(json.dumps(serve(args.run, args.checkpoint, args.replan, args.wait_seconds, not args.no_video)), flush=True)


if __name__ == '__main__':
    main()
