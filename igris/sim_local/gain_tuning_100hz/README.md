# 100 Hz vendor MuJoCo gain experiment

This uses the existing `igris-sim` container, actual vendor controller/bridge,
MuJoCo 3.2.7 and WALK_LOW. Its base remains free. It does not connect to hardware.
The fixed RCV hand model retains its original friction, backlash, parallel
linkages, torque limits, nominal hand inertia, and gravity compensation.

`run.py --prepare RESULTS` copies the validated 80.6 s / 8,061 point replay from
`outputs/synthetic_box_motion_20260914/motion_100hz.npz`. Each payload is the
total weight shared equally between the hands. The added rigid inertial masses
are centered at the EE frames and remain present for the entire trajectory.
This isolates arm tracking under load; it does not demonstrate frictional grasp
capacity. The vendor gravity model is unchanged and receives no payload model.

Use `--seed` with the saved complete settled native qpos to reproduce the final
campaign. It preserves all parallel couplers in a consistent configuration;
resetting only serial wrist angles caused startup linkage failures. The free
joint remains free. A common smooth approach and gain ramp precede each replay.
Each candidate receives the identical complete q sequence,
one command every 10 ms, followed by thirty seconds of constant final q. All
outgoing dq and feedforward torque are zero; only waist and arm gains are set.
Kd is limited to 5. A 600 Kp experimental bound is not a vendor rating.

`observe.cpp` interposes `mj_step`, calls the original library unchanged, then
records at every seventh physics step (nominal 300 Hz). It writes no model,
state, control or applied force. The binary header is three int32 values
`nq,nv,nu`; each float64 row contains `CLOCK_MONOTONIC,time,qpos,qvel,ctrl,
actuator_force,left_EE_xyz_wxyz,right_EE_xyz_wxyz,self_count,min_self_distance`.
`physics.bin.map` supplies model-derived joint and actuator indices.

`score.py` separately reports SDK PJS feedback, native serial joints, physical
output angles including collinear passive backlash, torque, base tilt and
observed timing. Tracking compares against the last command already published,
without compensating away latency. Spectral metrics use 300 Hz uniform wall
time resampling and a fourth order 8–45 Hz Butterworth bandpass. Hold statistics
exclude the first two settling seconds and include standard deviation and a 99% range, so slow
oscillation is not hidden by the bandpass. `ee_score.py` adds 45–120 Hz checks
and same-base EE position/orientation error. Its `fk.cpp` uses only
`mj_kinematics` in a separate process with the same MuJoCo 3.2.7 library; it does
not integrate an alternate trajectory.

Use only while the dedicated simulator is unused. The runner restarts its
simulation launch and pauses it after a trial. Xvfb :99 must be running in that
container; the experiment does not change the desktop X server's permissions.

```bash
cd /home/rcv/workspace/Project_Humanoid
docker exec -d igris-sim Xvfb :99 -screen 0 1280x720x24 -ac -nolisten tcp
docker exec igris-sim g++ -shared -fPIC -O2 /igris/sim_local/gain_tuning_100hz/observe.cpp \
  -I/igris/ws/install/igris_c_mujoco/include \
  -L/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -Wl,-rpath,/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -l:libmujoco.so.3.2.7 -ldl -o /igris/sim_local/gain_tuning_100hz/observe.so
docker exec igris-sim g++ -O2 /igris/sim_local/gain_tuning_100hz/fk.cpp \
  -I/igris/ws/install/igris_c_mujoco/include \
  -L/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -Wl,-rpath,/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -l:libmujoco.so.3.2.7 -o /igris/sim_local/gain_tuning_100hz/fk
/home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/gain_tuning_100hz/run.py \
  --prepare igris/sim_local/gain_tuning_100hz/results/NEW_CAMPAIGN \
  --seed igris/sim_local/gain_tuning_100hz/results/20260914_payload12_v3/seed_qpos.npy
/home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/gain_tuning_100hz/run.py \
  --trial igris/sim_local/gain_tuning_100hz/results/NEW_CAMPAIGN/precision_w400_2kg
/home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/gain_tuning_100hz/score.py \
  igris/sim_local/gain_tuning_100hz/results/NEW_CAMPAIGN/precision_w400_2kg
/home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/gain_tuning_100hz/ee_score.py \
  igris/sim_local/gain_tuning_100hz/results/NEW_CAMPAIGN/precision_w400_2kg
```

Each trial preserves inputs, command/state arrays, native physics and logs.
Existing trial results and original replay/gain/vendor files are not overwritten.
