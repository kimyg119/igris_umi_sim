# VR box evaluation

Runs the existing `run_vr_teleop` entrypoint against the vendor IGRIS simulation. These are synthetic Quest-contract inputs, not a recording of someone wearing a headset. The evaluation uses its own HOME motion and reset keyframe with WALK_STANDBY legs and raised inward-facing palms and elbows out (shoulder pitch -0.65, elbow about -1.85, shoulder roll about +/-0.48 rad); reset is followed immediately by WALK_LOW with cached vendor arm/waist hold commands; it checks the measured arm pose after entering WALK_LOW (maximum deviation 0.35 rad), and clutch calibration uses that measured pose. Every trial restarts the dedicated `igris-sim` simulation and its command bridge, enters WALK_LOW, and pauses physics when finished. Do not run alongside another operator of this simulator.

The robot retains `floating_base`; the 15 cm cube has its own free joint. Hands retain their fixed geometry and receive no HandCmd. The cube centre starts at 0.90 m height and 0.40 m horizontal radius. New runs default to HMD-yaw coupled waist/arm IK. Use `--hold-waist` to reproduce the previous hold-only protocol; the saved nine original trials used that flag and do not validate the new HMD path. WALK_LOW owns the legs in either mode. The current synthetic publisher compensates the measured base pose each frame, and generates elbow swivel with constant segment lengths. This ideal world registration uses MuJoCo odometry; it is not a feature verified on the live Quest publisher. Mass and friction are explicit trial arguments. The current unloaded diagnostic uses 0.01 kg and sliding friction 1.2; this does not validate 0.5, 1, or 2 kg transport. Two 10 cm square support surfaces leave 2.5 cm access below each cube side and have normal robot/environment collision enabled. The evaluator fails on robot/support force above 0.5 N. Existing fixed-hand meshes are enabled for cube/support contact; hand-versus-robot self-contact remains outside this added mesh contact model. Existing robot self-collision geometry remains active.

## Run

Requires the existing `igris-sim` container, built vendor workspace, cuRobo virtual environment, and ffmpeg. Container endpoints in this local setup are `172.17.0.2:5560/5561`; synthetic inputs use `127.0.0.1:5577`. The live Quest publisher on 5557 is separate.

```bash
cd /home/rcv/workspace/Project_Humanoid
docker exec igris-sim g++ -shared -fPIC -O2 \
  /igris/sim_local/vr_box_eval/record_physics.cpp \
  -I/igris/ws/install/igris_c_mujoco/include \
  -L/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -Wl,-rpath,/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -l:libmujoco.so.3.2.7 -ldl -o /tmp/igris_record_physics_next.so
docker exec igris-sim mv /tmp/igris_record_physics_next.so /tmp/igris_record_physics.so
/home/rcv/workspace/curobo/.venv/bin/python -u \
  igris/sim_local/vr_box_eval/evaluate.py elbow 0.5
/home/rcv/workspace/curobo/.venv/bin/python \
  igris/sim_local/vr_box_eval/render_results.py \
  igris/sim_local/vr_box_eval/results/<timestamp>_box_elbow_0.5kg
```

After changing `assets.FIXED_HAND_JOINT_VALUES`, run `python igris/sim_local/vr_box_eval/bake_fixed_hands.py` to regenerate vendor hand meshes/inertia and IK hand collision spheres. Run `python igris/sim_local/vr_box_eval/test_scene.py` for the scene preflight. Repeat for off/elbow/arm and each mass. Each run creates a timestamped result directory and preserves previous trials. Use `--motion yaw` for an unloaded torso diagnostic; it is not a box-task result. `--metrics-only` on the renderer skips video encoding.

## Evidence and limits

`record_physics.cpp` interposes only `mj_step`: it calls the original vendor MuJoCo 3.2.7 function, then observes qpos, qvel, and contact forces at approximately 30 Hz wall time. It never writes robot/box state or control. The binary starts with two native int32 values (nq, nv), followed by float64 rows:

`wall_time, sim_time, qpos[nq], qvel[nv], left_box_normal, right_box_normal, other_box_normal, robot_self_contact_count, minimum_self_distance, robot_support_normal, box_environment_normal`.

The renderer uses recorded qpos, including world base pose and box pose, at 24 fps. Cyan/magenta dots show the intended world URDF end-effector goals; arm mode can retarget them further. They are not measured wrists. `ready.png` and `preflight.json` preserve the measured post-settle clearance check; both the reset keyframe and actual settled pose require at least 3 cm from every robot geom to the cube and both supports. Host MuJoCo 3.12 is used only for FK/rendering; it does not integrate an alternate trajectory. Contacts/forces are read from the vendor recording, not recalculated by the renderer. The host loader removes a redundant `kv=0` attribute where the old model also specifies dampratio. This compatibility edit does not affect vendor physics.

The open approach uses a waypoint 5 cm forward, 8 cm outward per hand, and 6 cm above the final approach pose, then lowers beside the box. Fixed hands use MCP +10 degrees, DIP 0, and thumb CMC/MCP -60 degrees. The fingers point down/forward at 45 degrees within each box face (left/right hand-root roll -45/+45 degrees). A bounded 2 mm placement grid retains the whole palm patch with 1 cm edge margin and maximizes the exposed MCP patch while excluding the projected distal tongue with a 2 mm lateral buffer. The right contact window mirrors the left window, so normal forces act at matching radial/vertical positions; independent area optima had separated the two palm centroids by about 5 cm. The palm bump (`__1_5`) touches first with the broad MCP plate (`__1_6`) nominally 5 mm behind it. The contact-driven sequence rotates each wrist about its palm contact. Nominal coplanarity is 2.65 degrees; each hand can seek contact independently up to 10.6 degrees of virtual angular preload and stops tilting once its bump and plate support the box. The palm patch and an exposed portion of the MCP plate fit the 15 cm box face. A convex-hull check places the distal link's protruding central tongue outside that face throughout the unloaded 5.3-degree geometry arc. The larger command range compensates loaded wrist deflection and is not a claim of unloaded distal-link clearance over the whole commanded arc; actual box contacts are evaluated separately. This advances the hands further and reduces the available plate area; covered surfaces are not counted as usable contact. These mesh-derived projected areas are geometric proxies, not measured pressure/contact areas. The distal fingertip (`__1_4`) is supplementary. `--timed` retains the initial contact orientation and does not perform contact-driven seating.

`physics.bin.contacts` retains its existing float64 format: wall time, left palm/tip/other, right palm/tip/other (7 values). `physics.bin.pads` adds wall time and left/right broad MCP plate normal forces (3 values). The plate remains included in the old file's `other` field for compatibility; do not add it twice when summing forces. Both files observe native vendor contact forces without modifying physics.

Task success requires a completed carry/release, no loss of balance, and at least 95% valid carry frames. New `contact_target=palm_seated` recordings require both palm bumps and both MCP plates to retain contact, with combined bump+plate support exceeding the mass/friction force threshold. Tip force is recorded as an auxiliary metric. Missing plate-force evidence cannot pass the new criterion. Historical recordings retain their metadata-specific palm-only or palm/tip criteria. None of these screening checks proves force closure; contact samples can miss brief impacts.


Controller-only IK tracks both URDF tools with position/orientation axis factors 1/0.5 and torso yaw factor 0.625, with soft roll/pitch posture factors 2.5. Retract weight is 5/64, elbow-out factor 0.05, and acceleration regularization 0.001/64. This changes the balance against feasibility penalties; it is not an equivalent rescaling of the whole objective. The required HMD task checks yaw only; roll/pitch remain soft posture preferences within the vendor joint limits. Hands retain 8 mm / 0.08 rad residual gates; torso yaw retains 0.087 rad. Local candidates are projected within the per-cycle joint step bounds and then rechecked for all pose and static constraints. The command path also checks the 0.1 rad branch guard and samples the joint corridor at <=0.01 rad increments. A rejected Cartesian waypoint can rebase to FK of the last accepted command without replacing contact preload by measured q. Only a passing configuration updates the command and warm start. No automatic global branch switch occurs during tracking. These sampled checks do not add the cube/supports to the IK collision world or prove continuous swept clearance. `test_approach.py` checks the staged waypoint, frame-correct open clearance, and unchanged final palm/tip squeeze targets.


## Contact-driven execution (default for box trials)

Box trials now use observed vendor contact forces and an IK acknowledgement for every candidate target. `--timed` reproduces the previous fixed schedule. The publisher advances its shared grasp pose only after the teleop entrypoint accepts the candidate. A rejected candidate leaves the command cache and warm start unchanged; the publisher pauses briefly and reduces its next step. A valid Cartesian waypoint still approaching its endpoint waits without this penalty. The task uses twice the original per-update trajectory increments; all Cartesian and joint rate gates still apply. Rejected candidates retain FOLLOWING while the held configuration remains feasible; missing teleop feedback for 0.5 seconds fails the evaluation. Deadman, state freshness, mode and tracking-error gates remain enabled.

`contact_carry.py` increases symmetric preload from -20 mm clearance toward at most +35 mm per hand. Each palm bump must sustain at least `max(1, 1.5 * mass * 9.81 / (2 * friction))` N for 0.3 seconds before `seat_palm` starts. During seating, only IK-acknowledged rotations accumulate; each contact pivot remains fixed. Task values are clock, preload, lift, yaw, left seating fraction, right seating fraction. A missing plate contact increases tilt, while plate-only contact reverses it. Preload increases only if the total contact force of a hand is insufficient. Auxiliary contact therefore limits additional squeeze but never qualifies the grasp as seated. Loss of a palm bump under sufficient total force reverses that hand's tilt to recover the intended contact surface. Both tilt and preload remain bounded and require an IK acknowledgement.

Once each bump and plate reaches `max(0.1, min(0.5, 0.1 * mass * 9.81))` N and each hand's combined bump+plate force exceeds that preload threshold for 0.3 seconds, a probe lift of at most 4 cm starts. The actual box must rise >12 mm, lose support contact, and retain those contacts before the 8 cm lift and 90-degree carry can proceed. Probe/lift/transport hold height and yaw while restoring a missing surface; transport fails if actual box lift falls below 5 mm. Tips do not gate progress. The base stays free and wrist PD gains are unchanged.

The evaluator reads the last complete physics, contacts and pads records with a 150 ms freshness limit. This is a simulation-only observer and synthetic publisher; no physics state is written. Live Quest input does not automatically acquire contact feedback or this staged wrist motion.

Explicitly authorized wrist comparison:

```bash
/home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/vr_box_eval/evaluate.py off 0.5 --wrist-kp 30
/home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/vr_box_eval/evaluate.py off 0.5 --wrist-kp 60
```

Both keep wrist Kd=3, other gains, collision buffers, branch guard and speed/acceleration limits identical. `contact_task.json` stores transitions and the failure/success state; `target_feedback.json` stores accepted/rejected input sequence numbers. Rendered stages and scoring use actual contact-driven transition times, including trials which never reach carry. Render each returned result folder using the command above.


For a gain comparison with the same target schedule, `evaluate_wrist_hold.py` runs the timed approach, commands 30 mm preload per hand, and holds that pose from 23 to 42 seconds. It deliberately requests no lift. Compare the 25–40 second settled window, not its task-success score:

```bash
DISPLAY=:1 /home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/vr_box_eval/evaluate_wrist_hold.py 30
DISPLAY=:1 /home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/vr_box_eval/evaluate_wrist_hold.py 60
# Render each returned output directory, then summarize the same window:
/home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/vr_box_eval/render_results.py OUTPUT_DIRECTORY
/home/rcv/workspace/curobo/.venv/bin/python igris/sim_local/vr_box_eval/compare_wrist.py KP30_OUTPUT_DIRECTORY KP60_OUTPUT_DIRECTORY
```

The evaluation starts its simulation bridge with `--sim-wrist-kp60`, allowing only the six wrist Kp ceilings to reach 60 in namespace `igris_c_001`. Without this explicit option, the existing ceiling remains 45 and a Kp 60 frame is rejected in its entirety. Kd and every other axis keep their existing ceilings. The native hardware SDK bridge is unchanged. Restart `igris/sim_local/start_dds_bridge.sh` without the option after the experiment to restore the bridge default. Normal teleop still defaults to wrist Kp 30 / Kd 3.

The box evaluation overrides `noslip_iterations` to 10 while leaving mass, friction coefficients, contact geometry, gains and timestep unchanged. This refines tangential friction after the main contact solve; it does not attach the box. Native MuJoCo 3.2.7 contact isolation with recorded stationary hand geometry showed roughly 3 mm drift in 6 seconds at 0 iterations versus 0.02 mm at 10. This is a separate contact diagnostic, not a carry result. See [MuJoCo preventing slip](https://mujoco.readthedocs.io/en/3.2.7/modeling.html#preventing-slip). Full-stack validation must also check the measured physics realtime ratio because this option adds CPU work. The general walking scene is unchanged.


## Simulation arm gain comparison

`tune_gains.py` reuses the same vendor trial with explicit shoulder/elbow/wrist
gain groups. It keeps WALK_LOW, the free base, fixed hands, contact gates, IK and
actuator limits. It records the applied SDK gains and rejects profiles outside
the existing bridge experiment limits (shoulder/elbow: 1.5 times vendor;
wrist: Kp at most 60, Kd at most 4.5). Normal VR/hardware defaults are unchanged.

```bash
MUJOCO_GL=egl /home/rcv/workspace/curobo/.venv/bin/python \
  igris/sim_local/vr_box_eval/tune_gains.py --profile wrist45_damped --radius .37
```

Profiles are `vendor`, `wrist45`, `wrist45_damped`, `wrist60_damped`, `arm150`,
and `arm180`. The default box mass is 0.01 kg; use `--mass .5` explicitly for a
payload trial. Friction is 1.2. `--radius .37` places the box/supports 3 cm nearer
than the original .40 m task; do not compare its completion with .40 m runs as
a gain-only change. `--match-hand-inertia` additionally creates an evaluation-local
33-link controller URDF with only the two terminal wrist inertias matched to the
native model's fixed hand assembly. Moving wrist linkages are excluded.

`gain_score.json` separates SDK command/measurement errors, passive backlash
angles, sampled native actuator effort, and full contact-carry success.
`physics.bin.effort` has two int32 header values `(nu,nv)`, then float64 rows:
`wall_time, sim_time, ctrl[nu], actuator_force[nu], qfrc_actuator[nv]`.
It is a read-only record at the observer's approximately 30 Hz rate, not a
continuous peak-torque measurement. Rebuild the observer using the command above
before the experiment. Each completed experiment leaves the simulator paused.

## Native output-angle feedback

The native model places passive `*_backlash` joints after the SDK motor joints. The SDK's `q` is therefore not the physical link angle. A matched-state audit found 20–30 mm / about 3 degrees of hand FK error when these angles were omitted; adding motor and passive angles reproduces the native EE poses. The observer additionally writes `physics.bin.offsets`: 32 float64 values per record (`wall_time, offset[31]` in SDK order). The evaluation bridge uses `--sim-joint-offsets PATH`, restricted to namespace `igris_c_001`, and publishes only complete finite records no older than 150 ms.

`RobotState` optionally carries `joint_offsets` and `joint_offsets_t_wall` together. Teleop inserts observed offsets into cuRobo's existing joint-offset map and locked-joint transforms before FK/IK; solver variables, SDK commands, and vendor joint-limit checks remain motor angles. All global/local rollouts receive the update without rebuilding CUDA graphs. Once offset telemetry is present, stale or missing observations stop following. These are simulator observations, not a hardware backlash estimator: hardware requires equivalent output sensing or a separately validated estimate. The native passive dynamics and PD gains are unchanged.

## Prescribed waist comparison

`evaluate.py off 0.01 --friction 1.2 --separate-waist` uses the same scene, target generator, contact gates and gains with an experimental arm-only IK. `SeparatedWaistIK` prescribes motor yaw from the HMD-derived torso heading and retains the clutched motor roll/pitch. Only 14 arm joints are optimization variables; head/torso pose cost factors are zero. FK uses observed backlash for both arms and the prescribed waist. Prescribed waist and solved arms are cached together only after the original joint-step, vendor limit, branch and sampled collision-path checks pass. A rejected waypoint therefore holds both. Waist collision geometry is updated for each sampled command posture. The base remains free.

This is exposed as `--separate-waist` on the VR CLI as well, requiring tracking OFF and no `--hold-waist`. Default teleop remains the coupled 17-axis solver. The target/state contract and safety gates are unchanged; the comparison does not supply hardware output-angle or contact sensing. `metadata.json`, `teleop_summary.json`, and the rendered video identify the selected mode.


## Simulation MCP compliance (XC330-T288-T)

`--compliant-hands` replaces the evaluation's two fixed MCP joints with
actuated hinges (default 30 degrees, range 0–60 degrees), and both DIP joints with
passive torsion-spring hinges. Palm, MCP and distal
inertias divide the assumed 0.6 kg hand mass; vendor arm/leg axes and actuators
are preserved. Normal VR defaults and hardware configuration are unchanged.

```bash
docker exec igris-sim g++ -shared -fPIC -O2 \
  /igris/sim_local/vr_box_eval/hand_motor.cpp \
  -I/igris/ws/install/igris_c_mujoco/include \
  -L/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -Wl,-rpath,/igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco \
  -l:libmujoco.so.3.2.7 -ldl -o /tmp/igris_hand_motor_next.so
docker exec igris-sim mv /tmp/igris_hand_motor_next.so /tmp/igris_hand_motor.so
MUJOCO_GL=egl /home/rcv/workspace/curobo/.venv/bin/python -u \
  igris/sim_local/vr_box_eval/tune_gains.py \
  --profile arm180_wrist60 --radius .37 --height .8 --mass 1 --compliant-hands --preload-limit .05
```

Rebuild the read-only observer above as well. `hand_motor.cpp` is a separate,
explicitly actuating simulation plugin: it writes only the two additional MCP
`qfrc_applied` entries before native `mj_step`. It never writes qpos/qvel or vendor
arm commands. The Python process reuses `HAND` admittance/estimator code at 50 Hz,
with native free-load baseline and simulated input current. `HAND` runtime limits
are 180 mA, Goal PWM 133, and external torque 0.5 Nm. The existing admittance
stiffness is 6 Nm/rad, virtual inertia 0.02, damping ratio 1, and closing bias
0.03 Nm. A stale controller/physics stream stops the evaluation; native MCP holds
the last reference until the simulator is paused.

`--hand-face-rotation 0` evaluates fingers aligned with the box face edges;
the default remains mirrored 45-degree placement. Horizontal placement centers
the 15 cm face between the palm and distal support patches. `--palm-force N`
can raise the palm-first contact force target above the mass/friction requirement.
Compliant probe lifts allow up to 8 cm of vertical target motion but still require
observed lift, unsupported contact and both selected hand surfaces before carrying.
The 2026-09-12 horizontal 1 kg trials did not complete transport; these options
are evaluation controls, not a validated replacement for the diagonal pose.

The user confirmed XC330-T288-T and an external 1:1 shaft/MCP connection. Historical
`HAND` M288 calibration is deliberately not used. The [official T288 manual](https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/)
specifies input-supply current measurement, so current cannot be treated as
winding current with a constant torque multiplier. The **uncalibrated** motor
approximation uses the 11.1 V stall point (0.92 Nm, 0.80 A), 65 rpm no-load speed,
and assumed factory P900/I0/D0. It assumes ideal transmission/driver efficiency
and zero idle current; temperature, friction identification and firmware behavior
are not reproduced. At rest it models copper loss as
`I = 0.80 * (tau / 0.92)^2`; positive mechanical power adds `max(tau*dq,0)/11.1`.
Position error requests PWM, bounded by 133/885, with back-EMF damping. Thus the
modeled static PWM torque ceiling is 0.138 Nm, reached before 180 mA.
These are model assumptions, not measured force capability or continuous ratings.
The hardware runtime uses Position Mode 3 and a current **stop threshold**; the
simulator's torque saturation is an additional modeling approximation, not proof
that hardware Mode 3 enforces Goal Current.

Approach orientation uses the 30-degree hand shape and places fingertips at
least 5 mm behind the palm support plane. The palm bump and distal phalanx are deliberately placed together inside the
15 cm face. The fingers rotate -45/+45 degrees within the left/right box face. The radial box
center lies 1 cm forward of the palm support-patch centroid, so the weak distal spring does
not carry half the load to balance a 14.8 cm palm-to-tip lever. The vertical
window still centers the combined patches. At the nominal loaded shape, cropped
palm/distal support-band areas are about 228/201 mm2 per side. The last approach
segment moves laterally at box height, rather than descending diagonally. The loaded contact geometry uses MCP 30 / DIP 27 degrees. DIP 27 is the intended
**loaded** shape, never the spring rest angle or a commanded motor position. The
approach uses the unloaded DIP-zero shape and seats about its palm contact;
virtual seating is capped at 12 degrees and stopped by surface feedback. After both palms contact
at 0.5 N for a 1 kg box, each wrist rotates about its palm pivot to acquire distal
contact before adding further preload. MCP angle remains determined by the
motor/admittance and contact; it is not overwritten to achieve the target.
IK uses 64 hand spheres per side covering sampled geometry over the entire
MCP 20–40 / DIP -10–45 degree sampled geometry, with angular interpolation padding.
Safety buffers are not disabled. DIP has no invented mechanical hard stops;
The MCP and DIP ranges are monitored operating envelopes around MCP 30 / DIP 27,
not assertions about hardware travel. The DIP range surrounds the 27-degree loaded
shape. MCP physical stops remain 0–60 degrees. A -30–90 degree all-pose DIP envelope
blocked seating in a 1 kg trial before substantial DIP deflection. The all-pose
MCP 0–60 envelope also blocked further seating while actual MCP stayed near 30. Leaving
the evaluated angular envelope stops the trial instead of producing a
rigid-stop grasp force.
For new passive-DIP trials, each palm alone must exceed 6.131 N for 1 kg /
friction 1.2, while each distal tip provides auxiliary contact of at least 0.2 N.
Earlier recordings required 0.5 N on each patch and 6.131 N combined; their
metadata-specific scoring remains unchanged. A better approach target does not
itself establish successful carrying. Friction 1.2 is an assumption, not a
measured sandpaper/cardboard coefficient.

`compliance.json` records motor assumptions and IK envelopes.
`hand_motor.bin` has 23 float64 values (see metadata for older 15-value records) per native 50 Hz record:
`wall, sim, reference_wall, L_MCP(q,dq,tau,free_tau,reference,constraint_tau), R_MCP(...), L_DIP(q,dq,passive_tau,constraint_tau), R_DIP(...)`.
`hand_compliance_log.json` records filtered external torque, reference, contact
state, and modeled current. Video overlays distinguish modeled current from
native angles/contact forces. Broad MCP plate force is still recorded as an auxiliary metric. Pressure distribution/contact area is not measured
by MuJoCo's point contacts. Real force mapping needs T288 calibration with the
actual voltage, PID/PWM settings, mounting direction, known load and moment arm.


### Passive DIP spring assumptions

User-specified spring: SUS304, wire 0.8 mm, outside diameter 5 mm, six active coils,
180-degree unloaded leg opening. The installed preload is unknown. Using mean
coil diameter 4.2 mm and Young's modulus 193 GPa, the coil-only beam estimate is
`k = E*d^4/(64*D*N) = 0.04902 Nm/rad`. This neglects end-leg compliance and is not
a calibrated spring rate. [Spring-wire material reference](https://leecospring.com/springs/spring-materials-information/).

The baseline assumes unloaded legs correspond to URDF DIP zero. Native MuJoCo
supplies `tau = -k*(q + preload) - 0.0001*dq` with no DIP actuator, joint-position
overwrite or hard stop. Thus zero-preload 27-degree deformation gives about
0.0231 Nm of restoring torque; it does not mean that 27 degrees must occur at
any arbitrary applied force. Damping is an uncalibrated small value. Positive
`--dip-preload-deg` adds restoring preload (sensitivity range 0–30 degrees), not
an inferred fact about the installed hand. To establish hardware force, measure
its unloaded joint angle and torque/load at a known DIP moment arm.

Both palm and distal projected patches must retain at least 100 mm2 within the
15 cm face. These 1 mm support-band projections are geometry filters, not
measured pressure areas. The MCP middle plate remains an auxiliary contact.
The video and `metrics.json` include measured passive DIP angles separately from
the 27-degree design geometry.

Freshness is checked against a timestamp acquired **after** reading the records.
Checking against loop-start time could incorrectly reject samples published
between loop start and the reads, repeatedly resetting the contact stability
window. Inputs now record `contact_fresh`, `surface_force`, and `stable_contact_s`
so acquisition validity and insufficient contact can be distinguished.


For the passive-DIP box trial, robot elbow-out preference is 0.1 with a 0.12 m
lateral shoulder-to-elbow offset; the general VR defaults remain 0.05 / 0.06 m.
This reduced the forearm/torso boundary failures in the recorded approach.
`tune_gains.py --preload-limit .05` bounds each hand's virtual inward EE offset
from its contact pose at 50 mm. Contact feedback can stop compression earlier;
50 mm is not an unconditional target or a relaxation of self-collision checks.
The general evaluator and `ContactCarry` still default to 35 mm. Applied arm
gains, passive finger angles, force gates and the actual preload are recorded
for each profile; a force-gate pass alone is not a successful carry.


The additional PD profiles `arm150_wrist60` and `arm180_wrist60` retain wrist
60/4.5 and compare shoulder/elbow 150/7.5 and 180/9.0. They remain within the
existing simulation bridge gain bounds. In compliant trials, if tip contact is
weak after the seating-angle limit is reached, bounded normal preload remains
available even when total palm force is sufficient. The original surface-force
gates are retained. Before rotation, the compliant lift now checks actual box
rise above 35 mm and loss of support contact as well as stable surface contact.
The nominal 80 mm vertical command can extend to 120 mm to compensate compliant
deflection; recontact during carry pauses yaw and requests additional bounded
lift. This vertical allowance is separate from the 50 mm inward squeeze limit.

The 45-degree passive-DIP candidate is evaluated at box center height 0.8 m with
`--profile arm180_wrist60 --radius .37 --height .8 --mass 1 --compliant-hands
--preload-limit .05`. A 0.9 m initial center placed wrist coupled limits near
saturation before lifting. Height changes both supports and all task targets.
These are simulation candidates; consult each run's metrics for task success.

During compliant probe/lift/carry, an actual IK rejection can request a 0.5 mm
normal retreat if both selected patches remain in contact and each hand has
more than 110% of its required palm force. Lift/yaw are held for this proposal, and
it must pass the same IK gates. This releases excess virtual preload at the
hand-sphere boundary instead of accumulating squeeze that prevents transport.
No pressure retreat is allowed from this rule without observed force reserve.

The final 0.37 m / 0.8 m configuration completed pickup, 90-degree carry and
release in two native vendor trials (`20260912_034901_box_off_1kg` and
`20260912_035136_box_off_1kg`). Peak inward offsets were 18.47 and 24.67 mm;
the latter retreated to 21.67 mm for carrying. This is bounded simulation
evidence, not a hardware gain/force calibration or a general reliability claim.
A stable support contact within 10 mm of landing height for 0.5 s can trigger
opening before virtual vertical preload reaches zero. Landed opening retains
its 24 mm/s proposal rate despite prior lower-goal retries, with all IK gates
still active. Any hand-controller safety error invalidates task success.

## Arm damping limit correction (2026-09-12)

Saved `igris_ee_pose_v1` JSONL can be evaluated without a box using
`quest_trajectory_replay.py CAPTURE --profile 400_5` or `--profile vendor_kd5`.
The latter keeps vendor arm Kp (121.51457 shoulder/elbow, 30 wrist) with Kd 5/5/3.
Both use a forward HOME, free-base WALK_LOW, separated HMD waist yaw, current
fixed hands, and existing Cartesian/IK/SDK guards. Default `--mapping absolute`
preserves robot-pelvis EE poses and uses absolute pelvis HMD yaw for the waist.
Use `--mapping clutch` only for observations deliberately reanchored to HOME.
Absolute replay first attempts 300 bounded Cartesian steps toward the first pose;
`entry.jsonl` and metadata report whether it reached that pose, separately from
the original source clock. A controller trip stays latched during the remaining
input audit. Metrics distinguish FOLLOWING samples from post-stop holds.
The constant `seed_qpos` field is not sent to the robot. This is an ordered
file trajectory: all source targets are processed, sampling gaps hold the prior
target, and slow solves extend playback. It does not reconstruct missing grip
events or reproduce live packet-timeout/reference-jump behavior. Source jumps
are audited separately. Use `quest_trajectory_replay.py RESULT_DIR --render`
for per-target CSV errors and a target-overlay MP4 from recorded native qpos.

The gain experiment now accepts arm **Kd <= 5** only, including ready/clutch
commands. `--arm400-kd` rejects larger values before restarting the simulator;
the VR override and simulation bridge enforce the same bound. Historical Kp
presets in `tune_gains.py` retain their Kp but cap each arm group's Kd at 5;
`--profile vendor` therefore means vendor Kp with bounded Kd, not an unchanged
vendor gain set. The waist and leg balance-controller gains are unchanged.

Previous Kd 12/20/30 results and the old Kd20 recommendation are outside this
limit. Even previous wrist Kd3/4.5 trials used shoulder/elbow Kd above 5.
See [the Kd<=5 re-evaluation](results/20260912_kd5_review/report_ko.md) for the
historical audit, actual logged gains and fresh native vendor-stack trials.

## Box moment and MCP diagnosis (2026-09-12)

The read-only native observer now adds `physics.bin.wrenches`: float64 rows of
21 values **per box contact**, sampled with the existing 30 Hz observer:
`wall, sim, group, normal_N, position_world[3], box_COM_world[3], force_on_box_world[3], contact_torque_on_box_world[3], MCP_load_Nm, DIP_load_Nm, normal_on_box_world[3]`.
Groups 0–3 are left palm/tip/MCP plate/other; 4–7 are the right equivalents;
8 is static environment (stands or floor), 9 is other dynamic bodies. Sum only
one representation of the contacts. Box COM moment is
`cross(position - COM, force) + contact_torque`; MCP/DIP fields are the opposite
contact wrench projected onto the native joint axis, only for descendants of that
joint. These are native constraint forces, not host `mj_forward` force estimates.
The [MuJoCo contact convention](https://mujoco.readthedocs.io/en/stable/computation/#contact)
uses forces on geom 1 and contact-frame rows as world axes.

`test_wrench_observer.cpp` verifies the recorded world wrench against the native
free-box constraint force, and the projected nonzero MCP load against its joint
constraint. Compile it with the same include/library/rpath arguments as the
observer above, using `-o /tmp/test_wrench_observer` instead of `-shared -fPIC`.
Run via `docker exec -e LD_PRELOAD=/tmp/igris_record_physics.so
-e IGRIS_EVAL_LOG=/tmp/wrench_test -e IGRIS_EVAL_TRIGGER=/tmp/wrench_test.trigger
igris-sim /tmp/test_wrench_observer`.

ContactCarry now receives observed `box_tilt_rad` from the evaluator. A tilt over
10 degrees holds lift/yaw/preload immediately and fails after 0.15 s continuously
above the threshold. At landing, physical support contact stops downward target
accumulation; release requires support, COM height within 10 mm of the landing
height, and tilt at most 5 degrees for 0.5 s. Persistent unstable support contact
fails after 3 s. Zero **target** height no longer proves actual landing. These are
evaluation screening thresholds, not a general force-closure certificate.
Metrics also include box orientation when scoring stable carry and final landing.

MCP logs now separate native reference, outer admittance reference/contact anchor,
free torque, PWM limiting and current limiting; exceptions preserve the exact
native sample that tripped the hand guard. Arm Kp400 does not tune MCP: its
separate model uses position P900, PWM133 (or the explicit PWM400 sensitivity)
and the 180 mA runtime limit. The selected `HAND` compliance profile is **Mode 3**;
`igris/Hand` boot Mode 5 is a different entrypoint. The simulator's ideal current
clamp is stronger than the external current-stop behavior of Mode 3 hardware.
XC330 measures input supply current, as documented by
[ROBOTIS](https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/#current-limit38),
so the DC supply-current approximation is not a calibrated linear torque constant.
Neither real motor settings nor the passive DIP spring parameters were changed.
