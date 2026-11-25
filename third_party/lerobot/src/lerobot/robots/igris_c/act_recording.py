"""One direct-ACT run -> one local LeRobot episode, outside the DDS process."""

import json
import queue
import signal
import time
from pathlib import Path

import numpy as np

from .act_attention import (
    ATTENTION_DESCRIPTION,
    attention_image_names,
    attention_overlay,
    validate_attention_maps,
)
from .direct_dds import vector
from .igris_c import joint_names

SHOULDER_TAU_AXES = (16, 17, 23, 24)


def recording_features(contract, *, shoulder_torque=False, head_cam_attention=False, tactile_attention=False,
                       annotate_policy=False):
    names = joint_names()
    state_names = [names[i] + ".pos" for i in contract.state_indices]
    if shoulder_torque:
        state_names += [names[i] + ".tau_est_Nm" for i in SHOULDER_TAU_AXES]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(contract.action_indices),),
            "names": [names[i] + ".pos" for i in contract.action_indices],
        },
        "complementary_info.policy_action": {
            "dtype": "float32",
            "shape": (len(contract.owned),),
            "names": [names[i] + ".pos" for i in contract.owned],
        },
        "complementary_info.monotonic_s": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["oldest_observation", "observation_read", "last_command", "inference_finished"],
        },
        "complementary_info.policy_seq": {"dtype": "int64", "shape": (1,), "names": ["sequence"]},
    }
    if annotate_policy:
        features["complementary_info.policy_index"] = {
            "dtype": "int64", "shape": (1,), "names": ["rollout_provenance_policy_index"]}
    if getattr(contract, "joint_tau_indices", ()):
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (len(contract.joint_tau_indices),),
            "names": [names[i] for i in contract.joint_tau_indices],
        }
    for name, shape in contract.images.items():
        features["observation.images." + name] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    for name in attention_image_names(
        contract.images, head_cam_attention=head_cam_attention, tactile_attention=tactile_attention
    ):
        features[f"observation.images.{name}_attention"] = dict(features[f"observation.images.{name}"])
        features[f"observation.{name}_attention_mass"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": [f"{name}_token_mass"],
        }
    return features


def recording_frame(
    contract,
    observation,
    action,
    command,
    *,
    stamp,
    started,
    finished,
    seq,
    task,
    joint_tau_est=None,
    attention_maps=None,
    policy=None,
):
    # command is captured with the observation, before inference can advance the robot.
    if command[31] <= 0:
        return None  # Model warmup/preflight never becomes rollout data.
    state = observation["observation.state"].copy()
    if joint_tau_est is not None:
        tau = vector(joint_tau_est, "joint_tau_est")[list(SHOULDER_TAU_AXES)]
        state = np.concatenate((state, tau)).astype(np.float32)
    return {
        **{k: v.copy() for k, v in observation.items()},
        "observation.state": state,
        "action": np.asarray(command[:31], dtype=np.float32)[list(contract.action_indices)],
        "complementary_info.policy_action": np.asarray(action, dtype=np.float32),
        "complementary_info.monotonic_s": np.array([stamp, started, command[31], finished]),
        "complementary_info.policy_seq": np.array([seq], dtype=np.int64),
        **(
            {"_attention_maps": validate_attention_maps(attention_maps, attention_maps)}
            if attention_maps is not None
            else {}
        ),
        **({"_policy": dict(policy)} if policy is not None else {}),
        "task": (f"{task} | model={policy['id']} ({policy['name']}) | "
                 f"checkpoint={policy['model_sha256']}" if policy is not None else task),
    }


def recording_worker(cfg, contract, output, frames, ready, failed):
    # The parent stops DDS first and then requests a flush, including on Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    dataset = None
    display_started = False
    result = {"frames": 0, "saved_episodes": 0, "error": None, "encoder_dropped_frames": 0}
    stamps = []
    policies = []
    try:
        attention_names = attention_image_names(
            contract.images,
            head_cam_attention=cfg.get("head_cam_attention", False),
            tactile_attention=cfg.get("tactile_attention", False),
        )
        if cfg.get("record_dataset"):
            from lerobot.configs.video import RGBEncoderConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            fps = int(contract.fps)
            if fps != contract.fps:
                raise ValueError("LeRobot video recording requires an integer dataset fps")
            root = Path(cfg.get("dataset_root") or output / "dataset")
            dataset = LeRobotDataset.create(
                cfg.get("recording_repo_id", cfg["dataset_repo_id"]),
                fps=fps,
                features=recording_features(
                    contract,
                    shoulder_torque=cfg["input_mode"] == "live",
                    head_cam_attention=cfg.get("head_cam_attention", False),
                    tactile_attention=cfg.get("tactile_attention", False),
                    annotate_policy=cfg.get("annotate_policy", False),
                ),
                root=root,
                robot_type="igris_c",
                streaming_encoding=True,
                rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=23),
                encoder_threads=1,
            )
            result["dataset_root"] = str(root.resolve())
            (root / "rollout_provenance.json").write_text(
                json.dumps(
                    {
                        "input_mode": cfg["input_mode"],
                        "output_mode": cfg["output_mode"],
                        "checkpoint": str(Path(cfg["checkpoint"]).resolve()),
                        "run_config": str((output / "config.json").resolve()),
                        "command_log": str((output / "commands.jsonl").resolve()),
                        "state": (
                            "policy-order PJS positions (rad), then LowState.joint_state[].tau_est (Nm) "
                            "at axes 16,17,23,24: left shoulder roll/yaw, right shoulder roll/yaw; "
                            "positions and torque from the same LowState"
                            if cfg["input_mode"] == "live"
                            else "sample policy-order PJS positions (rad); no measured torque available"
                        ),
                        "policy_joint_tau_input": (
                            "observation.environment_state: LowState.joint_state[].tau_est (Nm), "
                            "in the checkpoint's named joint order; saved training normalization"
                            if getattr(contract, "joint_tau_indices", ())
                            else None
                        ),
                        "action": "latest successfully issued absolute PJS command at observation read (shadow: serialized only)",
                        "policy_action": "unnormalized absolute policy target before projection/interpolation",
                        "image_attention": ATTENTION_DESCRIPTION if attention_names else None,
                        "attention_images": attention_names,
                        "timing": "nominal-fps frame timestamps; actual clocks in complementary_info.monotonic_s",
                    },
                    indent=2,
                )
                + "\n"
            )
        if cfg.get("display_data"):
            from lerobot.utils.visualization_utils import init_visualization, log_visualization_data

            init_visualization("rerun", session_name="igris_act_rollout")
            display_started = True
        ready.set()
        while (frame := frames.get()) is not None:
            if cfg.get("annotate_policy"):
                policy = frame.pop("_policy")
                if policy not in policies:
                    policies.append(policy)
                    if dataset is not None:
                        provenance_path = dataset.root / "rollout_provenance.json"
                        provenance = json.loads(provenance_path.read_text())
                        provenance["policies"] = policies
                        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
                frame["complementary_info.policy_index"] = np.array([policies.index(policy)], np.int64)
            if attention_names:
                maps = validate_attention_maps(frame.pop("_attention_maps"), attention_names)
                for name, attention in maps.items():
                    frame[f"observation.images.{name}_attention"] = attention_overlay(
                        frame[f"observation.images.{name}"], attention
                    )
                    frame[f"observation.{name}_attention_mass"] = np.array([attention.sum()], np.float32)
            if display_started:
                log_visualization_data(
                    "rerun",
                    observation={k: v for k, v in frame.items() if k.startswith("observation.")},
                    action={
                        "sent_pjs": frame["action"],
                        "policy_pjs": frame["complementary_info.policy_action"],
                    },
                    compress_images=True,
                )
            if dataset is not None:
                dataset.add_frame(dict(frame))
                # The existing streaming encoder drops on overload: never silently accept such an episode.
                result["encoder_dropped_frames"] = sum(
                    dataset.writer._streaming_encoder._dropped_frames.values()
                )
                if result["encoder_dropped_frames"]:
                    raise RuntimeError("video encoder dropped frames; this episode is incomplete")
            stamps.append(float(frame["complementary_info.monotonic_s"][1]))
            result["frames"] += 1
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        failed.set()
    finally:
        try:
            if dataset is not None:
                try:
                    if dataset.has_pending_frames() and not result["encoder_dropped_frames"]:
                        dataset.save_episode()
                        result["saved_episodes"] = dataset.num_episodes
                finally:
                    dataset.finalize()
            if display_started:
                from lerobot.utils.visualization_utils import shutdown_visualization

                shutdown_visualization("rerun")
        except BaseException as exc:
            result["error"] = f"{result['error'] or ''} {type(exc).__name__}: {exc}".strip()
            failed.set()
        if len(stamps) > 1:
            result["observation_hz"] = (len(stamps) - 1) / (stamps[-1] - stamps[0])
            result["max_observation_gap_s"] = float(np.max(np.diff(stamps)))
        (output / "recording_result.json").write_text(json.dumps(result, indent=2) + "\n")


class RolloutRecorder:
    def __init__(self, ctx, cfg, contract, output):
        self.frames = ctx.Queue(maxsize=16)
        self.command = ctx.Array("d", 32)
        self.ready, self.failed = ctx.Event(), ctx.Event()
        self.output = output
        self.process = ctx.Process(
            target=recording_worker,
            args=(cfg, contract, output, self.frames, self.ready, self.failed),
            name="act-recording",
        )
        self.process.start()

    def wait_ready(self):
        deadline = time.monotonic() + 120
        while not self.ready.wait(0.05):
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("rollout recorder startup timed out")
        self.check()

    def check(self):
        if self.failed.is_set() or not self.process.is_alive():
            raise RuntimeError(f"rollout recorder failed; see {self.output / 'recording_result.json'}")

    def close(self):
        # The inference producer must have exited/flushed its queue before this sentinel.
        while self.process.is_alive():
            try:
                self.frames.put(None, timeout=0.1)
                break
            except queue.Full:
                if self.failed.is_set():
                    break
        while self.process.is_alive():
            self.process.join(timeout=1)
        self.frames.cancel_join_thread()
        self.frames.close()
        path = self.output / "recording_result.json"
        if not path.exists():
            raise RuntimeError(f"recording process exited without a result: {self.process.exitcode}")
        return json.loads(path.read_text())
