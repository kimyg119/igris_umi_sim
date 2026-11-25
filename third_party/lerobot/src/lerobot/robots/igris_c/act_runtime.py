"""Joint-policy inference and sensor backends; no control backend imports."""

import json
import time
from pathlib import Path

from .act_policy import PolicyContract


def load_pi05_runtime_config(checkpoint):
    """Adapt only the two export fields absent from this inference runtime."""
    import draccus
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    raw = json.loads((checkpoint / "config.json").read_text())
    if not {"text_tokenizer_name", "optimizer_fused"} & raw.keys():
        return PI05Config.from_pretrained(checkpoint, local_files_only=True)
    if raw.pop("type") != "pi05":
        raise ValueError("Pi0.5 runtime config requires type pi05")
    tokenizer = raw.pop("text_tokenizer_name", "google/paligemma-3b-pt-224")
    if tokenizer != "google/paligemma-3b-pt-224" or not (checkpoint / "tokenizer").is_dir():
        raise ValueError("Pi0.5 export requires its saved PaliGemma tokenizer")
    if type(raw.pop("optimizer_fused", False)) is not bool:
        raise ValueError("Pi0.5 optimizer_fused must be a boolean training setting")
    # The saved processor selects the bundled tokenizer; no optimizer is built for inference.
    # draccus still rejects every other unsupported field.
    return draccus.decode(PI05Config, raw)


class ActInference:
    def __init__(
        self,
        checkpoint,
        dataset_info,
        device,
        *,
        temporal_ensemble_coeff=None,
        head_cam_attention=False,
        tactile_attention=False,
        task=None,
    ):
        import torch

        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.modeling_act import ACTPolicy

        config = json.loads((checkpoint / "config.json").read_text())
        self.contract = PolicyContract(config, dataset_info)
        self.device = torch.device(device)
        self.task = task
        overrides = {"device_processor": {"device": device}}
        if self.contract.policy_type in ("pi05", "groot"):
            if not isinstance(task, str) or not task.strip():
                raise ValueError("VLA inference requires the training task text")
            if temporal_ensemble_coeff is not None or head_cam_attention or tactile_attention:
                raise ValueError("ACT ensemble/attention options do not apply to VLA policies")
        if self.contract.policy_type == "groot":
            from .groot_runtime import load_groot_policy

            self.policy = load_groot_policy(checkpoint, config, self.device)
            overrides["groot_n1_7_vlm_encode_v1"] = {
                "model_name": str(checkpoint / "assets/vlm_processor"),
                "device": device,
            }
        elif self.contract.policy_type == "pi05":
            from accelerate import init_empty_weights
            from safetensors import safe_open
            from safetensors.torch import load_file

            from lerobot.policies.pi05.modeling_pi05 import PI05Policy

            policy_config = load_pi05_runtime_config(checkpoint)
            policy_config.device = None
            policy_config.dtype = "float32"
            policy_config.n_action_steps = 1  # Every result must use the current observation.
            policy_config.compile_model = False
            with init_empty_weights(include_buffers=False):
                self.policy = PI05Policy(policy_config)
            buffers = dict(self.policy.named_buffers())
            self.policy.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
            # Preserve computed rotary buffers in training precision; they are absent from the checkpoint.
            for name, value in buffers.items():
                owner, _, key = name.rpartition(".")
                setattr(self.policy.get_submodule(owner), key, value)
            self.policy.config.dtype = "bfloat16"
            weights = checkpoint / "model.safetensors"
            state = load_file(str(weights), device="cpu")
            with safe_open(weights, framework="pt", device="cpu") as stream:
                for alias, original in (stream.metadata() or {}).items():
                    if alias not in state and original in state:
                        state[alias] = state[original]
            expected = self.policy.state_dict()
            for name, value in state.items():
                if name in expected:
                    state[name] = value.to(dtype=expected[name].dtype)
            # The upstream Pi0.5 loader can swallow weight-loading failures; fail closed here.
            self.policy.load_state_dict(state, strict=True, assign=True)
            del state, expected
            self.policy.to(self.device).eval()
            self.policy.config.device = device
            if self.device.type == "cuda":
                torch.set_float32_matmul_precision("high")
                self.policy.model.sample_actions = torch.compile(
                    self.policy.model.sample_actions, mode="reduce-overhead"
                )
            overrides["tokenizer_processor"] = {"tokenizer_name": str(checkpoint / "tokenizer")}
        else:
            self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device).eval()
        self.attention_capture = None
        self.attention_maps = {}
        if head_cam_attention or tactile_attention:
            from .act_attention import ImageAttention, attention_image_names

            names = attention_image_names(
                self.contract.images,
                head_cam_attention=head_cam_attention,
                tactile_attention=tactile_attention,
            )
            self.attention_capture = ImageAttention(self.policy.model, names)
        if temporal_ensemble_coeff is not None:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            self.policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
            self.policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff, self.policy.config.chunk_size
            )
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides=overrides,
        )
        if self.contract.policy_type == "groot":
            from .groot_runtime import validate_groot_processors

            validate_groot_processors(self.pre, self.post, self.contract)
        self._install_forward_diagnostics()
        self.reset()

    def _install_forward_diagnostics(self):
        """Observe actual ACT forward or Pi0.5 chunk calls without changing action selection."""
        self._forward_hooks = []
        self._forward_diagnostics_available = False
        if self.contract.policy_type == "act":
            self._forward_hooks = [
                self.policy.model.register_forward_pre_hook(self._forward_started),
                self.policy.model.register_forward_hook(self._forward_returned),
            ]
            self._forward_diagnostics_available = True
        elif self.contract.policy_type == "pi05" and callable(getattr(self.policy, "predict_action_chunk", None)):
            predict = self.policy.predict_action_chunk

            def measured_predict(*args, **kwargs):
                self._forward_started(None, None)
                result = predict(*args, **kwargs)
                self._forward_returned(None, None, result)
                return result

            self.policy.predict_action_chunk = measured_predict
            self._forward_diagnostics_available = True

    def _forward_started(self, module, args):
        if self._call_forward_started_at is None:
            self._call_forward_started_at = time.monotonic()

    def _forward_returned(self, module, args, output):
        # PyTorch invokes this hook only when forward returns successfully.
        self._call_forwards += 1

    def reset(self):
        self.policy.reset()
        self._forward_count = 0
        self._call_forwards = 0
        self._call_forward_started_at = None
        self._forward_completed_at = None
        self._forward_duration_s = None
        self.last_diagnostics = {}
        self.attention_maps = {}
        if self.attention_capture is not None:
            self.attention_capture.clear()

    def __call__(self, observation):
        import torch

        from lerobot.policies.utils import prepare_observation_for_inference

        self.last_diagnostics = {}
        self._call_forwards = 0
        self._call_forward_started_at = None
        batch = prepare_observation_for_inference(
            dict(observation), self.device, task=getattr(self, "task", None), robot_type="igris_c"
        )
        self.attention_maps = {}
        if self.attention_capture is not None:
            self.attention_capture.clear()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda" and self.policy.config.use_amp,
            ),
        ):
            if self.contract.policy_type == "groot":
                # Decode every relative timestep against THIS observation, then take the fresh first target.
                chunk = self.post(self.policy.predict_action_chunk(self.pre(batch)))
                if tuple(chunk.shape) != (1, 16, 17) or not torch.isfinite(chunk).all():
                    raise ValueError("Invalid GR00T absolute action chunk")
                action = chunk[0, 0]
            else:
                # select_action preserves the saved temporal ensemble; post restores radians.
                action = self.post(self.policy.select_action(self.pre(batch)))[0]
        if self.attention_capture is not None:
            self.attention_maps = self.attention_capture.read()
        result = self.contract.action(action.float().cpu().numpy())
        if self._call_forwards:
            # CPU transfer synchronizes the selected CUDA action. This duration
            # includes postprocessing/transfer, not just GPU kernel execution.
            self._forward_completed_at = time.monotonic()
            self._forward_duration_s = self._forward_completed_at - self._call_forward_started_at
            self._forward_count += self._call_forwards
        source = "unknown"
        if getattr(self, "_forward_diagnostics_available", bool(self._forward_hooks)):
            source = "fresh_forward" if self._call_forwards else "cached_action_chunk"
        config = self.policy.config
        self.last_diagnostics = {
            "forward_count": self._forward_count,
            "forwards_this_call": self._call_forwards,
            "forward_completed_at": self._forward_completed_at,
            "forward_duration_s": self._forward_duration_s,
            "output_source": source,
            "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
            "n_action_steps": getattr(config, "n_action_steps", None),
            "chunk_size": getattr(config, "chunk_size", None),
        }
        return result


def make_sensors(path, contract):
    import draccus
    import yaml

    from lerobot.cameras.configs import CameraConfig, ColorMode
    from lerobot.cameras.gopro import GoProCameraConfig  # noqa: F401 (registry)
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 (registry)
    from lerobot.cameras.ros2 import Ros2CameraConfig  # noqa: F401 (registry)
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.robots.igris_c.tactile import Ros2TactileConfig, SerialTactileConfig, make_tactile_sensors
    from lerobot.utils.ros2_bundle import BUNDLE_FPS, Ros2Bundle
    from lerobot.utils.ros2_sensor import Ros2SensorConfig

    cfg = yaml.safe_load(path.read_text()) if isinstance(path, Path) else path
    if not isinstance(cfg, dict) or set(cfg) - {"cameras", "tactile", "ros2_bundle"}:
        raise ValueError("sensor YAML supports only cameras, tactile and ros2_bundle")
    cameras = make_cameras_from_configs(
        {k: draccus.decode(CameraConfig, v) for k, v in cfg.get("cameras", {}).items()}
    )
    if any(getattr(c.config, "color_mode", ColorMode.RGB) != ColorMode.RGB for c in cameras.values()):
        raise ValueError("ACT cameras must deliver RGB")
    tactile = make_tactile_sensors(
        {
            k: draccus.decode(Ros2TactileConfig | SerialTactileConfig, v)
            for k, v in cfg.get("tactile", {}).items()
        }
    )
    if set(cameras) & {k + "_heatmap" for k in tactile}:
        raise ValueError("duplicate camera/tactile feature")
    if set(cameras) | {k + "_heatmap" for k in tactile} != set(contract.images):
        raise ValueError("sensor names must exactly match checkpoint images")
    for key, sensor in tactile.items():
        shape = tuple(contract.images[key + "_heatmap"])
        pi05_tactile = (
            contract.policy_type == "pi05"
            and shape == (224, 224, 3)
            and contract.pi05_tactile_padding == (120, 320)
            and sensor.config.rows in (8, 12)
            and sensor.config.cols == 32
        )
        if shape != (80, 320, 3) and not pi05_tactile and not (
            contract.policy_type == "groot" and shape == (120, 320, 3) and sensor.config.rows == 12
        ):
            raise ValueError("Unsupported tactile sensor layout for the checkpoint image contract")
    bundle = None
    if cfg.get("ros2_bundle") is not None:
        if contract.fps != BUNDLE_FPS:
            raise ValueError("ROS2 bundle inference requires a 15 Hz checkpoint")
        bundle = Ros2Bundle.for_observation(
            draccus.decode(Ros2SensorConfig, cfg["ros2_bundle"]),
            {k: c.config for k, c in cameras.items()},
            {k: t.config for k, t in tactile.items()},
        )
    return cameras, tactile, bundle


def observe(
    backend, cameras, tactile, contract, *, bundle=None, include_joint_torque=False, preview_images=None,
    diagnostics=None,
):
    import cv2
    import numpy as np

    from lerobot.robots.igris_c.igris_c import _tactile_heatmap

    if preview_images is not None:
        preview_images.clear()
    if diagnostics is not None:
        diagnostics.clear()
    if bundle is not None:
        options = {"diagnostics": diagnostics} if diagnostics is not None else {}
        if preview_images is not None:
            options["include_all_tactile"] = True
        images, pressure, stamp = bundle.read(**options)
        frames = {name: (image, stamp) for name, image in images.items()}
        for name, (_, visual) in pressure.items():
            frames[name + "_heatmap"] = (_tactile_heatmap(visual, cv2.COLORMAP_VIRIDIS), stamp)
    else:
        from lerobot.cameras.ros2.camera_ros2 import Ros2Camera
        from .ros2_tactile import Ros2Tactile

        frames = {}
        if diagnostics is not None:
            diagnostics.update(mode="individual", sensors={})
        for name, camera in cameras.items():
            metadata = {}
            options = {"diagnostics": metadata} if diagnostics is not None and isinstance(camera, Ros2Camera) else {}
            frame, stamp = camera.async_read_with_timestamp(timeout_ms=150, **options)
            frames[name] = (frame, stamp)
            if diagnostics is not None:
                diagnostics["sensors"][name] = {"kind": "camera", "received_at": stamp, **metadata}
        for name, sensor in tactile.items():
            metadata = {}
            options = {"diagnostics": metadata} if diagnostics is not None and isinstance(sensor, Ros2Tactile) else {}
            _, visual, stamp = sensor.get_with_timestamp(**options)
            frames[name + "_heatmap"] = (_tactile_heatmap(visual, cv2.COLORMAP_VIRIDIS), stamp)
            if diagnostics is not None:
                diagnostics["sensors"][name + "_heatmap"] = {"kind": "tactile", "received_at": stamp, **metadata}
    if diagnostics is not None:
        for name, metadata in diagnostics.get("sensors", {}).items():
            metadata["model_input"] = name in contract.images
    if preview_images is not None:
        for name in set(frames) - set(contract.images):
            preview_images["observation.images." + name] = frames.pop(name)[0]
    state = backend.snapshot()
    now = backend.clock()
    stamp = now - state["ages_s"]["lowstate"]
    observation = contract.observation(
        state["joint_q"], frames, now=now, state_at=stamp, joint_tau_est=state.get("joint_tau_est")
    )
    if include_joint_torque:
        tau = np.asarray(state.get("joint_tau_est"), dtype=np.float32)
        if tau.shape != (31,) or not np.isfinite(tau).all():
            raise ValueError("joint_tau_est: expected 31 finite LowState torque values for recording")
        observation["observation.joint_tau_est_Nm"] = tau
    return observation, min(stamp, *(t for _, t in frames.values()))
