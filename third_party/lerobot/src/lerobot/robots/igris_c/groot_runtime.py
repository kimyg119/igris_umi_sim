"""Restore the portable IGRIS GR00T N1.7 bundle using local assets only."""

import json
from pathlib import Path

import draccus
import torch
from safetensors.torch import load_model

from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep, GrootN17PackInputsStep


class BundledGrootPolicy(GrootPolicy):
    def _create_groot_model(self):
        native = json.loads((Path(self.config.base_model_path) / "config.json").read_text())
        if native.get("architectures") != ["Gr00tN1d7"]:
            raise ValueError("Expected the bundled GR00T N1.7 architecture")
        for key in (
            "tune_llm",
            "tune_visual",
            "tune_projector",
            "tune_diffusion_model",
            "tune_vlln",
            "tune_top_llm_layers",
            "use_flash_attention",
            "num_inference_timesteps",
            "rtc_ramp_rate",
        ):
            value = getattr(self.config, key)
            if value is not None:
                native[key] = value
        model = GR00TN17(GR00TN17Config(**native), load_backbone_weights=False)
        self._cast_model_parameters_to_fp32(model)
        return model


def load_groot_policy(checkpoint, saved_config, device):
    from .act_policy import OWNED

    embodiment = json.loads((checkpoint / "embodiment/embodiment_contract.json").read_text())
    names = [f"waist_{i}.pos" for i in range(3)] + [
        f"{side}_arm_{i}.pos" for side in ("left", "right") for i in range(7)
    ]
    if (
        embodiment["joint_names"] != names
        or embodiment["fps"] != 15
        or embodiment["state_dim"] != 17
        or embodiment["action_dim"] != 17
        or embodiment["provenance"]["source_indices_zero_based"]
        != {"observation.state": list(OWNED), "action": list(OWNED)}
    ):
        raise ValueError("GR00T bundle does not describe the IGRIS waist/arms 17D mapping")
    config = dict(saved_config)
    # The bundled 0.6.2 config renamed model_params_fp32 to dtype; translate only that known schema.
    if config.pop("dtype", "float32") != "float32" or config.pop("optimizer_fused", None) is not None:
        raise ValueError("Expected the portable FP32 GR00T checkpoint configuration")
    config.pop("type")
    config.update(
        device=str(device), base_model_path=str(checkpoint / "assets/architecture"), model_params_fp32=True
    )
    policy = BundledGrootPolicy(draccus.decode(GrootConfig, config))
    missing, unexpected = load_model(policy, checkpoint / "model.safetensors", strict=True, device="cpu")
    if missing or unexpected:
        raise RuntimeError(f"GR00T checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return policy.requires_grad_(False).to(device).eval()


def validate_groot_processors(pre, post, contract):
    packs = [s for s in pre.steps if isinstance(s, GrootN17PackInputsStep)]
    decoders = [s for s in post.steps if isinstance(s, GrootN17ActionDecodeStep)]
    if len(packs) != 1 or len(decoders) != 1:
        raise ValueError("GR00T requires its saved state packer and relative-action decoder")
    pack, decode = packs[0], decoders[0]
    if (
        decode.pack_step is not pack
        or not decode.use_relative_action
        or decode.env_action_dim != 17
        or pack.embodiment_tag != "new_embodiment"
        or pack.embodiment_mapping.get("new_embodiment") != 10
        or pack.valid_action_horizon != 16
        or pack.state_horizon != 1
        or pack.training
        or set(pack.video_modality_keys or []) != set(contract.images)
        or not pack.normalize_min_max
        or not pack.use_percentiles
        or not pack.clip_outliers
        or pack.raw_stats != decode.raw_stats
        or pack.modality_config != decode.modality_config
    ):
        raise ValueError("GR00T saved preprocessing/relative decoding contract differs")
    groups = pack.modality_config["action"]
    if groups["modality_keys"] != ["single_arm"] or groups["action_configs"] != [
        {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
    ]:
        raise ValueError("GR00T must decode relative joint positions, not end-effector actions")
    state = pack.raw_stats["state"]["single_arm"]
    for flat_key, source_key in (("min", "q01"), ("max", "q99")):
        actual = torch.as_tensor(pack.stats["observation.state"][flat_key], dtype=torch.float32)
        expected = torch.as_tensor(state[source_key], dtype=torch.float32)
        if actual.shape != (17,) or not torch.isfinite(actual).all() or not torch.equal(actual, expected):
            raise ValueError("GR00T state normalization must use the saved training q01/q99")
    relative = pack.raw_stats["relative_action"]["single_arm"]
    low, high = (torch.tensor(relative[k], dtype=torch.float32) for k in ("min", "max"))
    if (
        low.shape != (40, 17)
        or high.shape != low.shape
        or not torch.isfinite(low).all()
        or not torch.isfinite(high).all()
        or (low > high).any()
    ):
        raise ValueError("GR00T requires finite per-timestep 40x17 relative-action min/max statistics")
