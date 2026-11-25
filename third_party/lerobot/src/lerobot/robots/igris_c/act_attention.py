"""Read-only decoder attention diagnostics; never fed back into the policy."""

import cv2
import numpy as np

ATTENTION_DESCRIPTION = (
    "Last decoder cross-attention, first action query, averaged over attention heads; "
    "selected image spatial memory tokens only. Each overlay uses independent per-frame min/max "
    "scaling, TURBO RGB colors and 50% opacity. Colors cannot compare modality importance. "
    "Mass is the sum of the original weights for that image. No black-border mask is applied. "
    "Encoder memory mixes modalities: this is not causal pixel attribution or ensemble attribution. "
    "Only the overlay video and token mass are saved, not raw attention weights."
)


def attention_image_names(images, *, head_cam_attention=False, tactile_attention=False):
    tactile = [name for name in images if "tactile" in name and name.endswith("_heatmap")]
    if head_cam_attention and "head_cam" not in images:
        raise ValueError("head_cam attention requires a head_cam checkpoint input")
    if tactile_attention and not tactile:
        raise ValueError("tactile attention requires tactile heatmap checkpoint inputs")
    return [
        name
        for name in images
        if (head_cam_attention and name == "head_cam") or (tactile_attention and name in tactile)
    ]


def validate_attention_map(value):
    grid = np.asarray(value, dtype=np.float32)
    if (
        grid.ndim != 2
        or not grid.size
        or not np.isfinite(grid).all()
        or np.any(grid < 0)
        or float(grid.sum()) > 1.0001
    ):
        raise ValueError("image attention must be a finite nonnegative spatial probability map")
    return grid


def validate_attention_maps(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("attention image names differ from enabled checkpoint inputs")
    maps = {name: validate_attention_map(value[name]).copy() for name in names}
    if sum(float(grid.sum()) for grid in maps.values()) > 1.0001:
        raise ValueError("image attention mass exceeds one")
    return maps


class ImageAttention:
    """Tap existing MHA output without changing its arguments, output or kernel."""

    def __init__(self, model, names):
        self.keys = [key.removeprefix("observation.images.") for key in model.config.image_features]
        if not names or set(names) - set(self.keys):
            raise ValueError("attention requires selected checkpoint image inputs")
        self.names = names
        self.prefix = 1 + bool(model.config.robot_state_feature) + bool(model.config.env_state_feature)
        self.clear()
        self.handles = [
            model.backbone.register_forward_hook(self._backbone),
            model.decoder.layers[-1].multihead_attn.register_forward_hook(self._attention),
        ]

    def clear(self):
        self.shapes = []
        self.weights = None

    def _backbone(self, _module, _inputs, output):
        self.shapes.append(tuple(output["feature_map"].shape[-2:]))

    def _attention(self, _module, _inputs, output):
        weights = output[1]
        if weights is None or weights.ndim != 3 or weights.shape[0] != 1:
            raise RuntimeError("image attention requires batch-one, head-averaged decoder weights")
        self.weights = weights[0, 0].detach()

    def read(self):
        if self.weights is None or len(self.shapes) != len(self.keys):
            raise RuntimeError("image attention was not captured for this inference")
        counts = [h * w for h, w in self.shapes]
        if self.weights.numel() != self.prefix + sum(counts):
            raise RuntimeError("image attention token layout differs from camera feature maps")
        weights = self.weights.float().cpu().numpy()
        maps, offset = {}, self.prefix
        for name, shape, count in zip(self.keys, self.shapes, counts, strict=True):
            if name in self.names:
                maps[name] = weights[offset : offset + count].reshape(shape)
            offset += count
        return validate_attention_maps(maps, self.names)

    def close(self):
        for handle in self.handles:
            handle.remove()


def attention_overlay(rgb, attention):
    grid = validate_attention_map(attention)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image attention overlay requires RGB uint8 HWC")
    span = float(np.ptp(grid))
    if span <= 1e-12:
        return rgb.copy()
    intensity = (grid - grid.min()) / span
    intensity = cv2.resize(intensity, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.rint(intensity * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.5, heatmap, 0.5, 0)
