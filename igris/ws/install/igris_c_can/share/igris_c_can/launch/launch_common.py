import os

import yaml


def flatten_dict(src, parent_key="", out=None):
    if out is None:
        out = {}

    for key, value in src.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flatten_dict(value, full_key, out)
        else:
            out[full_key] = value
    return out


def load_robot_params_as_ros_dict(global_parameters):
    with open(global_parameters, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    return flatten_dict(config)


def _resolve_dev_repo_root():
    env_root = os.environ.get("IGRIS_C_REPO_ROOT", "").strip()
    if env_root and os.path.isdir(os.path.join(env_root, "configs", "robot")):
        return env_root

    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    while True:
        if os.path.isdir(os.path.join(cur, "configs", "robot")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    install_marker = f"{os.sep}install{os.sep}"
    if install_marker in here:
        ws_root = here.split(install_marker, 1)[0]
        candidate = os.path.join(ws_root, "src", "igris_c")
        if os.path.isdir(os.path.join(candidate, "configs", "robot")):
            return candidate

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_robot_config_path(config_file):
    config_file = (config_file or "").strip()
    repo_root = _resolve_dev_repo_root()
    dev_config_hint = os.path.join(repo_root, "configs", "robot", "dev_robot.yaml")

    if not config_file:
        raise RuntimeError(
            "Missing required launch argument: config_file\n"
            "Hint (installed): ros2 launch igris_c_can igris_c_can_control.launch.py "
            "config_file:=/robros/configs/robot.yaml\n"
            "Hint (dev): ros2 launch igris_c_can igris_c_can_control.launch.py "
            f"config_file:={dev_config_hint}"
        )

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            "Hint (installed): /robros/configs/robot.yaml\n"
            f"Hint (dev): {dev_config_hint}"
        )
    return config_file
